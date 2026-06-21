# Report 2 — Data Tasks: Data Collection and Cleaning + Exploratory Data Analysis

> Đối tượng: bộ dữ liệu huấn luyện mô hình Text-to-Speech (kiến trúc VITS, dựa trên Piper), đặt tại [data/](data/) và pipeline xử lý trong [src/python/piper_train/](src/python/piper_train/).

---

## 1. Data Descriptions

### 1.1 Sources

- **Vị trí trong project**: [data/metadata.csv](data/metadata.csv) (transcript) + [data/wavs/](data/wavs/) (audio gốc). Thư mục `data/` được khai báo trong [.gitignore:12](.gitignore#L12) (`/data/`) nên **không được commit vào repo** — đây là dữ liệu được tải/đặt cục bộ bởi người dùng để chạy pipeline huấn luyện.
- **Cách đưa vào pipeline**: script [.wsl_run.sh:19-24](.wsl_run.sh#L19-L24) gọi `piper_train.preprocess` với `--dataset-format ljspeech --single-speaker`, tức dữ liệu được tổ chức theo đúng **LJSpeech format** (xem mục 1.2).
- **Nhận diện nguồn dữ liệu**: tên file theo mẫu `LJ0XX-0XXX.wav` và cấu trúc `metadata.csv` (id|text, không header) khớp chính xác với định dạng của **bộ ngữ liệu công khai LJSpeech** (các đoạn đọc audiobook phi hư cấu từ LibriVox, do một người đọc duy nhất thực hiện, thuộc phạm vi public domain). Bộ này được dùng phổ biến làm dataset chuẩn để train/fine-tune các mô hình TTS (bao gồm Piper).
- **Pipeline đọc dữ liệu** (mã nguồn thật): hàm `ljspeech_dataset()` tại [preprocess.py:421-473](src/python/piper_train/preprocess.py#L421-L473) — đọc `metadata.csv` bằng `csv.reader(delimiter="|")`, dò tìm file `.wav` tương ứng ở nhiều vị trí khả dĩ (`wav/`, `wavs/`, có/không hậu tố `.wav`).

### 1.2 Size and Format

Số liệu dưới đây được đo trực tiếp trên dữ liệu thật trong `data/` (không phải ước lượng):

| Thuộc tính | Giá trị |
|---|---|
| Định dạng transcript | CSV, delimiter `\|`, không header, 2 cột (`id`, `text`) — single-speaker LJSpeech format |
| Định dạng audio | WAV, PCM 16-bit, mono, **22 050 Hz** |
| Số utterance (dòng/file audio) | **13 100** |
| Dung lượng `metadata.csv` | **1.4 MB** |
| Dung lượng `wavs/` | **3.6 GB** |
| Tổng thời lượng audio | **≈ 23.92 giờ** (đo từ header toàn bộ 13 100 file `.wav`) |
| Số speaker | 1 (single-speaker) |

→ Sau khi qua `preprocess.py`, dữ liệu được chuyển thành format huấn luyện gồm:
- `config.json`: schema bộ dữ liệu (sample rate, `phoneme_id_map`, `num_speakers`,...)
- `dataset.jsonl`: 1 dòng JSON/utterance (`phoneme_ids`, `audio_norm_path`, `audio_spec_path`, `audio_f0_path`,...)
- cache `*.pt` (tensor PyTorch): audio đã chuẩn hoá, spectrogram, F0

(Chi tiết format này được mô tả trong [TRAINING.md:51-96](TRAINING.md#L51-L96).)

### 1.3 Features

**(a) Đặc trưng thô (raw, từ `metadata.csv` + `.wav`)**

| Feature | Kiểu dữ liệu | Mô tả |
|---|---|---|
| `id` | `string` | Tên file audio (không kèm `.wav`), ví dụ `LJ001-0001` |
| `text` | `string` (UTF-8) | Transcript gốc của câu nói |
| `audio` (waveform) | `int16` array, sample rate 22 050 Hz, mono | Tín hiệu âm thanh thô |

**(b) Đặc trưng sau xử lý (engineered features, dùng trực tiếp để train)** — định nghĩa tại [vits/dataset.py:14-31](src/python/piper_train/vits/dataset.py#L14-L31) (`Utterance`, `UtteranceTensors`):

| Feature | Kiểu dữ liệu | Sinh ra từ |
|---|---|---|
| `phoneme_ids` | `List[int]` → `LongTensor` | Phonemize text bằng espeak-ng ([preprocess.py:303-314](src/python/piper_train/preprocess.py#L303-L314)) |
| `audio_norm` | `FloatTensor [-1,1]` | Trim-silence (VAD) + resample ([norm_audio/__init__.py:23-94](src/python/piper_train/norm_audio/__init__.py#L23-L94)) |
| `spectrogram` | `FloatTensor [n_freq, n_frame]` | STFT trên `audio_norm` ([mel_processing.py:40-76](src/python/piper_train/vits/mel_processing.py#L40-L76)) |
| `f0` | `FloatTensor [n_frame]` | Pitch contour, pyworld DIO+StoneMask ([norm_audio/__init__.py:111-151](src/python/piper_train/norm_audio/__init__.py#L111-L151)) |
| `speaker_id` | `Optional[int]` | Gán theo thứ hạng số utterance/speaker ([preprocess.py:152-163](src/python/piper_train/preprocess.py#L152-L163)) — không dùng vì dataset là single-speaker |

---

## 2. Data Cleaning and Processing

### 2.1 Các bước làm sạch & xử lý (theo thứ tự pipeline thực tế)

1. **Loại bản ghi thiếu/hỏng file** — [preprocess.py:462-469](src/python/piper_train/preprocess.py#L462-L469): nếu `wav_path` không tồn tại hoặc `size == 0` → bỏ qua, log warning. Định nghĩa lý do loại tương ứng `ExcludeReason.MISSING` / `ExcludeReason.EMPTY` trong [filter_utterances.py:28-32](src/python/piper_train/filter_utterances.py#L28-L32).

2. **Lọc theo tốc độ nói bất thường (speaking-rate outlier)** — [filter_utterances.py](src/python/piper_train/filter_utterances.py):
   - Đo thời lượng "có lời nói" thực tế bằng `ffmpeg` decode → 16 kHz mono → Silero VAD (dòng 192-240).
   - Tính `rate = số ký tự không dấu câu / thời gian nói` (dòng 44-49).
   - Dùng **IQR (interquartile range)** theo speaker để xác định ngưỡng: `lower = Q1 - 2×IQR`, `upper = Q3 + 2×IQR` (dòng 109-131); utterance ngoài ngưỡng bị gắn `ExcludeReason.LOW`/`HIGH` và loại khỏi `metadata.csv` đã lọc.

3. **Voice Activity Detection (VAD) — cắt khoảng lặng đầu/cuối**:
   - Mô hình **Silero VAD** (ONNX) wrap trong [norm_audio/vad.py:8-55](src/python/piper_train/norm_audio/vad.py#L8-L55).
   - Thuật toán `trim_silence()` ([norm_audio/trim.py:8-55](src/python/piper_train/norm_audio/trim.py#L8-L55)) quét audio theo chunk 480 samples @16 kHz, xác định vùng có giọng nói (prob ≥ threshold), giữ thêm 2 chunk đệm trước/sau.
   - Áp dụng trong `cache_norm_audio()` ([norm_audio/__init__.py:23-94](src/python/piper_train/norm_audio/__init__.py#L23-L94)): trim trên bản 16 kHz, rồi load lại bằng sample rate đích (22 050 Hz) — vừa cắt lặng, vừa resample.

4. **Chuẩn hoá & kiểm tra văn bản / phoneme**:
   - Chuẩn hoá casing (`lower`/`upper`/`casefold`/giữ nguyên) — [preprocess.py:272-282](src/python/piper_train/preprocess.py#L272-L282).
   - Phonemize bằng espeak-ng, theo dõi `missing_phonemes` (phoneme không có trong bảng id) — [preprocess.py:298-334](src/python/piper_train/preprocess.py#L298-L334).
   - Kiểm tra rời: [check_phonemes.py](src/python/piper_train/check_phonemes.py) đọc `dataset.jsonl` đã xử lý, lập bảng phoneme dùng/thiếu kèm mã Unicode.

5. **Giới hạn độ dài câu** — [vits/dataset.py:92-122](src/python/piper_train/vits/dataset.py#L92-L122): khi load dataset để train, utterance có `len(phoneme_ids) > max_phoneme_ids` (tham số `--max-phoneme-ids`) bị bỏ qua, tránh OOM và loại câu dài bất thường.

6. **Dọn cache tensor hỏng** — [clean_cached_audio.py:30-44](src/python/piper_train/clean_cached_audio.py#L30-L44): quét toàn bộ `*.pt` trong cache, thử `torch.load()`, xoá file lỗi nếu chạy với `--delete`.

### 2.2 Khó khăn gặp phải & cách giải quyết

**Khó khăn 1 — Dấu ngoặc kép (`"`) trong transcript làm hỏng việc parse CSV.**
Khi phân tích `metadata.csv` thật bằng `csv.reader` với cấu hình quoting mặc định (giống `preprocess.py:436-438` đang dùng), số dòng đọc được tụt từ **13 100 xuống 12 911** — tức **189 dòng bị gộp sai** vào dòng trước. Nguyên nhân: file chứa **1 052 ký tự `"`** (số lẻ trong nhiều đoạn văn bản, ví dụ `"forty-two line Bible"`); module `csv` mặc định coi `"` là quote-char (`QUOTE_MINIMAL`), nên khi gặp dấu `"` mở mà không có dấu đóng tương ứng trước khi file kết thúc nhánh đó, nó **nuốt luôn các dòng `|`-delimiter tiếp theo vào một field**, làm sai lệch cả `id` và `text` của nhiều utterance liên tiếp.
→ **Cách giải quyết khi phân tích dữ liệu**: đọc file với `csv.reader(..., delimiter="|", quoting=csv.QUOTE_NONE)` để tắt cơ chế quote, giữ đúng 13 100 dòng và độ dài text hợp lý (xem mục 3). Đây cũng là điểm cần lưu ý nếu mở rộng `preprocess.py` cho các bộ dữ liệu khác có chứa dấu `"` trong văn bản, để tránh dataset.jsonl bị sinh sai lệch âm thầm (silent data corruption).

**Khó khăn 2 — Phân biệt "khoảng lặng" với "không có giọng nói" khi đo thời lượng thật.**
Nếu chỉ dùng `duration = nframes / framerate` của file `.wav`, số liệu sẽ bị lẫn khoảng lặng đầu/cuối (do người đọc dừng trước/sau khi ghi âm), khiến chỉ số "tốc độ nói" (rate) không phản ánh đúng người đọc.
→ Giải pháp trong code: dùng VAD (Silero) để xác định đoạn có giọng nói thật trước khi tính `rate` ([filter_utterances.py:222-240](src/python/piper_train/filter_utterances.py#L222-L240)), và cũng dùng VAD để trim khi chuẩn hoá audio cache ([norm_audio/__init__.py:56-64](src/python/piper_train/norm_audio/__init__.py#L56-L64)).

**Khó khăn 3 — F0 (pitch) bị lệch số frame so với spectrogram.**
Thư viện `pyworld` luôn trả về số frame F0 lệch +1 so với số frame mel-spectrogram cho cùng `hop_length`/`sample_rate` (xác minh thực nghiệm, ghi rõ trong docstring).
→ Giải pháp: cắt/pad (`edge` mode) chuỗi F0 để khớp đúng `num_mel_frames` ([norm_audio/__init__.py:142-146](src/python/piper_train/norm_audio/__init__.py#L142-L146)), và nội suy log-domain cho các đoạn vô thanh (F0=0) để mô hình dự đoán F0 không học cách hội tụ về 0 ([norm_audio/__init__.py:97-108](src/python/piper_train/norm_audio/__init__.py#L97-L108)).

**Khó khăn 4 — Độ dài audio/text không đồng nhất giữa các utterance.**
Audio dao động 1.1s–10.1s, text 12–187 ký tự (xem mục 3) — không thể gộp batch trực tiếp.
→ Giải pháp: `UtteranceCollate` ([vits/dataset.py:137-224](src/python/piper_train/vits/dataset.py#L137-L224)) zero-pad toàn batch theo độ dài lớn nhất (`max_phonemes_length`, `max_spec_length`, `max_audio_length`) và lưu kèm `*_lengths` để mô hình bỏ qua phần pad.

---

## 3. Exploratory Data Analysis (EDA)

EDA được thực hiện trực tiếp trên dữ liệu thật (`data/metadata.csv` + toàn bộ 13 100 file trong `data/wavs/`).

### 3.1 Phân phối thời lượng audio

| Thống kê | Giá trị |
|---|---|
| Số mẫu | 13 100 |
| Min | 1.11 s |
| Max | 10.10 s |
| Mean | 6.57 s |
| Median | 6.76 s |
| Std. dev | 2.19 s |
| Tổng thời lượng | 23.92 giờ |

Phân bố theo khoảng:

| Khoảng | Số utterance |
|---|---|
| < 2s | 272 |
| 2–5s | 3 041 |
| 5–8s | 5 720 |
| 8–10s | 3 877 |
| > 10s | 190 |

![Audio duration distribution](data_processing_assets/duration_hist.png)

**Nhận xét**: phân phối lệch nhẹ, tập trung quanh 5–8 giây; rất ít utterance dưới 2s hoặc trên 10s. Đây là dải thời lượng hợp lý cho audiobook reading, không có outlier cực đoan (ví dụ audio 0s hoặc hàng trăm giây) — cho thấy bộ dữ liệu đã được người tạo gốc (LJSpeech) phân đoạn (segment) khá sạch.

### 3.2 Phân phối độ dài transcript

(Đọc đúng với `quoting=QUOTE_NONE` — xem "Khó khăn 1" ở mục 2.2)

| Thống kê | Giá trị |
|---|---|
| Số mẫu | 13 100 |
| Min | 12 ký tự |
| Max | 187 ký tự |
| Mean | 99.9 ký tự |
| Median | 102 ký tự |

| Khoảng (ký tự) | Số utterance |
|---|---|
| < 50 | 1 147 |
| 50–100 | 5 012 |
| 100–150 | 6 189 |
| 150–200 | 752 |

![Transcript length distribution](data_processing_assets/text_length_hist.png)

**Nhận xét**: đa số câu dài 50–150 ký tự (~1 câu hoặc 1 mệnh đề), phù hợp với cấu trúc audiobook được cắt theo câu/cụm câu. Không có transcript rỗng hoặc cực dài (so với "outlier giả" 6293 ký tự gây ra bởi lỗi parse CSV ở mục 2.2 — đã loại trừ).

### 3.3 Tương quan độ dài transcript ↔ thời lượng audio

![Text length vs duration](data_processing_assets/text_len_vs_duration.png)

**Nhận xét**: có tương quan dương rõ rệt giữa số ký tự và thời lượng audio (câu dài hơn → đọc lâu hơn), đúng như kỳ vọng vật lý. Các điểm lệch xa khỏi xu hướng chính (ví dụ text ngắn nhưng audio dài, hoặc ngược lại) chính là các utterance mà `filter_utterances.py` sẽ gắn cờ qua chỉ số `rate` (mục 2.1, bước 2) — đây là cơ sở định lượng cho bước data cleaning theo IQR.

### 3.4 Ví dụ trực quan hoá một utterance (waveform & spectrogram)

![Example waveform and spectrogram](data_processing_assets/example_waveform_spectrogram.png)

**Nhận xét**: waveform cho thấy rõ khoảng lặng ở đầu/cuối file — đúng là phần mà bước VAD trim-silence (mục 2.1, bước 3) cần loại bỏ trước khi đưa vào model. Spectrogram thể hiện năng lượng tần số thấp chiếm ưu thế (giọng nói tự nhiên), không có dải nhiễu bất thường (ví dụ nhiễu tần số cao trải đều toàn bộ clip), cho thấy chất lượng ghi âm khá sạch.

### 3.5 Phát hiện khác từ thống kê pipeline (không cần chạy lại preprocessing)

| Phân tích | Vị trí code | Mục đích |
|---|---|---|
| Số utterance/speaker | [preprocess.py:140-155](src/python/piper_train/preprocess.py#L140-L155) | Dataset hiện tại là single-speaker (1 speaker, 13 100 utterance) |
| Phoneme dùng/thiếu | [check_phonemes.py:10-51](src/python/piper_train/check_phonemes.py#L10-L51) | Đánh giá độ phủ bảng phoneme sau khi phonemize bằng espeak-ng |
| Số utterance loại do vượt `max_phoneme_ids` | [vits/dataset.py:97-122](src/python/piper_train/vits/dataset.py#L97-L122) | Đánh giá ảnh hưởng của tham số giới hạn độ dài câu khi train |
| Train/Val/Test split | [vits/lightning.py:142-160](src/python/piper_train/vits/lightning.py#L142-L160) | `random_split` theo `validation_split` (mặc định 0.1) và `num_test_examples` (mặc định 5) |

### 3.6 Kết luận EDA

- Bộ dữ liệu (LJSpeech format, 13 100 utterance, ~23.92 giờ audio, 1 speaker, 22 050 Hz mono) có chất lượng tương đối đồng đều: không có audio rỗng/hỏng trong mẫu kiểm tra, thời lượng và độ dài transcript phân bố hợp lý, tương quan đúng kỳ vọng vật lý.
- Rủi ro chính nằm ở **bước parsing CSV** (ký tự `"` trong text) và **khoảng lặng đầu/cuối audio** — cả hai đã được pipeline xử lý (hoặc cần xử lý) như mô tả ở mục 2.
- Vì là single-speaker, các bước liên quan multi-speaker (gán `speaker_id`, cân bằng số utterance giữa speaker) trong pipeline không phát huy tác dụng trên bộ dữ liệu này nhưng vẫn tồn tại sẵn trong code để mở rộng.

---

## Phụ lục — Bảng tra cứu file code theo chức năng

| File | Chức năng chính |
|---|---|
| [preprocess.py](src/python/piper_train/preprocess.py) | Thu thập, parse metadata, phonemize, điều phối toàn bộ tiền xử lý |
| [filter_utterances.py](src/python/piper_train/filter_utterances.py) | Làm sạch theo speaking-rate (IQR) |
| [check_phonemes.py](src/python/piper_train/check_phonemes.py) | EDA bảng phoneme sử dụng/thiếu |
| [clean_cached_audio.py](src/python/piper_train/clean_cached_audio.py) | Dọn cache `.pt` hỏng |
| [select_speaker.py](src/python/piper_train/select_speaker.py) | Trích/lọc dữ liệu theo speaker |
| [norm_audio/vad.py](src/python/piper_train/norm_audio/vad.py) | Mô hình Silero VAD |
| [norm_audio/trim.py](src/python/piper_train/norm_audio/trim.py) | Thuật toán cắt khoảng lặng dựa trên VAD |
| [norm_audio/__init__.py](src/python/piper_train/norm_audio/__init__.py) | Chuẩn hoá audio, cache spectrogram & F0 |
| [vits/mel_processing.py](src/python/piper_train/vits/mel_processing.py) | STFT / mel-spectrogram |
| [vits/dataset.py](src/python/piper_train/vits/dataset.py) | `PiperDataset`, lọc theo độ dài phoneme, gộp batch (padding) |
| [vits/lightning.py](src/python/piper_train/vits/lightning.py) | Chia train/val/test, `DataLoader` |
| [TRAINING.md](TRAINING.md) | Mô tả định dạng dataset, `config.json`, `dataset.jsonl` |
