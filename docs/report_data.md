# BÁO CÁO 2: CÔNG TÁC DỮ LIỆU
## Thu thập và Làm sạch Dữ liệu, Phân tích Khám phá Dữ liệu (EDA)
### Ứng dụng: Hệ thống Tổng hợp Tiếng nói (Text-to-Speech) dựa trên kiến trúc VITS

---

## Tóm tắt

Báo cáo này trình bày quy trình thu thập, mô tả, làm sạch và phân tích khám phá đối với bộ dữ liệu được sử dụng để huấn luyện mô hình tổng hợp tiếng nói (Text-to-Speech – TTS) trong khuôn khổ đồ án. Dữ liệu sử dụng là một ngữ liệu giọng nói đơn người nói (single-speaker speech corpus) theo định dạng LJSpeech, gồm 13.100 cặp (văn bản, âm thanh), tổng thời lượng xấp xỉ 23,92 giờ. Báo cáo trình bày cơ sở lý thuyết cho từng bước xử lý dữ liệu — bao gồm phát hiện hoạt động giọng nói (Voice Activity Detection), loại bỏ giá trị ngoại lai bằng phương pháp khoảng liên phân vị (Interquartile Range), trích xuất đặc trưng âm học (spectrogram, đường bao cao độ F0) và biểu diễn ngữ âm (phoneme) — đồng thời tổng hợp các kết quả phân tích định lượng và trực quan hoá nhằm đánh giá chất lượng và tính phù hợp của dữ liệu đối với bài toán huấn luyện mô hình sinh tiếng nói.

---

## 1. Mô tả Dữ liệu (Data Descriptions)

### 1.1 Nguồn dữ liệu (Sources)

Trong các hệ thống tổng hợp tiếng nói dựa trên học sâu, dữ liệu huấn luyện đóng vai trò quyết định đối với chất lượng giọng nói tạo ra, do mô hình học cách ánh xạ từ không gian ngôn ngữ (văn bản/ngữ âm) sang không gian âm học (dạng sóng/spectrogram) hoàn toàn dựa trên các cặp mẫu quan sát được. Vì lý do đó, các ngữ liệu giọng nói dùng cho TTS thường được thu âm trong điều kiện kiểm soát (phòng cách âm, một người đọc duy nhất, văn bản được phân đoạn rõ ràng theo câu hoặc mệnh đề) nhằm giảm thiểu nhiễu và sự không đồng nhất giữa các mẫu.

Bộ dữ liệu sử dụng trong đồ án này được tổ chức theo **định dạng LJSpeech** — một chuẩn tổ chức dữ liệu phổ biến trong nghiên cứu TTS, gồm:

- Một tệp ánh xạ văn bản (transcript), trong đó mỗi dòng tương ứng với một định danh bản ghi âm và nội dung văn bản được đọc;
- Một thư mục chứa các tệp âm thanh dạng sóng (waveform) tương ứng 1–1 với từng dòng văn bản.

Đặc điểm của ngữ liệu là **đơn người nói (single-speaker)**: toàn bộ các đoạn ghi âm do một người đọc thực hiện, với nội dung văn bản được trích từ các tác phẩm phi hư cấu. Việc lựa chọn ngữ liệu đơn người nói phù hợp với mục tiêu huấn luyện một giọng nói (voice) cụ thể, thay vì mô hình đa người nói (multi-speaker) cần đặc trưng nhận dạng người nói (speaker embedding) bổ sung.

### 1.2 Kích thước và Định dạng (Size and Format)

Bộ dữ liệu được khảo sát có quy mô và định dạng như sau:

| Thuộc tính | Giá trị |
|---|---|
| Định dạng văn bản | Văn bản phân tách (delimited text), không có dòng tiêu đề, mỗi dòng gồm định danh và nội dung câu |
| Định dạng âm thanh | Tín hiệu số hoá dạng sóng (PCM), đơn kênh (mono), độ phân giải 16-bit |
| Tần số lấy mẫu (sample rate) | 22.050 Hz |
| Số lượng mẫu (utterance) | 13.100 |
| Số người nói | 1 |
| Tổng thời lượng âm thanh | Xấp xỉ 23,92 giờ |
| Dung lượng văn bản | Khoảng 1,4 MB |
| Dung lượng âm thanh | Khoảng 3,6 GB |

Sau khi qua giai đoạn tiền xử lý, dữ liệu gốc được chuyển đổi thành một biểu diễn trung gian phục vụ huấn luyện, bao gồm: (i) một tệp mô tả cấu hình ngữ liệu (chứa thông tin tần số lấy mẫu, bảng ánh xạ đơn vị ngữ âm, số người nói); (ii) một tệp danh mục mẫu huấn luyện theo định dạng JSON Lines, mỗi dòng tương ứng một mẫu đã được lượng hoá thành các trường đặc trưng; và (iii) các tệp tensor đã được tiền tính toán (cached tensors) lưu trữ dạng nhị phân nhằm tránh tính toán lại trong mỗi epoch huấn luyện.

### 1.3 Đặc trưng dữ liệu (Features)

Có thể phân loại đặc trưng của bộ dữ liệu thành hai nhóm: **đặc trưng thô** (thu được trực tiếp từ nguồn) và **đặc trưng đã qua xử lý** (được trích xuất phục vụ huấn luyện mô hình).

**Nhóm đặc trưng thô:**

| Đặc trưng | Kiểu dữ liệu | Diễn giải |
|---|---|---|
| Định danh mẫu | Chuỗi ký tự | Khoá liên kết giữa văn bản và tệp âm thanh |
| Văn bản (transcript) | Chuỗi ký tự (UTF-8) | Nội dung ngôn ngữ tự nhiên cần tổng hợp |
| Tín hiệu âm thanh | Mảng số nguyên 16-bit, miền thời gian | Tín hiệu giọng nói tương ứng với văn bản |

**Nhóm đặc trưng đã qua xử lý (đầu vào trực tiếp cho mô hình):**

| Đặc trưng | Kiểu dữ liệu | Cơ sở lý thuyết |
|---|---|---|
| Chuỗi đơn vị ngữ âm (phoneme sequence) | Dãy số nguyên (chỉ số trong từ điển ngữ âm) | Biểu diễn văn bản ở cấp độ âm vị học (phonological representation), giúp mô hình khái quát hoá tốt hơn so với biểu diễn ở cấp ký tự, đặc biệt với ngôn ngữ có quy tắc đọc không nhất quán |
| Tín hiệu đã chuẩn hoá (normalized waveform) | Mảng số thực trong [-1, 1] | Tín hiệu sau khi loại bỏ khoảng lặng và đưa về cùng thang biên độ |
| Phổ tần số (spectrogram) | Ma trận số thực 2 chiều (tần số × thời gian) | Biểu diễn miền tần số của tín hiệu thông qua biến đổi Fourier thời gian ngắn (Short-Time Fourier Transform), là mục tiêu huấn luyện cho thành phần sinh âm thanh |
| Đường bao cao độ (F0 / pitch contour) | Dãy số thực theo khung thời gian (frame) | Biểu diễn tần số cơ bản của tín hiệu giọng nói, phản ánh ngữ điệu (prosody/intonation) |
| Định danh người nói (speaker id) | Số nguyên (tuỳ chọn) | Dùng cho mô hình đa người nói; không có ý nghĩa thống kê trên ngữ liệu đơn người nói hiện tại |

---

## 2. Làm sạch và Xử lý Dữ liệu (Data Cleaning and Processing)

### 2.1 Cơ sở lý thuyết và quy trình thực hiện

Việc làm sạch dữ liệu trong các bài toán TTS hướng đến ba mục tiêu chính: (1) loại bỏ các mẫu không hợp lệ về mặt kỹ thuật (tệp thiếu, hỏng, hoặc rỗng); (2) loại bỏ các mẫu có sự không tương thích giữa văn bản và âm thanh (ví dụ tốc độ đọc bất thường, gợi ý khả năng ghép sai cặp văn bản–âm thanh); và (3) chuẩn hoá tín hiệu và văn bản về một không gian biểu diễn đồng nhất để mô hình học hiệu quả hơn. Quy trình xử lý được thực hiện theo các bước sau:

**Bước 1 — Kiểm tra tính hợp lệ của tệp tin.**
Mỗi bản ghi được kiểm tra sự tồn tại và kích thước của tệp âm thanh tương ứng. Các bản ghi tham chiếu đến tệp không tồn tại hoặc có kích thước bằng không được loại khỏi tập dữ liệu trước khi đưa vào các bước xử lý tiếp theo, nhằm tránh lỗi runtime và sai lệch số liệu thống kê ở các bước sau.

**Bước 2 — Phát hiện hoạt động giọng nói và cắt khoảng lặng (Voice Activity Detection).**
Một thách thức phổ biến trong dữ liệu thu âm là sự tồn tại của các khoảng lặng (silence) ở đầu và cuối mỗi đoạn ghi âm, phát sinh từ độ trễ phản xạ của người đọc hoặc thiết bị ghi âm. Các khoảng lặng này, nếu không được loại bỏ, sẽ làm sai lệch các thống kê liên quan đến thời lượng nói thực tế (ví dụ tốc độ đọc), đồng thời gây lãng phí tài nguyên tính toán trong quá trình huấn luyện do mô hình phải xử lý các đoạn không chứa thông tin ngữ âm. Để giải quyết vấn đề này, hệ thống áp dụng một mô hình học sâu phát hiện hoạt động giọng nói (cụ thể là kiến trúc Silero VAD, dựa trên mạng hồi quy có cổng — gated recurrent network), hoạt động trên tín hiệu lấy mẫu lại ở 16 kHz, để xác định ranh giới giữa các đoạn có và không có giọng nói. Phần tín hiệu nằm ngoài ranh giới phát hiện được sẽ bị loại bỏ trước khi tín hiệu được lấy mẫu lại theo tần số đích phục vụ huấn luyện.

**Bước 3 — Loại bỏ mẫu có tốc độ đọc bất thường (outlier detection).**
Tốc độ đọc của một mẫu được định nghĩa là tỷ số giữa số ký tự văn bản (loại trừ dấu câu, vì dấu câu không phát âm và không đóng góp vào thời lượng nói) và thời lượng nói thực tế (xác định qua VAD ở Bước 2). Về mặt thống kê, tốc độ đọc giữa các mẫu của cùng một người nói được kỳ vọng tuân theo một phân phối tương đối hẹp; các mẫu có tốc độ đọc lệch quá xa khỏi phân phối này thường là dấu hiệu của lỗi ghép cặp văn bản–âm thanh, lỗi cắt đoạn, hoặc nhiễu trong quá trình thu âm. Phương pháp **khoảng liên phân vị (Interquartile Range – IQR)** được áp dụng để xác định ngưỡng loại bỏ: gọi Q1, Q3 là phân vị thứ nhất và thứ ba của phân phối tốc độ đọc, IQR = Q3 − Q1, các mẫu có tốc độ nằm ngoài khoảng [Q1 − k·IQR, Q3 + k·IQR] (với hệ số k được lựa chọn thực nghiệm) bị loại khỏi tập huấn luyện. Đây là một phương pháp phát hiện ngoại lai phi tham số (không giả định phân phối chuẩn), phù hợp với dữ liệu thời lượng/tốc độ vốn thường có phân phối lệch.

**Bước 4 — Chuẩn hoá văn bản và biểu diễn ngữ âm.**
Văn bản được chuẩn hoá về một quy ước viết hoa/viết thường thống nhất, sau đó được chuyển đổi thành chuỗi đơn vị ngữ âm thông qua một công cụ chuyển văn bản–ngữ âm (grapheme-to-phoneme). Việc biểu diễn ở cấp độ ngữ âm, thay vì cấp độ ký tự, giúp mô hình tránh phải học các quy tắc đọc bất quy tắc của ngôn ngữ tự nhiên (ví dụ từ đồng âm khác nghĩa, từ vay mượn), từ đó cải thiện khả năng khái quát hoá. Trong quá trình này, hệ thống đồng thời theo dõi và ghi nhận các đơn vị ngữ âm không có trong từ điển ánh xạ đã định nghĩa trước (missing phonemes), nhằm phục vụ việc rà soát chất lượng văn bản gốc.

**Bước 5 — Giới hạn độ dài mẫu.**
Các mẫu có chuỗi ngữ âm vượt quá một ngưỡng độ dài tối đa được loại khỏi quá trình huấn luyện. Việc giới hạn này có hai tác dụng: giảm yêu cầu bộ nhớ khi gộp mẫu vào lô (batch) — do bộ nhớ tiêu thụ tỷ lệ với độ dài chuỗi dài nhất trong lô — và loại bỏ các mẫu có độ dài bất thường có thể là dấu hiệu lỗi phân đoạn dữ liệu.

**Bước 6 — Kiểm tra tính toàn vẹn của dữ liệu đã lưu đệm (cache).**
Do quá trình tính toán đặc trưng (chuẩn hoá âm thanh, phổ tần số, F0) được lưu đệm dưới dạng tệp nhị phân để tái sử dụng giữa các lần huấn luyện, các tệp này cần được kiểm tra định kỳ về tính toàn vẹn (ví dụ do tiến trình ghi tệp bị gián đoạn), và bị loại bỏ nếu không thể đọc lại được, buộc hệ thống tính toán lại từ dữ liệu gốc.

### 2.2 Thách thức gặp phải và giải pháp

**Thách thức về phân tích cú pháp văn bản chứa ký tự đặc biệt.**
Trong quá trình đọc tệp văn bản phân tách, một thách thức phát sinh từ sự xuất hiện của ký tự dấu ngoặc kép trong nội dung câu (ví dụ các đoạn trích dẫn trong văn bản gốc). Khi sử dụng cơ chế phân tích cú pháp CSV theo cấu hình mặc định — trong đó dấu ngoặc kép được hiểu là ký tự bao quanh một trường dữ liệu (quote character) — một số lượng lẻ các ký tự ngoặc kép trong toàn văn bản dẫn đến việc trình phân tích hiểu sai ranh giới giữa các trường, khiến nhiều dòng liên tiếp bị gộp nhầm thành một bản ghi duy nhất. Hệ quả là số lượng bản ghi đọc được bị suy giảm đáng kể so với số lượng thực tế, đồng thời nội dung của các bản ghi bị gộp trở nên không chính xác. Giải pháp được áp dụng là tắt cơ chế xử lý ký tự bao trường (vô hiệu hoá quote-handling) khi phân tích cú pháp văn bản dạng này, khôi phục đúng số lượng và nội dung bản ghi. Thách thức này minh chứng cho tầm quan trọng của việc kiểm tra chéo (cross-validation) giữa số lượng bản ghi văn bản và số lượng tệp âm thanh tương ứng như một bước kiểm soát chất lượng cơ bản nhưng thiết yếu.

**Thách thức về đo lường thời lượng nói thực tế.**
Việc sử dụng trực tiếp thời lượng tệp âm thanh (tổng số mẫu chia tần số lấy mẫu) làm chỉ số "thời lượng nói" dẫn đến sai lệch hệ thống, do bao gồm cả khoảng lặng không mang thông tin ngữ âm. Giải pháp là tách biệt hai khái niệm: thời lượng tệp (file duration) và thời lượng nói thực tế (speech duration, xác định qua VAD), và chỉ sử dụng khái niệm thứ hai cho các phân tích liên quan đến tốc độ đọc.

**Thách thức về sự lệch khung (frame misalignment) giữa các đặc trưng âm học.**
Các phương pháp trích xuất đặc trưng khác nhau (biến đổi Fourier cho phổ tần số, thuật toán DIO/StoneMask cho đường bao cao độ) có thể cho ra số khung thời gian (frame) không hoàn toàn khớp nhau cho cùng một đoạn tín hiệu, do khác biệt trong cách xử lý biên (padding) ở mỗi thuật toán. Sự lệch khung này, nếu không được xử lý, sẽ gây ra lỗi căn chỉnh khi kết hợp nhiều đặc trưng làm đầu vào mô hình. Giải pháp là áp dụng một bước căn chỉnh hậu kỳ (cắt hoặc đệm theo chiến lược lặp biên — edge padding) để đảm bảo mọi đặc trưng của cùng một mẫu có cùng số khung thời gian.

**Thách thức về tính không đồng nhất độ dài giữa các mẫu trong một lô huấn luyện.**
Do bản chất chuỗi (sequential) của cả văn bản và âm thanh, các mẫu trong cùng một lô huấn luyện có độ dài khác nhau. Giải pháp tiêu chuẩn trong học sâu là đệm không (zero-padding) toàn bộ mẫu trong lô theo độ dài lớn nhất, kết hợp lưu trữ độ dài thực tế của từng mẫu để mô hình có thể bỏ qua phần đệm trong quá trình tính toán hàm mất mát (loss masking).

---

## 3. Phân tích Khám phá Dữ liệu (Exploratory Data Analysis)

### 3.1 Mục tiêu và phương pháp

Phân tích khám phá dữ liệu (EDA) trong bối cảnh huấn luyện mô hình TTS nhằm trả lời ba câu hỏi cốt lõi: (i) dữ liệu có đặc điểm phân phối ra sao, có tồn tại điểm bất thường (anomaly) cần xử lý không; (ii) các giả định về tính nhất quán dữ liệu (ví dụ tương quan giữa độ dài văn bản và thời lượng âm thanh) có được thoả mãn không; và (iii) các tham số huấn luyện (kích thước lô, ngưỡng lọc) nên được lựa chọn như thế nào dựa trên đặc điểm thực nghiệm của dữ liệu. Phân tích được thực hiện trực tiếp trên toàn bộ 13.100 mẫu của bộ dữ liệu.

### 3.2 Phân phối thời lượng âm thanh

Thời lượng của các mẫu âm thanh trong bộ dữ liệu có các đặc trưng thống kê như sau: giá trị nhỏ nhất 1,11 giây, giá trị lớn nhất 10,10 giây, trung bình 6,57 giây, trung vị 6,76 giây, độ lệch chuẩn 2,19 giây.

| Khoảng thời lượng | Số lượng mẫu | Tỷ lệ |
|---|---|---|
| Dưới 2 giây | 272 | 2,1% |
| 2–5 giây | 3.041 | 23,2% |
| 5–8 giây | 5.720 | 43,7% |
| 8–10 giây | 3.877 | 29,6% |
| Trên 10 giây | 190 | 1,4% |

![Phân phối thời lượng âm thanh](data_processing_assets/duration_hist.png)

*Hình 1. Histogram phân phối thời lượng của 13.100 mẫu âm thanh.*

Kết quả cho thấy phân phối thời lượng tập trung trong khoảng 5–8 giây, với độ lệch (skewness) không đáng kể và không xuất hiện các giá trị ngoại lai cực đoan (ví dụ mẫu có thời lượng gần 0 hoặc vượt quá vài chục giây). Điều này hàm ý rằng dữ liệu gốc đã được phân đoạn theo đơn vị câu hoặc mệnh đề một cách tương đối nhất quán trước khi được công bố, là một tiền đề thuận lợi cho việc huấn luyện mô hình mà không cần can thiệp phân đoạn lại.

### 3.3 Phân phối độ dài văn bản

Độ dài văn bản (tính theo số ký tự) có các đặc trưng thống kê: giá trị nhỏ nhất 12 ký tự, giá trị lớn nhất 187 ký tự, trung bình 99,9 ký tự, trung vị 102 ký tự.

| Khoảng độ dài (ký tự) | Số lượng mẫu | Tỷ lệ |
|---|---|---|
| Dưới 50 | 1.147 | 8,8% |
| 50–100 | 5.012 | 38,3% |
| 100–150 | 6.189 | 47,2% |
| 150–200 | 752 | 5,7% |

![Phân phối độ dài văn bản](data_processing_assets/text_length_hist.png)

*Hình 2. Histogram phân phối độ dài văn bản (số ký tự) của 13.100 mẫu.*

Phân phối độ dài văn bản có dạng tương đối đối xứng, tập trung quanh khoảng 100–150 ký tự, tương ứng với độ dài của một câu hoặc một mệnh đề hoàn chỉnh trong văn bản tự nhiên. Không quan sát thấy mẫu văn bản rỗng hoặc có độ dài bất thường lớn, sau khi đã loại trừ ảnh hưởng của lỗi phân tích cú pháp được trình bày ở mục 2.2.

### 3.4 Tương quan giữa độ dài văn bản và thời lượng âm thanh

![Tương quan độ dài văn bản và thời lượng âm thanh](data_processing_assets/text_len_vs_duration.png)

*Hình 3. Biểu đồ phân tán (scatter plot) giữa độ dài văn bản và thời lượng âm thanh trên một mẫu con gồm 800 quan sát.*

Biểu đồ phân tán cho thấy một xu hướng tương quan dương rõ rệt giữa độ dài văn bản và thời lượng âm thanh, phù hợp với kỳ vọng vật lý: văn bản dài hơn đòi hỏi thời gian đọc lâu hơn. Mức độ phân tán quan sát được xung quanh xu hướng tuyến tính phản ánh sự khác biệt tự nhiên về tốc độ đọc giữa các đoạn (do ngữ điệu, độ phức tạp của từ vựng, hoặc khoảng ngắt nghỉ trong câu). Các điểm dữ liệu lệch xa khỏi xu hướng chính — tức các mẫu có tỷ lệ độ dài văn bản/thời lượng âm thanh bất thường — chính là đối tượng mà phương pháp lọc theo khoảng liên phân vị (Bước 3, mục 2.1) nhằm phát hiện và loại bỏ. Quan sát này cung cấp một cơ sở định lượng trực quan, hỗ trợ và xác nhận tính hợp lý của tiêu chí lọc theo tốc độ đọc đã trình bày trong phần làm sạch dữ liệu.

### 3.5 Phân tích trực quan tín hiệu ở cấp độ mẫu đơn lẻ

![Waveform và spectrogram của một mẫu](data_processing_assets/example_waveform_spectrogram.png)

*Hình 4. Dạng sóng (trên) và phổ tần số theo thời gian — spectrogram (dưới) của một mẫu âm thanh đại diện.*

Quan sát ở cấp độ mẫu đơn lẻ cho thấy sự tồn tại rõ rệt của các khoảng lặng ở đầu và cuối đoạn ghi âm — đúng như đã lập luận về sự cần thiết của bước phát hiện hoạt động giọng nói (VAD) trong quy trình làm sạch dữ liệu (mục 2.1, Bước 2). Phổ tần số (spectrogram) cho thấy năng lượng tập trung chủ yếu ở vùng tần số thấp đến trung bình, đặc trưng của tín hiệu giọng nói tự nhiên, không xuất hiện các dải nhiễu trải rộng đồng nhất theo toàn bộ phổ tần số (dấu hiệu thường gặp của nhiễu thiết bị hoặc nhiễu nền), cho thấy chất lượng thu âm của ngữ liệu là tương đối tốt.

### 3.6 Tổng hợp và nhận định

Kết quả phân tích khám phá dữ liệu cho phép rút ra các nhận định sau:

1. **Về tính đầy đủ và toàn vẹn**: bộ dữ liệu không tồn tại mẫu thiếu hoặc hỏng nghiêm trọng trong phạm vi khảo sát; thách thức chính phát sinh từ lỗi định dạng văn bản (ký tự đặc biệt trong CSV) hơn là từ chất lượng nội dung dữ liệu.
2. **Về tính nhất quán**: các phân phối thời lượng âm thanh và độ dài văn bản đều có dạng đơn đỉnh (unimodal), không lệch cực đoan, và thể hiện tương quan vật lý hợp lý giữa hai miền dữ liệu (văn bản và âm thanh) — đây là điều kiện cần để một mô hình sinh tiếng nói có thể học được ánh xạ ổn định.
3. **Về tính ứng dụng cho huấn luyện**: do bộ dữ liệu là đơn người nói, các cơ chế xử lý đa người nói trong hệ thống (gán định danh người nói, cân bằng số mẫu giữa người nói) không phát huy hiệu lực thống kê trên dữ liệu hiện tại, nhưng cần được lưu ý khi mở rộng sang ngữ liệu đa người nói trong các giai đoạn phát triển tiếp theo.
4. **Về hạn chế của phân tích**: các thống kê được trình bày dựa trên đặc trưng cấp độ tín hiệu và văn bản (thời lượng, độ dài), chưa bao gồm các phân tích sâu hơn về phân phối ngữ âm (ví dụ tần suất xuất hiện của từng đơn vị ngữ âm) hoặc đặc trưng cao độ giọng nói (phân phối F0 toàn cục) — đây là hướng phân tích bổ sung có thể thực hiện trong các báo cáo tiếp theo.

---

## Kết luận

Báo cáo đã trình bày một cách hệ thống quy trình mô tả, làm sạch và phân tích khám phá đối với bộ dữ liệu huấn luyện mô hình tổng hợp tiếng nói. Trên cơ sở lý thuyết về xử lý tín hiệu giọng nói (phát hiện hoạt động giọng nói, trích xuất đặc trưng âm học) và các phương pháp thống kê mô tả/loại bỏ ngoại lai (khoảng liên phân vị), kết hợp với kết quả phân tích định lượng và trực quan hoá trên dữ liệu thực tế, có thể kết luận rằng bộ dữ liệu hiện có đáp ứng được các yêu cầu cơ bản về chất lượng để phục vụ huấn luyện mô hình, đồng thời các thách thức phát sinh trong quá trình xử lý đã được nhận diện rõ nguyên nhân và có giải pháp xử lý tương ứng.
