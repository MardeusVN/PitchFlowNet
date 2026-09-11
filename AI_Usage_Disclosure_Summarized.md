# AI Usage Disclosure — Bản tóm tắt theo mục đích sử dụng

Tài liệu này công bố các tương tác AI trong Dự án "Đồ án" theo dạng mục đích sử dụng và tóm tắt nội dung prompt, thay vì xuất nguyên văn toàn bộ hội thoại. Cách trình bày này giúp thể hiện đầy đủ phạm vi sử dụng AI nhưng không làm lộ source code, đường dẫn, checkpoint, dữ liệu nội bộ hoặc cấu hình có thể dùng để tái tạo pipeline.

> **Ghi chú phạm vi:** Dự án được tổng hợp là hệ thống Text-to-Speech (Banhmi-TTS / EdgeTTS) — bao gồm nghiên cứu Related Work, thiết kế ablation study, tiền xử lý dữ liệu, huấn luyện mô hình và vận hành mã nguồn. Các nhóm mục đích bên dưới được xây dựng theo nội dung thực tế của các phiên làm việc mà hệ thống có thể truy cập được, thay vì một danh mục cố định.

## Phạm vi tài liệu

- Tổng số 59 tương tác thuộc quá trình thực hiện đồ án được tổng hợp, trải trên 3 phiên làm việc.
- Không công khai prompt gốc hoặc câu trả lời đầy đủ của AI.
- Các tương tác được nhóm theo mục đích sử dụng.
- Trong từng nhóm, các tương tác được sắp xếp theo thời gian.
- Nội dung tóm tắt không chứa chi tiết triển khai nội bộ.
- Thời gian được ghi theo timestamp của phiên vì hệ thống không cung cấp timestamp riêng cho từng prompt.

---

## 1. Nghiên cứu Related Work & viết Gap Statement

**Cách AI được sử dụng:**
AI hỗ trợ soạn thảo và tinh chỉnh các đoạn "gap statement" trong phần Related Work, mô tả hạn chế của các kiến trúc nền tảng, dựa trên yêu cầu và quyết định phạm vi nội dung của nhóm.

### Tương tác 1
- **Người thực hiện:** Trần Hoàng Tuấn Hưng
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Viết Related Work / Gap Analysis
- **Tóm tắt nội dung prompt:** Yêu cầu AI cung cấp lại bản gap statement song ngữ Anh–Việt về hạn chế của vocoder truyền thống đã thảo luận trước đó, để đưa vào phần Related Work.
- **Cách sử dụng và kiểm chứng:** Nhóm rà soát nội dung AI với tài liệu gốc và kết quả thực nghiệm.

### Tương tác 2
- **Người thực hiện:** Trần Hoàng Tuấn Hưng
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Viết Related Work / Gap Analysis
- **Tóm tắt nội dung prompt:** Xác nhận nội dung gap về vocoder đã ổn, đề nghị chuyển sang thảo luận gap liên quan đến kiến trúc VITS2.
- **Cách sử dụng và kiểm chứng:** Nhóm rà soát nội dung AI với tài liệu gốc và kết quả thực nghiệm.

### Tương tác 3
- **Người thực hiện:** Trần Hoàng Tuấn Hưng
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Viết Related Work / Gap Analysis
- **Tóm tắt nội dung prompt:** Hỏi có thể áp dụng toàn bộ các cải tiến kiến trúc thay vì một phần, và liệu có thể paraphrase phần mô tả hạn chế đã có sẵn trong tài liệu tham khảo để trích dẫn ngắn gọn hơn.
- **Cách sử dụng và kiểm chứng:** Nhóm rà soát nội dung AI với tài liệu gốc và kết quả thực nghiệm.

### Tương tác 4
- **Người thực hiện:** Trần Hoàng Tuấn Hưng
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Viết Related Work / Gap Analysis
- **Tóm tắt nội dung prompt:** Hỏi liệu kết quả trên tập dữ liệu một giọng nói có thể so sánh trực tiếp với các công bố tham chiếu vì cùng dùng chung một bộ dữ liệu công khai.
- **Cách sử dụng và kiểm chứng:** Nhóm đối chiếu với yêu cầu đồ án, nhu cầu người dùng và phạm vi triển khai.

### Tương tác 5
- **Người thực hiện:** Trần Hoàng Tuấn Hưng
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Viết Related Work / Gap Analysis
- **Tóm tắt nội dung prompt:** Đề xuất tách rõ phần "Dataset" (số liệu thống kê) và phần "Preprocessing" (mô tả phương pháp) trong paper theo cách trình bày phổ biến của các bài báo lớn.
- **Cách sử dụng và kiểm chứng:** Nhóm rà soát nội dung AI với tài liệu gốc và kết quả thực nghiệm.

### Tương tác 6
- **Người thực hiện:** Trần Hoàng Tuấn Hưng
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Viết Related Work / Gap Analysis
- **Tóm tắt nội dung prompt:** Yêu cầu viết gap statement (một câu, theo văn phong đã dùng ở phần trước) cho phần so sánh liên quan đến cơ chế alignment/mô hình hóa thời lượng.
- **Cách sử dụng và kiểm chứng:** Nhóm rà soát nội dung AI với tài liệu gốc và kết quả thực nghiệm.

### Tương tác 7
- **Người thực hiện:** Trần Hoàng Tuấn Hưng
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Viết Related Work / Gap Analysis
- **Tóm tắt nội dung prompt:** Đề xuất gap chỉ nên mô tả hạn chế của kiến trúc gốc, chưa cần nêu giải pháp cụ thể (để dành cho phần Methods).
- **Cách sử dụng và kiểm chứng:** Nhóm rà soát nội dung AI với tài liệu gốc và kết quả thực nghiệm.

### Tương tác 8
- **Người thực hiện:** Trần Hoàng Tuấn Hưng
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Viết Related Work / Gap Analysis
- **Tóm tắt nội dung prompt:** Xác nhận đã hoàn tất hai phần gap, còn thiếu gap cuối liên quan đến đặc trưng ngôn điệu (F0).
- **Cách sử dụng và kiểm chứng:** Nhóm rà soát nội dung AI với tài liệu gốc và kết quả thực nghiệm.

### Tương tác 9
- **Người thực hiện:** Trần Hoàng Tuấn Hưng
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Viết Related Work / Gap Analysis
- **Tóm tắt nội dung prompt:** Xác nhận bắt đầu viết gap thứ ba liên quan đến đặc trưng ngôn điệu.
- **Cách sử dụng và kiểm chứng:** Nhóm rà soát nội dung AI với tài liệu gốc và kết quả thực nghiệm.

### Tương tác 10
- **Người thực hiện:** Trần Hoàng Tuấn Hưng
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Viết Related Work / Gap Analysis
- **Tóm tắt nội dung prompt:** Yêu cầu viết thêm một phiên bản gap cho kịch bản tương lai, khi nhóm áp dụng đầy đủ các cải tiến kiến trúc đã thảo luận.
- **Cách sử dụng và kiểm chứng:** Nhóm rà soát nội dung AI với tài liệu gốc và kết quả thực nghiệm.

### Tương tác 11
- **Người thực hiện:** Trần Hoàng Tuấn Hưng
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Viết Related Work / Gap Analysis
- **Tóm tắt nội dung prompt:** Xác nhận chọn viết gap theo phiên bản áp dụng đầy đủ các cải tiến kiến trúc.
- **Cách sử dụng và kiểm chứng:** Nhóm rà soát nội dung AI với tài liệu gốc và kết quả thực nghiệm.

---

## 2. Phân tích kiến trúc mô hình & thuật ngữ kỹ thuật

**Cách AI được sử dụng:**
AI được dùng để tra cứu, đối chiếu và giải thích các thành phần kỹ thuật của kiến trúc mô hình (vocoder, decoder, đặc trưng ngôn điệu) nhằm phục vụ việc ra quyết định thiết kế và viết tài liệu.

### Tương tác 12
- **Người thực hiện:** Trần Hoàng Tuấn Hưng
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Phân tích kiến trúc mô hình
- **Tóm tắt nội dung prompt:** Yêu cầu liệt kê các thành phần cải tiến chính của một kiến trúc tham chiếu và xác nhận nhóm đã áp dụng những thành phần nào trong số đó.
- **Cách sử dụng và kiểm chứng:** Nhóm kiểm tra lại bằng tài liệu mô hình, source code hiện có và kết quả thử nghiệm.

### Tương tác 13
- **Người thực hiện:** Trần Hoàng Tuấn Hưng
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Phân tích kiến trúc mô hình
- **Tóm tắt nội dung prompt:** Hỏi kiến trúc tham chiếu còn có cải tiến kỹ thuật nào khác ngoài các điểm đã thảo luận.
- **Cách sử dụng và kiểm chứng:** Nhóm kiểm tra lại bằng tài liệu mô hình, source code hiện có và kết quả thử nghiệm.

### Tương tác 14
- **Người thực hiện:** Trần Hoàng Tuấn Hưng
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Phân tích kiến trúc mô hình
- **Tóm tắt nội dung prompt:** Hỏi sự khác biệt về kiến trúc giữa hệ thống nền (baseline hiện tại của nhóm) và kiến trúc tham chiếu gốc.
- **Cách sử dụng và kiểm chứng:** Nhóm kiểm tra lại bằng tài liệu mô hình, source code hiện có và kết quả thử nghiệm.

### Tương tác 15
- **Người thực hiện:** Trần Hoàng Tuấn Hưng
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Phân tích kiến trúc mô hình
- **Tóm tắt nội dung prompt:** Yêu cầu giải thích lý do việc bổ sung đặc trưng ngôn điệu (F0) mang lại đóng góp lớn về mặt kỹ thuật.
- **Cách sử dụng và kiểm chứng:** Nhóm kiểm tra lại bằng tài liệu mô hình, source code hiện có và kết quả thử nghiệm.

### Tương tác 16
- **Người thực hiện:** Trần Hoàng Tuấn Hưng
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Phân tích kiến trúc mô hình
- **Tóm tắt nội dung prompt:** Yêu cầu xác minh lại thành phần discriminator được sử dụng trong vocoder của nhóm (bao gồm những loại discriminator nào).
- **Cách sử dụng và kiểm chứng:** Nhóm kiểm tra lại bằng tài liệu mô hình, source code hiện có và kết quả thử nghiệm.

### Tương tác 17
- **Người thực hiện:** Trần Hoàng Tuấn Hưng
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Phân tích kiến trúc mô hình
- **Tóm tắt nội dung prompt:** Đề xuất phương pháp: liệt kê từng thành phần cải tiến kèm lý do kỹ thuật cụ thể, làm cơ sở để viết gap statement chính xác hơn.
- **Cách sử dụng và kiểm chứng:** Nhóm kiểm tra lại bằng tài liệu mô hình, source code hiện có và kết quả thử nghiệm.

---

## 3. Thiết kế Ablation Study & phương pháp thực nghiệm

**Cách AI được sử dụng:**
AI hỗ trợ phản biện và tinh chỉnh thiết kế thí nghiệm ablation, bao gồm phạm vi các biến thể, chiến lược huấn luyện (from-scratch vs. kế thừa trọng số) và cách xử lý phương sai do seed ngẫu nhiên.

### Tương tác 18
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-07-17 (phiên trước)
- **Mục đích sử dụng AI:** Thiết kế Ablation Study
- **Tóm tắt nội dung prompt:** Giải thích mục đích chạy một cấu hình huấn luyện rút gọn nhằm kiểm chứng riêng mức đóng góp của đặc trưng ngôn điệu (F0) vào kết quả tổng thể của cấu hình đầy đủ.
- **Cách sử dụng và kiểm chứng:** Nhóm đánh giá đề xuất theo nguồn lực, thời gian và tiêu chí chấm đồ án.

### Tương tác 19
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Thiết kế Ablation Study
- **Tóm tắt nội dung prompt:** Hỏi lý do vì sao một thành phần cải tiến kiến trúc (liên quan đến nhiễu trong quá trình alignment) không được đưa vào phạm vi ablation.
- **Cách sử dụng và kiểm chứng:** Nhóm đánh giá đề xuất theo nguồn lực, thời gian và tiêu chí chấm đồ án.

### Tương tác 20
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Thiết kế Ablation Study
- **Tóm tắt nội dung prompt:** Xác nhận quyết định giữ nguyên phạm vi ablation hiện tại, ghi nhận thành phần chưa áp dụng vào phần Limitations của paper.
- **Cách sử dụng và kiểm chứng:** Nhóm xem xét đề xuất của AI trước khi quyết định áp dụng.

### Tương tác 21
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Thiết kế Ablation Study
- **Tóm tắt nội dung prompt:** Đặt lại câu hỏi liệu phạm vi ablation hiện tại có phải lựa chọn hợp lý hay cần mở rộng thêm.
- **Cách sử dụng và kiểm chứng:** Nhóm đánh giá đề xuất theo nguồn lực, thời gian và tiêu chí chấm đồ án.

### Tương tác 22
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Thiết kế Ablation Study
- **Tóm tắt nội dung prompt:** Hỏi về cách một công trình tham chiếu thực hiện đánh giá ablation — liệu họ có huấn luyện riêng từng biến thể mô hình.
- **Cách sử dụng và kiểm chứng:** Nhóm đánh giá đề xuất theo nguồn lực, thời gian và tiêu chí chấm đồ án.

### Tương tác 23
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Thiết kế Ablation Study
- **Tóm tắt nội dung prompt:** Hỏi chi tiết cách công trình tham chiếu huấn luyện các biến thể ablation: từ đầu hay kế thừa trọng số.
- **Cách sử dụng và kiểm chứng:** Nhóm đánh giá đề xuất theo nguồn lực, thời gian và tiêu chí chấm đồ án.

### Tương tác 24
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Thiết kế Ablation Study
- **Tóm tắt nội dung prompt:** Hỏi ý kiến nên huấn luyện từ đầu hay kế thừa trọng số cho từng biến thể ablation, dựa trên kinh nghiệm thực tế cho thấy số bước huấn luyện ngắn không đủ để mô hình hội tụ.
- **Cách sử dụng và kiểm chứng:** Nhóm đánh giá đề xuất theo nguồn lực, thời gian và tiêu chí chấm đồ án.

### Tương tác 25
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Thiết kế Ablation Study
- **Tóm tắt nội dung prompt:** Xác nhận lựa chọn huấn luyện từ đầu tới hội tụ cho tất cả các biến thể trong thiết kế ablation.
- **Cách sử dụng và kiểm chứng:** Nhóm xem xét đề xuất của AI trước khi quyết định áp dụng.

### Tương tác 26
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Thiết kế Ablation Study
- **Tóm tắt nội dung prompt:** Đề xuất thiết kế ma trận ablation dạng tổ hợp đầy đủ giữa các nhóm cải tiến kiến trúc và xin xác nhận tính hợp lý của thiết kế.
- **Cách sử dụng và kiểm chứng:** Nhóm đánh giá đề xuất theo nguồn lực, thời gian và tiêu chí chấm đồ án.

### Tương tác 27
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Thiết kế Ablation Study
- **Tóm tắt nội dung prompt:** Hỏi liệu thiết kế ablation đề xuất có đủ chặt chẽ để phục vụ mục tiêu công bố paở tạp chí mục tiêu hay không.
- **Cách sử dụng và kiểm chứng:** Nhóm đánh giá đề xuất theo nguồn lực, thời gian và tiêu chí chấm đồ án.

### Tương tác 28
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Thiết kế Ablation Study
- **Tóm tắt nội dung prompt:** Yêu cầu chốt riêng tính hợp lý của thiết kế ablation trước khi bàn đến các vấn đề khác.
- **Cách sử dụng và kiểm chứng:** Nhóm đánh giá đề xuất theo nguồn lực, thời gian và tiêu chí chấm đồ án.

### Tương tác 29
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Thiết kế Ablation Study
- **Tóm tắt nội dung prompt:** Xác nhận quyết định huấn luyện từ đầu cho toàn bộ các biến thể trong thiết kế ablation.
- **Cách sử dụng và kiểm chứng:** Nhóm xem xét đề xuất của AI trước khi quyết định áp dụng.

### Tương tác 30
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Thiết kế Ablation Study
- **Tóm tắt nội dung prompt:** Hỏi khái niệm phương sai do chỉ dùng một seed ngẫu nhiên duy nhất (single-seed variance) nghĩa là gì.
- **Cách sử dụng và kiểm chứng:** Nhóm đánh giá đề xuất theo nguồn lực, thời gian và tiêu chí chấm đồ án.

### Tương tác 31
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Thiết kế Ablation Study
- **Tóm tắt nội dung prompt:** Hỏi hướng xử lý phù hợp cho vấn đề phương sai seed, trong điều kiện giới hạn về thời gian thực hiện đồ án.
- **Cách sử dụng và kiểm chứng:** Nhóm đánh giá đề xuất theo nguồn lực, thời gian và tiêu chí chấm đồ án.

### Tương tác 32
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Thiết kế Ablation Study
- **Tóm tắt nội dung prompt:** Xác nhận chốt phương án xử lý vấn đề phương sai seed.
- **Cách sử dụng và kiểm chứng:** Nhóm xem xét đề xuất của AI trước khi quyết định áp dụng.

### Tương tác 33
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Thiết kế Ablation Study
- **Tóm tắt nội dung prompt:** Cam kết sẽ bổ sung đánh giá chủ quan bằng con người (MOS) ở giai đoạn sau.
- **Cách sử dụng và kiểm chứng:** Nhóm đánh giá đề xuất theo nguồn lực, thời gian và tiêu chí chấm đồ án.

### Tương tác 34
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Thiết kế Ablation Study
- **Tóm tắt nội dung prompt:** Chia sẻ định hướng tương lai: sẽ áp dụng đầy đủ các cải tiến kiến trúc đã thảo luận để chuẩn bị cho một bộ dữ liệu đa giọng nói.
- **Cách sử dụng và kiểm chứng:** Nhóm đánh giá đề xuất theo nguồn lực, thời gian và tiêu chí chấm đồ án.

---

## 4. Tiền xử lý dữ liệu & vấn đề dataset

**Cách AI được sử dụng:**
AI hỗ trợ phân tích vấn đề trong bước tiền xử lý dữ liệu văn bản và đánh giá mức độ ảnh hưởng của các thay đổi tham số phân tích dữ liệu tới tập huấn luyện.

### Tương tác 35
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Tiền xử lý dữ liệu
- **Tóm tắt nội dung prompt:** Bắt đầu hỏi về bước xử lý ký tự đặc biệt (dấu ngoặc kép) trong pipeline tiền xử lý văn bản của hệ thống nền (prompt bị ngắt giữa chừng, xem Tương tác 36 cho bản đầy đủ).
- **Cách sử dụng và kiểm chứng:** Nhóm kiểm tra lại bằng tài liệu mô hình, source code hiện có và kết quả thử nghiệm.

### Tương tác 36
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Tiền xử lý dữ liệu
- **Tóm tắt nội dung prompt:** Hỏi rõ liệu pipeline tiền xử lý gốc đã xử lý dấu ngoặc kép như một bước riêng biệt, hay cần thay đổi tham số phân tích dữ liệu — và nếu thay đổi thì có còn được xem là cấu hình baseline hay không.
- **Cách sử dụng và kiểm chứng:** Nhóm kiểm tra lại bằng tài liệu mô hình, source code hiện có và kết quả thử nghiệm.

### Tương tác 37
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Tiền xử lý dữ liệu
- **Tóm tắt nội dung prompt:** Hỏi mức độ ảnh hưởng của việc thay đổi tham số phân tích dữ liệu văn bản đến chất lượng mô hình, với ví dụ câu có chứa dấu ngoặc kép.
- **Cách sử dụng và kiểm chứng:** Nhóm kiểm tra lại bằng tài liệu mô hình, source code hiện có và kết quả thử nghiệm.

### Tương tác 38
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Tiền xử lý dữ liệu
- **Tóm tắt nội dung prompt:** Hỏi nếu chấp nhận huấn luyện lại toàn bộ thì việc thay đổi tham số phân tích dữ liệu có đáng thực hiện hay không.
- **Cách sử dụng và kiểm chứng:** Nhóm kiểm tra lại bằng tài liệu mô hình, source code hiện có và kết quả thử nghiệm.

### Tương tác 39
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Tiền xử lý dữ liệu
- **Tóm tắt nội dung prompt:** Hỏi liệu pipeline tiền xử lý hiện tại đã lọc các ký tự không thuộc phạm vi ngôn ngữ mục tiêu và các dấu câu bất thường hay chưa.
- **Cách sử dụng và kiểm chứng:** Nhóm kiểm tra lại bằng tài liệu mô hình, source code hiện có và kết quả thử nghiệm.

### Tương tác 40
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Tiền xử lý dữ liệu
- **Tóm tắt nội dung prompt:** Đặt câu hỏi phản biện: việc thay đổi tham số phân tích dữ liệu có thực sự ảnh hưởng nhiều không, vì các mẫu lỗi cũng sẽ bị loại bỏ theo cách khác.
- **Cách sử dụng và kiểm chứng:** Nhóm kiểm tra lại bằng tài liệu mô hình, source code hiện có và kết quả thử nghiệm.

### Tương tác 41
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Tiền xử lý dữ liệu
- **Tóm tắt nội dung prompt:** Yêu cầu kiểm tra chi tiết các dòng dữ liệu bị mất trong quá trình phân tích tệp dữ liệu văn bản, và xác minh mối liên hệ với một nhóm dòng dữ liệu bị gộp đã phát hiện trước đó.
- **Cách sử dụng và kiểm chứng:** Nhóm so sánh output trung gian giữa các module để xác định nguyên nhân lỗi.

### Tương tác 42
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Tiền xử lý dữ liệu
- **Tóm tắt nội dung prompt:** Xác nhận lại mối quan hệ giữa nhóm dòng dữ liệu bị gộp và nhóm dòng dữ liệu bị mất.
- **Cách sử dụng và kiểm chứng:** Nhóm so sánh output trung gian giữa các module để xác định nguyên nhân lỗi.

### Tương tác 43
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Tiền xử lý dữ liệu
- **Tóm tắt nội dung prompt:** Xác nhận số lượng mẫu thực tế được đưa vào huấn luyện sau khi trừ đi các dòng dữ liệu bị mất.
- **Cách sử dụng và kiểm chứng:** Nhóm so sánh output trung gian giữa các module để xác định nguyên nhân lỗi.

### Tương tác 44
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Tiền xử lý dữ liệu
- **Tóm tắt nội dung prompt:** Đặt câu hỏi về tính đáng giá: việc thay đổi tham số phân tích dữ liệu có đóng góp đủ lớn để đánh đổi việc phải huấn luyện lại toàn bộ các biến thể ablation hay không.
- **Cách sử dụng và kiểm chứng:** Nhóm kiểm tra lại bằng tài liệu mô hình, source code hiện có và kết quả thử nghiệm.

---

## 5. Huấn luyện mô hình & theo dõi tiến độ thực nghiệm

**Cách AI được sử dụng:**
AI hỗ trợ nhóm theo dõi tiến độ huấn luyện, xác nhận trạng thái hoàn thành của từng cấu hình thực nghiệm và lập kế hoạch cho các đợt huấn luyện tiếp theo.

### Tương tác 45
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-07-17 (phiên trước)
- **Mục đích sử dụng AI:** Theo dõi huấn luyện mô hình
- **Tóm tắt nội dung prompt:** Xác nhận đã chuẩn bị xong cấu hình huấn luyện và quyết định chỉ chạy một cấu hình rút gọn cụ thể trong đợt này.
- **Cách sử dụng và kiểm chứng:** Nhóm đối chiếu với log huấn luyện và kết quả inference.

### Tương tác 46
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Theo dõi huấn luyện mô hình
- **Tóm tắt nội dung prompt:** Xác nhận nếu bổ sung thêm một thành phần cải tiến đang cân nhắc thì đồng nghĩa phải huấn luyện lại mô hình chính từ đầu.
- **Cách sử dụng và kiểm chứng:** Nhóm đối chiếu với log huấn luyện và kết quả inference.

### Tương tác 47
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Theo dõi huấn luyện mô hình
- **Tóm tắt nội dung prompt:** Đề xuất kế hoạch hai giai đoạn: để cấu hình đang huấn luyện chạy tới mốc tương đương cấu hình đầy đủ nhất để so sánh trước, sau đó mới quyết định thay đổi tiền xử lý và huấn luyện lại chính thức.
- **Cách sử dụng và kiểm chứng:** Nhóm đối chiếu với log huấn luyện và kết quả inference.

### Tương tác 48
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Theo dõi huấn luyện mô hình
- **Tóm tắt nội dung prompt:** Xác nhận nếu không thay đổi tiền xử lý thì cấu hình baseline đang huấn luyện có thể được giữ lại, tính là một phần đã hoàn thành trong kế hoạch ablation.
- **Cách sử dụng và kiểm chứng:** Nhóm đối chiếu với log huấn luyện và kết quả inference.

### Tương tác 49
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Theo dõi huấn luyện mô hình
- **Tóm tắt nội dung prompt:** Đặt vấn đề: cấu hình đầy đủ nhất cũng cần huấn luyện lại nếu áp dụng thay đổi kiến trúc trong tương lai.
- **Cách sử dụng và kiểm chứng:** Nhóm đối chiếu với log huấn luyện và kết quả inference.

### Tương tác 50
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Theo dõi huấn luyện mô hình
- **Tóm tắt nội dung prompt:** Làm rõ rằng câu hỏi trước đang đề cập đến kịch bản tương lai, không phải hiện tại.
- **Cách sử dụng và kiểm chứng:** Nhóm đối chiếu với log huấn luyện và kết quả inference.

### Tương tác 51
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Theo dõi huấn luyện mô hình
- **Tóm tắt nội dung prompt:** Xác nhận lại: trong kịch bản không thay đổi gì, một cấu hình huấn luyện đã được xem là hoàn thành.
- **Cách sử dụng và kiểm chứng:** Nhóm đối chiếu với log huấn luyện và kết quả inference.

### Tương tác 52
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên trước)
- **Mục đích sử dụng AI:** Theo dõi huấn luyện mô hình
- **Tóm tắt nội dung prompt:** Làm rõ câu hỏi trước đang hỏi về số lượng cấu hình hoàn thành trong kịch bản có áp dụng thay đổi kiến trúc trong tương lai.
- **Cách sử dụng và kiểm chứng:** Nhóm đối chiếu với log huấn luyện và kết quả inference.

---

## 6. Trực quan hóa & giám sát huấn luyện (TensorBoard)

**Cách AI được sử dụng:**
AI được dùng để thiết lập, khắc phục sự cố và mở công cụ trực quan hóa (TensorBoard) phục vụ theo dõi và trình bày kết quả huấn luyện.

### Tương tác 53
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-07-17 (phiên trước)
- **Mục đích sử dụng AI:** Trực quan hóa huấn luyện (TensorBoard)
- **Tóm tắt nội dung prompt:** Yêu cầu dựng công cụ trực quan hóa quá trình huấn luyện để theo dõi.
- **Cách sử dụng và kiểm chứng:** Nhóm xem xét đề xuất của AI trước khi quyết định áp dụng.

### Tương tác 54
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-07-17 (phiên trước)
- **Mục đích sử dụng AI:** Trực quan hóa huấn luyện (TensorBoard)
- **Tóm tắt nội dung prompt:** Báo cáo công cụ trực quan hóa không mở được và không cập nhật dữ liệu qua nhiều lần thử liên tiếp, yêu cầu khắc phục.
- **Cách sử dụng và kiểm chứng:** Nhóm xác nhận lỗi bằng log, console và kiểm thử lại sau khi chỉnh sửa.

### Tương tác 55
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên hiện tại)
- **Mục đích sử dụng AI:** Trực quan hóa huấn luyện (TensorBoard)
- **Tóm tắt nội dung prompt:** Yêu cầu mở đồng thời hai log huấn luyện trên công cụ trực quan hóa để so sánh cấu hình baseline và cấu hình áp dụng đầy đủ các cải tiến kiến trúc, phục vụ demo cho giảng viên hướng dẫn.
- **Cách sử dụng và kiểm chứng:** Nhóm xác nhận lỗi bằng log, console và kiểm thử lại sau khi chỉnh sửa.

---

## 7. Quản lý mã nguồn & vận hành (Git)

**Cách AI được sử dụng:**
AI được dùng để thực hiện các thao tác quản lý mã nguồn (rà soát thay đổi, commit, push) theo yêu cầu và xác nhận của nhóm.

### Tương tác 56
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên hiện tại)
- **Mục đích sử dụng AI:** Quản lý mã nguồn (Git)
- **Tóm tắt nội dung prompt:** Yêu cầu commit toàn bộ thay đổi mã nguồn hiện có trong working tree.
- **Cách sử dụng và kiểm chứng:** Nhóm xem xét đề xuất của AI trước khi quyết định áp dụng.

### Tương tác 57
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên hiện tại)
- **Mục đích sử dụng AI:** Quản lý mã nguồn (Git)
- **Tóm tắt nội dung prompt:** Xác nhận đồng ý với danh sách file sẽ được commit sau khi AI rà soát và loại trừ các file nghi ngờ không hợp lệ.
- **Cách sử dụng và kiểm chứng:** Nhóm xem xét đề xuất của AI trước khi quyết định áp dụng.

### Tương tác 58
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên hiện tại)
- **Mục đích sử dụng AI:** Quản lý mã nguồn (Git)
- **Tóm tắt nội dung prompt:** Yêu cầu đẩy commit vừa tạo lên remote repository.
- **Cách sử dụng và kiểm chứng:** Nhóm xem xét đề xuất của AI trước khi quyết định áp dụng.

### Tương tác 59
- **Người thực hiện:** Nguyễn Văn Anh Duy
- **Thời gian:** 2026-08-04 (phiên hiện tại)
- **Mục đích sử dụng AI:** Quản lý mã nguồn (Git)
- **Tóm tắt nội dung prompt:** Yêu cầu commit và đẩy toàn bộ phần còn lại lên remote, kể cả các file chưa chắc chắn về tính hợp lệ, sau khi được hỏi lại và xác nhận.
- **Cách sử dụng và kiểm chứng:** Nhóm xem xét đề xuất của AI trước khi quyết định áp dụng.

---

## Cam kết sử dụng AI

- AI được sử dụng để hỗ trợ phân tích, giải thích, phản biện, viết tài liệu, kiểm tra lỗi và chuẩn bị trình bày.
- Nhóm chịu trách nhiệm đối với kiến trúc, source code, dataset, thí nghiệm và kết quả cuối cùng.
- Đề xuất của AI được kiểm tra lại bằng tài liệu kỹ thuật, source code, log thực nghiệm hoặc đánh giá thủ công.
- Quyết định kỹ thuật cuối cùng do nhóm thực hiện đồ án đưa ra.
- Tài liệu này đã lược bỏ các chi tiết nội bộ không cần thiết cho mục đích công bố việc sử dụng AI.

## Lưu ý

Bản này là tài liệu công bố tóm tắt, không phải bản ghi hội thoại nguyên văn. Bản log đầy đủ chỉ được lưu nội bộ và không được đính kèm vào báo cáo công khai.
