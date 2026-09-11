# AI Usage Log — Dự án Banhmi-TTS (EdgeTTS)

**Tên dự án:** Banhmi-TTS — Tích hợp BigVGAN, VITS2 và FastPitch vào hệ thống TTS production nhẹ  
**Repository:** FPT-Graduation-Project / EdgeTTS  
**Branch:** dev  
**Công cụ AI sử dụng:** Claude (Anthropic) — Claude Sonnet 4.6  
**Người thực hiện:** Nguyễn Hoàng Duy (tranhoangtuanhung1997@gmail.com)  
**Ngày ghi:** 2026-08-04  

> **Lưu ý:** File này ghi nhận các prompt từ phiên làm việc ngày 2026-08-04. Các phiên khác sẽ được bổ sung sau. Phần đầu session (trước khi context bị nén) được tái tạo từ bản tóm tắt hệ thống; phần sau được ghi trực tiếp.

---

## Prompt 1

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04 (đầu session)
- **Mục đích:** Yêu cầu AI paste lại bản dịch EN+VI của phần gap statement §2.2 (BigVGAN/Snake+MRD) đã làm ở session trước
- **Nội dung prompt gốc:**
```text
gửi tôi phần tiếng anh và tiếng việt giống bạn làm ở phần BigVGan đi
```

---

## Prompt 2

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Chuyển sang bàn về phần VITS2 trong Related Work; xác nhận phần BigVGAN đã ổn
- **Nội dung prompt gốc:**
```text
Tiếp tục chúng ta sẽ bàn tới phần của VITS2 nhé, phần BIGVGan vậy là ổn rồi đấy
```

---

## Prompt 3

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Yêu cầu tóm tắt 4 thành phần của VITS2 và xác nhận dự án đã lấy 2 phần nào
- **Nội dung prompt gốc:**
```text
vậy tổng kết là 4 phần của VITS2 là gì và chúng tôi đã lấy 2 phần nào?
```

---

## Prompt 4

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Hỏi lý do không lấy thành phần thứ 3 (alignment noise) của VITS2
- **Nội dung prompt gốc:**
```text
vậy tại sao không lấy luôn phần 3 mà lại bỏ nó đi? mà chỉ lấy 2/4 phần này?
```

---

## Prompt 5

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Trả lời câu hỏi lựa chọn (AskUserQuestion) — quyết định giữ alignment noise như một Limitations note thay vì implement
- **Nội dung prompt gốc:**
```text
Giữ nguyên, ghi vào Limitations (Recommended)
```

---

## Prompt 6

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Tái đánh giá quyết định — liệu chỉ lấy 2/4 phần VITS2 có đúng không, có nên bổ sung phần 3 không
- **Nội dung prompt gốc:**
```text
vậy liệu chỉ lấy 2/4 phần đó có phải quyết định đúng đắn không hay phải bổ sung phần 3 ?"
```

---

## Prompt 7

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Xác nhận hệ quả: nếu thêm alignment noise thì Banhmi-TTS phải train lại từ đầu
- **Nội dung prompt gốc:**
```text
Vậy là thêm vô đồng nghĩa phải train lại BanhmiTTS đúng không?
```

---

## Prompt 8

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Hỏi VITS2 còn có cải tiến nào khác ngoài 4 điểm đã thảo luận không
- **Nội dung prompt gốc:**
```text
cho tôi hỏi thêm ngoài 4 cái mình bàn từ nãy tới giờ thì VITS2 còn cải thiện điểm nào so với VITS không?
```

---

## Prompt 9

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Hỏi có thể lấy cả 4/4 phần VITS2 không; và liệu có thể paraphrase gap-of-VITS từ paper VITS2 không
- **Nội dung prompt gốc:**
```text
vậy nếu giờ tôi sửa thì tôi lấy cả 4 phần VITS2 luôn được không? và trong paper VITS2 họ có trình bày gap của VITS sẵn không thì tôi sẽ paraphrase lại phần đó rồi ref cho gọn vì mình cũng đang áp dụng giống VITS2 mà.
```

---

## Prompt 10

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Hỏi liệu có thể so sánh kết quả single-speaker với VITS/VITS2 vì họ cũng có VCTK
- **Nội dung prompt gốc:**
```text
làm trên single-speaker liệu có thể so sánh với VITS và VITS2 không vì họ có cả VTCK nữa
```

---

## Prompt 11

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Hỏi sự khác biệt kiến trúc giữa Piper và VITS gốc
- **Nội dung prompt gốc:**
```text
vậy điểm khác biệt giữa Piper và VITS là gì?
```

---

## Prompt 12

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Hỏi về preprocessing của Piper gốc và xử lý dấu ngoặc kép (bị gián đoạn)
- **Nội dung prompt gốc:**
```text
Vậy tôi hỏi riêng phần Piper gốc, thì phần preprocessing thì cơ bản đã phải xử lý dấu"
```
*(Prompt bị gián đoạn, xem Prompt 13 cho phiên bản đầy đủ)*

---

## Prompt 13

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Hỏi phần Piper gốc đã xử lý dấu `"` bên ngoài chưa, hay phải thay QUOTE_NONE — và nếu thay thì có còn gọi là baseline không
- **Nội dung prompt gốc:**
```text
Vậy tôi hỏi riêng phần Piper gốc, thì phần preprocessing thì cơ bản đã phải xử lí dấu " như một bước xử lí bên ngoài rồi, hay tôi phải thay đổi phần QUOTE_NONE luôn thì có còn gọi là baseline không?
```

---

## Prompt 14

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Hỏi ảnh hưởng của việc đổi QUOTE_MINIMAL sang QUOTE_NONE và cách model xử lý dấu `"` trong inference
- **Nội dung prompt gốc:**
```text
về cơ bản thì sửa từ QUOTE_MINIMAL thành NONE thì có ảnh hưởng nhiều tới hiệu suất không? Ví dụ người dùng nhập text: He said: "It's great!" thì model có hiểu " khi chuyển thành NONE không?
```

---

## Prompt 15

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Hỏi nếu chấp nhận train lại toàn bộ thì có đáng để sửa QUOTE_NONE không
- **Nội dung prompt gốc:**
```text
nếu chấp nhận train lại tất cả thì bạn thấy đáng để sửa thành NONE không?
```

---

## Prompt 16

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Hỏi về phương pháp ablation của VITS2 — họ có train 4 model riêng biệt không
- **Nội dung prompt gốc:**
```text
Về VITS2 thì MOS họ chia ra là họ bỏ từng thành phần 1 trong 4 tức là họ có train cả 4 model hả?
```

---

## Prompt 17

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Hỏi chi tiết VITS2 train từ đầu hay pretrain, và các ablation variant được tạo ra như thế nào
- **Nội dung prompt gốc:**
```text
nhưng cách họ train thế nào? ví dụ VITS2 là họ train từ đầu hay pre-train từ VITS, và các phần họ bỏ từng phần thì họ có train từ đầu hay là pre-train?
```

---

## Prompt 18

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Hỏi nên chọn pretrain hay train from scratch cho ablation, dựa trên kinh nghiệm thực tế (30k steps không đủ hội tụ)
- **Nội dung prompt gốc:**
```text
vậy theo bạn thì nên pre-train hay chọn train from scratch từng phần? vì tôi đã train thử from scratch và khá chắc rằng một điều 30k steps thì không thể nào đủ để hội tụ để có được một audio tốt như 800k steps được.
```

---

## Prompt 19

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Trả lời câu hỏi lựa chọn (AskUserQuestion) — xác nhận có đủ compute để train from scratch tới hội tụ cho mọi config
- **Nội dung prompt gốc:**
```text
Đầy đủ — train from-scratch tới hội tụ cho mọi config (Recommended về phương pháp)
```

---

## Prompt 20

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Đề xuất thiết kế ablation 8 case theo 2³ factorial (baseline × BigVGAN × VITS2 × F0 dạng bundle), xin xác nhận tính hợp lý
- **Nội dung prompt gốc:**
```text
có 1 phần tôi cần verify lại với bạn, tôi sẽ chấp nhận việc train baseline, sau đó sẽ train tổ hợp từng phần để chia case như sau:
- baseline  
- baseline + BIGVGan 
- baseline + VITS2 
- baseline + F0 
- baseline + BigVGan + VITS2  
- baseline + BigVGan + F0
- baseline + VITS2 + F0
- baseline + All

Tôi sẽ sử dụng full BigVGan luôn chứ không tách lẻ từng phần, tương tự VITS2, tôi chỉ lấy tổ hợp cuối vì từ VITS2 nó cũng nói số liệu tốt hơn rồi, khi bạn nhìn vào đây sẽ thấy tổng cộng 8 case, và tôi chỉ cần so sánh baseline kết hợp với các phương pháp. Baseline sẽ là Piper. Bạn thấy thế này hợp lý chứ? vì tôi không có đủ thời gian để train from scratch 16 hay 32 hay 128 case.
```

---

## Prompt 21

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Hỏi liệu 8 case có đủ để paper Q2 được approve không nếu kết quả tốt
- **Nội dung prompt gốc:**
```text
khoan tôi đang hỏi về tính hợp lệ để bảo toàn thời gian cũng như logic, liệu 8 case trên đã đủ để report cho paper Q2 được approve nếu nó thật sự tốt hơn không?
```

---

## Prompt 22

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Yêu cầu chốt riêng về tính hợp lý của cấu trúc 8 case trước khi bàn đến các vấn đề khác
- **Nội dung prompt gốc:**
```text
từ từ trước tiên tôi muốn bạn chốt xem về 8 cases trên chia như thế đã hợp lí chưa đã
```

---

## Prompt 23

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Xác nhận quyết định train from scratch toàn bộ 8 case
- **Nội dung prompt gốc:**
```text
rồi, tất cả tôi sẽ quyết định train from scratch hết thì có hợp lý không?
```

---

## Prompt 24

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Hỏi giải thích khái niệm "single-seed variance" là gì
- **Nội dung prompt gốc:**
```text
single-seed variance là sao?
```

---

## Prompt 25

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Hỏi hướng xử lý seed variance phù hợp với giới hạn thời gian, target venue Q2 hoặc Q3
- **Nội dung prompt gốc:**
```text
vì tôi không có thời gian như bạn đã biết thì hướng nào hợp lý? có thể Q3 hoặc Q2 đều ổn
```

---

## Prompt 26

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Xác nhận chốt quyết định single-seed + Limitations note
- **Nội dung prompt gốc:**
```text
chốt đi
```

---

## Prompt 27

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Xác nhận sẽ bổ sung human MOS evaluation sau
- **Nội dung prompt gốc:**
```text
TÔi chắc chắn sẽ bổ sung MOS sau bạn yên tâm
```

---

## Prompt 28

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Đề xuất kế hoạch 2 giai đoạn: để baseline train tới ~1044 epoch rồi so với +All, sau đó mới sửa preprocessing và train lại chính thức
- **Nội dung prompt gốc:**
```text
Hiện tại đang train baseline của preprocessing cũ, thì bạn cứ để nó chạy cho xong tới khoảng 800k steps hoặc 1044 epoch giống với baseline + All để tôi evaluation xem liệu ghép các thứ lại với nhau có thật sự tốt chưa đã. Sau đó sẽ tiến hành sửa preprocessing thành QUOTE_None rồi train lại theo plan đo từ baseline, bạn thấy hợp lí chứ?
```

---

## Prompt 29

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Hỏi code preprocessing hiện tại có xử lý ký tự bất thường (ngoài Latin, punctuation lạ) chưa
- **Nội dung prompt gốc:**
```text
cho tôi hỏi, code preprocessing đã xử lí những mẫu có kí tự bất thường chưa? tại chúng ta đang xử lí tiếng Anh thôi nên các chữ ngoài chữ latin nên được bỏ hết cùng với các punctuation bất thường.
```

---

## Prompt 30

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Đặt câu hỏi phản biện: QUOTE_NONE đâu ảnh hưởng nhiều vì nó cũng loại các mẫu đó đi mà
- **Nội dung prompt gốc:**
```text
Khoan, nếu việc thay đổi QUOTE_NONE thì cũng đâu ảnh hưởng quá nhiều tới việc kết quả model đâu? dù sao nó cũng loại các mẫu đó đi mà?
```

---

## Prompt 31

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Yêu cầu xem 189 row bị lỗi là row nào và verify xem chúng có nằm trong 16 dòng bị gộp không
- **Nội dung prompt gốc:**
```text
Bạn xem thử 189 row bị lỗi là row nào và xem kĩ coi có phải là nằm trong 16 dòng bị gộp không?
```

---

## Prompt 32

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Xác nhận lại mối quan hệ: 16 row có chứa 189 row bên trong không
- **Nội dung prompt gốc:**
```text
vậy trong 16 row đó có chứa 189 row chứ?
```

---

## Prompt 33

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Hỏi xác nhận: khi model train thì số lượng mẫu thực tế là 13100-189 hay khác
- **Nội dung prompt gốc:**
```text
cần phải hiểu rõ là 13100 là bị mất 189 rồi, rồi nó trả về 16 entry đó, thì tức là khi model feed vô thì vẫn là 13100-189 chứ nhỉ?
```

---

## Prompt 34

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Đặt câu hỏi về tính đáng giá: sửa QUOTE_NONE không đóng góp nhiều mà còn phải train lại cả 8 case
- **Nội dung prompt gốc:**
```text
vậy sửa thành NONE cũng không đóng góp quá lớn vào việc model train đúng không? mà còn phải train lại cả 8 phần.
```

---

## Prompt 35

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Đề xuất hướng viết paper theo chuẩn: tách Dataset section (số mẫu) và Preprocessing section (mô tả phương pháp), không cần report con số chính xác sau filter
- **Nội dung prompt gốc:**
```text
tôi thấy các bài báo lớn họ đều có chia 2 phần dataset có nói bao gồm bao nhiêu mẫu, và phần preprocessing của họ nói về cách xử lí chứ không hoàn toàn viết con số cụ thể sẽ đưa vào paper. tôi nghĩ tôi sẽ xử lí theo hướng này.
```

---

## Prompt 36

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Thông báo kế hoạch tương lai: implement đủ 4/4 VITS2 để chuẩn bị cho dataset multi-speaker
- **Nội dung prompt gốc:**
```text
mục tiêu tương lại tôi sẽ fix cả phần VITS2, tức là tôi sẽ lấy cả 4/4 phần của họ luôn. Vì có thể tương lai tôi train thêm 1 dataset multi-speaker.
```

---

## Prompt 37

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Xác nhận: không sửa preprocessing → baseline hiện tại đang train có thể giữ lại là 1/8 case
- **Nội dung prompt gốc:**
```text
vậy coi như tôi sẽ không sửa preprocessing, baseline tôi đang train hiện tại có thể được giữ lại coi như hoàn thành 1/8 case cần train đúng không?
```

---

## Prompt 38

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Đặt vấn đề: baseline + All cũng phải train lại vì sửa VITS2
- **Nội dung prompt gốc:**
```text
baseline + All tôi nghĩ cũng phải train lại vì sửa VITS2 mà
```

---

## Prompt 39

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Làm rõ: "sửa VITS2" đang nói về kế hoạch tương lai, không phải hiện tại
- **Nội dung prompt gốc:**
```text
tôi đang nói tương lai
```

---

## Prompt 40

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Xác nhận lại: trong kịch bản hiện tại (không sửa gì), đã xong 1/8 case chưa
- **Nội dung prompt gốc:**
```text
vậy tương lai thì đồng nghĩa là tôi đã xong 1/8 case rồi đúng chưa
```

---

## Prompt 41

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Làm rõ câu hỏi: trong kịch bản tương lai (sửa VITS2 lên 4/4) thì số case đã xong là bao nhiêu
- **Nội dung prompt gốc:**
```text
ý tôi trường hợp sửa VITS2 ấy
```

---

## Prompt 42

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Quay lại Related Work — yêu cầu viết gap statement §2.3 (VITS vs VITS2) theo format giống §2.2 BigVGAN
- **Nội dung prompt gốc:**
```text
chúng ta quay lại phần Related Work phần gap của VITS so với VITS2 nhé. Tôi cần bạn viết 1 câu giống phần BigVGan vậy.
```

---

## Prompt 43

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Đề xuất hướng viết gap: chỉ cần nêu gap của VITS, không cần nói rõ VITS2 bổ sung gì (để dành cho Methods)
- **Nội dung prompt gốc:**
```text
Tôi nghĩ ở phần này thì bạn chỉ cần viết rằng VITS có gap gì chứ chưa cần nói rõ VITS2 bổ sung những gì chứ nhỉ?
```

---

## Prompt 44

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Xác nhận đã xong §2.2 và §2.3, gap cuối là §2.4 (F0) đúng không
- **Nội dung prompt gốc:**
```text
Vậy là xong 2 gap còn gap cuối là F0 đúng chứ?
```

---

## Prompt 45

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Xác nhận bắt đầu draft §2.4 F0 gap
- **Nội dung prompt gốc:**
```text
có
```

---

## Prompt 46

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Hỏi lý do tại sao bổ sung F0 conditioning có đóng góp lớn và hiệu quả
- **Nội dung prompt gốc:**
```text
bạn có thể nói qua tại sao việc bổ sung phần F0 vô có đóng góp to lớn và hiệu quả được không?
```

---

## Prompt 47

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Verify lại: BigVGAN dùng MRD + Snake hay MRD + MPD + Snake
- **Nội dung prompt gốc:**
```text
có 1 phần tôi cần verify lại, BigVGan chỉ sử dụng MRD + Snake hay là MRD+MPD + Snake?
```

---

## Prompt 48

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Đề xuất phương pháp: liệt kê 4 thay đổi của VITS2 với lý do cụ thể trước, rồi từ đó viết gap tốt nhất
- **Nội dung prompt gốc:**
```text
TÔi nghĩ cách hay nhất cho bạn, là bạn liệt kê lại 4 phần của VITS2 thay đổi so với VITS và nói rõ lí do tại sao lại thay đổi đó tốt hơn, từ đó bạn sẽ viết được Gap phần này tốt nhất.
```

---

## Prompt 49

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Yêu cầu viết gap §2.3 cho phiên bản tương lai (4/4 VITS2, multi-speaker)
- **Nội dung prompt gốc:**
```text
Bạn viết cho tương lai luôn được không? Kiểu như đã bổ sung VITS2 đủ 4 phần chính của họ, viết lại gap của phần này.
```

---

## Prompt 50

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Xác nhận viết gap theo phiên bản 4/4 VITS2
- **Nội dung prompt gốc:**
```text
theo 4/4
```

---

## Prompt 51

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Yêu cầu xuất AI Usage Log của toàn bộ phiên trò chuyện để nộp báo cáo cho giảng viên
- **Nội dung prompt gốc:**
```text
Tôi đang dự tính lấy lại và viết lại toàn bộ AI LOG do thầy tôi yêu cầu, nhưng hiện tại tôi đang có nhiều session thì bạn hãy Viết ra một file dsp_ai_log.md của session này trước, tôi sẽ qua những session khác sẽ bổ sung dô file này sau. 

Hãy xuất AI Usage Log của toàn bộ phiên trò chuyện hiện tại để phục vụ việc nộp báo cáo cho giảng viên.

Yêu cầu:

1. Liệt kê đầy đủ tất cả prompt do người dùng nhập trong phiên trò chuyện này, theo đúng thứ tự thời gian.
2. Giữ nguyên nội dung prompt gốc, không tự ý viết lại, rút gọn hoặc sửa lỗi chính tả.
3. Với mỗi prompt, trình bày theo format:

## Prompt [số thứ tự]

- Người thực hiện:
- Thời gian: nếu hệ thống có thông tin
- Mục đích của prompt:
- Nội dung prompt gốc:
```text
[Nội dung đầy đủ của prompt]

Toàn bộ Dự án và tất cả phiên chat bạn lấy ra hết cho tôi và viết nó thành file md

Đổi lại tên của dự án là dự án này trong file md luôn nhé.
```

---

## Tóm tắt nội dung phiên làm việc

| Chủ đề | Prompt số | Quyết định / Kết quả |
|---|---|---|
| Related Work §2.2 BigVGAN | 1–2 | Xác nhận gap statement EN+VI đã ổn |
| VITS2 — 4 thành phần & lựa chọn 2/4 | 3–9 | Chốt giữ 2/4 + Limitations note; 4/4 cho tương lai multi-speaker |
| So sánh cross-paper & Piper vs VITS | 10–11 | Không so sánh số liệu cross-paper; Piper ≠ VITS gốc về decoder |
| CSV quoting bug (QUOTE_MINIMAL vs NONE) | 12–14, 29–35 | Giữ QUOTE_MINIMAL; cite LJSpeech 13,100 theo chuẩn; không cần Limitations riêng |
| Ablation design: 8-case 2³ factorial | 20–23 | Chốt cross-line factorial; không tách lẻ within-line |
| Training strategy (from scratch vs pretrain) | 17–19 | From scratch mọi config; seed variance: single-seed + Limitations note |
| Evaluation (MOS) | 26–27 | Human MOS sẽ bổ sung sau; WER+UTMOS automatic là bước đầu |
| Kế hoạch training hiện tại | 28, 37–41 | Baseline (1/8) đang train; +All (epoch 863+) tái dùng; cần 6 run mới |
| Related Work §2.3 VITS2 gap | 42–43, 48–50 | Gap 2 câu: nêu hạn chế VITS, dành VITS2 solution cho Methods |
| Related Work §2.4 F0 gap | 44–46 | Draft xong gap F0 (FastPitch-style per-phoneme conditioning) |
| Verify BigVGAN discriminator | 47 | Xác nhận: BigVGAN dùng MPD + MRD + Snake (không phải MRD only) |

---

---

# Phiên làm việc 2 — Git commit/push & TensorBoard comparison setup

**Ngày:** 2026-08-04
**Công cụ AI sử dụng:** Claude (Anthropic) — Claude Sonnet 5
**Chủ đề chính:** Commit/push code lên git, dựng TensorBoard so sánh training log giữa Piper baseline và full-config (BigVGAN+VITS2+F0)

---

## Prompt 1

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Yêu cầu commit toàn bộ thay đổi trong working tree
- **Nội dung prompt gốc:**
```text
commit my code
```

---

## Prompt 2

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Xác nhận đồng ý sau khi AI liệt kê các file sẽ commit/loại trừ (ngữ cảnh: IDE đang chọn đoạn bảng ablation config A–H trong docs/draft_demo1.md)
- **Nội dung prompt gốc:**
```text
ok
```

---

## Prompt 3

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Yêu cầu push commit vừa tạo lên remote git
- **Nội dung prompt gốc:**
```text
push all to git
```

---

## Prompt 4

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Yêu cầu commit và push toàn bộ phần còn lại, kể cả các file nghi ngờ là artifact/junk (patch files, file rỗng)
- **Nội dung prompt gốc:**
```text
commit va push het code
```

---

## Prompt 5

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Yêu cầu mở TensorBoard cho 2 training run để demo cho giảng viên (prompt bị gián đoạn giữa chừng, xem Prompt 6 cho phiên bản đầy đủ). Ngữ cảnh: IDE đang mở file CSV kết quả eval baseline vs Piper-Modern.
- **Nội dung prompt gốc:**
```text
open for me 2 training log on tensor, i wanna show the result for my teacher see. 
1. Is Baseline of Piper
```

---

## Prompt 6

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Gửi lại đầy đủ yêu cầu mở TensorBoard so sánh 2 run: Piper baseline và full config (BigVGAN + VITS2 + F0) để demo cho giảng viên
- **Nội dung prompt gốc:**
```text
open for me 2 training log on tensor, i wanna show the result for my teacher see. 
1. Is Baseline of Piper
2. Is full apply BigVgan, vits2. fo.
```

---

## Prompt 7

- **Người thực hiện:** Sinh viên
- **Thời gian:** 2026-08-04
- **Mục đích:** Yêu cầu xuất toàn bộ AI Usage Log của phiên trò chuyện hiện tại (git operations + TensorBoard) để bổ sung vào file dsp_ai_log.md phục vụ báo cáo giảng viên. Ngữ cảnh: IDE đang mở file dsp_ai_log.md.
- **Nội dung prompt gốc:**
````text
Hãy xuất AI Usage Log của toàn bộ phiên trò chuyện hiện tại để bổ sung vào file dsp_ai_log.md (báo cáo cho giảng viên).

Yêu cầu:
1. Liệt kê đầy đủ tất cả prompt do tôi (người dùng) nhập trong session này, theo đúng thứ tự thời gian.
2. Giữ nguyên nội dung prompt gốc — không viết lại, không rút gọn, không sửa lỗi chính tả.
3. Với mỗi prompt, trình bày theo format:

## Prompt [số thứ tự]

- **Người thực hiện:** Sinh viên
- **Thời gian:** [ngày session nếu biết, hoặc N/A]
- **Mục đích:** [1 câu mô tả mục đích]
- **Nội dung prompt gốc:**
```text
[nội dung đầy đủ]

Bổ sung vào file md dưới.
````

---

## Tóm tắt nội dung phiên làm việc 2

| Chủ đề | Prompt số | Quyết định / Kết quả |
|---|---|---|
| Commit code lần 1 | 1–2 | Commit các thay đổi hợp lệ (banhmi_tts preprocessing, configs ablation B–G, eval scripts); loại trừ file `=0.29,` rỗng và 6 file `patch_*.patch` trỏ tới path `../BanhmiTTS/...` không tồn tại trong repo |
| Push lên git | 3 | Push nhánh `dev` lên `origin` (2 commit) |
| Commit + push toàn bộ | 4 | AI hỏi lại xác nhận trước khi commit các file nghi junk; sinh viên chọn commit hết, kể cả file rỗng và patch files → commit `b721762` + push |
| Mở TensorBoard so sánh baseline vs full-config | 5–6 | Xác định đúng 2 log dir qua `hparams.yaml`: `02_Baseline_VanillaVITS` = Piper baseline (`use_bigvgan=use_vits2=use_f0=false`), `01_Baseline_BigVgan_VITS2_FO/version_2` = full config (`use_bigvgan=use_vits2=use_f0=true`); dựng TensorBoard trong WSL trên port 6007 (port 6006 đã có tiến trình `BanhmiTTS` khác đang chạy), mở trình duyệt tại `localhost:6007` |
| Xuất AI Usage Log | 7 | Bổ sung log phiên làm việc 2 vào `dsp_ai_log.md` |

---

*File này sẽ được bổ sung thêm từ các phiên làm việc khác.*
