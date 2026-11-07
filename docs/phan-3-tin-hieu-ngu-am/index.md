# Phần III. Tín hiệu và ngữ âm

Một kỹ sư TTS giỏi không chỉ biết huấn luyện mạng nơ-ron mà còn hiểu **tiếng nói là gì**: nó được tạo ra thế nào trong bộ máy phát âm, được biểu diễn thế nào dưới dạng tín hiệu số, được tai người cảm nhận ra sao, và — với tiếng Việt — các thanh điệu, âm tiết và phương ngữ vận hành thế nào. Phần này cung cấp kiến thức chuyên ngành đó.

## Các chương

| Chương | Nội dung | Vì sao quan trọng cho TTS |
|---|---|---|
| [13. Âm học và cơ chế tạo tiếng nói](ch13-am-hoc-tieng-noi.md) | Mô hình nguồn – bộ lọc, formant, ngữ âm cấu âm, IPA, thính giác | Hiểu vì sao dùng thang mel, vì sao vocoder cần mô hình hóa F0 |
| [14. Xử lý tín hiệu số cho tiếng nói](ch14-xu-ly-tin-hieu-so.md) | Lấy mẫu, STFT, mel-spectrogram, Griffin–Lim, trích xuất F0 | Mel-spectrogram là "ngôn ngữ chung" giữa mô hình âm học và vocoder |
| [15. Ngữ âm và âm vị học tiếng Việt](ch15-ngu-am-tieng-viet.md) | Cấu trúc âm tiết, phụ âm, nguyên âm, sáu thanh điệu, phương ngữ | Lỗi thanh điệu là lỗi nghiêm trọng nhất của TTS tiếng Việt |
| [16. Ngôn điệu (Prosody)](ch16-ngon-dieu.md) | Ngữ điệu, nhịp, trọng âm, ngắt nghỉ; tương tác thanh điệu – ngữ điệu | Phân biệt giọng "đọc máy" với giọng tự nhiên |

!!! note "Góc tiếng Việt"
    Tiếng Việt là ngôn ngữ **có thanh điệu**: đường nét cao độ (F0) của mỗi âm tiết mang nghĩa từ vựng — "ma, mà, má, mả, mã, mạ" là sáu từ khác nhau. Điều này khiến việc mô hình hóa F0 trong TTS tiếng Việt khắt khe hơn nhiều so với tiếng Anh: thay đổi ngữ điệu câu mà làm méo thanh điệu là đọc sai từ.
