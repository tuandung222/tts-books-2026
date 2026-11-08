# Lời nói đầu

## Vì sao có cuốn giáo trình này?

Tổng hợp tiếng nói (*Text-to-Speech*, TTS) là một trong những lĩnh vực của trí tuệ nhân tạo thay đổi nhanh nhất mười năm qua. Năm 2016, WaveNet lần đầu cho thấy mạng nơ-ron có thể tạo ra sóng âm gần như không phân biệt được với giọng người. Năm 2017–2019, Tacotron 2 và FastSpeech đưa TTS neural vào sản phẩm. Năm 2021, VITS gộp cả hệ thống vào một mô hình end-to-end. Từ năm 2023, các mô hình dùng *neural audio codec* và *language model* như VALL-E, rồi các mô hình *flow matching* như Voicebox, F5-TTS, và các hệ thống lai như CosyVoice, cho phép nhân bản một giọng nói chỉ từ vài giây âm thanh mẫu. Ngày nay TTS là thành phần cốt lõi của trợ lý ảo, sách nói, tổng đài tự động, công cụ hỗ trợ người khiếm thị, và các *speech LLM* có thể trò chuyện song công thời gian thực.

Thế nhưng tài liệu học TTS một cách **hệ thống**, **từ gốc đến ngọn**, lại hiếm — và bằng **tiếng Việt** thì gần như chưa có. Người học thường phải tự ghép nối từ hàng trăm bài báo tiếng Anh, mỗi bài dùng một hệ ký hiệu, giả định một nền tảng khác nhau, và hầu như không bài nào bàn tới những vấn đề riêng của tiếng Việt: sáu thanh điệu, ba vùng phương ngữ, cách đọc số "hai mươi mốt" hay "một trăm linh năm", chữ viết tắt "TP.HCM", hay câu trộn tiếng Anh như "gửi email cho team".

Cuốn giáo trình này ra đời để lấp khoảng trống đó. Mục tiêu của chúng tôi là: **một người có nền tảng lập trình và toán phổ thông, đi hết cuốn sách và làm các dự án, sẽ hiểu tận gốc và tự xây dựng được hệ thống TTS tiếng Việt hiện đại.**

## Cuốn sách dành cho ai?

- **Sinh viên** ngành khoa học máy tính, điện tử – viễn thông, ngôn ngữ học tính toán muốn bước vào nghiên cứu xử lý tiếng nói.
- **Kỹ sư phần mềm và kỹ sư học máy** cần xây dựng hoặc tinh chỉnh hệ thống TTS cho sản phẩm.
- **Nhà nghiên cứu** ở lĩnh vực lân cận (nhận dạng tiếng nói, xử lý ngôn ngữ tự nhiên, thị giác máy tính) muốn nắm nhanh bức tranh toàn cảnh của TTS.
- **Giảng viên** cần một giáo trình mở để dạy môn xử lý tiếng nói hoặc mô hình sinh.

Bạn cần biết lập trình Python ở mức cơ bản và có kiến thức toán phổ thông (đại số, đạo hàm, xác suất cơ bản). Mọi thứ khác — từ đại số tuyến tính, lan truyền ngược cho tới phương trình vi phân ngẫu nhiên của mô hình khuếch tán — đều được trình bày lại trong sách.

## Triết lý biên soạn

1. **Hiểu tận gốc, không chỉ dùng công cụ.** Mỗi kiến trúc được giải thích theo nhịp *trực giác → toán học → mã nguồn → liên hệ TTS*. Chúng tôi luôn trả lời câu hỏi: *thiết kế này giải quyết vấn đề gì của phương pháp trước?*
2. **Một mạch xuyên suốt.** Ý tưởng trung tâm của cả cuốn sách là: TTS là bài toán **sinh có điều kiện một-đối-nhiều** — cùng một câu văn có vô số cách đọc đúng. Từ đó, lịch sử TTS hiện ra như một chuỗi lời giải ngày càng tốt cho bài toán này: từ hồi quy mel-spectrogram (bị "nhòe"), đến mô hình hóa trường độ, cao độ, rồi đến VAE, flow, GAN, diffusion, flow matching và mô hình ngôn ngữ trên token âm thanh.
3. **Tiếng Việt là công dân hạng nhất.** Các hộp *Góc tiếng Việt*, chương ngữ âm học tiếng Việt, chương chuẩn hóa văn bản, chương trộn mã Việt–Anh và các dự án đều xoay quanh những khó khăn thật khi làm TTS tiếng Việt.
4. **Mã nguồn chạy được.** Mã minh họa viết bằng PyTorch, được chạy thử trên CPU với dữ liệu nhỏ; các dự án có mã nguồn và kiểm thử tự động trong thư mục `code/` của kho mã.
5. **Trung thực về độ chắc chắn.** Lĩnh vực này thay đổi từng tháng. Những thông tin dễ lỗi thời (mô hình mới, kích thước dữ liệu, giấy phép) được đánh dấu "tính đến thời điểm biên soạn". Chúng tôi không bịa số liệu; khi không chắc, chúng tôi nói rõ là không chắc.

## Cấu trúc sách

Sách gồm 40 chương chia thành 9 phần lý thuyết, cộng 5 dự án thực hành và các phụ lục:

- **Phần I – Nền tảng học sâu** (Chương 1–6): công cụ toán học và kiến trúc mạng.
- **Phần II – Mô hình sinh** (Chương 7–12): toàn bộ "động cơ" sinh dữ liệu dùng trong TTS.
- **Phần III – Tín hiệu và ngữ âm** (Chương 13–16): âm học, xử lý tín hiệu, ngữ âm tiếng Việt, ngôn điệu.
- **Phần IV – Xử lý văn bản** (Chương 17–18): chuẩn hóa văn bản và G2P cho tiếng Việt.
- **Phần V – Mô hình âm học** (Chương 19–23): từ lịch sử TTS tới Tacotron, FastSpeech và các mô hình flow/diffusion.
- **Phần VI – Vocoder** (Chương 24–26): biến đặc trưng âm học thành sóng âm.
- **Phần VII – End-to-end và biểu cảm** (Chương 27–30): VITS, đa người nói, cảm xúc, đa ngôn ngữ.
- **Phần VIII – TTS thời mô hình lớn** (Chương 31–35): codec, codec LM, mô hình zero-shot, speech LLM.
- **Phần IX – Thực tiễn** (Chương 36–40): dữ liệu, huấn luyện, đánh giá, triển khai, đạo đức.
- **Phần X – Dự án**: năm dự án từ frontend tiếng Việt đến dịch vụ TTS streaming.

[Lộ trình học](lo-trinh-hoc.md) gợi ý thứ tự đọc cho từng nhóm người học và kế hoạch học theo tuần.

## Về phiên bản và đóng góp

Đây là phiên bản đầu tiên (2026) của một dự án mở, sẽ tiếp tục được cộng đồng hoàn thiện. Chắc chắn còn những chỗ sai sót, thiếu sót hoặc diễn đạt chưa tốt. Nếu phát hiện, xin bạn mở *issue* hoặc gửi *pull request* trên [GitHub](https://github.com/tuandung222/tts-books-2026) — mỗi đóng góp nhỏ đều giúp hàng nghìn người học sau.

Chúc bạn học tập hiệu quả, và mong sớm được nghe giọng nói tiếng Việt do chính bạn xây dựng!
