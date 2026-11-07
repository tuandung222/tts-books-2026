# Phần IV. Xử lý văn bản (Frontend)

Trước khi một mô hình nơ-ron "nhìn thấy" văn bản, văn bản phải được **chuẩn hóa** — biến "Ngày 2/9/1945, lúc 14h, tại TP.HCM có 1.250 người, giá 50.000đ" thành chuỗi chữ có thể đọc thành tiếng — và thường được chuyển thành **âm vị** kèm **thanh điệu**. Trong thực tế sản phẩm, rất nhiều lỗi "đọc sai" mà người dùng phàn nàn không đến từ mô hình âm học hay vocoder, mà đến từ frontend.

## Các chương

| Chương | Nội dung |
|---|---|
| [17. Chuẩn hóa văn bản tiếng Việt](ch17-chuan-hoa-van-ban.md) | Unicode NFC/NFD, cách đặt dấu thanh, đọc số, ngày giờ, tiền tệ, đơn vị, chữ viết tắt; phương pháp dựa trên luật, mạng nơ-ron và LLM |
| [18. G2P và biểu diễn đầu vào](ch18-g2p-bieu-dien-dau-vao.md) | Ký tự, âm vị, byte, BPE; G2P dựa trên luật cho tiếng Việt; biểu diễn thanh điệu; từ mượn |

!!! tip "Thực hành ngay"
    Sau khi học xong hai chương này, hãy làm [Dự án 1: Frontend tiếng Việt](../phan-10-du-an/du-an-1-frontend-tieng-viet.md) — bạn sẽ có một gói Python dùng lại được cho mọi dự án sau.

!!! info "Kiến thức cần có"
    [Chương 15](../phan-3-tin-hieu-ngu-am/ch15-ngu-am-tieng-viet.md) (ngữ âm và âm vị học tiếng Việt).
