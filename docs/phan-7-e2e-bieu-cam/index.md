# Phần VII. End-to-end và biểu cảm

Phần này đi qua hai hướng phát triển lớn sau thế hệ hai tầng: **gộp toàn bộ hệ thống vào một mô hình huấn luyện đầu-cuối** (VITS và họ hàng), và **làm cho giọng nói giàu biểu cảm, đa dạng**: nhiều người nói, nhiều phong cách, nhiều cảm xúc, nhiều ngôn ngữ — kể cả câu trộn tiếng Việt và tiếng Anh vốn rất phổ biến trong đời sống.

## Các chương

| Chương | Nội dung |
|---|---|
| [27. VITS và họ end-to-end](ch27-vits.md) | CVAE + flow + GAN + MAS; VITS2, JETS, MB-iSTFT-VITS |
| [28. Đa người nói và nhân bản giọng](ch28-da-nguoi-noi-voice-cloning.md) | Embedding người nói, mô hình xác minh người nói, thích nghi, zero-shot |
| [29. Ngôn điệu, cảm xúc và phong cách](ch29-prosody-cam-xuc-phong-cach.md) | Reference encoder, GST, StyleTTS 2, điều khiển bằng mô tả ngôn ngữ tự nhiên |
| [30. Đa ngôn ngữ và trộn mã Việt–Anh](ch30-da-ngon-ngu-code-switching.md) | Bộ ký hiệu chung, tách người nói – ngôn ngữ, xử lý từ tiếng Anh trong câu tiếng Việt |

!!! tip "Thực hành"
    [Dự án 3: VITS tiếng Việt](../phan-10-du-an/du-an-3-vits-tieng-viet.md) áp dụng trực tiếp Chương 27–28.
