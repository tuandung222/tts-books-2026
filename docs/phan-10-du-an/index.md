# Phần X. Dự án thực hành

Năm dự án dưới đây đưa bạn từ một frontend tiếng Việt tới một dịch vụ TTS streaming hoàn chỉnh. Mã nguồn nằm trong thư mục [`code/`](https://github.com/tuandung222/tts-books-2026/tree/main/code) của kho mã, có kiểm thử tự động chạy trên CPU.

| Dự án | Bạn sẽ xây dựng | Kiến thức cần có |
|---|---|---|
| [1. Frontend tiếng Việt](du-an-1-frontend-tieng-viet.md) | Gói Python chuẩn hóa văn bản + G2P tiếng Việt | Chương 15, 17, 18 |
| [2. FastSpeech 2 + HiFi-GAN](du-an-2-fastspeech2-hifigan.md) | Hệ thống hai tầng hoàn chỉnh | Chương 14, 21, 22, 25 |
| [3. VITS tiếng Việt](du-an-3-vits-tieng-viet.md) | Mô hình end-to-end, xuất ONNX chạy trên thiết bị biên | Chương 27, 28 |
| [4. Zero-shot TTS tiếng Việt](du-an-4-zero-shot-tts.md) | Tinh chỉnh một mô hình zero-shot mã nguồn mở cho tiếng Việt | Chương 33, 34, 36, 38 |
| [5. Dịch vụ TTS streaming](du-an-5-tts-service-streaming.md) | Dịch vụ FastAPI + WebSocket, đo độ trễ | Chương 35, 39 |

## Chạy mã nguồn

```bash
git clone https://github.com/tuandung222/tts-books-2026.git
cd tts-books-2026/code
pip install -r requirements.txt
python -m pytest -q
```

!!! warning "Dữ liệu và giấy phép"
    Các dự án không kèm dữ liệu âm thanh. Trước khi dùng bất kỳ bộ dữ liệu hay mô hình huấn luyện sẵn nào, hãy đọc kỹ giấy phép của nó, và **chỉ nhân bản giọng nói khi có sự đồng ý của người sở hữu giọng** (xem [Chương 40](../phan-9-thuc-tien/ch40-dao-duc-an-toan.md)).
