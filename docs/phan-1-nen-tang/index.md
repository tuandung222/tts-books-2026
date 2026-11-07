# Phần I. Nền tảng học sâu

Mọi hệ thống TTS hiện đại, từ Tacotron 2 tới các speech LLM, đều được lắp ghép từ một số ít khối cơ bản: phép biến đổi tuyến tính, hàm kích hoạt, tích chập một chiều, mạng hồi quy, cơ chế attention, cùng các thuật toán tối ưu dựa trên gradient. Phần này xây dựng nền móng đó một cách chắc chắn, và luôn chỉ ra mỗi công cụ sẽ được dùng ở đâu trong TTS.

!!! tip "Nếu bạn đã vững học sâu"
    Bạn có thể lướt nhanh phần này. Tuy vậy, hãy đọc kỹ các mục về **trường tiếp nhận của tích chập nhân giãn** (Chương 4), **KV cache, các chiến lược lấy mẫu, AdaLN** (Chương 5) và **xử lý batch âm thanh có độ dài thay đổi** (Chương 6): đây là những chi tiết rất hay gặp khi làm TTS.

## Các chương

| Chương | Nội dung | Dùng lại ở |
|---|---|---|
| [1. Toán nền tảng cho học sâu](ch01-toan-nen-tang.md) | Đại số tuyến tính, giải tích ma trận, xác suất, lý thuyết thông tin | Mọi chương; đặc biệt Phần II |
| [2. Học máy và mạng nơ-ron](ch02-hoc-may-va-mang-no-ron.md) | Hàm mất mát từ góc nhìn hợp lý cực đại, MLP, lan truyền ngược, khởi tạo | Chọn hàm mất mát L1/L2/CE trong TTS |
| [3. Tối ưu hóa và kỹ thuật huấn luyện](ch03-toi-uu-va-ky-thuat-huan-luyen.md) | SGD, Adam/AdamW, lịch tốc độ học, chuẩn hóa, mixed precision, EMA | Huấn luyện mọi mô hình; weight/spectral norm trong vocoder GAN |
| [4. CNN, RNN và mô hình chuỗi](ch04-cnn-rnn-mo-hinh-chuoi.md) | Conv1d, tích chập nhân quả, nhân giãn, LSTM, seq2seq | WaveNet, HiFi-GAN, Tacotron 2 |
| [5. Attention và Transformer](ch05-transformer.md) | Multi-head attention, mã hóa vị trí, RoPE, KV cache, Conformer, AdaLN | FastSpeech, VITS, codec LM, F5-TTS |
| [6. PyTorch cho xử lý tiếng nói](ch06-pytorch-cho-tieng-noi.md) | Pipeline dữ liệu âm thanh, mask, bucketing, checkpoint, AMP, DDP | Mọi dự án ở Phần X |

## Sau phần này, bạn có thể

- Đọc hiểu mọi công thức và sơ đồ kiến trúc trong các bài báo TTS.
- Tự cài đặt, huấn luyện và gỡ lỗi một mô hình PyTorch xử lý chuỗi có độ dài thay đổi.
