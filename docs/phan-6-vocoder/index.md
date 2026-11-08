# Phần VI. Vocoder

Mel-spectrogram đã vứt bỏ thông tin **pha** và nén trục tần số theo thang mel; việc khôi phục một sóng âm tự nhiên, ở 22–48 nghìn mẫu mỗi giây, từ biểu diễn "thiếu thông tin" đó là nhiệm vụ của **vocoder**. Đây cũng là nơi diễn ra nhiều đột phá nhất về tốc độ: từ WaveNet sinh từng mẫu một, chậm hơn thời gian thực hàng trăm lần, đến HiFi-GAN và Vocos nhanh hơn thời gian thực hàng trăm lần trên GPU.

## Các chương

| Chương | Nội dung |
|---|---|
| [24. Vocoder tự hồi quy và flow](ch24-vocoder-tu-hoi-quy-va-flow.md) | WaveNet, WaveRNN, LPCNet, Parallel WaveNet, WaveGlow, mô hình nguồn – bộ lọc nơ-ron |
| [25. Vocoder dựa trên GAN](ch25-gan-vocoder.md) | MelGAN, Parallel WaveGAN, HiFi-GAN, UnivNet, BigVGAN, Vocos |
| [26. Vocoder khuếch tán và so sánh](ch26-diffusion-vocoder-va-so-sanh.md) | WaveGrad, DiffWave, PriorGrad; so sánh toàn diện, chọn vocoder cho dự án |

!!! warning "Hợp đồng đặc trưng"
    Mô hình âm học và vocoder phải dùng **cùng một cấu hình trích xuất mel** (tần số lấy mẫu, kích thước FFT, bước nhảy, số kênh mel, `fmin`/`fmax`, công thức mel, cách lấy log và chuẩn hóa). Sai lệch nhỏ ở đây là nguyên nhân phổ biến hàng đầu của âm thanh rè, méo. Xem [Chương 14](../phan-3-tin-hieu-ngu-am/ch14-xu-ly-tin-hieu-so.md).
