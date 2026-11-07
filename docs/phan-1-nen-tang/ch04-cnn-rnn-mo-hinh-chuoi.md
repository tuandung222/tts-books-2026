# Chương 4. CNN, RNN và mô hình chuỗi

!!! abstract "Mục tiêu học tập"
    Sau chương này, bạn có thể:

    - **Tính** hình dạng đầu ra, số tham số và trường tiếp nhận (*receptive field*) của một chồng `Conv1d` bất kỳ (stride, padding, dilation, groups), rồi kiểm chứng bằng thực nghiệm.
    - **Cài đặt** tích chập nhân quả giãn nở (*causal dilated convolution*) và chứng minh bằng kiểm thử rằng nó không "nhìn trộm" tương lai.
    - **Giải thích** cơ chế của tích chập chuyển vị (*transposed convolution*), nguồn gốc hiện tượng bàn cờ (*checkerboard artifacts*) và các cách khắc phục trong vocoder.
    - **So sánh** các khối xây dựng: khối dư, đơn vị kích hoạt có cổng, tích chập tách theo chiều sâu, khối ConvNeXt 1D.
    - **Suy ra** lan truyền ngược theo thời gian (BPTT), giải thích vì sao gradient tiêu biến hoặc bùng nổ và vì sao LSTM, GRU giảm nhẹ được vấn đề này.
    - **Xây dựng** một mô hình seq2seq LSTM có attention; phân tích teacher forcing và exposure bias.
    - **So sánh** các chiến lược giải mã greedy, beam search, lấy mẫu và chỉ ra khi nào mỗi chiến lược phù hợp với TTS.

!!! info "Kiến thức cần có"
    - [Chương 1](ch01-toan-nen-tang.md): nhân ma trận, chuẩn ma trận, giá trị kỳ dị, quy tắc chuỗi cho véc-tơ.
    - [Chương 2](ch02-hoc-may-va-mang-no-ron.md): MLP, hàm kích hoạt, lan truyền ngược, entropy chéo.
    - [Chương 3](ch03-toi-uu-va-ky-thuat-huan-luyen.md): Adam, cắt gradient (*gradient clipping*), LayerNorm.
    - Các khối mã trong chương được thiết kế để **chạy tuần tự như một notebook** (khối sau có thể dùng lại hàm của khối trước), trên CPU, mỗi khối mất từ vài giây đến khoảng hai chục giây.

Tiếng nói là một chuỗi: chuỗi mẫu âm thanh, chuỗi khung mel-spectrogram, chuỗi âm vị, chuỗi token. Gần như mọi mô hình TTS mà bạn sẽ gặp trong cuốn sách này đều được lắp ráp từ một số ít "viên gạch" xử lý chuỗi: tích chập một chiều (trong WaveNet, HiFi-GAN, FastSpeech, Vocos), mạng hồi quy (trong Tacotron 2, WaveRNN, LPCNet), và attention (trong mọi thứ, từ Tacotron đến các codec language model). Chương này trình bày hai họ đầu tiên một cách kỹ lưỡng, rồi giới thiệu attention trong khung seq2seq như một bước đệm cho [Chương 5](ch05-transformer.md).

Mục tiêu không chỉ là biết "lớp này làm gì", mà là **tính toán được** chúng: một kỹ sư TTS cần trả lời ngay được những câu như "vocoder này nhìn được bao nhiêu mili-giây ngữ cảnh?", "vì sao lớp nâng mẫu này gây tiếng rè?", "vì sao attention của tôi lặp từ?".

## 4.1 Vì sao cần kiến trúc riêng cho dữ liệu chuỗi

Hãy bắt đầu bằng một phép tính. Một giây âm thanh ở tần số lấy mẫu $f_s = 22\,050$ Hz có 22 050 mẫu. Một lớp nối đầy đủ (*fully connected*) ánh xạ 22 050 mẫu sang 22 050 đơn vị ẩn cần $22\,050^2 \approx 4{,}9 \times 10^8$ trọng số — chỉ cho một lớp, và chỉ dùng được cho đúng những đoạn âm thanh dài đúng một giây. MLP của [Chương 2](ch02-hoc-may-va-mang-no-ron.md) rõ ràng không phải là công cụ phù hợp.

Ba tính chất của tín hiệu tiếng nói gợi ý những thiết kế tốt hơn:

1. **Tính cục bộ.** Các mẫu gần nhau tương quan rất mạnh. Nhiều cấu trúc quan trọng có độ dài ngắn: một chu kỳ cao độ chỉ dài 2–20 ms (ứng với F0 từ 500 Hz xuống 50 Hz), một chuyển tiếp formant kéo dài vài chục mili-giây.
2. **Tính bất biến (gần đúng) theo tịnh tiến.** Nguyên âm /a/ ở đầu câu hay cuối câu có đặc trưng phổ tương tự nhau. Một bộ phát hiện mẫu hình học được ở vị trí này nên dùng lại được ở vị trí khác — tức là **chia sẻ trọng số theo thời gian**.
3. **Độ dài thay đổi và phụ thuộc xa.** Câu dài ngắn khác nhau; ngữ điệu ở đầu câu có thể phụ thuộc vào một tiểu từ ở cuối câu.

Hai họ kiến trúc kinh điển khai thác các tính chất này theo hai cách khác nhau:

- **Mạng tích chập (CNN)** áp đặt tính cục bộ và chia sẻ trọng số; mọi vị trí được tính **song song**. Ngữ cảnh xa đạt được bằng cách xếp chồng nhiều lớp.
- **Mạng hồi quy (RNN)** duy trì một **trạng thái ẩn** tóm tắt toàn bộ quá khứ; về nguyên tắc nó nhìn được ngữ cảnh dài vô hạn, nhưng phải xử lý **tuần tự** từng bước.

Attention, chủ đề của [Chương 5](ch05-transformer.md), là họ thứ ba: mỗi vị trí truy cập trực tiếp mọi vị trí khác.

## 4.2 Tích chập một chiều (Conv1d)

### 4.2.1 Định nghĩa

**Trực giác.** Một bộ lọc tích chập là một "cửa sổ trượt": tại mỗi vị trí thời gian, nó lấy một đoạn ngắn của đầu vào, nhân từng phần tử với một bộ trọng số rồi cộng lại. Cùng một bộ trọng số được dùng ở mọi vị trí.

**Hình thức hóa.** Gọi đầu vào là $\mathbf{X} \in \mathbb{R}^{C_{\text{in}} \times T_{\text{in}}}$ (số kênh × số bước thời gian), trọng số $\mathcal{W} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times k}$ với $k$ là **kích thước kernel**, độ lệch $\mathbf{b} \in \mathbb{R}^{C_{\text{out}}}$. Với bước nhảy (*stride*) $s$, đệm (*padding*) $p$ và độ giãn (*dilation*) $d$, kênh ra $o$ tại vị trí $t$ là

$$
y_{o}[t] = b_o + \sum_{c=1}^{C_{\text{in}}} \sum_{m=0}^{k-1} \mathcal{W}_{o,c,m}\; x_c[\,s\,t + d\,m - p\,],
$$

trong đó $x_c[n] = 0$ nếu $n$ nằm ngoài $[0, T_{\text{in}} - 1]$ (đệm số không). Công thức nói rằng: đầu ra tại $t$ là tổ hợp tuyến tính của $k$ mẫu đầu vào, cách nhau $d$ bước, bắt đầu từ vị trí $s\,t - p$, trên **mọi** kênh vào.

Hai lưu ý về thuật ngữ:

- Trong xử lý tín hiệu, tích chập $(x \ast w)[t] = \sum_m w[m]\,x[t - m]$ có **lật** kernel. Các thư viện học sâu thực ra tính *tương quan chéo* (*cross-correlation*, không lật). Vì $\mathcal{W}$ được học, hai cách là tương đương — mô hình chỉ học ra kernel đã lật.
- Bố cục tensor của `nn.Conv1d` trong PyTorch là `(B, C, T)`: batch, kênh, thời gian. Mel-spectrogram thường được lưu dạng `(B, T, F)` nên phải `transpose(1, 2)` trước khi đưa vào `Conv1d`. Đây là nguồn lỗi im lặng số một (xem mục Lỗi thường gặp).

### 4.2.2 Độ dài đầu ra

Kernel có độ giãn $d$ phủ một khoảng $d(k-1) + 1$ mẫu, gọi là **kích thước kernel hiệu dụng**. Vị trí đầu ra cuối cùng $t$ hợp lệ khi điểm cuối của cửa sổ còn nằm trong đầu vào đã đệm: $s\,t + d(k-1) - p \le T_{\text{in}} - 1 + p$. Giải ra $t$ và đếm cả vị trí $t = 0$:

$$
T_{\text{out}} = \left\lfloor \frac{T_{\text{in}} + 2p - d(k-1) - 1}{s} \right\rfloor + 1 .
$$

Hệ quả quan trọng:

- Với $s = 1$, chọn $p = d(k-1)/2$ thì $T_{\text{out}} = T_{\text{in}}$ (đệm "same"). Điều này chỉ đối xứng được khi $d(k-1)$ chẵn — đó là lý do kernel lẻ (3, 5, 7…) phổ biến.
- Với $s > 1$, độ dài giảm khoảng $s$ lần: đây là cách **hạ mẫu** (*downsampling*) có học.

### 4.2.3 Số tham số, chi phí và tham số `groups`

Một lớp `Conv1d` có $C_{\text{out}} \cdot C_{\text{in}} \cdot k + C_{\text{out}}$ tham số và cần khoảng $T_{\text{out}} \cdot C_{\text{out}} \cdot C_{\text{in}} \cdot k$ phép nhân–cộng. Số tham số **không phụ thuộc vào độ dài chuỗi** — khác hẳn lớp nối đầy đủ ở mục 4.1.

Tham số `groups` $= g$ chia các kênh thành $g$ nhóm độc lập; mỗi kênh ra chỉ nhìn $C_{\text{in}}/g$ kênh vào, nên số tham số và chi phí giảm $g$ lần. Trường hợp cực đoan $g = C_{\text{in}}$ gọi là **tích chập theo chiều sâu** (*depthwise convolution*): mỗi kênh được lọc riêng, không trộn kênh. Ta sẽ dùng nó ở mục 4.6.

### 4.2.4 Tích chập là phép nhân ma trận

Vì $y_o[t]$ tuyến tính theo $\mathbf{X}$, ta có thể duỗi $\mathbf{X}$ thành véc-tơ $\mathbf{x}$ và viết $\mathbf{y} = \mathbf{C}\mathbf{x}$, với $\mathbf{C}$ là một **ma trận Toeplitz theo khối**, thưa (mỗi hàng chỉ có $C_{\text{in}} \cdot k$ phần tử khác không) và có các phần tử lặp lại dọc đường chéo (chia sẻ trọng số). Góc nhìn này hữu ích ở hai chỗ: (i) nó cho thấy CNN là một MLP bị **ràng buộc cấu trúc** — cục bộ và chia sẻ trọng số — chính là các thiên kiến quy nạp (*inductive bias*) ta muốn từ mục 4.1; (ii) nó cho ta định nghĩa sạch sẽ của tích chập chuyển vị ở mục 4.5: phép nhân với $\mathbf{C}^\top$.

**Mã minh họa.** Hàm dưới đây cài đặt công thức độ dài và đối chiếu với PyTorch trên một lưới tham số. Hàm cho tích chập chuyển vị sẽ được giải thích ở mục 4.5.

```python
import torch
import torch.nn as nn

def conv1d_out_len(t_in, k, s=1, p=0, d=1):
    """Độ dài đầu ra của Conv1d."""
    return (t_in + 2 * p - d * (k - 1) - 1) // s + 1

def conv_transpose1d_out_len(t_in, k, s=1, p=0, d=1, output_padding=0):
    """Độ dài đầu ra của ConvTranspose1d (xem mục 4.5)."""
    return (t_in - 1) * s - 2 * p + d * (k - 1) + output_padding + 1

# Đối chiếu với PyTorch trên một lưới tham số nhỏ
for t_in in [7, 16, 101]:
    for k, s, p, d in [(3, 1, 1, 1), (5, 2, 2, 1), (3, 1, 4, 4), (4, 3, 0, 2)]:
        x = torch.randn(1, 2, t_in)                          # (B, C, T)
        y = nn.Conv1d(2, 3, k, stride=s, padding=p, dilation=d)(x)
        assert y.shape[-1] == conv1d_out_len(t_in, k, s, p, d)
        z = nn.ConvTranspose1d(2, 3, k, stride=s, padding=p, dilation=d)(x)
        assert z.shape[-1] == conv_transpose1d_out_len(t_in, k, s, p, d)

# Lớp nâng mẫu kiểu HiFi-GAN: k = 2s, p = s/2  =>  T_out = s * T_in
for s in [8, 2]:
    assert conv_transpose1d_out_len(50, k=2 * s, s=s, p=s // 2) == s * 50
print("Công thức độ dài khớp với PyTorch")
```

## 4.3 Tích chập nhân quả và tích chập giãn nở

### 4.3.1 Tính nhân quả

Một mô hình **tự hồi quy** (*autoregressive*) sinh mẫu $x[t]$ dựa trên các mẫu trước đó $x[0], \ldots, x[t-1]$ (xem [Chương 7](../phan-2-mo-hinh-sinh/ch07-tong-quan-mo-hinh-sinh.md)). Khi huấn luyện, ta muốn tính dự đoán cho **mọi** $t$ song song bằng một lần chạy tích chập, nhưng phải bảo đảm đầu ra tại $t$ không phụ thuộc vào đầu vào tại các thời điểm sau $t$. Tích chập như vậy gọi là **nhân quả** (*causal*).

Với $s = 1$, cách cài đặt đơn giản là **đệm toàn bộ $(k-1)d$ mẫu ở bên trái, không đệm bên phải**. Khi đó, trong công thức ở mục 4.2.1 với $p$ thay bằng việc dịch trái, chỉ số đầu vào lớn nhất mà $y[t]$ dùng là $t$:

$$
y[t] = \sum_{m=0}^{k-1} w_m\, x[\,t - d\,(k-1-m)\,].
$$

Tích chập nhân quả là xương sống của WaveNet (van den Oord và cộng sự, 2016) và của mọi mô hình **streaming**: khi âm thanh hay token đến từng đoạn, mô hình không được phép chờ tương lai ([Chương 35](../phan-8-mo-hinh-lon/ch35-streaming-speech-llm.md)). Ngược lại, các mô hình ngoại tuyến như bộ sinh HiFi-GAN hay bộ mã hóa văn bản dùng tích chập **đối xứng** (không nhân quả), vì toàn bộ đầu vào đã có sẵn.

### 4.3.2 Tích chập giãn nở

Với kernel $k$ và $L$ lớp nhân quả thông thường ($d = 1$), mỗi đầu ra nhìn được $1 + L(k-1)$ mẫu — tăng **tuyến tính** theo độ sâu. Để nhìn 0,1 giây ở 22 050 Hz (2 205 mẫu) với $k = 2$ ta cần hơn 2 000 lớp. Không khả thi.

**Tích chập giãn nở** (*dilated convolution*) chèn khoảng cách $d$ giữa các nhịp của kernel. Nếu cho độ giãn tăng theo cấp số nhân $d_\ell = 2^{\ell}$ với $\ell = 0, \ldots, M-1$ và $k = 2$, trường tiếp nhận sau $M$ lớp là

$$
1 + \sum_{\ell=0}^{M-1} (2-1) \cdot 2^{\ell} = 2^{M},
$$

tức là tăng **theo hàm mũ** của độ sâu, trong khi số tham số vẫn tăng tuyến tính. WaveNet dùng chính lịch độ giãn $1, 2, 4, \ldots, 512$ và lặp lại khối này vài lần.

Có một cách nhìn đẹp vì sao lịch lũy thừa của 2 "phủ kín" quá khứ mà không bỏ sót hay trùng lặp: mỗi lớp $\ell$ chọn dịch thêm $0$ hoặc $2^{\ell}$ mẫu, nên tổng độ dịch là $\sum_\ell b_\ell 2^\ell$ với $b_\ell \in \{0, 1\}$ — chính là **biểu diễn nhị phân** của mọi số từ $0$ đến $2^M - 1$, mỗi số đúng một lần. Tương tự, kernel $k = 3$ với độ giãn $1, 3, 9, \ldots$ ứng với biểu diễn cơ số 3. Nếu chọn độ giãn không khéo (ví dụ mọi lớp đều $d = 4$), một số vị trí trong quá khứ không bao giờ được nhìn thấy — hiện tượng "lưới thưa" (*gridding*).

**Mã minh họa.** Lớp `CausalConv1d` dưới đây đệm trái rồi gọi `Conv1d` gốc. Hai phép kiểm thử quan trọng nhất cho mọi lớp nhân quả: (1) thay đổi tương lai không làm thay đổi quá khứ; (2) đầu ra tại $t$ chỉ phụ thuộc đúng cửa sổ trường tiếp nhận.

```python
import torch.nn.functional as F

class CausalConv1d(nn.Conv1d):
    """Conv1d nhân quả: đệm (k-1)*d mẫu ở bên trái, không đệm bên phải."""
    def __init__(self, c_in, c_out, kernel_size, dilation=1):
        super().__init__(c_in, c_out, kernel_size, dilation=dilation, padding=0)
        self.left_pad = (kernel_size - 1) * dilation

    def forward(self, x):                         # x: (B, C_in, T)
        x = F.pad(x, (self.left_pad, 0))          # (B, C_in, T + left_pad)
        return super().forward(x)                 # (B, C_out, T)

class DilatedCausalStack(nn.Module):
    """Chồng tích chập nhân quả giãn nở có kết nối dư."""
    def __init__(self, channels=16, kernel_size=3, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.layers = nn.ModuleList(
            CausalConv1d(channels, channels, kernel_size, d) for d in dilations)
        self.rf = 1 + sum((kernel_size - 1) * d for d in dilations)

    def forward(self, x):                         # x: (B, C, T)
        for conv in self.layers:
            x = x + torch.tanh(conv(x))           # giữ nguyên (B, C, T)
        return x

torch.manual_seed(0)
net = DilatedCausalStack()
x = torch.randn(2, 16, 100)                       # (B, C, T)
y = net(x)
assert y.shape == x.shape

# Kiểm thử 1 — tính nhân quả: thay đổi tương lai (t >= 60) không làm đổi quá khứ
x2 = x.clone()
x2[:, :, 60:] = torch.randn(2, 16, 40)
y2 = net(x2)
assert torch.allclose(y[:, :, :60], y2[:, :, :60])
assert not torch.allclose(y[:, :, 60:], y2[:, :, 60:])

# Kiểm thử 2 — trường tiếp nhận: đầu ra tại t chỉ phụ thuộc x[t-rf+1 .. t]
x3 = x.clone()
x3[:, :, : 80 - net.rf + 1] = 0.0                 # xóa mọi thứ ngoài cửa sổ của t = 80
assert torch.allclose(net(x3)[:, :, 80], y[:, :, 80], atol=1e-6)
print("Trường tiếp nhận:", net.rf, "mẫu; tính nhân quả: OK")
```

## 4.4 Trường tiếp nhận

**Trường tiếp nhận** (*receptive field*) của một đơn vị đầu ra là tập các mẫu đầu vào có thể ảnh hưởng tới nó. Đây là con số quan trọng bậc nhất khi thiết kế vocoder và bộ dự đoán ngôn điệu: một vocoder có trường tiếp nhận 5 ms không thể "biết" cao độ của một giọng nam trầm có chu kỳ 10 ms; một bộ dự đoán cao độ chỉ nhìn một âm vị không thể dựng được đường nét ngữ điệu của cả câu.

### 4.4.1 Công thức truy hồi

Xét một chồng $L$ lớp tích chập, lớp $\ell$ có kernel $k_\ell$, stride $s_\ell$, độ giãn $d_\ell$. Định nghĩa hai đại lượng, đo bằng **số mẫu đầu vào gốc**:

- $r_\ell$: trường tiếp nhận của một đơn vị ở đầu ra lớp $\ell$;
- $j_\ell$: **bước nhảy tích lũy** (*jump*) — khoảng cách (theo mẫu đầu vào gốc) giữa hai đơn vị kề nhau ở đầu ra lớp $\ell$.

Khởi đầu $r_0 = 1$, $j_0 = 1$ (mỗi mẫu đầu vào chỉ "nhìn" chính nó). Khi đó

$$
r_\ell = r_{\ell-1} + (k_\ell - 1)\, d_\ell\, j_{\ell-1}, \qquad j_\ell = j_{\ell-1}\, s_\ell .
$$

**Chứng minh.** Một đơn vị ở lớp $\ell$ đọc $k_\ell$ đơn vị của lớp $\ell-1$, cách nhau $d_\ell$ vị trí, tức là trải trên $(k_\ell - 1) d_\ell$ "bước" của lớp $\ell - 1$. Mỗi bước đó tương ứng với $j_{\ell-1}$ mẫu đầu vào gốc. Trường tiếp nhận của đơn vị ngoài cùng bên trái và ngoài cùng bên phải đều có độ rộng $r_{\ell-1}$, và chúng lệch nhau $(k_\ell - 1) d_\ell j_{\ell-1}$ mẫu. Hợp của chúng (cùng mọi đơn vị ở giữa) phủ một đoạn dài $r_{\ell-1} + (k_\ell - 1) d_\ell j_{\ell-1}$. Còn bước nhảy thì nhân thêm $s_\ell$ vì lớp $\ell$ chỉ giữ một trong mỗi $s_\ell$ vị trí. $\blacksquare$

Khai triển truy hồi cho dạng đóng:

$$
r_L = 1 + \sum_{\ell=1}^{L} (k_\ell - 1)\, d_\ell \prod_{i=1}^{\ell-1} s_i .
$$

Công thức nói rằng các lớp càng sâu (sau nhiều lần hạ mẫu) thì mỗi nhịp kernel càng "đáng giá" nhiều mẫu đầu vào. Ba nhận xét thực hành:

- **Kết nối dư** không làm tăng trường tiếp nhận: nhánh tắt có trường tiếp nhận nhỏ hơn, và hợp của hai nhánh bằng nhánh lớn hơn.
- Công thức cho **độ rộng** của vùng ảnh hưởng. Nếu lịch độ giãn tạo "lưới thưa", bên trong vùng này có thể có những mẫu không ảnh hưởng gì.
- Với lớp **nâng mẫu** (mục 4.5), bước nhảy bị chia cho hệ số nâng mẫu; khi đó cách chắc chắn nhất là **đo trực tiếp** bằng gradient như trong mã dưới đây.

### 4.4.2 Ví dụ tính toán

**WaveNet.** Với $k = 2$ và độ giãn $1, 2, \ldots, 512$ (10 lớp): $r = 1 + (1 + 2 + \cdots + 512) = 1024$ mẫu. Lặp lại khối này 3 lần: $r = 1 + 3 \times 1023 = 3070$ mẫu, tức khoảng 139 ms ở 22 050 Hz. (Cấu hình cụ thể trong bài báo gốc thay đổi theo thí nghiệm; ở đây ta chỉ dùng nó làm ví dụ tính.)

**Khối dư của HiFi-GAN.** Cấu hình V1 của HiFi-GAN (Kong và cộng sự, 2020) dùng ba khối dư song song với kernel $3, 7, 11$; mỗi khối gồm ba cặp tích chập, cặp thứ $i$ là một tích chập giãn $d_i \in \{1, 3, 5\}$ theo sau bởi một tích chập $d = 1$. Trường tiếp nhận của một khối với kernel $k$ là

$$
r = 1 + (k-1)\,(1 + 1 + 3 + 1 + 5 + 1) = 1 + 12\,(k-1),
$$

tức 25, 73 và 121 vị trí cho $k = 3, 7, 11$. Điều thú vị là cùng một số vị trí lại ứng với thời lượng rất khác nhau tùy độ phân giải của tầng. Với hop $H = 256$, sau lớp nâng mẫu đầu tiên ($\times 8$) mỗi vị trí ứng với $256/8 = 32$ mẫu âm thanh, nên 121 vị trí phủ $121 \times 32 = 3872$ mẫu ≈ 176 ms; ở tầng cuối cùng (đủ 22 050 Hz) thì 121 vị trí chỉ là ≈ 5,5 ms. Tầng thô nắm cấu trúc dài (cao độ, đường bao năng lượng), tầng mịn lo chi tiết dạng sóng — đó là tinh thần "đa trường tiếp nhận" (*multi-receptive field*) của HiFi-GAN ([Chương 25](../phan-6-vocoder/ch25-gan-vocoder.md)).

**Đổi đơn vị.** Trường tiếp nhận $r$ mẫu ứng với $r / f_s$ giây; $r$ khung mel ứng với $r \cdot H / f_s$ giây. Ví dụ 25 khung với $H = 256$, $f_s = 22\,050$ Hz là khoảng 0,29 s.

### 4.4.3 Trường tiếp nhận lý thuyết và hiệu dụng

Công thức trên cho trường tiếp nhận **lý thuyết**: vùng mà một mẫu *có thể* ảnh hưởng. Luo và cộng sự (2016) chỉ ra rằng ảnh hưởng thực tế (đo bằng độ lớn gradient) thường tập trung ở giữa vùng, có dạng gần với hình chuông, và **trường tiếp nhận hiệu dụng** chỉ chiếm một phần nhỏ của trường lý thuyết; với chồng các lớp giống nhau, độ rộng hiệu dụng tăng chậm hơn nhiều (theo phân tích của họ, xấp xỉ theo căn bậc hai của số lớp). Bài học thực hành: hãy thiết kế trường tiếp nhận lý thuyết **dư dả** hơn nhu cầu, và khi nghi ngờ, đo ảnh hưởng thực tế bằng gradient.

**Mã minh họa.** Bộ tính trường tiếp nhận và phép kiểm chứng thực nghiệm: gán trọng số dương cho mọi lớp (để tránh các đóng góp triệt tiêu ngẫu nhiên), lan truyền ngược từ một đầu ra ở giữa và đếm các mẫu đầu vào có gradient khác không.

```python
def receptive_field(layers):
    """layers: danh sách (kernel_size, stride, dilation) theo thứ tự từ đầu vào.
    Trả về (r, j): trường tiếp nhận và bước nhảy tích lũy, tính theo mẫu đầu vào."""
    r, j = 1, 1
    for k, s, d in layers:
        r = r + (k - 1) * d * j
        j = j * s
    return r, j

def empirical_rf(model, t_in=4096, c_in=1):
    """Đo trường tiếp nhận bằng gradient: những mẫu đầu vào nào ảnh hưởng tới 1 đầu ra."""
    x = torch.zeros(1, c_in, t_in, requires_grad=True)       # (B, C, T)
    y = model(x)                                               # (B, C', T')
    y[0, :, y.shape[-1] // 2].sum().backward()                 # chọn đầu ra ở giữa
    nz = (x.grad[0].abs().sum(0) > 0).nonzero().squeeze(-1)
    return int(nz.max() - nz.min() + 1)

def build_stack(layers, channels=4):
    mods, c_in = [], 1
    for k, s, d in layers:
        conv = nn.Conv1d(c_in, channels, k, stride=s, dilation=d)
        nn.init.constant_(conv.weight, 0.1)   # trọng số dương: tránh triệt tiêu ngẫu nhiên
        mods += [conv, nn.Tanh()]             # tanh'(0) = 1 nên gradient không bị chặn
        c_in = channels
    return nn.Sequential(*mods)

# Ví dụ 1: kiểu WaveNet, k = 2, độ giãn 1, 2, ..., 512 (10 lớp)
wavenet = [(2, 1, 2 ** i) for i in range(10)]
print("WaveNet 1 khối:", receptive_field(wavenet))           # (1024, 1)
print("WaveNet 3 khối:", receptive_field(wavenet * 3))       # (3070, 1)

# Ví dụ 2: có stride — mỗi nhịp kernel ở lớp sâu "đáng giá" nhiều mẫu hơn
strided = [(4, 2, 1), (4, 2, 1), (3, 1, 2)]
print("Có stride:", receptive_field(strided))                 # (26, 4)

# Ví dụ 3: khối dư HiFi-GAN với kernel 11, độ giãn (1, 3, 5), xen kẽ d = 1
hifigan_rb = [(11, 1, d) for pair in [(1, 1), (3, 1), (5, 1)] for d in pair]
print("Khối dư HiFi-GAN, k=11:", receptive_field(hifigan_rb)) # (121, 1)

for cfg in [wavenet, strided, hifigan_rb, [(5, 1, 1)] * 3]:
    assert receptive_field(cfg)[0] == empirical_rf(build_stack(cfg))
print("Công thức khớp với đo đạc bằng gradient")
```

!!! note "Góc tiếng Việt"
    Thanh điệu tiếng Việt là một đường nét cao độ trải trên **toàn bộ phần vần** của âm tiết, và hình dạng của nó còn chịu ảnh hưởng đồng cấu âm từ các âm tiết lân cận ([Chương 15](../phan-3-tin-hieu-ngu-am/ch15-ngu-am-tieng-viet.md), [Chương 16](../phan-3-tin-hieu-ngu-am/ch16-ngon-dieu.md)). Hệ quả cho thiết kế:

    - Bộ mã hóa văn bản theo **ký tự** phải có trường tiếp nhận đủ phủ cả âm tiết để "biết" dấu thanh thuộc về âm tiết nào. Ví dụ, bộ mã hóa của Tacotron 2 có ba lớp tích chập kernel 5, nên trường tiếp nhận của phần tích chập là $1 + 3 \times 4 = 13$ ký tự — phủ được một âm tiết dài như "nghiêng" (7 chữ cái) cùng phần lân cận. Phần BiLSTM phía sau (mục 4.7.6) mở rộng ngữ cảnh ra cả câu.
    - Bộ dự đoán cao độ/trường độ ở mức khung hoặc âm vị nên nhìn được ít nhất **âm tiết đứng trước và đứng sau**. Một bộ dự đoán chỉ nhìn vài khung sẽ sinh đường F0 đúng xu hướng chung nhưng sai hình dạng thanh (ví dụ thanh hỏi không "gãy" xuống rồi lên).
