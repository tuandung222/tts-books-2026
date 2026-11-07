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

## 4.5 Tích chập chuyển vị và hiện tượng bàn cờ

### 4.5.1 Từ khung mel tới mẫu âm thanh: bài toán nâng mẫu

Một vocoder nhận mel-spectrogram có khoảng $f_s / H$ khung mỗi giây (với $H = 256$, $f_s = 22\,050$ Hz là khoảng 86 khung/giây) và phải sinh ra $f_s$ mẫu mỗi giây: mỗi khung phải "nở" thành $H$ mẫu. Các vocoder GAN như MelGAN và HiFi-GAN làm việc này bằng một dãy lớp **nâng mẫu có học** (*learned upsampling*) với tích các hệ số bằng đúng $H$; ví dụ HiFi-GAN V1 dùng các hệ số $8 \times 8 \times 2 \times 2 = 256$.

### 4.5.2 Định nghĩa

Ở mục 4.2.4 ta thấy một tích chập stride $s$ là phép nhân với ma trận $\mathbf{C} \in \mathbb{R}^{T_{\text{out}} \times T_{\text{in}}}$ "béo" (ít hàng hơn cột), ánh xạ chuỗi dài thành chuỗi ngắn. **Tích chập chuyển vị** là phép nhân với $\mathbf{C}^\top$: nó ánh xạ chuỗi ngắn trở lại chuỗi dài. Lưu ý $\mathbf{C}^\top$ **không phải** nghịch đảo của $\mathbf{C}$; nó chỉ có cùng "mẫu hình kết nối" theo chiều ngược lại. (Tên gọi "deconvolution" vì thế là gây hiểu nhầm.)

Một cách hình dung tương đương và trực quan hơn là **góc nhìn "rải"** (*scatter*): mỗi mẫu đầu vào $x[n]$ cộng một bản sao của kernel, nhân với $x[n]$, vào các vị trí đầu ra $n s, n s + 1, \ldots, n s + k - 1$ (khi $d = 1$):

$$
y[t] = \sum_{n} \sum_{m=0}^{k-1} \mathbb{1}[\,t = n s + m\,]\; w_m\, x[n] .
$$

Từ đó suy ra độ dài đầu ra (cùng công thức ta đã dùng trong mã ở mục 4.2):

$$
T_{\text{out}} = (T_{\text{in}} - 1)\, s - 2p + d\,(k-1) + p_{\text{out}} + 1,
$$

với $p_{\text{out}}$ là tham số `output_padding` dùng để khử sự mơ hồ về độ dài. Một lựa chọn phổ biến là $k = 2s$, $p = s/2$, cho $T_{\text{out}} = s \cdot T_{\text{in}}$ đúng bằng hệ số nâng mẫu — đây cũng là quy tắc chọn kernel của HiFi-GAN (kernel 16 cho hệ số 8, kernel 4 cho hệ số 2).

### 4.5.3 Hiện tượng bàn cờ

Trong góc nhìn "rải", vị trí đầu ra $t$ nhận đóng góp từ mọi $n$ thỏa $0 \le t - n s \le k - 1$. Nếu $k$ **chia hết** cho $s$, mỗi vị trí (xa biên) nhận đúng $k/s$ đóng góp. Nếu không, số đóng góp **dao động tuần hoàn** giữa $\lfloor k/s \rfloor$ và $\lceil k/s \rceil$ với chu kỳ $s$. Với ảnh, điều này tạo ra họa tiết bàn cờ (Odena và cộng sự, 2016); với âm thanh, một thành phần tuần hoàn chu kỳ $s$ mẫu là một **âm thuần** ở tần số $f_{\text{out}}/s$ và các họa âm của nó — nghe như tiếng rè, tiếng "kim loại" cố định không đổi theo nội dung.

Điều kiện "$k$ chia hết cho $s$" là **cần nhưng chưa đủ**. Góc nhìn đa pha (*polyphase*) cho thấy vì sao: các vị trí đầu ra có cùng pha $r = t \bmod s$ chỉ dùng các nhịp kernel $w_r, w_{r+s}, w_{r+2s}, \ldots$. Vậy tích chập chuyển vị thực chất là $s$ bộ lọc con độc lập, mỗi bộ lo một pha, rồi đan xen kết quả. Nếu các bộ lọc con có độ lợi (*gain*) khác nhau — điều hoàn toàn có thể xảy ra với trọng số đã học — thì một đầu vào hằng số vẫn cho đầu ra tuần hoàn. Pons và cộng sự (2021) phân tích chi tiết các nguồn nhiễu do nâng mẫu trong tổng hợp âm thanh: theo họ, tích chập chuyển vị và sub-pixel dễ gây nhiễu dạng âm thuần (*tonal artifacts*), còn nâng mẫu bằng nội suy thì gây nhiễu kiểu lọc (*filtering artifacts*).

Các cách khắc phục thường gặp:

1. **Chọn $k$ là bội số của $s$** (ví dụ $k = 2s$), như HiFi-GAN.
2. **Nội suy rồi tích chập** (*resize-convolution*): nâng mẫu bằng láng giềng gần nhất hoặc nội suy tuyến tính, sau đó dùng `Conv1d` thường. Mọi vị trí dùng chung một bộ lọc nên không có thành phần tuần hoàn do cấu trúc.
3. **Sub-pixel 1D** (*pixel shuffle*): một `Conv1d` sinh $C \cdot s$ kênh rồi xếp lại thành $C$ kênh dài gấp $s$ lần. Về khả năng biểu diễn, nó tương đương tích chập chuyển vị, nên vẫn cần khởi tạo cẩn thận.
4. **Tránh nâng mẫu trong miền thời gian**: các vocoder như iSTFTNet, Vocos dự đoán hệ số STFT ở tốc độ khung rồi dùng iSTFT ([Chương 25](../phan-6-vocoder/ch25-gan-vocoder.md)).

Ngoài ra, các phi tuyến (như Leaky ReLU) áp lên tín hiệu đã nâng mẫu có thể sinh ra tần số vượt Nyquist và gây **chồng phổ** (*aliasing*); BigVGAN xử lý bằng các hàm kích hoạt chống chồng phổ — chi tiết ở [Chương 25](../phan-6-vocoder/ch25-gan-vocoder.md).

**Mã minh họa.** Cho đầu vào hằng 1 và kernel hằng 1, đầu ra của tích chập chuyển vị chính là "số lần chồng lấp" tại mỗi vị trí.

```python
def overlap_pattern(kernel_size, stride, t_in=8):
    """Đầu vào hằng 1, kernel hằng 1: đầu ra đếm số lần chồng lấp tại mỗi vị trí."""
    up = nn.ConvTranspose1d(1, 1, kernel_size, stride=stride, bias=False)
    nn.init.ones_(up.weight)
    with torch.no_grad():
        y = up(torch.ones(1, 1, t_in))            # (1, 1, (t_in-1)*s + k)
    k = kernel_size
    return y[0, 0, k:-k].int().tolist()           # bỏ vùng biên

print("k=3, s=2:", overlap_pattern(3, 2))   # không chia hết -> 1, 2, 1, 2, ...
print("k=4, s=2:", overlap_pattern(4, 2))   # chia hết -> đều
print("k=5, s=4:", overlap_pattern(5, 4))   # 1, 1, 1, 2, 1, 1, 1, 2, ...

# Khắc phục 1: nội suy láng giềng gần nhất rồi tích chập thường
x = torch.ones(1, 1, 8)
conv = nn.Conv1d(1, 1, 3, padding=1, bias=False)
nn.init.ones_(conv.weight)
with torch.no_grad():
    y = conv(F.interpolate(x, scale_factor=2, mode="nearest"))
print("nearest + conv:", y[0, 0, 2:-2].int().tolist())        # đều: 3, 3, 3, ...

# Khắc phục 2: sub-pixel 1D — tích chập ra C*s kênh rồi xếp lại theo thời gian
def pixel_shuffle_1d(x, s):                           # x: (B, C*s, T)
    b, cs, t = x.shape
    x = x.view(b, cs // s, s, t).permute(0, 1, 3, 2)  # (B, C, T, s)
    return x.reshape(b, cs // s, t * s)               # (B, C, T*s)

z = pixel_shuffle_1d(torch.arange(6.).view(1, 2, 3), 2)
assert z.tolist() == [[[0., 3., 1., 4., 2., 5.]]]
```

## 4.6 Các khối xây dựng hiện đại

### 4.6.1 Khối dư

Khi xếp chồng hàng chục lớp tích chập, việc tối ưu trở nên khó khăn: gradient phải đi qua một tích dài các Jacobian. **Khối dư** (*residual block*; He và cộng sự, 2016) học phần **hiệu chỉnh** thay vì toàn bộ ánh xạ:

$$
\mathbf{y} = \mathbf{x} + \mathcal{F}(\mathbf{x}), \qquad \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \mathbf{I} + \frac{\partial \mathcal{F}}{\partial \mathbf{x}} .
$$

Số hạng $\mathbf{I}$ trong Jacobian tạo một "đường cao tốc" cho gradient: kể cả khi $\partial \mathcal{F} / \partial \mathbf{x}$ nhỏ, gradient vẫn truyền nguyên vẹn về các lớp đầu. Khi số kênh vào và ra khác nhau, nhánh tắt dùng một tích chập $1 \times 1$. Gần như mọi vocoder và mô hình âm học hiện đại đều là chồng các khối dư.

### 4.6.2 Đơn vị kích hoạt có cổng

PixelCNN có điều kiện (van den Oord và cộng sự, 2016b) và WaveNet dùng **đơn vị kích hoạt có cổng** (*gated activation unit*):

$$
\mathbf{z} = \tanh(\mathcal{W}_f \ast \mathbf{x}) \odot \sigma(\mathcal{W}_g \ast \mathbf{x}),
$$

trong đó $\ast$ là tích chập, $\odot$ là tích từng phần tử. Nhánh $\tanh$ mang "nội dung", nhánh sigmoid là một **van** trong $(0, 1)$ quyết định bao nhiêu nội dung được đi qua, và van này phụ thuộc vào chính đầu vào. Phép nhân giữa hai nhánh tạo ra tương tác bậc hai mà một phi tuyến đơn (như ReLU) không có — cùng tinh thần với các cổng của LSTM ở mục 4.7.

Điều kiện hóa được đưa vào **bên trong** cả hai nhánh. Với điều kiện cục bộ $\mathbf{c}$ (ví dụ mel đã nâng mẫu lên tốc độ mẫu) và điều kiện toàn cục $\mathbf{g}$ (ví dụ embedding người nói), WaveNet dùng

$$
\mathbf{z} = \tanh\!\left(\mathcal{W}_f \ast \mathbf{x} + \mathcal{V}_f \ast \mathbf{c} + \mathbf{U}_f\, \mathbf{g}\right) \odot \sigma\!\left(\mathcal{W}_g \ast \mathbf{x} + \mathcal{V}_g \ast \mathbf{c} + \mathbf{U}_g\, \mathbf{g}\right).
$$

Một biến thể gần gũi là **GLU** (*Gated Linear Unit*; Dauphin và cộng sự, 2017): $(\mathbf{W}\mathbf{x}) \odot \sigma(\mathbf{V}\mathbf{x})$, bỏ $\tanh$ ở nhánh nội dung. Dauphin và cộng sự lập luận rằng nhánh tuyến tính giúp gradient không bị co lại bởi đạo hàm của $\tanh$. GLU xuất hiện trong mô-đun tích chập của Conformer ([Chương 5](ch05-transformer.md)) và trong biến thể SwiGLU của Transformer hiện đại.

WaveNet còn tách đầu ra mỗi khối thành hai nhánh: nhánh **dư** đi tiếp tới khối sau, và nhánh **skip** được cộng dồn từ mọi khối rồi đưa vào phần đầu ra. Nhánh skip cho phép đầu ra kết hợp trực tiếp đặc trưng ở mọi thang thời gian. Chi tiết WaveNet được trình bày ở [Chương 24](../phan-6-vocoder/ch24-vocoder-tu-hoi-quy-va-flow.md).

### 4.6.3 Tích chập tách theo chiều sâu

Một `Conv1d` thông thường làm đồng thời hai việc: trộn thông tin **theo thời gian** (trong cửa sổ $k$) và **theo kênh**. **Tích chập tách theo chiều sâu** (*depthwise-separable convolution*; xem Chollet, 2017) tách hai việc này: một tích chập theo chiều sâu ($g = C$, chỉ trộn thời gian) rồi một tích chập $1 \times 1$ (chỉ trộn kênh). Với $C$ kênh vào, $C'$ kênh ra, bỏ qua độ lệch:

$$
\frac{\text{tham số tách}}{\text{tham số thường}} = \frac{C k + C C'}{C C' k} = \frac{1}{C'} + \frac{1}{k}.
$$

Với $C' = 256$, $k = 7$, tỷ lệ này khoảng 0,147 — tiết kiệm gần 7 lần. Giả định ngầm là "mẫu hình thời gian" và "tổ hợp kênh" có thể học tách rời mà không mất nhiều năng lực biểu diễn; thực nghiệm trên nhiều miền cho thấy giả định này khá hợp lý, nhất là khi bù lại bằng nhiều kênh hơn.

### 4.6.4 Khối ConvNeXt 1D

ConvNeXt (Liu và cộng sự, 2022) "hiện đại hóa" khối tích chập bằng các lựa chọn thiết kế học từ Transformer:

1. Tích chập theo chiều sâu với **kernel lớn** (7) để trộn thời gian — vai trò giống attention cục bộ;
2. **LayerNorm** thay vì BatchNorm;
3. MLP từng vị trí kiểu **nút cổ chai ngược** (*inverted bottleneck*): mở rộng số kênh (bài báo gốc dùng hệ số 4), GELU, rồi thu hẹp lại — giống khối FFN của Transformer;
4. **Layer scale**: nhân nhánh dư với véc-tơ $\boldsymbol{\gamma}$ khởi tạo rất nhỏ, để mỗi khối gần như là ánh xạ đồng nhất lúc đầu huấn luyện;
5. Ít hàm kích hoạt và ít lớp chuẩn hóa hơn khối ResNet cổ điển.

Phiên bản 1D của khối này là xương sống của vocoder **Vocos** (Siuzdak, 2024), chạy ở tốc độ khung thay vì tốc độ mẫu nên rất nhanh; F5-TTS cũng dùng các khối ConvNeXt (phiên bản V2) để tinh chỉnh biểu diễn văn bản trước khi đưa vào Transformer ([Chương 33](../phan-8-mo-hinh-lon/ch33-nar-quy-mo-lon.md)).

**Mã minh họa.** Khối dư có cổng kiểu WaveNet và khối ConvNeXt 1D; cuối cùng so sánh số tham số của tích chập thường và tích chập tách.

```python
class GatedResidualBlock(nn.Module):
    """Khối dư kiểu WaveNet: tích chập giãn nở nhân quả + kích hoạt có cổng
    + điều kiện cục bộ (ví dụ mel đã nâng mẫu) + nhánh skip."""
    def __init__(self, channels, cond_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, 2 * channels, kernel_size, dilation=dilation)
        self.cond = nn.Conv1d(cond_channels, 2 * channels, 1)   # chiếu điều kiện 1x1
        self.res = nn.Conv1d(channels, channels, 1)
        self.skip = nn.Conv1d(channels, channels, 1)

    def forward(self, x, c):                      # x: (B, C, T), c: (B, C_cond, T)
        h = self.conv(F.pad(x, (self.pad, 0))) + self.cond(c)    # (B, 2C, T)
        h_f, h_g = h.chunk(2, dim=1)                             # 2 x (B, C, T)
        z = torch.tanh(h_f) * torch.sigmoid(h_g)                 # cổng: (B, C, T)
        return x + self.res(z), self.skip(z)                     # (dư, skip)

class ConvNeXtBlock1d(nn.Module):
    """Khối ConvNeXt 1D: depthwise conv -> LayerNorm -> MLP mở rộng với GELU
    -> layer scale -> cộng dư."""
    def __init__(self, dim, kernel_size=7, expansion=4, layer_scale=1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, expansion * dim)
        self.pw2 = nn.Linear(expansion * dim, dim)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim))

    def forward(self, x):                         # x: (B, C, T)
        h = self.dwconv(x).transpose(1, 2)        # (B, T, C): LayerNorm trên trục kênh
        h = self.pw2(F.gelu(self.pw1(self.norm(h))))
        return x + (self.gamma * h).transpose(1, 2)   # (B, C, T)

def n_params(m):
    return sum(p.numel() for p in m.parameters())

B, C, T = 2, 64, 50
blk = GatedResidualBlock(C, cond_channels=80, dilation=4)
res, skip = blk(torch.randn(B, C, T), torch.randn(B, 80, T))
assert res.shape == skip.shape == (B, C, T)
cnx = ConvNeXtBlock1d(C)
assert cnx(torch.randn(B, C, T)).shape == (B, C, T)

# Tích chập thường vs tách theo chiều sâu (C_in = C_out = 256, k = 7)
full = nn.Conv1d(256, 256, 7, padding=3)
sep = nn.Sequential(nn.Conv1d(256, 256, 7, padding=3, groups=256), nn.Conv1d(256, 256, 1))
print("Conv thường:", n_params(full), "| tách theo chiều sâu:", n_params(sep))
print("Tỷ lệ:", round(n_params(sep) / n_params(full), 3), "≈ 1/7 + 1/256 =", round(1 / 7 + 1 / 256, 3))
```

## 4.7 Mạng nơ-ron hồi quy

### 4.7.1 RNN cơ bản

**Trực giác.** Thay vì nhìn một cửa sổ cố định, RNN đọc chuỗi từng bước và cập nhật một **trạng thái ẩn** $\mathbf{h}_t$ — một "bản tóm tắt" về mọi thứ đã thấy. Cùng một hàm cập nhật được dùng ở mọi bước, tức là trọng số được chia sẻ theo thời gian như CNN, nhưng ngữ cảnh không bị giới hạn bởi cửa sổ.

**Hình thức hóa.** RNN Elman với đầu vào $\mathbf{x}_t \in \mathbb{R}^{D}$, trạng thái $\mathbf{h}_t \in \mathbb{R}^{H}$:

$$
\mathbf{a}_t = \mathbf{W}\mathbf{h}_{t-1} + \mathbf{U}\mathbf{x}_t + \mathbf{b}, \qquad
\mathbf{h}_t = \tanh(\mathbf{a}_t), \qquad
\hat{\mathbf{y}}_t = \mathbf{V}\mathbf{h}_t + \mathbf{e},
$$

với $\mathbf{W} \in \mathbb{R}^{H \times H}$ là ma trận hồi quy, $\mathbf{U} \in \mathbb{R}^{H \times D}$, $\mathbf{V}$ và $\mathbf{e}$ là phần đầu ra. "Mở" (*unroll*) RNN theo thời gian ta được một mạng truyền thẳng rất sâu — sâu bằng độ dài chuỗi — trong đó mọi lớp dùng chung $(\mathbf{W}, \mathbf{U}, \mathbf{b})$.

### 4.7.2 Lan truyền ngược theo thời gian (BPTT)

Hàm mất mát là tổng theo thời gian $\mathcal{L} = \sum_{t=1}^{T} \ell_t(\hat{\mathbf{y}}_t)$. Áp dụng lan truyền ngược ([Chương 2](ch02-hoc-may-va-mang-no-ron.md)) cho mạng đã mở gọi là **lan truyền ngược theo thời gian** (*backpropagation through time*, BPTT; Werbos, 1990). Ta suy ra từng bước.

**Bước 1 — gradient theo trạng thái.** $\mathbf{h}_t$ ảnh hưởng tới $\mathcal{L}$ qua hai đường: trực tiếp qua $\ell_t$, và gián tiếp qua $\mathbf{a}_{t+1}$ (rồi mọi bước sau). Đặt $\mathbf{g}_t = \partial \mathcal{L} / \partial \mathbf{h}_t$ và $\boldsymbol{\delta}_t = \partial \mathcal{L} / \partial \mathbf{a}_t$:

$$
\mathbf{g}_t = \mathbf{V}^\top \frac{\partial \ell_t}{\partial \hat{\mathbf{y}}_t} + \mathbf{W}^\top \boldsymbol{\delta}_{t+1}, \qquad \boldsymbol{\delta}_{T+1} = \mathbf{0} .
$$

**Bước 2 — qua phi tuyến.** Vì $\mathbf{h}_t = \tanh(\mathbf{a}_t)$ tác động từng phần tử và $\tanh'(a) = 1 - \tanh^2(a)$:

$$
\boldsymbol{\delta}_t = (1 - \mathbf{h}_t \odot \mathbf{h}_t) \odot \mathbf{g}_t .
$$

Hai công thức trên được tính **ngược** từ $t = T$ về $t = 1$.

**Bước 3 — gradient theo tham số.** Vì tham số được chia sẻ, gradient là tổng đóng góp của mọi bước:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \sum_{t=1}^{T} \boldsymbol{\delta}_t\, \mathbf{h}_{t-1}^\top, \qquad
\frac{\partial \mathcal{L}}{\partial \mathbf{U}} = \sum_{t=1}^{T} \boldsymbol{\delta}_t\, \mathbf{x}_t^\top, \qquad
\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \sum_{t=1}^{T} \boldsymbol{\delta}_t .
$$

Chi phí bộ nhớ của BPTT tỷ lệ với $T$ vì phải lưu mọi $\mathbf{h}_t$. Với chuỗi rất dài, người ta dùng **BPTT cắt cụt** (*truncated BPTT*): chia chuỗi thành các đoạn $k$ bước, truyền trạng thái ẩn sang đoạn sau nhưng **cắt gradient** (`h = h.detach()` trong PyTorch) ở ranh giới đoạn.

### 4.7.3 Gradient tiêu biến và bùng nổ

Ghép hai bước trên, gradient truyền từ bước $t$ về bước $k < t$ phải đi qua tích các Jacobian:

$$
\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} = \prod_{i=k+1}^{t} \frac{\partial \mathbf{h}_i}{\partial \mathbf{h}_{i-1}} = \prod_{i=k+1}^{t} \operatorname{diag}\!\left(1 - \mathbf{h}_i \odot \mathbf{h}_i\right) \mathbf{W}
$$

(tích lấy theo thứ tự $i = t, t-1, \ldots, k+1$ từ trái sang phải). Dùng tính chất $\lVert \mathbf{A}\mathbf{B} \rVert \le \lVert \mathbf{A} \rVert \lVert \mathbf{B} \rVert$ với chuẩn phổ, và $|\tanh'| \le \gamma = 1$:

$$
\left\lVert \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} \right\rVert \le \left(\gamma\, \sigma_{\max}(\mathbf{W})\right)^{t-k},
$$

với $\sigma_{\max}(\mathbf{W})$ là giá trị kỳ dị lớn nhất. Suy ra (Bengio và cộng sự, 1994; Pascanu và cộng sự, 2013):

- Nếu $\sigma_{\max}(\mathbf{W}) < 1/\gamma$, gradient **tiêu biến** theo hàm mũ của khoảng cách $t - k$ — điều kiện **đủ**. Mô hình gần như không học được phụ thuộc xa.
- Để gradient **bùng nổ**, **cần** $\sigma_{\max}(\mathbf{W}) > 1/\gamma$. Khi đó một vài bước cập nhật khổng lồ có thể phá hỏng quá trình huấn luyện.

Bùng nổ thì chữa tương đối dễ bằng **cắt gradient theo chuẩn** ([Chương 3](ch03-toi-uu-va-ky-thuat-huan-luyen.md)) — gần như bắt buộc khi huấn luyện RNN. Tiêu biến thì khó hơn nhiều, và đó là động lực của LSTM.

**Mã minh họa.** Ta chọn $\mathbf{W} = \rho\,\mathbf{Q}$ với $\mathbf{Q}$ trực giao, nên mọi giá trị kỳ dị của $\mathbf{W}$ đều bằng $\rho$. Với RNN tuyến tính, chuẩn gradient đúng bằng $\rho^{t-k}$ nhân hằng số; với $\tanh$, đạo hàm $\tanh' < 1$ còn làm gradient co thêm.

```python
def grad_norm_through_time(rho, steps, hidden=32, nonlinear=True, seed=0):
    """Chuẩn của dL/dh_0 với L = <u, h_T>, W = rho * Q (Q trực giao)."""
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(hidden, hidden, generator=g))
    W = rho * q                                   # mọi giá trị kỳ dị của W đều bằng rho
    h0 = (0.1 * torch.randn(hidden, generator=g)).requires_grad_()
    x = 0.1 * torch.randn(steps, hidden, generator=g)   # đầu vào nhỏ
    u = torch.randn(hidden, generator=g)
    h = h0
    for t in range(steps):
        a = W @ h + x[t]
        h = torch.tanh(a) if nonlinear else a
    (u @ h).backward()
    return h0.grad.norm().item()

for rho in [0.8, 1.0, 1.2]:
    lin = [grad_norm_through_time(rho, T, nonlinear=False) for T in (10, 50, 100)]
    nl = [grad_norm_through_time(rho, T, nonlinear=True) for T in (10, 50, 100)]
    print(f"rho={rho}: tuyến tính {['%.1e' % v for v in lin]} | tanh {['%.1e' % v for v in nl]}")
```

Kết quả điển hình: với $\rho = 0{,}8$, chuẩn gradient sau 100 bước giảm xuống cỡ $10^{-9}$–$10^{-10}$; với $\rho = 1{,}2$ ở mạng tuyến tính, nó tăng lên cỡ $10^{8}$. Với $\tanh$ và $\rho = 1{,}2$ thì gradient *không* bùng nổ vì trạng thái bị bão hòa ($\tanh' \approx 0$) — nhưng chính sự bão hòa này lại là một nguồn tiêu biến khác. Ngay cả $\rho = 1$ (trường hợp "lý tưởng" của mạng tuyến tính) cũng tiêu biến khi có $\tanh$.

### 4.7.4 LSTM

**Long Short-Term Memory** (Hochreiter và Schmidhuber, 1997; cổng quên được bổ sung bởi Gers và cộng sự, 2000) thêm một **ô nhớ** $\mathbf{c}_t$ được cập nhật theo kiểu **cộng** và được điều khiển bởi ba cổng:

$$
\begin{aligned}
\mathbf{i}_t &= \sigma(\mathbf{W}_i \mathbf{h}_{t-1} + \mathbf{U}_i \mathbf{x}_t + \mathbf{b}_i) && \text{(cổng vào)}\\
\mathbf{f}_t &= \sigma(\mathbf{W}_f \mathbf{h}_{t-1} + \mathbf{U}_f \mathbf{x}_t + \mathbf{b}_f) && \text{(cổng quên)}\\
\mathbf{o}_t &= \sigma(\mathbf{W}_o \mathbf{h}_{t-1} + \mathbf{U}_o \mathbf{x}_t + \mathbf{b}_o) && \text{(cổng ra)}\\
\tilde{\mathbf{c}}_t &= \tanh(\mathbf{W}_c \mathbf{h}_{t-1} + \mathbf{U}_c \mathbf{x}_t + \mathbf{b}_c) && \text{(nội dung mới)}\\
\mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
\end{aligned}
$$

**Vì sao LSTM giảm tiêu biến gradient?** Xét riêng đường đi qua ô nhớ (bỏ qua ảnh hưởng gián tiếp qua các cổng, vốn phụ thuộc $\mathbf{h}_{t-1}$):

$$
\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} \approx \operatorname{diag}(\mathbf{f}_t)
\quad\Longrightarrow\quad
\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_k} \approx \operatorname{diag}\!\Big(\prod_{i=k+1}^{t} \mathbf{f}_i\Big).
$$

Không còn phép nhân với ma trận $\mathbf{W}$ và đạo hàm $\tanh'$ ở mỗi bước, chỉ còn tích của các giá trị cổng quên — **do mô hình tự quyết định**. Khi $\mathbf{f}_i \approx 1$, thông tin và gradient đi qua hàng trăm bước gần như nguyên vẹn. Đây cũng chính là ý tưởng của khối dư ở mục 4.6.1, nhưng theo trục thời gian: cập nhật cộng $\mathbf{c}_t = \mathbf{c}_{t-1} + (\ldots)$ khi $\mathbf{f}_t = 1$.

Hai chi tiết thực hành:

- **Khởi tạo độ lệch cổng quên dương** (thường là 1) để lúc đầu mạng "nhớ lâu", một khuyến nghị được Jozefowicz và cộng sự (2015) xác nhận bằng thực nghiệm.
- Trong PyTorch, `nn.LSTM` xếp trọng số bốn cổng theo thứ tự **(i, f, g, o)** — `g` là $\tilde{\mathbf{c}}_t$ — và có **hai** véc-tơ độ lệch `bias_ih`, `bias_hh` (tổng của chúng mới là $\mathbf{b}$). Số tham số một lớp là $4H(D + H) + 8H$.

### 4.7.5 GRU

**Gated Recurrent Unit** (Cho và cộng sự, 2014) gộp ô nhớ vào trạng thái ẩn và chỉ dùng hai cổng. Theo quy ước của PyTorch:

$$
\begin{aligned}
\mathbf{r}_t &= \sigma(\mathbf{W}_r \mathbf{h}_{t-1} + \mathbf{U}_r \mathbf{x}_t + \mathbf{b}_r) && \text{(cổng đặt lại)}\\
\mathbf{z}_t &= \sigma(\mathbf{W}_z \mathbf{h}_{t-1} + \mathbf{U}_z \mathbf{x}_t + \mathbf{b}_z) && \text{(cổng cập nhật)}\\
\tilde{\mathbf{h}}_t &= \tanh\!\big(\mathbf{U}_n \mathbf{x}_t + \mathbf{b}_n + \mathbf{r}_t \odot (\mathbf{W}_n \mathbf{h}_{t-1} + \mathbf{b}'_n)\big)\\
\mathbf{h}_t &= (1 - \mathbf{z}_t) \odot \tilde{\mathbf{h}}_t + \mathbf{z}_t \odot \mathbf{h}_{t-1}
\end{aligned}
$$

Cổng cập nhật $\mathbf{z}_t$ nội suy giữa trạng thái cũ và ứng viên mới, đóng cả vai trò cổng quên và cổng vào của LSTM. (Một số tài liệu dùng quy ước ngược lại, đổi vai trò của $\mathbf{z}_t$ và $1 - \mathbf{z}_t$; bản chất không đổi.) GRU ít tham số hơn LSTM khoảng một phần tư, và các so sánh thực nghiệm (ví dụ Chung và cộng sự, 2014) cho thấy hai kiến trúc thường ngang ngửa nhau, tùy bài toán. Trong TTS, các vocoder tự hồi quy hiệu quả như WaveRNN và LPCNet dựa trên GRU ([Chương 24](../phan-6-vocoder/ch24-vocoder-tu-hoi-quy-va-flow.md)), còn Tacotron 2 dùng LSTM.

**Mã minh họa.** Cài đặt một bước LSTM bằng tay, dùng chính trọng số của `nn.LSTM`, và kiểm tra hai bên cho kết quả giống hệt — cách tốt nhất để chắc chắn bạn hiểu đúng thứ tự cổng.

```python
torch.manual_seed(0)
D, H = 8, 16
lstm = nn.LSTM(D, H, batch_first=True)

def lstm_step(x_t, h, c, m):
    """Một bước LSTM viết tay, dùng trọng số của nn.LSTM (thứ tự cổng: i, f, g, o)."""
    z = x_t @ m.weight_ih_l0.T + m.bias_ih_l0 + h @ m.weight_hh_l0.T + m.bias_hh_l0  # (B, 4H)
    i, f, g, o = z.chunk(4, dim=-1)
    i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
    c = f * c + i * torch.tanh(g)                 # cập nhật ô nhớ: đường "cộng"
    h = o * torch.tanh(c)
    return h, c

x = torch.randn(3, 5, D)                          # (B, T, D)
h = c = torch.zeros(3, H)
outs = []
for t in range(x.shape[1]):
    h, c = lstm_step(x[:, t], h, c, lstm)
    outs.append(h)
ref, (h_n, c_n) = lstm(x)                         # ref: (B, T, H)
assert torch.allclose(torch.stack(outs, 1), ref, atol=1e-6)
assert torch.allclose(c, c_n[0], atol=1e-6)

n = sum(p.numel() for p in lstm.parameters())
assert n == 4 * H * (D + H) + 8 * H               # PyTorch có 2 véc-tơ bias
print("LSTM viết tay khớp nn.LSTM; số tham số =", n)

# Mẹo: khởi tạo độ lệch cổng quên = 1 (đặt ở bias_ih, còn bias_hh để 0)
with torch.no_grad():
    for name, p in lstm.named_parameters():
        if "bias" in name:
            p.zero_()
            if "bias_ih" in name:
                p[H:2 * H] = 1.0                  # đoạn [H, 2H) ứng với cổng quên f
```

### 4.7.6 RNN hai chiều và chuỗi có độ dài thay đổi

Một RNN một chiều tại bước $t$ chỉ biết quá khứ. Với bài toán mà **toàn bộ đầu vào có sẵn** — như mã hóa một câu văn bản trước khi tổng hợp — ta muốn mỗi vị trí biết cả ngữ cảnh phía sau. **RNN hai chiều** (Schuster và Paliwal, 1997) chạy một RNN xuôi và một RNN ngược rồi ghép trạng thái: $\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t]$. Bộ mã hóa của Tacotron 2 dùng một lớp BiLSTM 512 đơn vị (256 mỗi chiều) sau ba lớp tích chập (Shen và cộng sự, 2018). RNN hai chiều **không dùng được** cho phần sinh tự hồi quy hay cho streaming, vì chiều ngược cần tương lai.

Khi huấn luyện theo batch, các câu có độ dài khác nhau được **đệm** (*padding*) tới cùng độ dài. Với RNN một chiều, phần đệm nằm *sau* nội dung thật nên không làm hỏng trạng thái tại các vị trí thật. Nhưng với chiều **ngược** của BiLSTM, RNN bắt đầu đọc từ phần đệm — trạng thái tại các vị trí thật bị "nhiễm" phần đệm. Cách đúng trong PyTorch là **đóng gói** chuỗi bằng `pack_padded_sequence`.

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

torch.manual_seed(0)
bilstm = nn.LSTM(input_size=4, hidden_size=8, batch_first=True, bidirectional=True)
lengths = torch.tensor([6, 3])
x = torch.randn(2, 6, 4)                          # (B, N, D) — câu thứ 2 dài 3, còn lại là đệm
x[1, 3:] = 0.0

# Cách SAI: chạy thẳng trên chuỗi đã đệm -> chiều ngược "đọc" cả phần đệm
naive, _ = bilstm(x)                              # (B, N, 2H)

# Cách ĐÚNG: đóng gói theo độ dài thật
packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
out, _ = bilstm(packed)
out, _ = pad_packed_sequence(out, batch_first=True, total_length=6)   # (B, N, 2H)

# Đối chứng: chạy riêng câu thứ 2 không đệm
alone, _ = bilstm(x[1:2, :3])                     # (1, 3, 2H)
assert torch.allclose(out[1, :3], alone[0], atol=1e-6)
print("Sai lệch chiều xuôi (cách sai):", (naive[1, :3, :8] - alone[0, :, :8]).abs().max().item())
print("Sai lệch chiều ngược (cách sai):", (naive[1, :3, 8:] - alone[0, :, 8:]).abs().max().item())
```

Chiều xuôi khớp (sai lệch cỡ $10^{-8}$, chỉ là sai số dấu phẩy động), còn chiều ngược sai lệch đáng kể. Lỗi này **không gây ra ngoại lệ nào**: mô hình vẫn huấn luyện được, chỉ kém hơn và hành xử khác nhau tùy độ dài câu khác trong batch.

!!! note "Góc tiếng Việt"
    Kiểu câu tiếng Việt thường được đánh dấu bằng **tiểu từ cuối câu**: "Anh đi à?", "Anh đi nhé.", "Anh có đi không?". Ngữ điệu của cả câu — kể cả các âm tiết đầu — phụ thuộc vào tiểu từ này ([Chương 16](../phan-3-tin-hieu-ngu-am/ch16-ngon-dieu.md)). Một bộ mã hóa văn bản hai chiều (BiLSTM, hoặc self-attention ở [Chương 5](ch05-transformer.md)) nhìn được tiểu từ ngay từ đầu. Ngược lại, trong TTS streaming chỉ được nhìn trước một ít token, mô hình có thể đã đọc xong nửa đầu câu với ngữ điệu trần thuật trước khi "thấy" chữ "không?" ở cuối — một lý do để các hệ thống streaming chọn ranh giới đoạn theo cụm từ và dấu câu ([Chương 35](../phan-8-mo-hinh-lon/ch35-streaming-speech-llm.md)).
