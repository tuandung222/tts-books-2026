# Chương 21. Bài toán căn chỉnh (Alignment)

!!! abstract "Mục tiêu học tập"
    Sau chương này, bạn có thể:

    - **Phát biểu** chính xác bài toán căn chỉnh văn bản–tiếng nói, phân biệt căn chỉnh mềm và cứng, và đếm số căn chỉnh đơn điệu hợp lệ.
    - **So sánh** các cơ chế attention có ràng buộc vị trí hoặc đơn điệu: GMM attention, forward attention, stepwise monotonic attention và Dynamic Convolution Attention.
    - **Cài đặt** guided attention loss và trích xuất trường độ từ attention của mô hình thầy (kèm thước đo *focus rate*).
    - **Giải thích** căn chỉnh cưỡng bức dựa trên HMM (Montreal Forced Aligner) và dựa trên CTC, và chuyển kết quả của chúng thành trường độ theo khung.
    - **Suy ra** thuật toán quy hoạch động Monotonic Alignment Search (MAS), cài đặt nó và kiểm chứng bằng vét cạn trên ví dụ nhỏ.
    - **Cài đặt** forward-sum loss bằng `F.ctc_loss`, tiên nghiệm beta–nhị thức và binarization loss của phương pháp học căn chỉnh không giám sát.
    - **Chẩn đoán** chất lượng căn chỉnh bằng thước đo định lượng và bằng cách đọc hình ảnh ma trận căn chỉnh.

!!! info "Kiến thức cần có"
    - Tacotron 2, attention nhạy vị trí và các lỗi đặc trưng của TTS dựa trên attention: [Chương 20](ch20-tacotron.md).
    - Attention, softmax, mặt nạ: [Chương 4](../phan-1-nen-tang/ch04-cnn-rnn-mo-hinh-chuoi.md), [Chương 5](../phan-1-nen-tang/ch05-transformer.md).
    - Phân phối Gauss nhiều chiều, log-likelihood: [Chương 1](../phan-1-nen-tang/ch01-toan-nen-tang.md).
    - Khung, bước nhảy và mel-spectrogram: [Chương 14](../phan-3-tin-hieu-ngu-am/ch14-xu-ly-tin-hieu-so.md).
    - Cấu trúc âm tiết và âm vị tiếng Việt (để hiểu căn chỉnh cho tiếng Việt): [Chương 15](../phan-3-tin-hieu-ngu-am/ch15-ngu-am-tieng-viet.md).

Mọi hệ thống TTS đều phải trả lời một câu hỏi: **khung âm thanh nào thuộc về token văn bản nào?** Hệ thống tham số thống kê trả lời bằng HMM; Tacotron trả lời ngầm bằng attention; FastSpeech hỏi một mô hình thầy; Glow-TTS và VITS tự tìm bằng quy hoạch động; các mô hình lớn như E2/F5-TTS để mạng tự xoay xở bên trong. Câu trả lời cho câu hỏi này — **căn chỉnh** (*alignment*) — quyết định phần lớn độ bền vững của mô hình: hầu hết lỗi bỏ từ, lặp từ, đọc lướt, ngắt sai chỗ đều là lỗi căn chỉnh. Chương này hệ thống hóa các cách tiếp cận, với trọng tâm là hai công cụ được dùng nhiều nhất hiện nay: **Monotonic Alignment Search** và **học căn chỉnh không giám sát bằng forward-sum loss**.

## 21.1 Phát biểu bài toán

### Căn chỉnh cứng

Cho chuỗi token $\mathbf{x} = (x_1, \ldots, x_N)$ và chuỗi khung âm học $\mathbf{Y} = (\mathbf{y}_1, \ldots, \mathbf{y}_T)$ với $T \ge N$. Một **căn chỉnh cứng** là một hàm $a : \{1, \ldots, T\} \to \{1, \ldots, N\}$ gán mỗi khung cho đúng một token. Ta biểu diễn nó bằng ma trận $\mathbf{A} \in \{0, 1\}^{N \times T}$ với $A_{n,t} = \mathbb{1}[a(t) = n]$ — mỗi cột có đúng một số 1. Với TTS, ta đòi hỏi ba ràng buộc:

1. **Đơn điệu** (*monotonic*): $a(t) \le a(t+1)$ — đọc theo thứ tự.
2. **Không nhảy cóc** (*no skipping*): $a(t+1) - a(t) \in \{0, 1\}$ — không bỏ token nào.
3. **Biên**: $a(1) = 1$ và $a(T) = N$ — bắt đầu từ token đầu, kết thúc ở token cuối.

Ba ràng buộc này tương đương với việc mỗi token chiếm một **đoạn liên tiếp** gồm $d_n \ge 1$ khung, với $\sum_n d_n = T$. Nói cách khác, căn chỉnh cứng hợp lệ **chính là** véc-tơ trường độ $\mathbf{d}$ trong bảng ký hiệu, và $d_n = \sum_t A_{n,t}$.

**Có bao nhiêu căn chỉnh hợp lệ?** Xếp $T$ khung thành một hàng; chọn một căn chỉnh tương đương với đặt $N - 1$ "vách ngăn" vào $T - 1$ khe giữa các khung (mỗi khe nhiều nhất một vách, để mọi $d_n \ge 1$). Do đó số căn chỉnh là

$$
\left|\mathcal{S}_{N,T}\right| = \binom{T-1}{N-1}.
$$

Con số này tăng cực nhanh: với $N = 50$ âm vị và $T = 400$ khung (một câu khoảng 4–5 giây), $\binom{399}{49}$ có hơn 60 chữ số. Vét cạn là bất khả thi; ta cần quy hoạch động.

### Căn chỉnh mềm

Căn chỉnh **mềm** cho phép mỗi khung "chia" mình cho nhiều token: $\mathbf{A} \in [0,1]^{N \times T}$, mỗi cột là một phân phối xác suất trên token. Attention trong Tacotron sinh ra căn chỉnh mềm. Căn chỉnh mềm khả vi nên học được bằng lan truyền ngược, nhưng không có gì bảo đảm nó đơn điệu. Từ căn chỉnh mềm có thể rút ra căn chỉnh cứng bằng argmax từng cột (đơn giản nhưng có thể vi phạm ràng buộc) hoặc bằng thuật toán Viterbi/MAS (luôn hợp lệ).

### Vì sao đơn điệu là giả thiết đúng?

Người đọc phát âm văn bản theo thứ tự — nhưng chỉ khi văn bản **đã được chuẩn hóa**. Chuỗi "$5" trong tiếng Anh đọc là "five dollars": ký hiệu đứng trước lại được đọc sau. Đây là lý do chuẩn hóa văn bản ([Chương 17](../phan-4-frontend/ch17-chuan-hoa-van-ban.md)) phải chạy *trước* mô hình âm học: sau chuẩn hóa, giả thiết đơn điệu gần như luôn đúng. Tiếng Việt thuận lợi hơn ở điểm này ("5 USD" đọc là "năm đô-la Mỹ", cùng thứ tự), nhưng các trường hợp như "TP.HCM", "%", "km/h" vẫn phải được triển khai thành chữ.

### Bản đồ các cách tiếp cận

| Cách tiếp cận | Kiểu căn chỉnh | Học khi nào | Hệ thống tiêu biểu |
|---|---|---|---|
| Attention (có/không ràng buộc) | Mềm | Cùng lúc với mô hình | Tacotron 2, Transformer-TTS |
| Trường độ từ mô hình thầy | Cứng (rút từ attention) | Trước, bằng mô hình khác | FastSpeech |
| Căn chỉnh cưỡng bức HMM/CTC | Cứng | Trước, bằng công cụ ngoài | FastSpeech 2 (MFA), nhiều hệ thống sản phẩm |
| Monotonic Alignment Search | Cứng | Cùng lúc, không khả vi | Glow-TTS, Grad-TTS, VITS, Matcha-TTS |
| Forward-sum + MAS | Mềm rồi cứng | Cùng lúc, khả vi | RAD-TTS, FastPitch (bản sau), nhiều bản FastSpeech 2 |
| Ngầm định | Không tường minh | Cùng lúc | E2 TTS, F5-TTS ([Chương 33](../phan-8-mo-hinh-lon/ch33-nar-quy-mo-lon.md)) |

## 21.2 Attention có ràng buộc vị trí và tính đơn điệu

[Chương 20](ch20-tacotron.md) đã cho thấy attention dựa trên nội dung dễ bỏ từ, lặp từ và không tổng quát hóa với câu dài. Một loạt cơ chế attention đã được đề xuất để **cài sẵn** tính đơn điệu hoặc tính vị trí vào chính cấu trúc của attention. Trong mục này, $t$ là bước giải mã, $n$ là vị trí token, $\mathbf{s}_t$ là trạng thái truy vấn và $\boldsymbol{\alpha}_t \in \mathbb{R}^N$ là phân phối attention.

### GMM attention

Graves (2013), trong công trình sinh chữ viết tay, đề xuất attention dựa **hoàn toàn trên vị trí**: trọng số attention là một hỗn hợp $K$ hàm Gauss trên trục token, và tâm của mỗi thành phần chỉ được phép **tiến về phía trước**:

$$
(\hat{\boldsymbol{\omega}}_t, \hat{\boldsymbol{\beta}}_t, \hat{\boldsymbol{\kappa}}_t) = \mathbf{W}\mathbf{s}_t + \mathbf{b}, \qquad
\omega^k_t = e^{\hat{\omega}^k_t}, \quad \beta^k_t = e^{\hat{\beta}^k_t}, \quad \kappa^k_t = \kappa^k_{t-1} + e^{\hat{\kappa}^k_t},
$$

$$
\alpha_{t,n} = \sum_{k=1}^{K} \omega^k_t \exp\left(-\beta^k_t \left(\kappa^k_t - n\right)^2\right).
$$

Ở đây $\omega^k_t$ là trọng số trộn, $\beta^k_t$ điều khiển độ hẹp, $\kappa^k_t$ là tâm. Vì $e^{\hat{\kappa}} > 0$, tâm luôn tăng: **tính đơn điệu được bảo đảm bởi cấu trúc**. Năng lượng không phụ thuộc nội dung $\mathbf{h}_n$, nên attention không thể bị "hút" sang một token ở xa có nội dung giống — đổi lại, nó cũng không thể tự sửa khi đã đi lệch. Battenberg và cộng sự (2020) so sánh nhiều biến thể và báo cáo rằng việc chuẩn hóa trọng số trộn bằng softmax và dùng softplus (thay vì $\exp$) cho bước dịch và độ rộng giúp huấn luyện ổn định hơn, và các biến thể GMM này tổng quát hóa sang văn bản dài tốt hơn nhiều so với attention nhạy vị trí của Tacotron 2.

### Forward attention

Zhang, Ling và Dai (2018) lấy cảm hứng từ **thuật toán tiến** (*forward algorithm*) của HMM. Gọi $\tilde{\alpha}_{t,n}$ là trọng số attention "thô" do một cơ chế bất kỳ tính ra (ví dụ attention nhạy vị trí). Forward attention chỉ giữ lại khối lượng xác suất có thể đến được bằng cách **đứng yên hoặc tiến một bước** từ phân phối trước:

$$
\hat{\alpha}_{t,n} = \left(\alpha_{t-1,n} + \alpha_{t-1,n-1}\right)\tilde{\alpha}_{t,n}, \qquad
\alpha_{t,n} = \frac{\hat{\alpha}_{t,n}}{\sum_{m} \hat{\alpha}_{t,m}},
$$

khởi tạo $\boldsymbol{\alpha}_0 = (1, 0, \ldots, 0)$. Phiên bản có **tác tử chuyển tiếp** (*transition agent*) học thêm một xác suất $u_t \in (0, 1)$ (từ một mạng nhỏ với đầu ra sigmoid) cho việc "tiến":

$$
\hat{\alpha}_{t,n} = \left((1 - u_{t-1})\,\alpha_{t-1,n} + u_{t-1}\,\alpha_{t-1,n-1}\right)\tilde{\alpha}_{t,n}.
$$

Dễ thấy giá của $\boldsymbol{\alpha}_t$ chỉ có thể mở rộng về bên phải tối đa một vị trí mỗi bước, nên attention không thể nhảy cóc. Các tác giả cũng cho thấy có thể điều khiển tốc độ nói khi suy luận bằng cách dịch $u_t$ lên hoặc xuống.

### Stepwise monotonic attention

He, Deng và He (2019) mở rộng *monotonic attention* (Raffel và cộng sự, 2017) theo hướng "từng bước": ở mỗi bước giải mã, attention hoặc **đứng yên** hoặc **tiến đúng một token**. Với năng lượng $e_{t,n}$ từ một cơ chế bất kỳ, xác suất dừng lại ở token $n$ là $p_{t,n} = \sigma(e_{t,n})$, và phân phối kỳ vọng được cập nhật:

$$
\alpha_{t,n} = \alpha_{t-1,n}\, p_{t,n} + \alpha_{t-1,n-1}\left(1 - p_{t,n-1}\right).
$$

Số hạng thứ nhất: đang ở $n$ và ở lại; số hạng thứ hai: đang ở $n - 1$ và tiến lên. Khi huấn luyện dùng dạng kỳ vọng (mềm) này; thường cộng nhiễu Gauss vào $e_{t,n}$ trước sigmoid để đẩy $p_{t,n}$ về gần 0 hoặc 1, nhờ đó khi suy luận có thể dùng **chế độ cứng** (tiến khi $p_{t,n} < 0.5$) mà không lệch phân phối. Cơ chế này về mặt cấu trúc không thể bỏ hay lặp token — rất hợp với TTS, nơi mỗi token phải được đọc đúng một lần.

### Dynamic Convolution Attention

Battenberg và cộng sự (2020) đề xuất **Dynamic Convolution Attention** (DCA), một cơ chế **thuần vị trí tương đối**: năng lượng không có số hạng nội dung $\mathbf{V}\mathbf{h}_n$, chỉ phụ thuộc vào phân phối attention trước đó:

$$
e_{t,n} = \mathbf{w}^\top \tanh\left(\mathbf{U}\mathbf{f}_{t,n} + \mathbf{T}\mathbf{g}_{t,n} + \mathbf{b}\right) + p_{t,n},
$$

$$
\mathbf{f}_t = \mathbf{F} \ast \boldsymbol{\alpha}_{t-1}, \qquad
\mathbf{g}_t = \mathbf{G}(\mathbf{s}_t) \ast \boldsymbol{\alpha}_{t-1}, \qquad
\mathbf{p}_t = \log\left(\mathbf{P} \ast \boldsymbol{\alpha}_{t-1}\right).
$$

Ba loại bộ lọc: $\mathbf{F}$ là bộ lọc **tĩnh** học được (như attention nhạy vị trí); $\mathbf{G}(\mathbf{s}_t)$ là bộ lọc **động** do một mạng nhỏ sinh ra từ truy vấn — cho phép "bước đi" phụ thuộc trạng thái giải mã hiện tại; $\mathbf{P}$ là bộ lọc **tiên nghiệm** cố định, nhân quả, lấy giá trị từ một phân phối beta–nhị thức để khuyến khích dịch nhẹ về phía trước (giá trị $\log$ được kẹp dưới để tránh $\log 0$). Vì không phụ thuộc vào vị trí tuyệt đối hay nội dung, DCA hoạt động như nhau ở token thứ 10 hay thứ 1000. Theo báo cáo gốc, DCA và các biến thể GMM tốt nhất duy trì căn chỉnh ổn định trên những đoạn văn dài hơn nhiều so với dữ liệu huấn luyện, trong khi vẫn giữ chất lượng tương đương trên câu thường.

| Cơ chế | Đơn điệu bảo đảm? | Dùng nội dung? | Điểm mạnh | Điểm yếu |
|---|---|---|---|---|
| Nhạy vị trí (Tacotron 2) | Không | Có | Linh hoạt, phổ biến | Bỏ/lặp từ, kém với câu dài |
| GMM | Có (tâm chỉ tăng) | Không | Tổng quát hóa độ dài tốt | Khó sửa khi đã lệch; nhạy cách tham số hóa |
| Forward attention | Có (đứng/tiến 1) | Qua $\tilde{\alpha}$ | Hội tụ nhanh, điều khiển tốc độ | Vẫn tự hồi quy, tuần tự |
| Stepwise monotonic | Có | Qua $e_{t,n}$ | Không thể bỏ/lặp; chế độ cứng | Khối lượng xác suất cần xử lý ở biên |
| DCA | Gần như (nhờ tiên nghiệm) | Không | Rất bền với văn bản dài | Thiết kế phức tạp hơn |

## 21.3 Guided attention loss

Thay vì sửa cấu trúc attention, Tachibana, Uenoyama và Aihara (2018) — trong mô hình DCTTS — đề xuất một cách rất đơn giản: **phạt attention nằm xa đường chéo**. Lý do: trong đa số câu, tốc độ đọc gần đều, nên token ở vị trí tương đối $n/N$ thường được phát âm quanh thời điểm tương đối $t/T$. Định nghĩa ma trận trọng số

$$
W_{n,t} = 1 - \exp\left(-\frac{\left(n/N - t/T\right)^2}{2g^2}\right),
$$

gần 0 trên đường chéo và tiến về 1 ở xa; $g$ điều khiển độ rộng dải "được phép" (bài báo dùng $g = 0.2$). Mất mát là trung bình của $A_{n,t} W_{n,t}$:

$$
\mathcal{L}_{\text{att}} = \frac{1}{NT} \sum_{n=1}^{N} \sum_{t=1}^{T} A_{n,t} \, W_{n,t}.
$$

Mất mát này được cộng (với một hệ số) vào mất mát chính. Nó không *ép* attention phải đúng đường chéo — nó chỉ tạo một "phễu" hướng attention về vùng hợp lý trong giai đoạn đầu, giúp căn chỉnh hình thành nhanh hơn nhiều. Khi căn chỉnh đã ổn định, người ta thường giảm dần hoặc tắt hệ số để không ép mô hình theo tốc độ đọc đều một cách giả tạo (câu có khoảng ngừng dài giữa chừng sẽ lệch khỏi đường chéo một cách hợp lý).

```python
import torch


def guided_attention_weights(N: int, T: int, g: float = 0.2) -> torch.Tensor:
    """W_{n,t} = 1 - exp(-(n/N - t/T)^2 / (2 g^2)); gần 0 trên đường chéo, gần 1 ở xa."""
    n = torch.arange(N).float().unsqueeze(1) / N          # (N, 1)
    t = torch.arange(T).float().unsqueeze(0) / T          # (1, T)
    return 1.0 - torch.exp(-((n - t) ** 2) / (2 * g * g))  # (N, T)


def guided_attention_loss(A, text_lens, mel_lens, g=0.2):
    """A: (B, N, T) attention mềm. Trung bình của W ⊙ A trên vùng hợp lệ của từng câu."""
    W = torch.zeros_like(A)
    valid = torch.zeros_like(A)
    for b, (nb, tb) in enumerate(zip(text_lens.tolist(), mel_lens.tolist())):
        W[b, :nb, :tb] = guided_attention_weights(nb, tb, g)
        valid[b, :nb, :tb] = 1.0
    return (A * W).sum() / valid.sum()


N, T = 10, 40
diag = torch.zeros(1, N, T)
diag[0, torch.arange(T) * N // T, torch.arange(T)] = 1.0      # căn chỉnh chéo lý tưởng
flat = torch.full((1, N, T), 1.0 / N)                          # attention "đều" lúc mới khởi tạo
rev = diag.flip(2)                                             # đọc ngược: sai hoàn toàn
lens_n, lens_t = torch.tensor([N]), torch.tensor([T])
losses = [guided_attention_loss(a, lens_n, lens_t).item() for a in (diag, flat, rev)]
print("diag / flat / reversed:", [f"{v:.4f}" for v in losses])
assert losses[0] < losses[1] < losses[2]
```

Kết quả in ra khoảng `0.0027 / 0.0579 / 0.0749`: căn chỉnh chéo bị phạt rất ít, attention đều và căn chỉnh ngược bị phạt nặng hơn nhiều. Lưu ý giá trị tuyệt đối nhỏ vì ta chia cho số ô $N \times T$ trong khi mỗi cột chỉ có tổng bằng 1; hệ số nhân của mất mát này phải được chỉnh theo cách chuẩn hóa bạn chọn.

## 21.4 Trường độ từ mô hình thầy tự hồi quy

FastSpeech (Ren và cộng sự, 2019) cần trường độ cho từng âm vị nhưng không muốn phụ thuộc công cụ ngoài, nên **rút trường độ từ attention của một mô hình Transformer-TTS tự hồi quy đã huấn luyện** ([Chương 20](ch20-tacotron.md)). Transformer-TTS có nhiều đầu attention ở nhiều lớp; không phải đầu nào cũng có dạng chéo. FastSpeech chọn đầu "tập trung" nhất theo **focus rate**:

$$
F = \frac{1}{T} \sum_{t=1}^{T} \max_{n} \alpha_{t,n},
$$

tức trung bình (theo khung) của trọng số lớn nhất trong mỗi cột. Một đầu attention sắc nét, chéo có $F$ gần 1; một đầu nhòe có $F$ nhỏ. Trường độ của token $n$ là số khung mà nó nhận trọng số lớn nhất:

$$
d_n = \sum_{t=1}^{T} \mathbb{1}\left[\arg\max_{m} \alpha_{t,m} = n\right].
$$

Cách này đơn giản nhưng có hai nhược điểm: (i) căn chỉnh rút ra có thể **vi phạm đơn điệu** (argmax nhảy lùi) và (ii) một token ngắn bị "lướt qua" sẽ có $d_n = 0$. Đoạn mã dưới minh họa cả trường hợp tốt lẫn trường hợp xấu:

```python
import torch


def focus_rate(A: torch.Tensor) -> torch.Tensor:
    """A: (N, T), mỗi cột là phân phối trên token. F = (1/T) * sum_t max_n A[n, t]."""
    return A.max(dim=0).values.mean()


def durations_from_attention(A: torch.Tensor) -> torch.Tensor:
    """d_n = số khung t có argmax_m A[m, t] = n (cách làm của FastSpeech)."""
    return torch.bincount(A.argmax(dim=0), minlength=A.size(0))   # (N,)


def soft_alignment_from_durations(d, blur=0.5):
    """Attention mềm 'giả lập': Gauss quanh tâm mỗi token (độ rộng tỉ lệ d), chuẩn hóa theo n."""
    centers = torch.cumsum(d, 0) - d / 2.0                         # (N,) tâm token, đơn vị khung
    t = torch.arange(int(d.sum())).float() + 0.5                   # (T,)
    logits = -((t[None, :] - centers[:, None]) / (blur * d[:, None])) ** 2
    return torch.softmax(logits, dim=0)                            # (N, T)


d_true = torch.tensor([3, 5, 1, 6, 4]).float()
A_good = soft_alignment_from_durations(d_true)
print("tốt   : F =", round(focus_rate(A_good).item(), 3), durations_from_attention(A_good).tolist())
assert durations_from_attention(A_good).tolist() == [3, 5, 1, 6, 4]

# Giả lập một đầu attention "lướt qua" token ngắn (chỉ số 2, ứng với khung 8)
A_skip = A_good.clone()
A_skip[:, 8] = torch.tensor([0.0, 0.5, 0.3, 0.2, 0.0])
d_skip = durations_from_attention(A_skip)
print("lướt  : F =", round(focus_rate(A_skip).item(), 3), d_skip.tolist())
assert d_skip.sum() == d_true.sum() and d_skip[2] == 0     # tổng vẫn = T nhưng token 2 bị "nuốt"
```

Kết quả: căn chỉnh tốt cho `[3, 5, 1, 6, 4]`, còn khi chỉ một cột bị lệch, ta được `[3, 6, 0, 6, 4]` — token thứ ba biến mất khỏi đầu ra của FastSpeech, trong khi focus rate gần như không đổi. Bài học: focus rate là thước đo **trung bình**, không phát hiện được lỗi cục bộ. Thay argmax từng cột bằng MAS (mục 21.6) trên $\log \mathbf{A}$ sẽ luôn cho căn chỉnh hợp lệ với mọi $d_n \ge 1$. Cách tiếp cận mô hình thầy ngày nay ít dùng vì cần huấn luyện cả một mô hình tự hồi quy chỉ để lấy trường độ ([Chương 22](ch22-fastspeech.md)).

## 21.5 Căn chỉnh cưỡng bức (forced alignment)

**Căn chỉnh cưỡng bức** dùng một mô hình nhận dạng tiếng nói đã biết trước bản chép lời để tìm ranh giới thời gian của từng đơn vị. "Cưỡng bức" vì chuỗi đơn vị đã cố định; chỉ cần tìm *khi nào* chúng xảy ra.

### Dựa trên HMM

Mỗi âm vị được mô hình hóa bằng một HMM trái-sang-phải (thường 3 trạng thái); nối các HMM của chuỗi âm vị trong câu ta được một HMM lớn với các trạng thái $j = 1, \ldots, J$. Với xác suất phát xạ $b_j(\mathbf{y}_t)$ (GMM hoặc mạng nơ-ron) và xác suất chuyển $a_{jj}$ (ở lại), $a_{j-1,j}$ (tiến), thuật toán **Viterbi** tìm chuỗi trạng thái xác suất cao nhất bằng đệ quy

$$
\delta_t(j) = \log b_j(\mathbf{y}_t) + \max\left\{\delta_{t-1}(j) + \log a_{jj}, \; \delta_{t-1}(j-1) + \log a_{j-1,j}\right\}.
$$

Hãy để ý cấu trúc "ở lại hoặc tiến một bước" — ta sẽ gặp lại nó y hệt trong MAS.

Công cụ phổ biến nhất cho TTS là **Montreal Forced Aligner** (MFA; McAuliffe và cộng sự, 2017), xây trên Kaldi. MFA huấn luyện GMM-HMM theo nhiều giai đoạn (đơn âm vị, tam âm vị, thích nghi người nói) trực tiếp trên dữ liệu của bạn, cần một **từ điển phát âm** ánh xạ từ sang chuỗi âm vị, và xuất ranh giới dưới dạng TextGrid (định dạng của Praat). FastSpeech 2 dùng MFA để lấy trường độ ([Chương 22](ch22-fastspeech.md)). Ưu điểm: ranh giới khá chính xác, không cần GPU, kiểm soát được bộ âm vị. Nhược điểm: thêm một công cụ ngoài vào quy trình, cần từ điển, và lỗi từ điển (từ mượn, tên riêng) lan thẳng vào trường độ.

Khi chuyển ranh giới (tính bằng giây) sang số khung, đừng làm tròn **từng trường độ**: hãy làm tròn **các ranh giới** rồi lấy hiệu, $d_n = \operatorname{round}(e_n f_s / H) - \operatorname{round}(b_n f_s / H)$ với $b_n, e_n$ là thời điểm bắt đầu và kết thúc. Làm vậy sai số không cộng dồn, và tổng trường độ khớp với số khung mel (sau khi xử lý khung lẻ ở cuối).

### Dựa trên CTC

CTC (Graves và cộng sự, 2006) huấn luyện mô hình nhận dạng với một ký hiệu đặc biệt **blank** (ký hiệu $\varnothing$). Một đường đi mức khung $\boldsymbol{\pi} = (\pi_1, \ldots, \pi_T)$ được ánh xạ về chuỗi nhãn bằng phép $\mathcal{B}$: gộp các nhãn lặp liền kề rồi xóa blank (ví dụ $\mathcal{B}(\text{a a} \varnothing \text{b b}) = \text{ab}$). Xác suất của chuỗi nhãn là tổng trên mọi đường đi tương thích:

$$
P(\mathbf{x} \mid \mathbf{Y}) = \sum_{\boldsymbol{\pi} \in \mathcal{B}^{-1}(\mathbf{x})} \prod_{t=1}^{T} P(\pi_t \mid \mathbf{Y}).
$$

**Căn chỉnh cưỡng bức CTC** thay tổng bằng max (Viterbi) để tìm đường đi tốt nhất. Thư viện torchaudio cung cấp sẵn `torchaudio.functional.forced_align` (chúng tôi đã kiểm tra với torchaudio 2.11; API của torchaudio thay đổi khá nhanh, hãy kiểm tra phiên bản của bạn). Đầu ra là nhãn của từng khung; ta cần chuyển nó thành trường độ **theo vị trí token** (không phải theo mã nhãn, vì một nhãn có thể xuất hiện nhiều lần):

```python
import torch
import torchaudio.functional as AF


def ctc_path_to_durations(path, n_tokens, blank=0):
    """path: (T,) nhãn từng khung (đường Viterbi CTC). Trả về d: (N,) theo VỊ TRÍ token.
    Token mới bắt đầu khi nhãn khác blank và khác nhãn của khung liền trước;
    khung blank được gán cho token đứng trước (blank đầu câu gán cho token đầu)."""
    d = torch.zeros(n_tokens, dtype=torch.long)
    pos, prev = -1, blank
    for lab in path.tolist():
        if lab != blank and lab != prev:
            pos += 1
        d[max(pos, 0)] += 1
        prev = lab
    assert pos == n_tokens - 1, "đường căn chỉnh không đi qua đủ N token"
    return d


torch.manual_seed(0)
T, V = 30, 6                                     # 30 khung, từ vựng 6 nhãn (0 = blank)
log_probs = torch.randn(1, T, V).log_softmax(-1)  # thay cho đầu ra của một mô hình CTC thật
targets = torch.tensor([[2, 3, 3, 5, 1]], dtype=torch.int32)   # chú ý: nhãn 3 lặp lại
path, scores = AF.forced_align(log_probs, targets, blank=0)   # Viterbi trên lưới CTC
d = ctc_path_to_durations(path[0], targets.size(1))
print(path[0].tolist())
print("durations:", d.tolist())
assert d.sum() == T and d.min() >= 1 and len(d) == 5
```

Mô hình CTC thật có xu hướng **nhọn** (*peaky*): mỗi token chỉ "bắn" ở một hoặc vài khung, phần còn lại là blank. Do đó vị trí của các đỉnh khá đáng tin, nhưng *ranh giới* thì không — quy tắc "gán blank cho token đứng trước" ở trên chỉ là một lựa chọn (có thể chia đôi blank cho hai token hai bên, hoặc gán đoạn blank dài cho một token khoảng lặng). Với TTS, hãy kiểm tra phân phối trường độ thu được, đặc biệt ở token cuối câu và quanh khoảng ngừng.

Ưu điểm lớn của CTC là có thể dùng **mô hình nhận dạng đa ngôn ngữ huấn luyện sẵn**. Ví dụ, torchaudio có gói `torchaudio.pipelines.MMS_FA` dựa trên dự án MMS (Pratap và cộng sự, 2023); bộ nhãn của nó là chữ Latin không dấu, nên văn bản tiếng Việt phải được bỏ dấu (latin hóa) trước khi căn chỉnh rồi ánh xạ ngược về âm tiết gốc. Chất lượng ranh giới cho tiếng Việt với các công cụ này cần được bạn tự đánh giá trên một mẫu có gán nhãn tay.

## 21.6 Monotonic Alignment Search (MAS)

### Bối cảnh

Glow-TTS (Kim và cộng sự, 2020; xem [Chương 23](ch23-glow-grad-matcha.md)) đặt câu hỏi: có thể **tự tìm căn chỉnh cứng tốt nhất** ngay trong lúc huấn luyện, không cần attention lẫn công cụ ngoài? Trong Glow-TTS, encoder văn bản sinh cho mỗi token $n$ một phân phối Gauss với tham số $(\boldsymbol{\mu}_n, \boldsymbol{\sigma}_n)$, và mel được ánh xạ khả nghịch thành biến ẩn $\mathbf{z}_t$. Với một căn chỉnh $a$, log-likelihood là $\sum_t \log \mathcal{N}(\mathbf{z}_t; \boldsymbol{\mu}_{a(t)}, \boldsymbol{\sigma}_{a(t)})$. Căn chỉnh tốt nhất là căn chỉnh làm tổng này lớn nhất. Trừu tượng hóa: cho **ma trận điểm** $\mathbf{S} \in \mathbb{R}^{N \times T}$ với $S_{n,t}$ là "độ hợp" giữa token $n$ và khung $t$ (ở đây $S_{n,t} = \log \mathcal{N}(\mathbf{z}_t; \boldsymbol{\mu}_n, \boldsymbol{\sigma}_n)$), tìm

$$
\mathbf{A}^\star = \arg\max_{\mathbf{A} \in \mathcal{S}_{N,T}} \sum_{n,t} A_{n,t} S_{n,t} = \arg\max_{a} \sum_{t=1}^{T} S_{a(t),t}.
$$

### Suy ra quy hoạch động

Định nghĩa $Q_{n,t}$ là **điểm lớn nhất** của các căn chỉnh *bộ phận* gán khung $1, \ldots, t$ cho token $1, \ldots, n$ (đơn điệu, không nhảy cóc), với khung $t$ thuộc token $n$.

*Bước 1 — trường hợp cơ sở.* Khung đầu tiên bắt buộc thuộc token đầu tiên: $Q_{1,1} = S_{1,1}$, và $Q_{n,1} = -\infty$ với $n > 1$.

*Bước 2 — đệ quy.* Xét một căn chỉnh bộ phận tối ưu kết thúc tại $(n, t)$ với $t > 1$. Khung $t - 1$ phải thuộc token $n$ (ở lại) hoặc token $n - 1$ (vừa tiến) — không có khả năng nào khác do ràng buộc không nhảy cóc. Phần căn chỉnh trên khung $1..t-1$ phải là tối ưu cho trạng thái kết thúc của nó (nếu không, thay nó bằng phần tối ưu sẽ cho tổng lớn hơn — mâu thuẫn). Đây là tính chất **cấu trúc con tối ưu**, và ta được:

$$
Q_{n,t} = S_{n,t} + \max\left(Q_{n,t-1}, \; Q_{n-1,t-1}\right),
$$

với quy ước $Q_{0,t} = -\infty$.

*Bước 3 — kết quả.* Do ràng buộc biên, căn chỉnh đầy đủ phải kết thúc ở $(N, T)$, nên điểm tối ưu là $Q_{N,T}$.

*Bước 4 — truy vết ngược.* Bắt đầu từ $n = N$, $t = T$. Ở mỗi bước, gán khung $t$ cho token $n$; rồi quyết định khung $t - 1$ thuộc token nào: chuyển sang $n - 1$ nếu $Q_{n-1,t-1} > Q_{n,t-1}$ (hoặc nếu $n = t$, khi đó token $n$ không còn chỗ, buộc phải lùi), ngược lại giữ $n$.

**Độ phức tạp** là $O(NT)$ về thời gian và bộ nhớ — so với $\binom{T-1}{N-1}$ của vét cạn. Chú ý ô $(n, t)$ chỉ khả thi khi $n \le t$ (đủ khung cho $n$ token đầu) và $N - n \le T - t$ (đủ khung cho các token còn lại); các cài đặt tối ưu chỉ duyệt vùng khả thi này.

!!! example "Ví dụ tính tay"
    Cho $N = 3$, $T = 5$ và ma trận điểm (hàng là token, cột là khung):

    | | $t=1$ | $t=2$ | $t=3$ | $t=4$ | $t=5$ |
    |---|---|---|---|---|---|
    | $S_{1,t}$ | 2 | 1 | −1 | −2 | −3 |
    | $S_{2,t}$ | −1 | 3 | 2 | −1 | −2 |
    | $S_{3,t}$ | −3 | −2 | 0 | 2 | 1 |

    Áp dụng đệ quy, ta được bảng $Q$:

    | | $t=1$ | $t=2$ | $t=3$ | $t=4$ | $t=5$ |
    |---|---|---|---|---|---|
    | $Q_{1,t}$ | 2 | 3 | 2 | 0 | −3 |
    | $Q_{2,t}$ | $-\infty$ | 5 | 7 | 6 | 4 |
    | $Q_{3,t}$ | $-\infty$ | $-\infty$ | 5 | 9 | **10** |

    Chẳng hạn $Q_{2,3} = S_{2,3} + \max(Q_{2,2}, Q_{1,2}) = 2 + \max(5, 3) = 7$. Truy vết từ $(3, 5)$: $Q_{3,4} = 9 > Q_{2,4} = 6$ nên khung 4 vẫn thuộc token 3; $Q_{2,3} = 7 > Q_{3,3} = 5$ nên khung 3 thuộc token 2; tại $(2, 2)$ có $n = t$ nên khung 1 thuộc token 1. Kết quả $\mathbf{d} = (1, 2, 2)$ với tổng điểm $2 + 3 + 2 + 2 + 1 = 10$.

### Cài đặt và kiểm chứng bằng vét cạn

```python
import numpy as np


def monotonic_alignment_search(S: np.ndarray) -> np.ndarray:
    """S: (N, T) điểm (log-likelihood) của cặp token n – khung t.
    Trả về A: (N, T) nhị phân, đơn điệu, không bỏ token, tối đa hóa sum(S * A)."""
    N, T = S.shape
    assert T >= N, "mỗi token cần ít nhất một khung"
    Q = np.full((N, T), -np.inf)
    Q[0, 0] = S[0, 0]                                   # khung đầu buộc thuộc token đầu
    for t in range(1, T):
        stay = Q[:, t - 1]                              # khung t-1 cũng thuộc token n
        move = np.concatenate(([-np.inf], Q[:-1, t - 1]))  # khung t-1 thuộc token n-1
        Q[:, t] = S[:, t] + np.maximum(stay, move)
    A = np.zeros((N, T), dtype=np.int64)
    n = N - 1                                           # khung cuối buộc thuộc token cuối
    for t in range(T - 1, -1, -1):                      # truy vết ngược
        A[n, t] = 1
        if n > 0 and (n == t or Q[n - 1, t - 1] > Q[n, t - 1]):
            n -= 1
    return A
```

Để chắc chắn cài đặt đúng, ta so với **vét cạn** trên các ví dụ nhỏ: liệt kê mọi bộ trường độ hợp lệ (đúng $\binom{T-1}{N-1}$ bộ), tính điểm và lấy max.

```python
import itertools
from math import comb


def all_durations(N, T):
    """Liệt kê mọi d với d_n >= 1, sum d = T: chọn N-1 'vết cắt' trong T-1 khe."""
    for cuts in itertools.combinations(range(1, T), N - 1):
        yield np.diff((0, *cuts, T))


def durations_to_alignment(d):
    N, T = len(d), int(np.sum(d))
    A = np.zeros((N, T), dtype=np.int64)
    A[np.repeat(np.arange(N), d), np.arange(T)] = 1     # token n chiếm d_n khung liên tiếp
    return A


def brute_force_best(S):
    best, best_A, count = -np.inf, None, 0
    for d in all_durations(*S.shape):
        A = durations_to_alignment(d)
        score, count = (S * A).sum(), count + 1
        if score > best:
            best, best_A = score, A
    return best, best_A, count


rng = np.random.default_rng(0)
for _ in range(300):
    N = int(rng.integers(1, 6))
    T = int(rng.integers(N, 12))
    S = rng.normal(size=(N, T))
    A = monotonic_alignment_search(S)
    best, A_bf, count = brute_force_best(S)
    assert count == comb(T - 1, N - 1)                 # số căn chỉnh hợp lệ = C(T-1, N-1)
    assert np.isclose((S * A).sum(), best)             # DP đạt đúng giá trị tối ưu
    assert np.array_equal(A, A_bf)                     # ... và đúng cùng một đường đi
    d = A.sum(1)
    assert d.min() >= 1 and d.sum() == T and np.all(A.sum(0) == 1)
print("MAS khớp với vét cạn trên 300 ví dụ ngẫu nhiên")
```

Phép thử dùng số thực ngẫu nhiên nên gần như không có hai căn chỉnh cùng điểm; nhờ vậy ta kiểm tra được cả *giá trị* lẫn *đường đi*. Ngoài ra, phép thử xác nhận công thức đếm $\binom{T-1}{N-1}$ của mục 21.1.

### Phiên bản theo batch cho GPU

Glow-TTS chính thức cài MAS bằng Cython chạy trên CPU; nhiều dự án dùng Numba. Một cách khác là vector hóa theo batch và theo token, chỉ lặp theo thời gian — chạy được trên GPU và đủ nhanh cho đa số trường hợp vì vòng lặp chỉ có $T$ bước:

```python
import torch


@torch.no_grad()
def mas_batch(S, text_lens, mel_lens):
    """S: (B, N, T) điểm; độ dài thật text_lens, mel_lens: (B,). Trả về A: (B, N, T).
    Vector hóa theo B và N, chỉ lặp theo T -> chạy được trên GPU."""
    B, N, T = S.shape
    neg_inf = torch.tensor(float("-inf"), dtype=S.dtype, device=S.device)
    tok_ok = torch.arange(N, device=S.device)[None, :] < text_lens[:, None]   # (B, N)
    S = torch.where(tok_ok[:, :, None], S, neg_inf)     # token đệm không bao giờ được chọn
    Q = torch.full_like(S, float("-inf"))
    Q[:, 0, 0] = S[:, 0, 0]
    for t in range(1, T):
        prev = Q[:, :, t - 1]                                         # (B, N)
        move = torch.cat([neg_inf.expand(B, 1), prev[:, :-1]], dim=1)
        Q[:, :, t] = S[:, :, t] + torch.maximum(prev, move)
    A = torch.zeros_like(S)
    b_idx = torch.arange(B, device=S.device)
    n = text_lens - 1                                                 # (B,) token hiện tại
    for t in range(T - 1, -1, -1):
        active = t < mel_lens                                         # khung t còn hợp lệ?
        A[b_idx[active], n[active], t] = 1.0
        if t == 0:
            break
        stay = Q[b_idx, n, t - 1]
        move = Q[b_idx, (n - 1).clamp(min=0), t - 1]
        go_back = active & (n > 0) & ((n == t) | (move > stay))
        n = n - go_back.long()
    return A


# So khớp với bản numpy ở trên trên một batch có độ dài khác nhau
torch.manual_seed(0)
B, N, T = 4, 6, 15
S = torch.randn(B, N, T, dtype=torch.float64)
text_lens = torch.tensor([6, 4, 5, 2])
mel_lens = torch.tensor([15, 9, 12, 2])
A = mas_batch(S, text_lens, mel_lens)
for b in range(B):
    nb, tb = int(text_lens[b]), int(mel_lens[b])
    ref = monotonic_alignment_search(S[b, :nb, :tb].numpy())
    assert np.array_equal(A[b, :nb, :tb].long().numpy(), ref)
    assert A[b].sum() == tb                       # đúng một token cho mỗi khung hợp lệ
print("mas_batch khớp với bản numpy; durations:", A.sum(-1).long().tolist())
```

### MAS trong bức tranh lớn

- **Quan hệ với Viterbi.** MAS chính là Viterbi trên một HMM trái-sang-phải mỗi token một trạng thái, với "xác suất phát xạ" $e^{S_{n,t}}$ và xác suất chuyển giống nhau cho mọi token. Khi đó mọi căn chỉnh hợp lệ có đúng $N - 1$ lần "tiến" và $T - N$ lần "ở lại", nên tổng log-xác suất chuyển là một hằng số và biến mất khỏi argmax (Bài tập 6).
- **Quan hệ với DTW.** Dynamic Time Warping (Sakoe và Chiba, 1978) cũng là quy hoạch động trên lưới hai chuỗi, nhưng cho phép cả bước "dọc" (một khung ứng với nhiều token). MAS cấm bước đó: mỗi khung thuộc đúng một token.
- **Không khả vi.** MAS là một phép argmax rời rạc, không có gradient. Glow-TTS dùng nó theo kiểu **EM cứng** (*hard EM*, hay huấn luyện Viterbi): cố định tham số, tìm $\mathbf{A}^\star$; cố định $\mathbf{A}^\star$, cập nhật tham số bằng gradient của log-likelihood. Chi tiết ở [Chương 23](ch23-glow-grad-matcha.md); VITS dùng cùng cơ chế ([Chương 27](../phan-7-e2e-bieu-cam/ch27-vits.md)).
- **Nguy cơ kẹt.** Vì là quyết định cứng ngay từ đầu, MAS có thể khóa vào một căn chỉnh kém khi mô hình còn ngẫu nhiên. VITS2 bổ sung nhiễu vào ma trận điểm trong giai đoạn đầu để tăng khám phá — xem [Chương 27](../phan-7-e2e-bieu-cam/ch27-vits.md).

## 21.7 Học căn chỉnh không giám sát: forward-sum, tiên nghiệm và nhị phân hóa

Badlani và cộng sự (2022), trong bài "One TTS Alignment To Rule Them All", đề xuất một mô-đun căn chỉnh **độc lập với kiến trúc** có thể gắn vào mô hình tự hồi quy lẫn không tự hồi quy. Ý tưởng kết hợp ưu điểm của cả hai thế giới: một căn chỉnh **mềm, khả vi** để học, và một căn chỉnh **cứng, hợp lệ** để dùng.

### Mô-đun căn chỉnh

Hai mạng tích chập nhỏ mã hóa văn bản thành $\mathbf{h}^{\text{text}}_n$ và mel thành $\mathbf{h}^{\text{mel}}_t$ trong cùng một không gian. Căn chỉnh mềm là softmax theo trục token của khoảng cách âm:

$$
D_{n,t} = \left\lVert \mathbf{h}^{\text{text}}_n - \mathbf{h}^{\text{mel}}_t \right\rVert_2, \qquad
A^{\text{soft}}_{n,t} = \frac{\exp(-D_{n,t})}{\sum_{m} \exp(-D_{m,t})}.
$$

Như vậy $A^{\text{soft}}_{n,t}$ được hiểu là $P(a(t) = n \mid \mathbf{y}_t)$ — xác suất khung $t$ thuộc token $n$.

### Forward-sum loss

Ta muốn tối đa hóa xác suất rằng **có một** căn chỉnh đơn điệu hợp lệ giải thích dữ liệu — tức là tổng (chứ không phải max) trên mọi căn chỉnh:

$$
P(\mathcal{S} \mid \mathbf{x}, \mathbf{Y}) = \sum_{a \in \mathcal{S}_{N,T}} \prod_{t=1}^{T} A^{\text{soft}}_{a(t), t}, \qquad
\mathcal{L}_{\text{ForwardSum}} = -\log P(\mathcal{S} \mid \mathbf{x}, \mathbf{Y}).
$$

Tổng trên $\binom{T-1}{N-1}$ căn chỉnh tính được hiệu quả bằng **thuật toán tiến**. Gọi $\ell_{n,t} = \log A^{\text{soft}}_{n,t}$ và $R_{n,t}$ là log của tổng xác suất các căn chỉnh bộ phận kết thúc tại $(n, t)$. Lặp lại đúng lập luận của MAS nhưng thay "chọn tốt nhất" bằng "cộng tất cả":

$$
R_{n,t} = \ell_{n,t} + \log\left(e^{R_{n,t-1}} + e^{R_{n-1,t-1}}\right)
\quad \text{so với} \quad
Q_{n,t} = \ell_{n,t} + \max\left(Q_{n,t-1}, Q_{n-1,t-1}\right).
$$

Nói gọn: **forward-sum là phiên bản "mềm" (log-sum-exp) của MAS**, và MAS là giới hạn "cứng" (max) của forward-sum. Vì log-sum-exp khả vi, gradient chảy về $\ell_{n,t}$ và về hai bộ mã hóa.

Điểm tinh tế trong cài đặt: không cần tự viết thuật toán tiến, vì **CTC loss đã làm đúng việc này**. Coi mỗi token $n$ là một "lớp" riêng mang nhãn $n$ (nhãn khác nhau từng đôi một), mục tiêu là chuỗi nhãn $(1, 2, \ldots, N)$. Vì các nhãn liền kề luôn khác nhau, CTC cho phép đi thẳng từ nhãn $n$ sang $n + 1$, và tổng trên các đường đi CTC chính là tổng trên các căn chỉnh đơn điệu — cộng thêm các đường có khung blank xen giữa. Bản cài đặt tham chiếu của NVIDIA gán cho blank một log-xác suất cố định (ví dụ $-1$) làm "chỗ trống" cho các khung không khớp token nào. Mã dưới đây kiểm chứng: khi blank gần như bị cấm, CTC loss trùng khớp với tổng vét cạn trên mọi căn chỉnh, và với thuật toán tiến tự viết.

```python
import torch
import torch.nn.functional as F


def forward_sum_loss(attn_logits, text_lens, mel_lens, blank_logprob=-1.0):
    """attn_logits: (B, T, N) điểm chưa chuẩn hóa giữa khung t và token n.
    Trả về -log sum_{A đơn điệu} prod_t P(a(t) | t), trung bình trên batch (qua CTC)."""
    B = attn_logits.size(0)
    padded = F.pad(attn_logits, (1, 0), value=blank_logprob)        # (B, T, N+1), cột 0 = blank
    total = attn_logits.new_zeros(())
    for b in range(B):
        n, t = int(text_lens[b]), int(mel_lens[b])
        log_probs = padded[b, :t, : n + 1].log_softmax(-1)          # (t, n+1): chuẩn hóa trên token
        target = torch.arange(1, n + 1).unsqueeze(0)                # nhãn 1..n: mỗi token một nhãn
        total = total + F.ctc_loss(log_probs.unsqueeze(1), target,  # CTC cần (T, B=1, C)
                                   torch.tensor([t]), torch.tensor([n]),
                                   blank=0, reduction="sum", zero_infinity=True)
    return total / B


def log_forward(logp):
    """Thuật toán tiến trong miền log; logp: (N, T). Thay logaddexp bằng max -> chính là MAS."""
    N, T = logp.shape
    alpha = torch.full((N,), float("-inf"), dtype=logp.dtype)
    alpha[0] = logp[0, 0]
    for t in range(1, T):
        move = torch.cat([alpha.new_full((1,), float("-inf")), alpha[:-1]])
        alpha = logp[:, t] + torch.logaddexp(alpha, move)
    return alpha[-1]


# Kiểm tra: khi blank gần như bị cấm, CTC = tổng trên mọi căn chỉnh đơn điệu (vét cạn)
torch.manual_seed(0)
N, T = 4, 9
logits = torch.randn(1, T, N, dtype=torch.float64, requires_grad=True)
loss = forward_sum_loss(logits, torch.tensor([N]), torch.tensor([T]), blank_logprob=-1e4)
logp = logits[0].log_softmax(-1).T.detach()                          # (N, T)
brute = torch.logsumexp(torch.stack([
    (logp * torch.from_numpy(durations_to_alignment(d))).sum()
    for d in all_durations(N, T)]), dim=0)
assert torch.isclose(-loss, brute) and torch.isclose(log_forward(logp), brute)
loss.backward()                                                      # khả vi theo logits
assert logits.grad is not None and torch.isfinite(logits.grad).all()
print(f"forward-sum = {loss.item():.4f}  (vét cạn: {-brute.item():.4f})")
```

Hai chi tiết thực hành. (i) `reduction="sum"` cho đúng $-\log P$; mặc định `"mean"` của PyTorch còn **chia cho độ dài chuỗi đích** ($N$), làm thay đổi trọng số tương đối giữa câu dài và câu ngắn — nhiều bản cài đặt dùng mặc định này, bạn cần biết để chỉnh hệ số mất mát. (ii) `zero_infinity=True` tránh NaN khi một câu có $T < N$ (thường do lỗi dữ liệu).

### Tiên nghiệm beta–nhị thức

Ở đầu huấn luyện, hai bộ mã hóa là ngẫu nhiên và forward-sum hội tụ chậm. Badlani và cộng sự cộng thêm một **tiên nghiệm tĩnh** khuyến khích căn chỉnh gần đường chéo: với khung $t$ ($1 \le t \le T$), phân phối tiên nghiệm trên token là beta–nhị thức

$$
P_{\text{prior}}(a(t) = n) = \operatorname{BetaBinomial}\left(n - 1;\; N - 1,\; \omega t,\; \omega (T - t + 1)\right),
$$

với $\omega$ là hệ số tỉ lệ (thường bằng 1). Kỳ vọng của phân phối này là $(N - 1)\, t / (T + 1)$ — tăng tuyến tính theo $t$, tức nằm trên đường chéo. Trong miền log, tiên nghiệm được **cộng** vào log-xác suất trước khi tính forward-sum (tương đương nhân $A^{\text{soft}}$ với $P_{\text{prior}}$ rồi chuẩn hóa lại). Khác guided attention loss — một số hạng phạt riêng — tiên nghiệm này thay đổi trực tiếp phân phối mà forward-sum nhìn thấy.

```python
import numpy as np
import torch
from scipy.stats import betabinom


def beta_binomial_prior(N, T, omega=1.0):
    """Tiên nghiệm căn chỉnh (T, N): hàng t (1..T) là BetaBinomial(N-1, ω t, ω (T - t + 1))."""
    k = np.arange(N)
    P = np.stack([betabinom(N - 1, omega * t, omega * (T - t + 1)).pmf(k)
                  for t in range(1, T + 1)])
    return torch.from_numpy(P).float()                     # (T, N)


P = beta_binomial_prior(N=8, T=40)
assert torch.allclose(P.sum(1), torch.ones(40))           # mỗi hàng là một phân phối trên token
mean_pos = (P * torch.arange(8)).sum(1)                   # kỳ vọng = (N-1) t / (T+1): đường chéo
assert torch.allclose(mean_pos, 7 * torch.arange(1, 41) / 41, atol=1e-5)
print("mode theo khung:", P.argmax(1).tolist())
```

### Nhị phân hóa

Mô hình TTS không tự hồi quy cần căn chỉnh **cứng** để mở rộng chuỗi ẩn (length regulator, [Chương 22](ch22-fastspeech.md)) và để lấy trường độ huấn luyện bộ dự đoán trường độ. Căn chỉnh cứng $\mathbf{A}^{\text{hard}}$ được lấy bằng MAS trên $\log \mathbf{A}^{\text{soft}}$. Để căn chỉnh mềm "sắc" dần và khớp với căn chỉnh cứng, ta thêm **binarization loss** — về bản chất là phân kỳ KL từ $\mathbf{A}^{\text{hard}}$ tới $\mathbf{A}^{\text{soft}}$:

$$
\mathcal{L}_{\text{bin}} = -\sum_{(n,t) : A^{\text{hard}}_{n,t} = 1} \log A^{\text{soft}}_{n,t},
\qquad
\mathcal{L}_{\text{align}} = \mathcal{L}_{\text{ForwardSum}} + \lambda_{\text{bin}} \mathcal{L}_{\text{bin}}.
$$

$\mathcal{L}_{\text{bin}}$ thường chỉ được bật sau khi forward-sum đã hội tụ phần nào (một dạng *curriculum*): nhị phân hóa quá sớm khóa mô hình vào căn chỉnh tồi, giống nguy cơ của MAS thuần.

### Ghép lại: học căn chỉnh từ con số không

Ví dụ đồ chơi dưới đây ghép mọi mảnh: dữ liệu giả gồm các "âm" lý tưởng cho từng ký hiệu, trường độ thật bị **ẩn** khỏi mô hình; một mô-đun căn chỉnh tí hon được huấn luyện chỉ bằng forward-sum (có tiên nghiệm); sau đó MAS trên câu chưa thấy phải khôi phục được căn chỉnh thật. Đoạn mã dùng lại các hàm đã định nghĩa ở trên.

```python
import torch.nn as nn

torch.manual_seed(0)
V, F_MEL, C = 12, 16, 32
protos = torch.randn(V, F_MEL)                         # "âm thanh" lý tưởng của từng ký hiệu


def make_utt():
    n = int(torch.randint(5, 11, ()))
    x = torch.randint(0, V, (n,))                      # chuỗi token (N,)
    d = torch.randint(2, 8, (n,))                      # trường độ thật (mô hình không được thấy)
    Y = protos[x].repeat_interleave(d, 0) + 0.5 * torch.randn(int(d.sum()), F_MEL)  # (T, F)
    log_prior = beta_binomial_prior(n, len(Y)).log().clamp(min=-30)                # (T, N)
    return x, d, Y, log_prior


class AlignerToy(nn.Module):
    """Mô-đun căn chỉnh tí hon theo tinh thần Badlani và cộng sự (2022).
    Bản thật dùng Conv1d nhân 3; dữ liệu đồ chơi ở đây chỉ cần mạng theo từng điểm."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, C)
        self.text_net = nn.Linear(C, C)
        self.mel_net = nn.Sequential(nn.Linear(F_MEL, C), nn.ReLU(), nn.Linear(C, C))

    def forward(self, x, Y):                            # x: (N,), Y: (T, F)
        ht, hm = self.text_net(self.emb(x)), self.mel_net(Y)   # (N, C), (T, C)
        return -torch.cdist(hm, ht) ** 2                # (T, N): -khoảng cách bình phương


def frame_accuracy(score_fn, utts):
    """Tỉ lệ khung được MAS gán đúng token so với căn chỉnh thật."""
    correct = total = 0
    for x, d, Y, _ in utts:
        A_hat = monotonic_alignment_search(score_fn(x, Y).log_softmax(-1).T.numpy())  # (N, T)
        correct += int((A_hat * durations_to_alignment(d.numpy())).sum())
        total += len(Y)
    return correct / total


data = [make_utt() for _ in range(48)]
train, test = data[:32], data[32:]
model = AlignerToy()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for step in range(100):
    loss = sum(forward_sum_loss((model(x, Y).log_softmax(-1) + lp)[None],   # + log tiên nghiệm
                                torch.tensor([len(x)]), torch.tensor([len(Y)]))
               for x, d, Y, lp in train) / len(train)
    opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():
    acc = frame_accuracy(model, test)                   # không dùng tiên nghiệm khi đánh giá
    oracle = frame_accuracy(lambda x, Y: -torch.cdist(Y, protos[x]) ** 2, test)
print(f"forward-sum: {loss.item():.3f} | khung gán đúng: {acc:.3f} (trần lý tưởng: {oracle:.3f})")
assert acc > 0.85
```

Trên máy của chúng tôi, sau 100 bước, khoảng 94% khung của các câu chưa thấy được gán đúng, trong khi "trần" — MAS với chính các âm lý tưởng — khoảng 95%. Trần không đạt 100% vì khi hai token giống nhau đứng cạnh nhau, ranh giới giữa chúng vốn không xác định được từ âm thanh. Cuối cùng là binarization loss:

```python
def binarization_loss(A_hard, log_soft):
    """L_bin = -(1/T) * sum_{(n,t): A_hard=1} log A_soft[n, t]; A_hard, log_soft: (N, T)."""
    return -(A_hard * log_soft).sum() / A_hard.sum()


x, d, Y, _ = test[0]
log_soft = model(x, Y).log_softmax(-1).T                         # (N, T), có gradient
A_hard = torch.from_numpy(monotonic_alignment_search(log_soft.detach().numpy())).float()
l_bin = binarization_loss(A_hard, log_soft)
l_bin.backward()                                                 # gradient chỉ đi qua A_soft
print(f"L_bin = {l_bin.item():.3f}")
assert l_bin.item() >= 0
```

Mô-đun căn chỉnh này được dùng trong RAD-TTS (Shih và cộng sự, 2021), trong các phiên bản sau của FastPitch, và trong nhiều bản cài đặt FastSpeech 2 mã nguồn mở thay cho MFA — giúp quy trình huấn luyện **không còn phụ thuộc công cụ ngoài** ([Dự án 2](../phan-10-du-an/du-an-2-fastspeech2-hifigan.md) sẽ so sánh hai lựa chọn).

## 21.8 Đánh giá và chẩn đoán căn chỉnh

### Thước đo định lượng

| Thước đo | Định nghĩa / cách tính | Dùng để |
|---|---|---|
| Focus rate | $\frac{1}{T}\sum_t \max_n \alpha_{t,n}$ | Độ sắc nét trung bình của attention |
| Entropy attention | $-\frac{1}{T}\sum_t \sum_n \alpha_{t,n} \log \alpha_{t,n}$ | Độ "nhòe"; tăng đột ngột là dấu hiệu lạc hướng |
| Vi phạm đơn điệu | Số $t$ có $\arg\max_n \alpha_{t+1,n} < \arg\max_n \alpha_{t,n}$ | Phát hiện lặp lại |
| Token bị bỏ | Số $n$ có $d_n = 0$ khi rút trường độ bằng argmax | Phát hiện bỏ từ |
| Trường độ bất thường | Điểm $z$ của $d_n$ so với thống kê của cùng âm vị | Phát hiện âm vị bị kéo dài/nuốt; lỗi dữ liệu |
| Sai số ranh giới | Trung bình $\lvert \hat{b}_n - b_n \rvert$ (ms) so với nhãn tay; tỉ lệ ranh giới lệch dưới 20 ms | Đánh giá aligner ngoài (MFA, CTC) |
| CER/WER qua ASR | Nhận dạng âm thanh tổng hợp rồi so với văn bản | Đo gián tiếp mọi lỗi bỏ/lặp/lảm nhảm ([Chương 38](../phan-9-thuc-tien/ch38-danh-gia-tts.md)) |

Thước đo cuối cùng là quan trọng nhất trong thực tế: mọi lỗi căn chỉnh nghiêm trọng đều làm tăng CER. Tuy nhiên ASR có thể "tự sửa" lỗi nhỏ, nên hãy kết hợp với nghe thử.

### Đọc hình ảnh ma trận căn chỉnh

- **Dải chéo sắc, liên tục, đơn điệu**: tốt. Độ dốc thay đổi theo tốc độ đọc từng đoạn là bình thường.
- **Dải nằm ngang dài** (nhiều khung trên cùng một token): khoảng ngừng (hợp lý nếu token là dấu câu) hoặc mô hình "kẹt" (nếu token là âm vị thường).
- **Dải bị gãy nhảy lên trên**: bỏ token. **Dải quay xuống rồi đi lại**: lặp.
- **Vùng mờ trải rộng**: mô hình chưa học được căn chỉnh hoặc gặp đầu vào lạ.
- **Dải không chạm góc trên phải**: dừng sớm. **Đoạn ngang dài ở token cuối**: không dừng.
- Với căn chỉnh cứng (MAS, MFA), hãy vẽ cả **trường độ theo âm vị** chồng lên mel-spectrogram: âm vị nguyên âm thường dài hơn phụ âm tắc; một phụ âm tắc kéo dài 40 khung gần như chắc chắn là lỗi.

!!! note "Góc tiếng Việt"
    - **Đơn vị căn chỉnh.** Có thể căn chỉnh theo âm tiết (đơn vị tự nhiên của tiếng Việt, mỗi âm tiết là một "chữ") hoặc theo âm vị. Căn chỉnh âm tiết ổn định hơn và ít phụ thuộc từ điển; căn chỉnh âm vị cho phép mô hình điều khiển trường độ tinh hơn (ví dụ kéo dài nguyên âm mà giữ nguyên phụ âm đầu). Một lựa chọn thực dụng là căn chỉnh âm vị nhưng kiểm tra ranh giới âm tiết.
    - **Thanh điệu và trường độ.** Thanh điệu ảnh hưởng trường độ: âm tiết thanh nặng thường ngắn và có thể kết thúc bằng hiện tượng thanh quản hóa; âm tiết tắc (kết thúc bằng -p, -t, -c, -ch) ngắn hơn âm tiết mở. Nếu thanh điệu là token riêng trong chuỗi đầu vào, hãy quyết định rõ token thanh được gán khung nào (ví dụ gắn vào nguyên âm, hoặc trường độ 0 nhưng đưa vào như đặc trưng), vì MAS mặc định đòi mọi token có ít nhất một khung.
    - **Từ điển và phương ngữ.** Với MFA, từ điển sinh bằng G2P ([Chương 18](../phan-4-frontend/ch18-g2p-bieu-dien-dau-vao.md)) phải khớp phương ngữ của người đọc: dùng từ điển giọng Bắc cho người đọc giọng Nam (ví dụ khác biệt ở phụ âm cuối -n/-ng, -t/-c) sẽ làm ranh giới lệch. Tính đến thời điểm biên soạn (2026), hãy kiểm tra danh sách mô hình và từ điển huấn luyện sẵn của MFA cho tiếng Việt; nếu không có cái phù hợp, MFA cho phép tự huấn luyện aligner trên chính dữ liệu của bạn.
    - **Trộn mã Việt–Anh.** Từ tiếng Anh trong câu tiếng Việt ("email", "Facebook") thường là nơi căn chỉnh hỏng nhất, vì G2P và aligner không biết cách phát âm thực tế của người đọc ([Chương 30](../phan-7-e2e-bieu-cam/ch30-da-ngon-ngu-code-switching.md)). Hãy thống kê riêng lỗi căn chỉnh quanh các từ này.

## Lỗi thường gặp và mẹo thực hành

- **Lệch đơn vị thời gian.** Trường độ từ MFA tính theo giây phải được đổi sang khung với *đúng* bước nhảy và tần số lấy mẫu của mel dùng để huấn luyện. Luôn `assert d.sum() == mel.shape[0]` cho từng câu; nếu lệch một vài khung ở cuối, sửa ở token cuối (thường là khoảng lặng) chứ đừng bỏ qua.
- **Làm tròn từng trường độ.** Làm tròn từng $d_n$ độc lập khiến tổng lệch nhiều khung với câu dài. Làm tròn ranh giới tích lũy (mục 21.5).
- **Token không có âm thanh.** Dấu câu, khoảng trắng, token thanh điệu riêng… có thể không ứng với khung nào. MAS và forward-sum giả định $d_n \ge 1$; nếu bộ ký hiệu của bạn có những token như vậy, hãy cho phép blank hoặc thiết kế lại chuỗi đầu vào (ví dụ chỉ giữ dấu câu ứng với khoảng ngừng thật).
- **Khoảng lặng đầu/cuối dài.** Chúng bị gán cho token đầu hoặc cuối, làm trường độ của token đó phình to và làm bộ dự đoán trường độ học sai. Cắt khoảng lặng nhất quán hoặc thêm token khoảng lặng ở hai đầu.
- **MAS trên số thực 16 bit.** Ma trận điểm log-likelihood có thể rất âm; tính MAS ở fp32 hoặc fp64 và luôn dưới `torch.no_grad()`.
- **Nhị phân hóa quá sớm.** Bật $\mathcal{L}_{\text{bin}}$ hoặc dùng căn chỉnh cứng cho decoder khi căn chỉnh mềm còn nhòe sẽ khóa mô hình vào lỗi. Theo dõi hình ảnh căn chỉnh mềm và chỉ bật khi nó đã có dạng chéo rõ.
- **Tin vào focus rate.** Focus rate cao không loại trừ lỗi bỏ token cục bộ (mục 21.4). Đếm trực tiếp số token có $d_n = 0$ và số vi phạm đơn điệu.
- **Đánh giá aligner ngoài mà không có nhãn tay.** Hãy gán nhãn tay ranh giới cho vài chục câu (có thể bằng Praat) — đây là khoản đầu tư nhỏ nhưng cho bạn con số sai số ranh giới thật, đặc biệt quan trọng cho tiếng Việt khi chưa có aligner chuẩn.

## Tóm tắt

- Căn chỉnh cứng đơn điệu, không nhảy cóc tương đương với véc-tơ trường độ $\mathbf{d}$ ($d_n \ge 1$, $\sum_n d_n = T$); có $\binom{T-1}{N-1}$ căn chỉnh như vậy.
- Các attention có ràng buộc (GMM, forward, stepwise monotonic, DCA) cài tính đơn điệu hoặc tính vị trí tương đối vào cấu trúc, giúp mô hình tự hồi quy bền hơn và tổng quát hóa sang câu dài.
- Guided attention loss phạt attention xa đường chéo, giúp căn chỉnh hình thành nhanh với chi phí gần như bằng không.
- FastSpeech rút trường độ từ attention của mô hình thầy (chọn đầu có focus rate cao nhất); cách này đơn giản nhưng có thể sinh $d_n = 0$ và vi phạm đơn điệu.
- Căn chỉnh cưỡng bức dùng Viterbi trên HMM (MFA) hoặc trên lưới CTC; cần cẩn thận khi chuyển ranh giới sang khung và khi xử lý blank của CTC.
- MAS là quy hoạch động $Q_{n,t} = S_{n,t} + \max(Q_{n,t-1}, Q_{n-1,t-1})$ với độ phức tạp $O(NT)$, tương đương Viterbi trên HMM trái-sang-phải; nó không khả vi và được dùng theo kiểu EM cứng.
- Forward-sum loss là phiên bản log-sum-exp của MAS, tính được bằng `F.ctc_loss` với mỗi token là một nhãn riêng; kết hợp tiên nghiệm beta–nhị thức và binarization loss, nó học căn chỉnh không giám sát mà không cần công cụ ngoài.
- Chẩn đoán căn chỉnh bằng cả thước đo (entropy, vi phạm đơn điệu, token bị bỏ, CER qua ASR) lẫn hình ảnh ma trận căn chỉnh.

## Bài tập

**Bài 1 [Cơ bản].** Với $N = 3$ và $T = 6$: (a) tính số căn chỉnh hợp lệ; (b) liệt kê tất cả dưới dạng véc-tơ trường độ; (c) có bao nhiêu căn chỉnh nếu cho phép $d_n \ge 0$ (token có thể bị bỏ)?

??? success "Lời giải"
    (a) $\binom{5}{2} = 10$. (b) $(1,1,4), (1,2,3), (1,3,2), (1,4,1), (2,1,3), (2,2,2), (2,3,1), (3,1,2), (3,2,1), (4,1,1)$. (c) Đặt $d'_n = d_n + 1 \ge 1$ với $\sum d'_n = T + N = 9$: số cách là $\binom{8}{2} = 28$ (bài toán "chia kẹo" có phần tử rỗng).

**Bài 2 [Cơ bản].** Với guided attention, $g = 0.2$: tính $W_{n,t}$ tại (a) $n/N = t/T$; (b) $\lvert n/N - t/T \rvert = 0.2$; (c) $\lvert n/N - t/T \rvert = 0.5$. Điều gì xảy ra với mất mát nếu ta chọn $g = 0.05$ hoặc $g = 1$?

??? success "Lời giải"
    (a) $W = 0$. (b) $W = 1 - e^{-0.5} \approx 0.39$. (c) $W = 1 - e^{-3.125} \approx 0.96$. Với $g = 0.05$, dải cho phép rất hẹp: mọi sai lệch nhỏ so với tốc độ đều bị phạt, ép mô hình đọc đều một cách giả tạo. Với $g = 1$, $W$ gần 0 ở khắp nơi (tối đa $1 - e^{-0.5}$ ở góc), mất mát gần như không còn tác dụng dẫn hướng.

**Bài 3 [Nâng cao].** Giả sử $S_{n,t} = \log \mathcal{N}(\mathbf{z}_t; \boldsymbol{\mu}_n, \mathbf{I})$. Chứng minh rằng MAS tương đương với bài toán **phân đoạn chuỗi** $\mathbf{z}_1, \ldots, \mathbf{z}_T$ thành $N$ đoạn liên tiếp sao cho tổng bình phương khoảng cách từ mỗi $\mathbf{z}_t$ tới "tâm" $\boldsymbol{\mu}_n$ của đoạn chứa nó là nhỏ nhất. So sánh với thuật toán k-means.

??? success "Lời giải"
    $\log \mathcal{N}(\mathbf{z}_t; \boldsymbol{\mu}_n, \mathbf{I}) = -\frac{F}{2}\log(2\pi) - \frac{1}{2}\lVert \mathbf{z}_t - \boldsymbol{\mu}_n \rVert^2$. Số hạng đầu là hằng số với mọi căn chỉnh (mỗi căn chỉnh có đúng $T$ số hạng), nên $\arg\max_a \sum_t S_{a(t),t} = \arg\min_a \sum_t \lVert \mathbf{z}_t - \boldsymbol{\mu}_{a(t)} \rVert^2$. Đây giống bước gán cụm của k-means, nhưng (i) các tâm $\boldsymbol{\mu}_n$ đến từ văn bản chứ không phải trung bình cụm và (ii) phép gán bị ràng buộc đơn điệu, liên tiếp — nên cần quy hoạch động thay vì gán độc lập từng điểm. Huấn luyện Glow-TTS xen kẽ MAS và cập nhật $\boldsymbol{\mu}$ chính là một dạng "k-means có ràng buộc thứ tự".

**Bài 4 [Nâng cao].** Với forward attention (không có tác tử chuyển tiếp) và $\boldsymbol{\alpha}_0 = (1, 0, \ldots, 0)$, chứng minh bằng quy nạp rằng $\alpha_{t,n} = 0$ với mọi $n > t + 1$. Giải thích vì sao tính chất này ngăn được lỗi bỏ từ nhưng *không* ngăn được lỗi "kẹt" (attention đứng yên quá lâu).

??? tip "Gợi ý"
    Bước quy nạp: $\hat{\alpha}_{t,n} = (\alpha_{t-1,n} + \alpha_{t-1,n-1})\tilde{\alpha}_{t,n}$; nếu $n > t + 1$ thì cả $n$ và $n - 1$ đều lớn hơn $(t - 1) + 1 = t$, nên theo giả thiết quy nạp cả hai số hạng bằng 0. Chuẩn hóa không làm số 0 thành khác 0. Về lỗi "kẹt": số hạng $\alpha_{t-1,n}\tilde{\alpha}_{t,n}$ cho phép ở lại vô hạn bước nếu $\tilde{\alpha}$ ưu tiên vị trí hiện tại.

**Bài 5 [Nâng cao].** Trong cài đặt forward-sum bằng CTC, blank có log-xác suất cố định $b$ (trước khi log-softmax). Phân tích điều gì xảy ra khi (a) $b \to -\infty$; (b) $b$ rất lớn; (c) $b = -1$ như bản cài đặt tham chiếu. Vì sao một chút "chỗ trống" cho blank có thể có ích với dữ liệu thật?

??? tip "Gợi ý"
    (a) Trở về đúng tổng trên các căn chỉnh đơn điệu (như phép thử trong bài). (b) Gần như mọi khung bị gán cho blank; mất mát có thể giảm mà không cần căn chỉnh tốt — mô hình "lười". (c) Thỏa hiệp: các khung không khớp token nào (khoảng lặng không có token tương ứng, tiếng thở, nhiễu) có thể được blank "hấp thụ" thay vì ép vào một token sai. Nghĩ tới dữ liệu tự nhiên có khoảng ngừng không được đánh dấu trong văn bản.

**Bài 6 [Lập trình].** Mở rộng `monotonic_alignment_search` thành Viterbi cho HMM trái-sang-phải với log-xác suất chuyển $\log a^{(n)}_{\text{stay}}$ và $\log a^{(n)}_{\text{move}}$ **phụ thuộc token** $n$. Viết hàm vét cạn tương ứng và kiểm tra hai hàm khớp nhau trên 200 ví dụ ngẫu nhiên. Sau đó cho xác suất chuyển giống nhau cho mọi token và quan sát: kết quả có khác MAS không?

??? tip "Gợi ý"
    Đệ quy: $Q_{n,t} = S_{n,t} + \max(Q_{n,t-1} + \log a^{(n)}_{\text{stay}}, \; Q_{n-1,t-1} + \log a^{(n-1)}_{\text{move}})$. Điểm của một căn chỉnh trong hàm vét cạn: $\sum_t S_{a(t),t} + \sum_n (d_n - 1)\log a^{(n)}_{\text{stay}} + \sum_{n<N} \log a^{(n)}_{\text{move}}$. Khi xác suất chuyển giống nhau cho mọi token, mọi căn chỉnh hợp lệ có đúng $N - 1$ lần "tiến" và $T - N$ lần "ở lại", nên số hạng chuyển là hằng số và **không làm thay đổi** argmax — kết quả trùng với MAS. Xác suất chuyển chỉ có tác dụng khi chúng khác nhau giữa các token (ví dụ nguyên âm có xác suất ở lại cao hơn phụ âm tắc), đóng vai trò một tiên nghiệm trường độ theo loại âm vị.

**Bài 7 [Lập trình].** Viết hàm `alignment_report(A)` nhận ma trận attention mềm $(N, T)$ và trả về focus rate, entropy trung bình, số vi phạm đơn điệu của argmax, số token có $d_n = 0$. Tạo ba ma trận giả lập: căn chỉnh tốt, căn chỉnh có lặp (dải quay lại) và căn chỉnh bỏ token, rồi kiểm tra báo cáo phân biệt được chúng.

??? tip "Gợi ý"
    Dùng `soft_alignment_from_durations` để tạo căn chỉnh tốt. Để giả lập lặp, ghép thêm một đoạn cột của token $3$–$4$ sau khi đã tới token $5$; để giả lập bỏ token, đặt trường độ của một token thành 0 khi tạo các tâm Gauss. Entropy: `-(A * A.clamp_min(1e-9).log()).sum(0).mean()`.

## Tài liệu tham khảo

1. Graves, A. (2013). *Generating Sequences With Recurrent Neural Networks*. arXiv:1308.0850.
2. Zhang, J.-X., Ling, Z.-H., Dai, L.-R. (2018). *Forward Attention in Sequence-to-Sequence Acoustic Modeling for Speech Synthesis*. ICASSP 2018.
3. Raffel, C. và cộng sự (2017). *Online and Linear-Time Attention by Enforcing Monotonic Alignments*. ICML 2017.
4. He, M., Deng, Y., He, L. (2019). *Robust Sequence-to-Sequence Acoustic Modeling with Stepwise Monotonic Attention for Neural TTS*. Interspeech 2019.
5. Battenberg, E. và cộng sự (2020). *Location-Relative Attention Mechanisms for Robust Long-Form Speech Synthesis*. ICASSP 2020.
6. Tachibana, H., Uenoyama, K., Aihara, S. (2018). *Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention*. ICASSP 2018.
7. Ren, Y. và cộng sự (2019). *FastSpeech: Fast, Robust and Controllable Text to Speech*. NeurIPS 2019.
8. Rabiner, L. R. (1989). *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition*. Proceedings of the IEEE.
9. McAuliffe, M. và cộng sự (2017). *Montreal Forced Aligner: Trainable Text-Speech Alignment Using Kaldi*. Interspeech 2017.
10. Graves, A. và cộng sự (2006). *Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks*. ICML 2006.
11. Pratap, V. và cộng sự (2023). *Scaling Speech Technology to 1,000+ Languages*. arXiv preprint.
12. Sakoe, H., Chiba, S. (1978). *Dynamic Programming Algorithm Optimization for Spoken Word Recognition*. IEEE Transactions on Acoustics, Speech, and Signal Processing.
13. Kim, J. và cộng sự (2020). *Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search*. NeurIPS 2020.
14. Badlani, R. và cộng sự (2022). *One TTS Alignment To Rule Them All*. ICASSP 2022.
15. Shih, K. J. và cộng sự (2021). *RAD-TTS: Parallel Flow-Based TTS with Robust Alignment Learning and Diverse Synthesis*. ICML 2021 Workshop on Invertible Neural Networks, Normalizing Flows, and Explicit Likelihood Models.
16. Łańcucki, A. (2021). *FastPitch: Parallel Text-to-Speech with Pitch Prediction*. ICASSP 2021.
17. Kim, J., Kong, J., Son, J. (2021). *Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech*. ICML 2021.
