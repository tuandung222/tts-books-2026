# Chương 5. Attention và Transformer

!!! abstract "Mục tiêu học tập"
    Sau chương này, bạn có thể:

    - **Suy ra** vì sao scaled dot-product attention chia điểm cho $\sqrt{d_k}$, và **cài đặt** multi-head attention từ đầu, đối chiếu chính xác với `torch.nn.functional.scaled_dot_product_attention`.
    - **Xây dựng** đúng các mặt nạ padding và nhân quả, phân biệt self-attention và cross-attention, tránh các bẫy về quy ước mặt nạ.
    - **So sánh** các mã hóa vị trí (sinusoidal, học được, tương đối, ALiBi, RoPE) và **chứng minh** tính chất tương đối của RoPE.
    - **Giải thích** các lựa chọn thiết kế của khối Transformer hiện đại: Pre-LN và Post-LN, RMSNorm, SwiGLU.
    - **Cài đặt** giải mã tự hồi quy với KV cache và các chiến lược lấy mẫu (nhiệt độ, top-k, top-p, phạt lặp) dùng trong codec language model.
    - **Phân tích** độ phức tạp $O(N^2)$ của attention và ý tưởng của FlashAttention (softmax trực tuyến).
    - **Mô tả** khối Conformer và các cơ chế điều kiện hóa FiLM, AdaLN, adaLN-Zero dùng trong TTS dựa trên DiT.

!!! info "Kiến thức cần có"
    - [Chương 1](ch01-toan-nen-tang.md): ma trận, ma trận quay, số phức và công thức Euler.
    - [Chương 3](ch03-toi-uu-va-ky-thuat-huan-luyen.md): LayerNorm, RMSNorm, warmup tốc độ học.
    - [Chương 4](ch04-cnn-rnn-mo-hinh-chuoi.md): seq2seq, attention cộng và nhân, teacher forcing, các chiến lược giải mã.
    - Các khối mã trong chương **chạy tuần tự như một notebook**; khối sau có thể dùng lại hàm của khối trước.

Ở [Chương 4](ch04-cnn-rnn-mo-hinh-chuoi.md), attention xuất hiện như một "phụ kiện" giúp bộ giải mã RNN nhìn lại bộ mã hóa. Năm 2017, Vaswani và cộng sự đặt câu hỏi ngược lại: nếu attention đã làm được việc nặng nhất, liệu có cần RNN nữa không? Câu trả lời — Transformer — đã trở thành kiến trúc chủ đạo của gần như toàn bộ học sâu, và TTS không ngoại lệ: bộ mã hóa văn bản của FastSpeech và VITS, bộ giải mã của Transformer-TTS, các codec language model như VALL-E, CosyVoice, và các mô hình flow matching dựa trên DiT như F5-TTS đều là các biến thể của Transformer.

Chương này đi từ phép toán cốt lõi (scaled dot-product attention) đến những chi tiết kỹ thuật quyết định một mô hình TTS hiện đại có chạy được hay không: mã hóa vị trí cho chuỗi rất dài, KV cache cho sinh tự hồi quy, chiến lược lấy mẫu cho token âm thanh, và cơ chế điều kiện hóa cho mô hình khuếch tán/flow.

## 5.1 Từ RNN có attention đến Transformer

Một RNN có hai hạn chế cấu trúc:

1. **Tuần tự.** Muốn tính $\mathbf{h}_t$ phải có $\mathbf{h}_{t-1}$. Với chuỗi 1 000 khung mel, ta có 1 000 bước nối tiếp nhau — không tận dụng được hàng nghìn lõi của GPU.
2. **Đường đi dài.** Thông tin từ vị trí $i$ tới vị trí $j$ phải đi qua $|i - j|$ bước cập nhật, mỗi bước là một cơ hội để bị "quên" hay để gradient tiêu biến.

Attention giải quyết cả hai. Mọi cặp vị trí được nối **trực tiếp** (đường đi độ dài 1), và mọi vị trí được tính **song song** bằng phép nhân ma trận. Cái giá phải trả là chi phí tăng theo **bình phương** độ dài chuỗi — chủ đề của mục 5.10.

Trực giác hữu ích nhất cho attention là một **từ điển mềm**. Một từ điển thông thường nhận một khóa và trả về đúng một giá trị. Attention nhận một **truy vấn** (*query*) $\mathbf{q}$, so sánh nó với mọi **khóa** (*key*) $\mathbf{k}_j$, rồi trả về **trung bình có trọng số** của các **giá trị** (*value*) $\mathbf{v}_j$ — khóa nào càng giống truy vấn, giá trị tương ứng càng được coi trọng.

## 5.2 Scaled dot-product attention

### 5.2.1 Định nghĩa

Xếp $N_q$ truy vấn, $N_k$ khóa và $N_k$ giá trị thành các ma trận $\mathbf{Q} \in \mathbb{R}^{N_q \times d_k}$, $\mathbf{K} \in \mathbb{R}^{N_k \times d_k}$, $\mathbf{V} \in \mathbb{R}^{N_k \times d_v}$ (mỗi hàng là một véc-tơ). **Scaled dot-product attention** là

$$
\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V},
$$

trong đó softmax lấy theo từng hàng, và $\mathbf{M} \in \{0, -\infty\}^{N_q \times N_k}$ (hoặc một ma trận thực bất kỳ) là **mặt nạ cộng** dùng để cấm hoặc ưu tiên một số cặp (mục 5.4). Đọc từ trong ra ngoài:

- $\mathbf{S} = \mathbf{Q}\mathbf{K}^\top / \sqrt{d_k} \in \mathbb{R}^{N_q \times N_k}$: **ma trận điểm**, phần tử $(i, j)$ đo độ giống giữa truy vấn $i$ và khóa $j$.
- $\mathbf{P} = \operatorname{softmax}(\mathbf{S} + \mathbf{M})$: **ma trận trọng số attention**, mỗi hàng là một phân phối xác suất trên các khóa.
- $\mathbf{P}\mathbf{V}$: mỗi hàng đầu ra là trung bình các giá trị theo phân phối đó.

So với attention cộng của Bahdanau ([Chương 4](ch04-cnn-rnn-mo-hinh-chuoi.md)), hàm điểm ở đây chỉ là tích vô hướng — toàn bộ $N_q \times N_k$ điểm được tính bằng **một phép nhân ma trận**, phép toán mà GPU làm nhanh nhất.

### 5.2.2 Vì sao chia cho $\sqrt{d_k}$?

Giả sử (như lúc khởi tạo, khi đầu vào đã được chuẩn hóa) các thành phần $q_i, k_i$ là các biến ngẫu nhiên độc lập, kỳ vọng 0, phương sai 1. Khi đó:

$$
\mathbb{E}[\mathbf{q}^\top \mathbf{k}] = \sum_{i=1}^{d_k} \mathbb{E}[q_i]\,\mathbb{E}[k_i] = 0,
\qquad
\operatorname{Var}(\mathbf{q}^\top \mathbf{k}) = \sum_{i=1}^{d_k} \operatorname{Var}(q_i k_i) = \sum_{i=1}^{d_k} \mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] = d_k .
$$

(Bước thứ hai dùng tính độc lập giữa các số hạng; bước thứ ba dùng $\operatorname{Var}(q_i k_i) = \mathbb{E}[q_i^2 k_i^2] - (\mathbb{E}[q_i k_i])^2 = 1 \cdot 1 - 0$.) Vậy độ lệch chuẩn của điểm tăng như $\sqrt{d_k}$: với $d_k = 64$, các điểm có độ lớn cỡ 8; với $d_k = 512$, cỡ 23.

Điểm lớn thì có hại vì softmax **bão hòa**. Jacobian của $\mathbf{p} = \operatorname{softmax}(\mathbf{s})$ là

$$
\frac{\partial p_i}{\partial s_j} = p_i(\delta_{ij} - p_j), \qquad \mathbf{J} = \operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top .
$$

Khi các điểm cách nhau xa, $\mathbf{p}$ tiến về một véc-tơ one-hot $\mathbf{e}_j$, và $\mathbf{J} \to \operatorname{diag}(\mathbf{e}_j) - \mathbf{e}_j\mathbf{e}_j^\top = \mathbf{0}$: gradient qua softmax **tiêu biến**, attention "đông cứng" vào một lựa chọn ngẫu nhiên ngay từ đầu huấn luyện. Chia cho $\sqrt{d_k}$ đưa phương sai của điểm về 1 **bất kể** $d_k$, giữ softmax ở vùng có gradient tốt.

**Mã minh họa.** Đo entropy của phân phối attention và chuẩn Frobenius của Jacobian softmax, có và không có hệ số tỷ lệ.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def softmax_stats(scores):
    """Entropy trung bình và chuẩn Frobenius trung bình của Jacobian softmax J = diag(p) - p p^T."""
    p = scores.softmax(-1)                                         # (M, n)
    ent = -(p * p.clamp_min(1e-12).log()).sum(-1).mean().item()
    jac = torch.diag_embed(p) - p[:, :, None] * p[:, None, :]       # (M, n, n)
    return ent, jac.flatten(1).norm(dim=-1).mean().item()

torch.manual_seed(0)
n_keys = 32
print(f"Entropy cực đại (đều trên {n_keys} khóa) = {math.log(n_keys):.2f}")
for d_k in [4, 64, 512]:
    q = torch.randn(2000, d_k)                    # mỗi thành phần ~ N(0, 1)
    k = torch.randn(2000, n_keys, d_k)
    scores = torch.einsum("md,mnd->mn", q, k)     # (2000, 32)
    e_raw, j_raw = softmax_stats(scores)
    e_sc, j_sc = softmax_stats(scores / math.sqrt(d_k))
    print(f"d_k={d_k:3d} | std(q·k)={scores.std().item():5.1f} (√d_k={math.sqrt(d_k):4.1f}) | "
          f"không chia: H={e_raw:.2f}, ‖J‖={j_raw:.3f} | chia √d_k: H={e_sc:.2f}, ‖J‖={j_sc:.3f}")
```

Kết quả: độ lệch chuẩn của điểm khớp đúng $\sqrt{d_k}$; khi không chia, entropy giảm từ khoảng 2,3 xuống 0,15 và chuẩn Jacobian giảm gần 4 lần khi $d_k$ tăng từ 4 lên 512; khi có chia, cả hai gần như không đổi theo $d_k$.

### 5.2.3 Attention là một căn chỉnh mềm

Trong cross-attention giữa bộ giải mã mel và bộ mã hóa văn bản, hàng $t$ của $\mathbf{P}$ cho biết khung $t$ "đang đọc" token nào. Chuyển vị lại, ta có đúng ma trận căn chỉnh mềm $\mathbf{A} \in [0, 1]^{N \times T}$ của [bảng ký hiệu](../gioi-thieu/ky-hieu.md). Vẽ ma trận này là công cụ chẩn đoán số một của các mô hình TTS dựa trên attention: một đường chéo sắc nét nghĩa là mô hình đã học được căn chỉnh; một vệt mờ hoặc đứt đoạn báo trước các lỗi lặp từ, bỏ từ ([Chương 20](../phan-5-mo-hinh-am-hoc/ch20-tacotron.md), [Chương 21](../phan-5-mo-hinh-am-hoc/ch21-alignment.md)).

## 5.3 Multi-head attention

Một phép attention cho mỗi truy vấn **một** phân phối trên các khóa, và đầu ra là **một** trung bình. Nếu truy vấn cần đồng thời hai loại thông tin — ví dụ một khung mel cần biết cả âm vị đang được đọc *và* dấu câu ở cuối cụm từ để định ngữ điệu — một phân phối duy nhất buộc phải "chia đôi" sự chú ý và trộn lẫn hai thông tin.

**Multi-head attention** chạy $h$ phép attention song song trên các phép chiếu khác nhau, rồi ghép kết quả:

$$
\operatorname{head}_i = \operatorname{Attention}(\mathbf{X}_q \mathbf{W}_i^{Q},\ \mathbf{X}_{kv} \mathbf{W}_i^{K},\ \mathbf{X}_{kv} \mathbf{W}_i^{V}),
\qquad
\operatorname{MHA}(\mathbf{X}_q, \mathbf{X}_{kv}) = [\operatorname{head}_1; \ldots; \operatorname{head}_h]\, \mathbf{W}^{O},
$$

với $\mathbf{W}_i^{Q}, \mathbf{W}_i^{K}, \mathbf{W}_i^{V} \in \mathbb{R}^{d \times d/h}$ và $\mathbf{W}^{O} \in \mathbb{R}^{d \times d}$ ($d$ là chiều mô hình, ở đây ghép theo chiều đặc trưng). Mỗi head làm việc trong không gian con $d/h$ chiều, nên tổng chi phí xấp xỉ một head đầy đủ $d$ chiều; số tham số là $4d^2$ (cộng $4d$ độ lệch). Trong cài đặt, ta chiếu một lần ra $d$ chiều rồi **tách** thành $h$ head bằng `view` + `transpose`, thay vì giữ $h$ ma trận riêng.

Trong các mô hình TTS dựa trên attention, người ta thường quan sát thấy chỉ một số head của cross-attention có dạng đường chéo rõ (làm nhiệm vụ căn chỉnh), các head khác phân tán hơn. Điều này có hệ quả thực hành: khi dùng attention để trích trường độ hay để áp đặt ràng buộc đường chéo, ta thường chọn hoặc ràng buộc **một vài head** cụ thể thay vì tất cả.

## 5.4 Mặt nạ; self-attention và cross-attention

### 5.4.1 Self-attention và cross-attention

- **Self-attention**: $\mathbf{X}_q = \mathbf{X}_{kv} = \mathbf{X}$ — mỗi vị trí của một chuỗi nhìn các vị trí khác của **chính chuỗi đó**. Dùng trong bộ mã hóa văn bản (mỗi âm vị nhìn cả câu) và trong bộ giải mã (mỗi khung nhìn các khung khác).
- **Cross-attention**: truy vấn từ chuỗi này, khóa/giá trị từ chuỗi khác — ví dụ truy vấn từ khung mel, khóa/giá trị từ đầu ra bộ mã hóa văn bản. Đây là cơ chế căn chỉnh của Transformer-TTS (Li và cộng sự, 2019).

Không phải mọi mô hình TTS hiện đại đều cần cross-attention. Các codec language model kiểu decoder-only (mục 5.7) đặt văn bản và token âm thanh vào **cùng một chuỗi** và chỉ dùng self-attention; E2 TTS và F5-TTS đệm chuỗi ký tự cho bằng độ dài mel rồi ghép theo kênh ([Chương 33](../phan-8-mo-hinh-lon/ch33-nar-quy-mo-lon.md)). Căn chỉnh khi đó được học **ngầm** bên trong self-attention.

### 5.4.2 Mặt nạ padding và mặt nạ nhân quả

**Mặt nạ padding.** Trong một batch, các câu ngắn được đệm. Khóa ở vị trí đệm phải bị cấm: $M_{ij} = -\infty$ nếu khóa $j$ là đệm. Truy vấn ở vị trí đệm thì vẫn tính ra *một* giá trị (vô nghĩa), nên phải loại các vị trí đó khỏi hàm mất mát và khỏi mọi phép gộp (*pooling*) phía sau.

**Mặt nạ nhân quả.** Trong bộ giải mã tự hồi quy, vị trí $i$ chỉ được nhìn các vị trí $j \le i$: $M_{ij} = -\infty$ nếu $j > i$ — ma trận tam giác dưới. Nhờ mặt nạ này, ta huấn luyện **mọi** bước của một chuỗi song song trong một lần chạy, mà vẫn bảo đảm tính tự hồi quy — cùng tinh thần với tích chập nhân quả ở [Chương 4](ch04-cnn-rnn-mo-hinh-chuoi.md). Hai mặt nạ được kết hợp bằng phép "và" (với mặt nạ bool) hoặc phép cộng (với mặt nạ số thực).

!!! warning "Bẫy: quy ước mặt nạ bool ngược nhau giữa các API"
    - Trong `F.scaled_dot_product_attention`, mặt nạ bool `True` nghĩa là **được chú ý**; mặt nạ số thực được **cộng** vào điểm.
    - Trong `nn.MultiheadAttention`, `key_padding_mask` và `attn_mask` dạng bool thì `True` nghĩa là **bị cấm**.

    Truyền nhầm quy ước không gây ra lỗi nào; mô hình chỉ âm thầm chú ý vào phần đệm và bỏ qua nội dung thật. Hãy luôn viết một kiểm thử đối chiếu như khối mã dưới đây.

**Mã minh họa.** Multi-head attention viết từ đầu; đối chiếu với `F.scaled_dot_product_attention` (SDPA) và với `nn.MultiheadAttention` (sau khi chép trọng số). Khối mã cũng kiểm chứng tính đồng biến hoán vị sẽ bàn ở mục 5.5.

```python
def attention(q, k, v, mask=None):
    """Scaled dot-product attention viết tay.
    q: (B, h, N_q, d), k, v: (B, h, N_k, d); mask: bool, True = ĐƯỢC chú ý."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])   # (B, h, N_q, N_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return scores.softmax(dim=-1) @ v                             # (B, h, N_q, d)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d = n_heads, d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def split(self, x):                           # (B, N, d_model) -> (B, h, N, d)
        return x.view(x.shape[0], x.shape[1], self.h, self.d).transpose(1, 2)

    def forward(self, x_q, x_kv, mask=None, use_sdpa=False):
        q, k, v = self.split(self.w_q(x_q)), self.split(self.w_k(x_kv)), self.split(self.w_v(x_kv))
        if use_sdpa:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            o = attention(q, k, v, mask)          # (B, h, N_q, d)
        o = o.transpose(1, 2).reshape(x_q.shape[0], x_q.shape[1], -1)   # (B, N_q, d_model)
        return self.w_o(o)

def make_masks(lengths, n_max, causal=False):
    """Mặt nạ bool (B, 1, N, N): True = được chú ý. Kết hợp padding và (tùy chọn) nhân quả."""
    valid = torch.arange(n_max)[None] < lengths[:, None]           # (B, N)
    mask = valid[:, None, None, :]                                 # che KHÓA là đệm
    if causal:
        mask = mask & torch.ones(n_max, n_max, dtype=torch.bool).tril()
    return mask

torch.manual_seed(0)
B, N, D, H = 2, 6, 32, 4
x = torch.randn(B, N, D)                          # (B, N, d_model)
lengths = torch.tensor([6, 4])
mha = MultiHeadAttention(D, H)
for causal in [False, True]:
    m = make_masks(lengths, N, causal)
    assert torch.allclose(mha(x, x, m), mha(x, x, m, use_sdpa=True), atol=1e-5)

# Đối chiếu với nn.MultiheadAttention: chép trọng số; CHÚ Ý quy ước mặt nạ ngược lại
torch_mha = nn.MultiheadAttention(D, H, batch_first=True)
with torch.no_grad():
    torch_mha.in_proj_weight.copy_(torch.cat([mha.w_q.weight, mha.w_k.weight, mha.w_v.weight]))
    torch_mha.in_proj_bias.copy_(torch.cat([mha.w_q.bias, mha.w_k.bias, mha.w_v.bias]))
    torch_mha.out_proj.weight.copy_(mha.w_o.weight)
    torch_mha.out_proj.bias.copy_(mha.w_o.bias)
pad = torch.arange(N)[None] >= lengths[:, None]   # True = BỎ QUA (ngược với SDPA!)
ref, _ = torch_mha(x, x, x, key_padding_mask=pad)
assert torch.allclose(mha(x, x, make_masks(lengths, N)), ref, atol=1e-5)

# Tự chú ý không có mã hóa vị trí là "đồng biến hoán vị" (mục 5.5.1)
perm = torch.randperm(N)
assert torch.allclose(mha(x[:, perm], x[:, perm]), mha(x, x)[:, perm], atol=1e-5)
print("MHA tự cài đặt khớp SDPA và nn.MultiheadAttention")
```

## 5.5 Mã hóa vị trí

### 5.5.1 Vì sao cần mã hóa vị trí?

Gọi $\mathbf{\Pi}$ là một ma trận hoán vị (đổi chỗ các hàng). Với self-attention không có thông tin vị trí, $\mathbf{Q} = \mathbf{X}\mathbf{W}^Q$, ..., ta có

$$
\operatorname{softmax}\!\left(\frac{\mathbf{\Pi}\mathbf{X}\mathbf{W}^Q (\mathbf{\Pi}\mathbf{X}\mathbf{W}^K)^\top}{\sqrt{d_k}}\right)\mathbf{\Pi}\mathbf{X}\mathbf{W}^V
= \mathbf{\Pi}\,\operatorname{softmax}\!\left(\frac{\mathbf{X}\mathbf{W}^Q (\mathbf{X}\mathbf{W}^K)^\top}{\sqrt{d_k}}\right)\mathbf{\Pi}^\top \mathbf{\Pi}\mathbf{X}\mathbf{W}^V
= \mathbf{\Pi}\,\operatorname{Attention}(\mathbf{X}),
$$

vì softmax theo hàng giao hoán với việc hoán vị đồng thời hàng và cột ($\operatorname{softmax}(\mathbf{\Pi}\mathbf{S}\mathbf{\Pi}^\top) = \mathbf{\Pi}\operatorname{softmax}(\mathbf{S})\mathbf{\Pi}^\top$) và $\mathbf{\Pi}^\top\mathbf{\Pi} = \mathbf{I}$. Nói cách khác, self-attention là **đồng biến hoán vị** (*permutation equivariant*): xáo trộn đầu vào thì đầu ra bị xáo trộn y hệt, ngoài ra không có gì thay đổi. Với nó, một câu chỉ là một "túi token". Khối mã ở mục trước đã kiểm chứng điều này bằng số.

!!! note "Góc tiếng Việt"
    Tiếng Việt là ngôn ngữ **đơn lập**: từ không biến hình, quan hệ ngữ pháp được thể hiện chủ yếu bằng **trật tự từ** và hư từ. "Con mèo cắn con chó" và "con chó cắn con mèo" có cùng một túi âm tiết nhưng nghĩa ngược nhau — và với TTS, cả cách nhấn, cách ngắt cũng khác. Một bộ mã hóa không có thông tin vị trí sẽ không phân biệt được hai câu này. Ở mức âm thanh, vấn đề còn hiển nhiên hơn: hoán vị các khung mel là phá hủy tiếng nói.

Có hai cách đưa vị trí vào: **tuyệt đối** (cộng một véc-tơ phụ thuộc vị trí vào embedding) và **tương đối** (sửa điểm attention để phụ thuộc vào khoảng cách $i - j$).

### 5.5.2 Mã hóa sinusoidal

Transformer gốc cộng vào embedding ở vị trí $\text{pos}$ véc-tơ

$$
\mathrm{PE}(\text{pos}, 2i) = \sin(\text{pos}\cdot\omega_i), \qquad \mathrm{PE}(\text{pos}, 2i+1) = \cos(\text{pos}\cdot\omega_i), \qquad \omega_i = 10000^{-2i/d}.
$$

Mỗi cặp chiều là một "kim đồng hồ" quay với tần số $\omega_i$; các tần số trải theo cấp số nhân, từ chu kỳ $2\pi$ đến $2\pi \cdot 10000$ vị trí — giống cách các chữ số nhị phân của một bộ đếm lật với tần số khác nhau.

**Tính chất quan trọng:** dịch vị trí là một **phép biến đổi tuyến tính không phụ thuộc vị trí**. Từ công thức cộng góc $\sin(a + b) = \sin a \cos b + \cos a \sin b$ và $\cos(a + b) = \cos a \cos b - \sin a \sin b$, với $a = \text{pos}\,\omega_i$, $b = k\,\omega_i$:

$$
\begin{bmatrix} \mathrm{PE}(\text{pos} + k, 2i) \\ \mathrm{PE}(\text{pos} + k, 2i+1) \end{bmatrix}
=
\begin{bmatrix} \cos k\omega_i & \sin k\omega_i \\ -\sin k\omega_i & \cos k\omega_i \end{bmatrix}
\begin{bmatrix} \mathrm{PE}(\text{pos}, 2i) \\ \mathrm{PE}(\text{pos}, 2i+1) \end{bmatrix}.
$$

Vậy $\mathrm{PE}(\text{pos} + k) = \mathbf{M}_k\, \mathrm{PE}(\text{pos})$ với $\mathbf{M}_k$ là ma trận quay khối chỉ phụ thuộc $k$: về nguyên tắc, mô hình có thể học cách "nhìn sang trái $k$ vị trí" bằng một phép tuyến tính dùng được ở mọi vị trí. Đây chính là mầm mống của RoPE (mục 5.5.6).

Transformer-TTS (Li và cộng sự, 2019) dùng mã hóa sinusoidal nhưng nhân thêm một **hệ số tỷ lệ học được** trước khi cộng vào embedding, vì embedding âm vị và embedding khung mel có thang đo khác nhau — một chi tiết nhỏ cho thấy việc "cộng vị trí vào nội dung" nhạy cảm với tỷ lệ như thế nào.

### 5.5.3 Mã hóa vị trí học được

Thay công thức bằng một bảng embedding `nn.Embedding(max_len, d)` học từ dữ liệu (như BERT, GPT-2). Đơn giản và linh hoạt, nhưng **không ngoại suy** được: vị trí lớn hơn `max_len` không có embedding, và các vị trí hiếm gặp khi huấn luyện (cuối các câu dài) được học kém. Với TTS, nơi một câu dài có thể vượt nghìn khung mel, đây là một nhược điểm thật sự.

### 5.5.4 Biểu diễn vị trí tương đối (Shaw và cộng sự)

Shaw, Uszkoreit và Vaswani (2018) đưa khoảng cách trực tiếp vào attention. Với mỗi khoảng cách $j - i$ (bị cắt trong $[-k, k]$), học hai véc-tơ $\mathbf{a}^K_{j-i}, \mathbf{a}^V_{j-i}$:

$$
e_{ij} = \frac{\mathbf{q}_i^\top (\mathbf{k}_j + \mathbf{a}^K_{\operatorname{clip}(j-i, k)})}{\sqrt{d_k}}, \qquad
\mathbf{z}_i = \sum_j \alpha_{ij}\, (\mathbf{v}_j + \mathbf{a}^V_{\operatorname{clip}(j-i, k)}).
$$

Mô hình chỉ cần học $2k + 1$ véc-tơ; điểm phụ thuộc vào khoảng cách chứ không vào vị trí tuyệt đối, nên **bất biến tịnh tiến** và dùng được cho chuỗi dài hơn lúc huấn luyện. Với văn bản đầu vào TTS, thông tin quan trọng nhất thường là "âm vị bên cạnh là gì", nên một cửa sổ nhỏ là đủ. Bộ mã hóa văn bản của Glow-TTS và VITS dùng biến thể của cách tiếp cận này (theo mã nguồn chính thức, với cửa sổ khoảng cách nhỏ, cỡ 4 vị trí mỗi phía). Các biến thể khác: T5 (Raffel và cộng sự, 2020) chỉ học một **độ lệch vô hướng** cho mỗi nhóm khoảng cách và mỗi head; Transformer-XL (Dai và cộng sự, 2019) dùng mã hóa tương đối dạng sinusoidal — biến thể được Conformer sử dụng (mục 5.11).

### 5.5.5 ALiBi

ALiBi (*Attention with Linear Biases*; Press, Smith và Lewis, 2022) đơn giản đến bất ngờ: không cộng gì vào embedding, chỉ trừ vào điểm một lượng tỷ lệ với khoảng cách:

$$
e_{ij}^{(h)} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_k}} - m_h\,(i - j), \qquad j \le i,
$$

với độ dốc $m_h$ khác nhau cho từng head, lấy theo cấp số nhân; với 8 head là $\tfrac{1}{2}, \tfrac{1}{4}, \ldots, \tfrac{1}{256}$ (tổng quát $m_h = 2^{-8h/H}$ khi số head $H$ là lũy thừa của 2). Head có độ dốc lớn chỉ nhìn gần, head có độ dốc nhỏ nhìn xa. Press và cộng sự cho thấy mô hình ngôn ngữ huấn luyện với ALiBi ngoại suy tốt sang chuỗi dài hơn lúc huấn luyện. Trong TTS, ALiBi ít phổ biến hơn RoPE, nhưng ý tưởng "thiên vị theo khoảng cách" rất gần với các kỹ thuật thiên vị đường chéo dùng để ổn định căn chỉnh.

**Mã minh họa.** Mã hóa sinusoidal (kiểm chứng tính chất $\mathrm{PE}(\text{pos} + k) = \mathbf{M}_k \mathrm{PE}(\text{pos})$) và độ lệch ALiBi dùng như mặt nạ số thực của SDPA.

```python
def sinusoidal_pe(n, d, base=10000.0):
    """PE[pos, 2i] = sin(pos * w_i), PE[pos, 2i+1] = cos(pos * w_i), w_i = base^(-2i/d)."""
    w = base ** (-torch.arange(0, d, 2).float() / d)              # (d/2,)
    ang = torch.arange(n).float()[:, None] * w[None]               # (n, d/2)
    pe = torch.zeros(n, d)
    pe[:, 0::2], pe[:, 1::2] = ang.sin(), ang.cos()
    return pe                                                      # (n, d)

# Tính chất: PE[pos + k] = M_k @ PE[pos], M_k là ma trận quay khối KHÔNG phụ thuộc pos
d, k = 16, 7
pe = sinusoidal_pe(64, d)
w = 10000.0 ** (-torch.arange(0, d, 2).float() / d)
M = torch.zeros(d, d)
for i, wi in enumerate(w):
    c, s = math.cos(k * wi), math.sin(k * wi)
    M[2 * i:2 * i + 2, 2 * i:2 * i + 2] = torch.tensor([[c, s], [-s, c]])
for pos in [0, 10, 40]:
    assert torch.allclose(M @ pe[pos], pe[pos + k], atol=1e-5)

def alibi_bias(n, n_heads):
    """ALiBi (bản nhân quả): bias[h, i, j] = -m_h * (i - j) với j <= i, -inf với j > i."""
    slopes = 2.0 ** (-8.0 * torch.arange(1, n_heads + 1) / n_heads)     # (h,)
    rel = (torch.arange(n)[None, :] - torch.arange(n)[:, None]).float()  # rel[i, j] = j - i
    bias = slopes[:, None, None] * rel.clamp(max=0)                     # (h, n, n)
    causal = torch.ones(n, n, dtype=torch.bool).tril()
    return bias.masked_fill(~causal, float("-inf"))

print("Độ dốc ALiBi với 8 head:", (2.0 ** (-8.0 * torch.arange(1, 9) / 8)).tolist())
q = k_ = v = torch.randn(1, 8, 5, 16)            # (B, h, N, d)
out = F.scaled_dot_product_attention(q, k_, v, attn_mask=alibi_bias(5, 8))  # mặt nạ thực: CỘNG
manual = ((q @ k_.transpose(-2, -1)) / 4.0 + alibi_bias(5, 8)).softmax(-1) @ v
assert torch.allclose(out, manual, atol=1e-5)
print("Sinusoidal và ALiBi: OK")
```

### 5.5.6 RoPE: mã hóa vị trí bằng phép quay

RoPE (*Rotary Position Embedding*; Su và cộng sự, 2024) là cách mã hóa vị trí phổ biến nhất trong các mô hình ngôn ngữ lớn hiện nay (LLaMA, Qwen…), và do đó trong các TTS dùng LLM làm xương sống; nhiều mô hình TTS dạng DiT như F5-TTS cũng dùng nó (theo mã nguồn công bố). Ý tưởng: thay vì **cộng** vị trí vào nội dung, ta **quay** truy vấn và khóa một góc tỷ lệ với vị trí.

**Trường hợp 2 chiều.** Gọi $\mathbf{R}(\phi) = \begin{bmatrix} \cos\phi & -\sin\phi \\ \sin\phi & \cos\phi \end{bmatrix}$. Đặt truy vấn ở vị trí $m$ thành $\mathbf{R}(m\theta)\mathbf{q}$ và khóa ở vị trí $n$ thành $\mathbf{R}(n\theta)\mathbf{k}$. Khi đó

$$
\left(\mathbf{R}(m\theta)\mathbf{q}\right)^\top \left(\mathbf{R}(n\theta)\mathbf{k}\right)
= \mathbf{q}^\top \mathbf{R}(m\theta)^\top \mathbf{R}(n\theta)\, \mathbf{k}
= \mathbf{q}^\top \mathbf{R}\big((n - m)\theta\big)\, \mathbf{k},
$$

vì ma trận quay là trực giao, $\mathbf{R}(\alpha)^\top = \mathbf{R}(-\alpha)$, và hợp của hai phép quay là phép quay với tổng góc. Điểm attention **chỉ phụ thuộc vào khoảng cách** $n - m$, dù ta chỉ áp dụng thông tin vị trí tuyệt đối cho từng véc-tơ riêng lẻ.

**Góc nhìn số phức.** Đồng nhất $\mathbf{q} = (q_1, q_2)$ với $\tilde q = q_1 + \mathrm{i}\, q_2$. Phép quay là nhân với $e^{\mathrm{i} m\theta}$, và tích vô hướng thực bằng phần thực của $\tilde q\, \overline{\tilde k}$:

$$
\operatorname{Re}\!\left[(\tilde q\, e^{\mathrm{i} m\theta})\,\overline{(\tilde k\, e^{\mathrm{i} n\theta})}\right] = \operatorname{Re}\!\left[\tilde q\,\overline{\tilde k}\, e^{\mathrm{i}(m - n)\theta}\right].
$$

**Nhiều chiều.** Chia $d$ chiều thành $d/2$ cặp, cặp thứ $i$ quay với tần số riêng $\theta_i = 10000^{-2i/d}$ — đúng các tần số của mã hóa sinusoidal. Ma trận quay toàn phần là khối chéo, nên lập luận trên áp dụng cho từng cặp và tổng lại. Các tính chất:

- Điểm attention chỉ phụ thuộc $m - n$ (vị trí tương đối), nhưng cài đặt chỉ cần biết vị trí tuyệt đối của từng token — rất tiện cho KV cache: khóa đã quay một lần thì lưu luôn.
- Phép quay **bảo toàn chuẩn**, không làm thay đổi thang đo của $\mathbf{q}, \mathbf{k}$.
- Chỉ quay $\mathbf{q}$ và $\mathbf{k}$, **không** quay $\mathbf{v}$: thông tin vị trí chỉ cần đi vào điểm.
- Có nhiều kỹ thuật mở rộng ngữ cảnh cho RoPE bằng cách co giãn góc quay (nội suy vị trí và các biến thể); chúng nằm ngoài phạm vi chương này.

!!! warning "Bẫy: hai quy ước ghép cặp của RoPE"
    Bài báo gốc ghép cặp các chiều **kề nhau** $(2i, 2i+1)$. Nhiều mã nguồn (ví dụ họ GPT-NeoX/LLaMA trong một số thư viện) ghép chiều $i$ với chiều $i + d/2$ (hàm `rotate_half`). Hai quy ước tương đương sai khác **một hoán vị kênh cố định** — nhưng một checkpoint huấn luyện với quy ước này sẽ cho kết quả sai nếu chạy với quy ước kia. Khi nạp trọng số từ nơi khác, hãy kiểm tra quy ước; các script chuyển đổi checkpoint thường phải hoán vị hàng của $\mathbf{W}^Q, \mathbf{W}^K$ chính vì lý do này.

**Mã minh họa.** Cài đặt RoPE (quy ước cặp kề nhau) và kiểm chứng: điểm chỉ phụ thuộc khoảng cách, chuẩn được bảo toàn, và hai quy ước sai khác một hoán vị.

```python
def rope_cos_sin(n, d, base=10000.0, offset=0):
    """Góc quay m * theta_i cho vị trí m = offset..offset+n-1, theta_i = base^(-2i/d)."""
    theta = base ** (-torch.arange(0, d, 2).float() / d)          # (d/2,)
    ang = torch.arange(offset, offset + n).float()[:, None] * theta[None]   # (n, d/2)
    return ang.cos(), ang.sin()

def apply_rope(x, cos, sin):
    """Quay từng cặp (x_{2i}, x_{2i+1}) một góc m*theta_i. x: (..., n, d)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]                            # (..., n, d/2)
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)                                         # (..., n, d)

torch.manual_seed(0)
d = 64
q, k = torch.randn(d), torch.randn(d)

def score(m, n):
    """<RoPE(q, m), RoPE(k, n)> với q đặt ở vị trí m, k ở vị trí n."""
    cq, sq = rope_cos_sin(1, d, offset=m)
    ck, sk = rope_cos_sin(1, d, offset=n)
    return (apply_rope(q[None], cq, sq) @ apply_rope(k[None], ck, sk).T).item()

# 1) Điểm attention chỉ phụ thuộc vào khoảng cách m - n
for m, n in [(3, 1), (10, 2), (100, 0)]:
    assert abs(score(m, n) - score(m + 37, n + 37)) < 1e-3
# 2) Phép quay bảo toàn chuẩn; 3) vị trí 0 là phép đồng nhất
cos, sin = rope_cos_sin(50, d)
x = torch.randn(50, d)
assert torch.allclose(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), atol=1e-4)
assert torch.allclose(apply_rope(x[:1], cos[:1], sin[:1]), x[:1])

# Quy ước "rotate_half" (ghép cặp i với i + d/2) tương đương sai khác một hoán vị kênh
def apply_rope_half(x, cos, sin):
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
perm = torch.cat([torch.arange(0, d, 2), torch.arange(1, d, 2)])  # xen kẽ -> tách nửa
assert torch.allclose(apply_rope_half(x[:, perm], cos, sin), apply_rope(x, cos, sin)[:, perm], atol=1e-5)
print("RoPE: tính tương đối, bảo toàn chuẩn, tương đương giữa hai quy ước — OK")
```

### 5.5.7 So sánh

| Phương pháp | Đưa vào ở đâu | Tương đối? | Ngoại suy độ dài | Ví dụ trong TTS |
|---|---|---|---|---|
| Sinusoidal | Cộng vào embedding | Gián tiếp (qua phép quay tuyến tính) | Trung bình | Transformer-TTS, FastSpeech |
| Học được | Cộng vào embedding | Không | Không vượt `max_len` | Các TTS dựa trên GPT-2 như Tortoise, XTTS (theo mã nguồn công bố) |
| Shaw / T5 | Sửa điểm (và giá trị) | Có, trong cửa sổ cắt | Tốt | Bộ mã hóa văn bản Glow-TTS, VITS |
| ALiBi | Trừ vào điểm | Có | Tốt (theo báo cáo gốc) | Ít gặp |
| RoPE | Quay $\mathbf{q}, \mathbf{k}$ | Có | Khá, có kỹ thuật mở rộng | LM xương sống của TTS dựa trên LLM, F5-TTS |

## 5.6 Khối Transformer

### 5.6.1 Các thành phần

Một khối Transformer gồm hai **khối con**, mỗi khối con bọc trong kết nối dư và chuẩn hóa:

1. **Multi-head self-attention** — trộn thông tin **giữa các vị trí**;
2. **Mạng truyền thẳng từng vị trí** (*position-wise feed-forward network*, FFN) — biến đổi phi tuyến **tại từng vị trí**, độc lập nhau:

$$
\operatorname{FFN}(\mathbf{x}) = \mathbf{W}_2\, \phi(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2,
\qquad \mathbf{W}_1 \in \mathbb{R}^{d_{\text{ff}} \times d},\ d_{\text{ff}} = 4d \text{ (thông dụng)},
$$

với $\phi$ là ReLU (bản gốc) hoặc GELU. Attention "thu thập", FFN "xử lý" — một cách hình dung thô nhưng hữu ích. Số tham số mỗi khối: $4d^2$ (attention) $+\ 8d^2$ (FFN với $d_{\text{ff}} = 4d$) $= 12d^2$, bỏ qua độ lệch và chuẩn hóa.

FastSpeech thay FFN bằng **hai lớp tích chập 1D** (gọi là khối FFT, *feed-forward Transformer*), vì trong chuỗi âm vị và khung mel, các vị trí kề nhau liên quan chặt chẽ — một thiên kiến cục bộ mà FFN từng vị trí không có ([Chương 22](../phan-5-mo-hinh-am-hoc/ch22-fastspeech.md)).

### 5.6.2 Post-LN và Pre-LN

Transformer gốc đặt LayerNorm **sau** phép cộng dư (Post-LN); các mô hình hiện đại (từ GPT-2 trở đi) đặt nó **trước** khối con (Pre-LN):

$$
\text{Post-LN: } \mathbf{x}_{l+1} = \operatorname{LN}\!\big(\mathbf{x}_l + \mathcal{F}_l(\mathbf{x}_l)\big),
\qquad
\text{Pre-LN: } \mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}_l\big(\operatorname{LN}(\mathbf{x}_l)\big).
$$

Khai triển Pre-LN: $\mathbf{x}_L = \mathbf{x}_0 + \sum_{l} \mathcal{F}_l(\operatorname{LN}(\mathbf{x}_l))$ — có một **đường đồng nhất** thông suốt từ đầu ra về mọi lớp, giống khối dư thuần túy ở [Chương 4](ch04-cnn-rnn-mo-hinh-chuoi.md). Ở Post-LN, mỗi LayerNorm nằm **trên** đường chính và co giãn lại gradient đi qua. Xiong và cộng sự (2020) phân tích rằng với Post-LN, gradient của các tham số gần đầu ra lúc khởi tạo rất lớn, nên cần **warmup** tốc độ học ([Chương 3](ch03-toi-uu-va-ky-thuat-huan-luyen.md)) để không phân kỳ; với Pre-LN gradient được kiểm soát tốt hơn và có thể huấn luyện với ít hoặc không warmup.

Đổi lại, Pre-LN cần thêm một LayerNorm **cuối cùng** (vì $\mathbf{x}_L$ chưa được chuẩn hóa), và chuẩn của "dòng dư" $\mathbf{x}_l$ có xu hướng tăng theo độ sâu, làm đóng góp tương đối của các lớp sâu nhỏ đi. Một số nghiên cứu báo cáo Post-LN (nếu huấn luyện ổn định được) cho chất lượng cuối tốt hơn; trong thực hành hiện nay, Pre-LN là lựa chọn mặc định vì độ bền. Các mô hình TTS đời đầu (Transformer-TTS, FastSpeech) dùng Post-LN như bản gốc; các mô hình lớn hơn gần đây dùng Pre-LN hoặc biến thể như adaLN (mục 5.12).

### 5.6.3 RMSNorm và SwiGLU

**RMSNorm** (Zhang và Sennrich, 2019) bỏ bước trừ trung bình của LayerNorm, chỉ chia cho căn trung bình bình phương:

$$
\operatorname{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\tfrac{1}{d}\sum_{i} x_i^2 + \epsilon}} \odot \mathbf{g}.
$$

Rẻ hơn một chút, và thực nghiệm cho thấy chất lượng tương đương; nó được dùng rộng rãi trong các LLM ([Chương 3](ch03-toi-uu-va-ky-thuat-huan-luyen.md) trình bày chi tiết các phép chuẩn hóa).

**SwiGLU** (Shazeer, 2020) thay FFN bằng một biến thể có cổng của GLU ([Chương 4](ch04-cnn-rnn-mo-hinh-chuoi.md)):

$$
\operatorname{FFN}_{\text{SwiGLU}}(\mathbf{x}) = \mathbf{W}_2\big(\operatorname{SiLU}(\mathbf{W}_1 \mathbf{x}) \odot \mathbf{W}_3 \mathbf{x}\big),
\qquad \operatorname{SiLU}(z) = z\,\sigma(z).
$$

Vì có ba ma trận thay vì hai, chiều ẩn thường đặt khoảng $\tfrac{2}{3} \cdot 4d = \tfrac{8}{3}d$ để giữ nguyên số tham số $8d^2$. Bộ ba **Pre-LN + RMSNorm + SwiGLU** (cùng RoPE) là "công thức chuẩn" của các LLM như LLaMA (Touvron và cộng sự, 2023), và vì vậy cũng là của các TTS dựng trên LLM.

**Mã minh họa.** Khối Pre-LN với RMSNorm, SwiGLU và self-attention nhân quả dùng RoPE, có hỗ trợ KV cache (giải thích ở mục 5.8). Ta lắp chúng thành một mô hình ngôn ngữ decoder-only tí hon.

```python
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))

    def forward(self, x):                         # (..., d)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class SwiGLU(nn.Module):
    """FFN(x) = W2 (SiLU(W1 x) ⊙ W3 x)."""
    def __init__(self, d, hidden):
        super().__init__()
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x, cache=None):             # x: (B, n, d); cache: dict hoặc None
        B, n, _ = x.shape
        q, k, v = self.qkv(x).view(B, n, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)  # (B, h, n, dh)
        t0 = cache["k"].shape[2] if cache and "k" in cache else 0     # số token đã có trong cache
        cos, sin = rope_cos_sin(n, self.dh, offset=t0)                 # vị trí t0 .. t0+n-1
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            if "k" in cache:
                k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v                              # (B, h, t0+n, dh)
        S = k.shape[2]
        mask = torch.ones(n, S, dtype=torch.bool).tril(diagonal=S - n)  # nhân quả "căn dưới-phải"
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)    # (B, h, n, dh)
        return self.proj(o.transpose(1, 2).reshape(B, n, -1))

class PreLNBlock(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
        self.attn = CausalSelfAttention(d, n_heads)
        self.ffn = SwiGLU(d, hidden=int(8 * d / 3))  # giữ số tham số ≈ FFN 4d thông thường

    def forward(self, x, cache=None):
        x = x + self.attn(self.norm1(x), cache)      # Pre-LN: chuẩn hóa TRƯỚC khối con
        return x + self.ffn(self.norm2(x))

class TinyLM(nn.Module):
    def __init__(self, vocab=100, d=64, n_heads=4, n_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(PreLNBlock(d, n_heads) for _ in range(n_layers))
        self.norm, self.head = RMSNorm(d), nn.Linear(d, vocab, bias=False)

    def forward(self, ids, caches=None):          # ids: (B, n) -> logits (B, n, V)
        x = self.emb(ids)
        for i, blk in enumerate(self.blocks):
            x = blk(x, None if caches is None else caches[i])
        return self.head(self.norm(x))            # LayerNorm cuối — bắt buộc với Pre-LN
```
