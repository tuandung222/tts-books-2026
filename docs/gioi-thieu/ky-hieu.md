# Ký hiệu toán học

Giáo trình dùng một bộ ký hiệu thống nhất. Khi một chương buộc phải theo ký hiệu của bài báo gốc, chương đó sẽ nói rõ.

## Ký hiệu chung

| Ký hiệu | Ý nghĩa |
|---|---|
| $a, b, x$ | Đại lượng vô hướng (chữ thường, nghiêng) |
| $\mathbf{x}, \mathbf{h}$ | Véc-tơ cột (chữ thường, đậm) |
| $\mathbf{W}, \mathbf{A}$ | Ma trận (chữ hoa, đậm) |
| $\mathbf{x}^\top$ | Chuyển vị |
| $\mathbb{R}^{d}$, $\mathbb{R}^{m \times n}$ | Không gian véc-tơ thực $d$ chiều, ma trận thực $m \times n$ |
| $\lVert \mathbf{x} \rVert_p$ | Chuẩn $\ell_p$ (mặc định $p = 2$) |
| $\odot$ | Tích từng phần tử (Hadamard) |
| $\ast$ | Tích chập |
| $[\mathbf{a}; \mathbf{b}]$ | Ghép (nối) hai véc-tơ |
| $\theta, \phi, \psi$ | Tham số mô hình |
| $f_\theta, g_\phi$ | Hàm/mạng có tham số |
| $\nabla_\theta \mathcal{L}$ | Gradient của hàm mất mát theo $\theta$ |
| $\mathcal{L}$ | Hàm mất mát |
| $\eta$ | Tốc độ học |
| $\sigma(\cdot)$ | Hàm sigmoid (cũng dùng $\sigma$ cho độ lệch chuẩn, tùy ngữ cảnh) |
| $\operatorname{softmax}(\cdot)$ | Hàm softmax |
| $\mathbb{1}[\cdot]$ | Hàm chỉ thị |

## Xác suất và thông tin

| Ký hiệu | Ý nghĩa |
|---|---|
| $p(\mathbf{x})$, $p_\theta(\mathbf{x})$ | Phân phối (mô hình) trên $\mathbf{x}$ |
| $p_{\text{data}}$ | Phân phối dữ liệu thật |
| $q_\phi(\mathbf{z} \mid \mathbf{x})$ | Phân phối hậu nghiệm xấp xỉ (encoder của VAE) |
| $\mathbb{E}_{p}[\cdot]$ | Kỳ vọng theo phân phối $p$ |
| $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ | Phân phối chuẩn |
| $D_{\mathrm{KL}}(q \,\Vert\, p)$ | Phân kỳ Kullback–Leibler |
| $H(p)$, $H(p, q)$ | Entropy, entropy chéo |
| $\mathbf{z}$ | Biến ẩn (*latent*) |
| $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ | Nhiễu chuẩn |

## Ký hiệu cho tổng hợp tiếng nói

| Ký hiệu | Ý nghĩa |
|---|---|
| $\mathbf{x} = (x_1, \ldots, x_N)$ | Chuỗi đầu vào văn bản (ký tự, âm vị hoặc token), độ dài $N$ |
| $\mathbf{Y} = (\mathbf{y}_1, \ldots, \mathbf{y}_T) \in \mathbb{R}^{T \times F}$ | Chuỗi đặc trưng âm học (thường là log-mel-spectrogram): $T$ khung, $F$ kênh mel |
| $s[n]$, $n = 0, \ldots, L-1$ | Dạng sóng (*waveform*) rời rạc gồm $L$ mẫu |
| $f_s$ | Tần số lấy mẫu (Hz), ví dụ 22 050 hoặc 24 000 |
| $N_{\text{FFT}}$, $W$, $H$ | Kích thước FFT, độ dài cửa sổ, bước nhảy (*hop*) tính theo số mẫu; $L \approx T \cdot H$ |
| $S(t, k)$ hoặc $S_{t,k}$ | Hệ số STFT tại khung $t$, bin tần số $k$ |
| $f_0$ hoặc F0 | Tần số cơ bản |
| $\mathbf{d} = (d_1, \ldots, d_N)$ | Trường độ: số khung ứng với mỗi token, $\sum_i d_i = T$ |
| $\mathbf{A} \in \{0, 1\}^{N \times T}$ | Ma trận căn chỉnh cứng giữa token và khung (hoặc $\mathbf{A} \in [0,1]^{N \times T}$ khi là attention mềm) |
| $\mathbf{g}$ | Điều kiện toàn cục (embedding người nói, phong cách, ngôn ngữ) |
| $\mathbf{h}^{\text{text}}$, $\mathbf{H}$ | Biểu diễn ẩn của bộ mã hóa văn bản |
| $G$, $D$ | Bộ sinh và bộ phân biệt trong GAN |
| $t \in [0, 1]$ | Biến thời gian liên tục của diffusion/flow matching |
| $K$ | Tổng số bước khuếch tán rời rạc (dùng $K$ để không trùng với số khung $T$) |
| $\mathbf{v}_\theta(\mathbf{x}_t, t)$ | Trường véc-tơ vận tốc (flow matching) |
| $\mathbf{q} \in \{1, \ldots, V\}^{T' \times Q}$ | Token âm thanh rời rạc: $T'$ khung codec, $Q$ tầng lượng tử (RVQ), sổ mã kích thước $V$ |

## Quy ước hình dạng tensor trong mã nguồn

| Chữ cái | Ý nghĩa |
|---|---|
| `B` | Kích thước batch |
| `N` | Số token văn bản |
| `T` | Số khung âm học |
| `F` | Số kênh mel (ví dụ 80 hoặc 100) |
| `C` | Số kênh ẩn |
| `L` | Số mẫu âm thanh |
| `Q` | Số tầng codebook |

PyTorch thường dùng bố cục `(B, C, T)` cho `Conv1d` và `(B, T, C)` cho Transformer; mã minh họa luôn chú thích bố cục đang dùng.
