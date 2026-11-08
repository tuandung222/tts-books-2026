# Chương 7. Tổng quan mô hình sinh

!!! abstract "Mục tiêu học tập"
    Sau chương này, bạn có thể:

    - **Phân biệt** mô hình phân biệt với mô hình sinh, và hai nhiệm vụ cốt lõi của mô hình sinh: ước lượng mật độ và lấy mẫu.
    - **Phân loại** các họ mô hình sinh (tự hồi quy, VAE, GAN, flow, diffusion, flow matching, mô hình ngôn ngữ trên token rời rạc) theo cách chúng xử lý hàm hợp lý.
    - **Suy ra** sự tương đương giữa ước lượng hợp lý cực đại và cực tiểu hóa phân kỳ KL thuận, và giải thích hành vi "phủ mode" của nó.
    - **Chứng minh** rằng hồi quy L2 (L1) hội tụ về trung bình (trung vị) có điều kiện, rồi dùng kết quả đó để giải thích vì sao mel-spectrogram dự đoán bằng hồi quy bị "trơn quá mức" (*over-smoothing*).
    - **Cài đặt** một mô hình tự hồi quy nhân quả có kiểm tra rò rỉ thông tin, và một mạng mật độ hỗn hợp (MDN).
    - **Đánh giá** ưu nhược điểm của các thước đo cho mô hình sinh (log-likelihood, FID/FAD, precision/recall, MOS).
    - **Định vị** mỗi hệ thống TTS nổi tiếng vào họ mô hình sinh tương ứng và vào các chương sau của giáo trình.

!!! info "Kiến thức cần có"
    - [Chương 1](../phan-1-nen-tang/ch01-toan-nen-tang.md): phân phối Gauss và Laplace, kỳ vọng và phương sai, entropy, phân kỳ KL, ước lượng Monte Carlo.
    - [Chương 2](../phan-1-nen-tang/ch02-hoc-may-va-mang-no-ron.md): cách nhìn hàm mất mát MSE/L1/CE như ước lượng hợp lý cực đại.
    - [Chương 4](../phan-1-nen-tang/ch04-cnn-rnn-mo-hinh-chuoi.md): tích chập nhân quả, seq2seq, teacher forcing.
    - [Chương 5](../phan-1-nen-tang/ch05-transformer.md): Transformer decoder, mặt nạ nhân quả, các chiến lược lấy mẫu.
    - Biết sơ lược mel-spectrogram là gì (chi tiết ở [Chương 14](../phan-3-tin-hieu-ngu-am/ch14-xu-ly-tin-hieu-so.md)).

Phần I trang bị cho bạn bộ công cụ để huấn luyện mạng nơ-ron ánh xạ đầu vào sang đầu ra. Phần II đặt ra một câu hỏi khó hơn: khi đầu ra không phải là *một* đáp án đúng mà là *cả một phân phối* các đáp án hợp lệ, ta học và lấy mẫu từ phân phối đó như thế nào? Tổng hợp tiếng nói chính là một bài toán như vậy. Chương này xây dựng khung khái niệm chung cho toàn bộ Phần II, và chứng minh một kết quả mà giáo trình sẽ nhắc lại nhiều lần ở các Phần V–VIII: **vì sao hồi quy L1/L2 thuần túy làm mel-spectrogram bị nhòe**, và vì sao điều đó đẩy cộng đồng TTS sang các mô hình sinh ngày càng mạnh.

## 7.1. Mô hình sinh là gì?

### 7.1.1. Mô hình phân biệt và mô hình sinh

Một **mô hình phân biệt** (*discriminative model*) học một ánh xạ từ dữ liệu sang nhãn, hay tổng quát hơn là phân phối $p(y \mid \mathbf{x})$ trên một nhãn $y$ ít chiều: ảnh này là mèo hay chó, câu nói này thuộc người nói nào. Không gian đầu ra nhỏ, và thường chỉ có một đáp án đúng.

Một **mô hình sinh** (*generative model*) học phân phối của chính dữ liệu, $p_\theta(\mathbf{x})$, hoặc phân phối có điều kiện $p_\theta(\mathbf{x} \mid \mathbf{c})$ khi có thông tin điều kiện $\mathbf{c}$. Ở đây $\mathbf{x}$ là một đối tượng nhiều chiều và có cấu trúc: một bức ảnh, một đoạn dạng sóng, một mel-spectrogram. Mục tiêu là $p_\theta$ đủ gần với phân phối dữ liệu thật $p_{\text{data}}$, phân phối mà ta không biết công thức nhưng có một tập mẫu $\mathcal{D} = \{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)}\}$ rút độc lập từ nó.

!!! note "Quy ước ký hiệu trong Phần II"
    Trong các mục tổng quát của Phần II, $\mathbf{x}$ chỉ **một điểm dữ liệu bất kỳ** (theo thông lệ của tài liệu học máy và theo [bảng ký hiệu](../gioi-thieu/ky-hieu.md)). Khi áp dụng vào TTS, ta quay về ký hiệu của sách: $\mathbf{x}$ là chuỗi văn bản đầu vào và $\mathbf{Y} \in \mathbb{R}^{T \times F}$ là chuỗi đặc trưng âm học cần sinh; mô hình khi đó là $p_\theta(\mathbf{Y} \mid \mathbf{x})$. Mỗi khi chuyển ngữ cảnh, văn bản sẽ nhắc lại.

Sự khác biệt giữa hai loại mô hình thể hiện rõ khi so sánh nhận dạng tiếng nói (ASR) với tổng hợp tiếng nói (TTS). ASR là ánh xạ **nhiều–một**: vô số cách đọc khác nhau của câu "xin chào" (giọng nam, giọng nữ, nhanh, chậm, ở phòng thu hay ngoài đường) đều phải cho ra cùng một chuỗi chữ. TTS là ánh xạ **một–nhiều**: cùng câu "xin chào" có vô số cách đọc đều đúng. Mô hình ASR có thể "vứt bỏ" mọi biến thiên không liên quan; mô hình TTS phải **tạo ra** những biến thiên đó một cách hợp lý. Đây là lý do tuy TTS có dạng "đầu vào → đầu ra" như một bài toán học có giám sát, bản chất của nó là một bài toán **sinh có điều kiện**.

### 7.1.2. Ba việc ta muốn làm với một mô hình sinh

1. **Ước lượng mật độ** (*density estimation*): với một $\mathbf{x}$ cho trước, tính được $p_\theta(\mathbf{x})$ (hoặc $\log p_\theta(\mathbf{x})$). Khả năng này cho phép huấn luyện bằng hợp lý cực đại, so sánh mô hình bằng log-likelihood trên tập kiểm tra, phát hiện mẫu bất thường, nén dữ liệu.
2. **Lấy mẫu** (*sampling*): sinh ra $\mathbf{x} \sim p_\theta$ mới, chưa từng có trong dữ liệu. Với TTS, đây là việc quan trọng nhất: ta cần *nghe* được tiếng nói.
3. **Học biểu diễn và điều khiển**: tìm biến ẩn $\mathbf{z}$ giải thích dữ liệu (phong cách, ngữ điệu, người nói), hoặc thực hiện các thao tác có điều kiện như điền vào chỗ trống (*infilling*), chỉnh sửa một đoạn tiếng nói, chuyển phong cách.

Không có họ mô hình nào giỏi đều cả ba việc. Một số họ tính được $p_\theta(\mathbf{x})$ chính xác nhưng lấy mẫu chậm (tự hồi quy); một số lấy mẫu rất nhanh nhưng không cho biết mật độ (GAN). Người ta hay tóm gọn sự đánh đổi này thành "bộ ba bất khả" (*generative learning trilemma*) giữa **chất lượng mẫu**, **độ đa dạng (phủ mode)** và **tốc độ lấy mẫu** (Xiao và cộng sự, 2022). Lịch sử TTS mười năm qua, ở một mức độ đáng kể, là lịch sử tìm cách thoát khỏi bộ ba này.

### 7.1.3. Phân loại các họ mô hình sinh

Cách phân loại hữu ích nhất dựa vào câu hỏi: **mô hình đối xử với hàm hợp lý $p_\theta(\mathbf{x})$ như thế nào?** (Goodfellow, 2016).

```mermaid
flowchart TD
    R["Mô hình sinh"] --> E["Mật độ tường minh"]
    R --> I["Mật độ ẩn (chỉ có bộ lấy mẫu)"]
    E --> T["Tính được chính xác"]
    E --> A["Xấp xỉ / cận dưới"]
    T --> AR["Tự hồi quy (WaveNet, codec LM)"]
    T --> NF["Normalizing flow (Glow, WaveGlow)"]
    A --> VAE["VAE: cận dưới ELBO"]
    A --> DM["Diffusion: ELBO / score matching"]
    I --> GAN["GAN: bộ sinh + bộ phân biệt"]
    NF --> CNF["Flow liên tục"]
    CNF --> FM["Flow matching: hồi quy trường vận tốc"]
```

- **Mật độ tường minh, tính được chính xác.** Mô hình *tự hồi quy* (mục 7.3) phân rã $p_\theta(\mathbf{x})$ thành tích các xác suất có điều kiện; *normalizing flow* ([Chương 10](ch10-normalizing-flows.md)) biến đổi khả nghịch một phân phối đơn giản và dùng công thức đổi biến. Cả hai cho phép huấn luyện bằng hợp lý cực đại đúng nghĩa.
- **Mật độ tường minh nhưng xấp xỉ.** *VAE* ([Chương 8](ch08-vae.md)) có $p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x} \mid \mathbf{z}) p(\mathbf{z})\, d\mathbf{z}$ không tính được, nên tối ưu một cận dưới. *Mô hình khuếch tán* ([Chương 11](ch11-diffusion.md)) cũng có thể được huấn luyện qua một cận dưới, hoặc tương đương qua khớp hàm score. Các mô hình dựa trên năng lượng (*energy-based*) có mật độ chưa chuẩn hóa; giáo trình chỉ nhắc đến chúng khi cần.
- **Mật độ ẩn.** *GAN* ([Chương 9](ch09-gan.md)) chỉ định nghĩa một quy trình lấy mẫu $\mathbf{x} = G(\mathbf{z})$ và không bao giờ tính $p_\theta(\mathbf{x})$; nó được huấn luyện bằng một bộ phân biệt đóng vai "thước đo khoảng cách" được học.
- **Flow matching** ([Chương 12](ch12-flow-matching.md)) là trường hợp lai thú vị: mô hình là một flow liên tục (về nguyên tắc tính được mật độ bằng cách giải một phương trình vi phân), nhưng được huấn luyện bằng một bài toán *hồi quy* trường vận tốc, không dùng đến likelihood.

Mô hình ngôn ngữ trên **token âm thanh rời rạc** (VALL-E và các hậu duệ, [Chương 32](../phan-8-mo-hinh-lon/ch32-codec-language-model.md)) thực chất là mô hình tự hồi quy với đầu ra softmax; nhưng vì vai trò nổi bật của nó trong TTS hiện đại, ta sẽ xếp nó thành một họ riêng trong bản đồ ở mục 7.7.

## 7.2. Ước lượng hợp lý cực đại và phân kỳ KL

### 7.2.1. Từ trung bình mẫu đến phân kỳ KL

Nguyên lý huấn luyện phổ biến nhất là **ước lượng hợp lý cực đại** (*maximum likelihood estimation*, MLE): chọn tham số làm cho dữ liệu quan sát được "có khả năng xảy ra nhất",

$$
\hat{\theta} = \arg\max_\theta \frac{1}{n} \sum_{i=1}^{n} \log p_\theta\big(\mathbf{x}^{(i)}\big).
$$

Theo luật số lớn, khi $n \to \infty$ trung bình mẫu hội tụ về kỳ vọng $\mathbb{E}_{p_{\text{data}}}[\log p_\theta(\mathbf{x})]$. Viết phân kỳ KL thuận từ dữ liệu đến mô hình ra:

$$
D_{\mathrm{KL}}(p_{\text{data}} \,\Vert\, p_\theta)
= \mathbb{E}_{p_{\text{data}}}\big[\log p_{\text{data}}(\mathbf{x})\big] - \mathbb{E}_{p_{\text{data}}}\big[\log p_\theta(\mathbf{x})\big]
= -H(p_{\text{data}}) - \mathbb{E}_{p_{\text{data}}}\big[\log p_\theta(\mathbf{x})\big].
$$

Số hạng entropy $H(p_{\text{data}})$ không phụ thuộc $\theta$. Do đó

$$
\arg\max_\theta \; \mathbb{E}_{p_{\text{data}}}\big[\log p_\theta(\mathbf{x})\big] = \arg\min_\theta \; D_{\mathrm{KL}}(p_{\text{data}} \,\Vert\, p_\theta).
$$

Nói bằng lời: **MLE chính là cực tiểu hóa KL thuận**. Một hệ quả nhỏ nhưng hữu ích: giá trị log-likelihood tuyệt đối không có nghĩa tự thân, vì nó bị dịch đi một hằng số $H(p_{\text{data}})$ mà ta không biết. Chỉ *hiệu* log-likelihood giữa hai mô hình trên cùng một tập dữ liệu, cùng một cách tiền xử lý, mới so sánh được. (Với dữ liệu liên tục, $H$ là entropy vi phân và có thể âm; lập luận trên không thay đổi.)

### 7.2.2. KL thuận "phủ mode", KL ngược "tìm mode"

Phân kỳ KL không đối xứng, và hai chiều của nó trừng phạt những lỗi khác nhau:

- **KL thuận** $D_{\mathrm{KL}}(p_{\text{data}} \Vert p_\theta) = \mathbb{E}_{p_{\text{data}}}[\log p_{\text{data}} - \log p_\theta]$ lấy kỳ vọng theo *dữ liệu*. Nếu ở đâu đó dữ liệu có mật độ đáng kể mà mô hình gán mật độ gần 0, số hạng $-\log p_\theta$ bùng nổ. Vì vậy mô hình bị ép **phủ mọi vùng có dữ liệu** (*mode-covering*, hay *zero-avoiding*), kể cả khi phải trải mật độ sang những vùng không có dữ liệu.
- **KL ngược** $D_{\mathrm{KL}}(p_\theta \Vert p_{\text{data}})$ lấy kỳ vọng theo *mô hình*. Mô hình bị phạt nặng nếu đặt mật độ ở nơi dữ liệu hiếm, nhưng không bị phạt nếu bỏ sót một mode. Kết quả là nó **chọn một mode và bám chặt vào đó** (*mode-seeking*, *zero-forcing*).

Thí nghiệm sau khớp một phân phối Gauss đơn lẻ vào dữ liệu có hai mode, theo cả hai chiều KL, bằng tìm kiếm lưới:

```python
import numpy as np

# Lưới 1-D để tính tích phân bằng tổng Riemann
y = np.linspace(-6.0, 6.0, 4001)
dy = y[1] - y[0]

def normal_pdf(y, mu, sigma):
    return np.exp(-0.5 * ((y - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

# "Dữ liệu": hỗn hợp hai mode cách xa nhau (ví dụ: hai cách ngắt nhịp của cùng một câu)
p = 0.5 * normal_pdf(y, -2.0, 0.5) + 0.5 * normal_pdf(y, 2.0, 0.5)

def kl(a, b):
    """D_KL(a || b) bằng tích phân số; eps tránh log(0)."""
    eps = 1e-300
    return np.sum(a * (np.log(a + eps) - np.log(b + eps))) * dy

best_fwd, best_rev = None, None
for mu in np.linspace(-3, 3, 121):
    for sigma in np.linspace(0.2, 3.0, 57):
        q = normal_pdf(y, mu, sigma)
        f, r = kl(p, q), kl(q, p)          # forward KL (= MLE) và reverse KL
        if best_fwd is None or f < best_fwd[0]:
            best_fwd = (f, mu, sigma)
        if best_rev is None or r < best_rev[0]:
            best_rev = (r, mu, sigma)

print(f"Forward KL (MLE): mu={best_fwd[1]:+.2f}, sigma={best_fwd[2]:.2f}")
print(f"Reverse KL      : mu={best_rev[1]:+.2f}, sigma={best_rev[2]:.2f}")
# MLE với một Gaussian khớp đúng trung bình và phương sai của dữ liệu:
# mean = 0, var = 0.5^2 + 2^2 = 4.25 -> sigma ≈ 2.06
assert abs(best_fwd[1]) < 0.06 and abs(best_fwd[2] - np.sqrt(4.25)) < 0.06
assert abs(abs(best_rev[1]) - 2.0) < 0.06 and abs(best_rev[2] - 0.5) < 0.06
```

Kết quả: KL thuận cho $\mu \approx 0$, $\sigma \approx 2.05$, một Gauss rộng đặt **giữa** hai mode, đúng chỗ dữ liệu gần như không có; KL ngược cho $\mu \approx \pm 2$, $\sigma \approx 0.5$, khớp hoàn hảo **một** mode và bỏ quên mode còn lại. (Việc KL thuận cho đúng trung bình và phương sai của dữ liệu không phải ngẫu nhiên: với họ phân phối mũ, MLE tương đương *khớp mô-men*, xem Bài tập 3.)

Hai bài học rút ra từ thí nghiệm nhỏ này sẽ theo chúng ta suốt Phần II:

1. **Vấn đề không nằm ở MLE mà ở họ mô hình quá nghèo.** Nếu họ mô hình đủ biểu đạt (ví dụ một hỗn hợp hai Gauss), MLE sẽ khớp đúng cả hai mode. Toàn bộ nỗ lực của Chương 8–12 có thể hiểu là: xây dựng những họ $p_\theta$ đủ giàu cho dữ liệu cực kỳ nhiều chiều như tiếng nói, mà vẫn huấn luyện và lấy mẫu được.
2. **Hai kiểu thất bại khác nhau.** Mô hình MLE nghèo cho mẫu *đa dạng nhưng sai* (nhiều mẫu rơi vào vùng không có dữ liệu, nghe như nhiễu hoặc lơ lớ); mô hình tìm mode cho mẫu *đẹp nhưng đơn điệu* (mất đa dạng, hiện tượng sụp mode ở GAN, [Chương 9](ch09-gan.md)). Trong thực tế, người ta thường "hạ nhiệt độ" (*temperature*) khi lấy mẫu từ các mô hình MLE (Glow-TTS, codec LM) chính là để đổi một ít đa dạng lấy chất lượng.

## 7.3. Mô hình tự hồi quy

### 7.3.1. Phân rã theo quy tắc chuỗi

Với một chuỗi $\mathbf{s} = (s_1, \ldots, s_L)$, quy tắc chuỗi của xác suất cho phép viết **chính xác**, không cần giả định gì:

$$
p_\theta(\mathbf{s}) = \prod_{t=1}^{L} p_\theta(s_t \mid s_1, \ldots, s_{t-1}) = \prod_{t=1}^{L} p_\theta(s_t \mid \mathbf{s}_{<t}),
\qquad
\log p_\theta(\mathbf{s}) = \sum_{t=1}^{L} \log p_\theta(s_t \mid \mathbf{s}_{<t}).
$$

Bài toán mô hình hóa một phân phối $L$ chiều trở thành bài toán mô hình hóa $L$ phân phối **một chiều** (hoặc ít chiều), mỗi phân phối có điều kiện trên quá khứ. Một mạng nơ-ron dùng chung tham số $\theta$ đảm nhận mọi bước. Với mỗi điều kiện ta chọn một họ phân phối đơn giản:

- Dữ liệu **rời rạc**: softmax trên $V$ lớp. WaveNet bản gốc lượng tử hóa mỗi mẫu âm thanh về 256 mức bằng nén $\mu$-law rồi dùng softmax 256 lớp (van den Oord và cộng sự, 2016a); codec LM dùng softmax trên sổ mã token.
- Dữ liệu **liên tục**: Gauss, Laplace, hoặc hỗn hợp phân phối logistic như PixelCNN++ (Salimans và cộng sự, 2017). Tacotron 2 dùng một WaveNet với đầu ra là hỗn hợp 10 phân phối logistic theo báo cáo gốc (Shen và cộng sự, 2018).

Chú ý điểm mấu chốt: **softmax trên $V$ lớp có thể biểu diễn bất kỳ phân phối nào trên $V$ giá trị**, kể cả đa mode. Chính vì vậy, khi bài toán được rời rạc hóa thành token, vấn đề "trung bình hóa" mà ta sẽ phân tích ở mục 7.4 biến mất ở cấp độ từng bước. Đây là một lý do sâu xa khiến các mô hình codec LM cho ngôn điệu tự nhiên.

### 7.3.2. Huấn luyện song song, suy luận tuần tự

Khi huấn luyện, toàn bộ chuỗi thật $\mathbf{s}$ đã có sẵn. Ta đưa chuỗi **dịch phải một bước** vào mạng và tính *đồng thời* mọi điều kiện $p_\theta(s_t \mid \mathbf{s}_{<t})$ trong một lượt truyền xuôi, miễn là mạng **nhân quả**: đầu ra tại $t$ không được phụ thuộc vào $s_t, s_{t+1}, \ldots$ Có hai cách thực hiện phổ biến: tích chập nhân quả với đệm một phía (WaveNet, PixelCNN dùng tích chập có mặt nạ), và self-attention với mặt nạ tam giác (Transformer decoder, [Chương 5](../phan-1-nen-tang/ch05-transformer.md)). Cách huấn luyện "cho mạng thấy tiền tố thật" này chính là **teacher forcing**.

Khi suy luận, không có chuỗi thật. Ta phải sinh $s_1$, đưa nó vào để sinh $s_2$, và cứ thế: **$L$ lượt gọi mạng tuần tự**. Đoạn mã sau cài đặt một mô hình tự hồi quy Gauss bằng tích chập nhân quả, đồng thời *kiểm chứng* tính nhân quả bằng cách so log-likelihood tính song song với log-likelihood tính lại từng bước. Kiểu kiểm tra này đáng đưa vào bộ test của mọi dự án có mô hình tự hồi quy: một lỗi đệm sai phía có thể làm mô hình "nhìn trộm" tương lai, cho loss rất thấp khi huấn luyện và rác hoàn toàn khi suy luận.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fn

class CausalConvAR(nn.Module):
    """Mô hình tự hồi quy nhỏ: p(s_t | s_<t) = N(mu_t, sigma_t^2), dùng tích chập nhân quả."""
    def __init__(self, channels=16, kernel=3, n_layers=3):
        super().__init__()
        self.kernel = kernel
        self.inp = nn.Conv1d(1, channels, 1)
        self.convs = nn.ModuleList(
            [nn.Conv1d(channels, channels, kernel, dilation=2 ** i) for i in range(n_layers)])
        self.out = nn.Conv1d(channels, 2, 1)                  # (mu, log_sigma)

    def forward(self, s_prev):                                # s_prev: (B, 1, L)
        h = self.inp(s_prev)
        for conv in self.convs:
            pad = (self.kernel - 1) * conv.dilation[0]        # chỉ đệm bên trái -> nhân quả
            h = h + torch.tanh(conv(Fn.pad(h, (pad, 0))))
        mu, log_sigma = self.out(h).chunk(2, dim=1)           # mỗi cái (B, 1, L)
        return mu, log_sigma

def log_likelihood(model, s):                                 # s: (B, 1, L)
    """Teacher forcing: dịch phải một bước, tính MỌI điều kiện trong MỘT lượt song song."""
    s_prev = Fn.pad(s, (1, 0))[..., :-1]                      # (B, 1, L): s_prev[t] = s[t-1]
    mu, log_sigma = model(s_prev)
    log_n = -0.5 * ((s - mu) / log_sigma.exp()) ** 2 - log_sigma - 0.5 * math.log(2 * math.pi)
    return log_n.sum(dim=(1, 2))                              # (B,): log p(s) theo quy tắc chuỗi

torch.manual_seed(0)
model = CausalConvAR().eval()
s = torch.randn(2, 1, 50)                                     # (B=2, 1, L=50)

with torch.no_grad():
    lp_parallel = log_likelihood(model, s)
    # Kiểm tra tính nhân quả: tính lại từng điều kiện p(s_t | s_<t) một cách tuần tự
    lp_seq = torch.zeros(2)
    for t in range(s.shape[-1]):
        prefix = Fn.pad(s[..., :t], (1, 0))                   # (B, 1, t+1): [0, s_0..s_{t-1}]
        mu, log_sigma = model(prefix)
        mu_t, ls_t = mu[..., -1], log_sigma[..., -1]          # (B, 1)
        lp_seq += (-0.5 * ((s[..., t] - mu_t) / ls_t.exp()) ** 2 - ls_t
                   - 0.5 * math.log(2 * math.pi)).squeeze(1)
    assert torch.allclose(lp_parallel, lp_seq, atol=1e-4), "mô hình rò rỉ thông tin tương lai!"

    # Lấy mẫu: bắt buộc tuần tự, L lượt gọi mạng; đầu vào là CHÍNH mẫu mô hình vừa sinh
    gen = torch.zeros(1, 1, 1)
    for t in range(50):
        mu, log_sigma = model(gen)
        nxt = mu[..., -1:] + log_sigma[..., -1:].exp() * torch.randn(1, 1, 1)
        gen = torch.cat([gen, nxt], dim=-1)
    print("log p song song:", lp_parallel.tolist(), "| độ dài mẫu sinh:", gen.shape[-1] - 1)
```

Sự bất đối xứng "huấn luyện song song, suy luận tuần tự" có giá rất đắt với âm thanh. Một giây âm thanh ở 24 kHz cần 24 000 lượt gọi mạng nối tiếp nhau; dù mỗi lượt chỉ mất vài chục micro giây trên GPU thì cũng khó đạt thời gian thực nếu không có kỹ thuật đặc biệt. Đây chính là động lực cho WaveRNN, LPCNet, Parallel WaveNet và cuối cùng là các vocoder GAN không tự hồi quy ([Chương 24](../phan-6-vocoder/ch24-vocoder-tu-hoi-quy-va-flow.md), [Chương 25](../phan-6-vocoder/ch25-gan-vocoder.md)). Với mel-spectrogram (vài chục đến khoảng một trăm khung mỗi giây) hay token codec (cỡ vài chục đến vài trăm token mỗi giây tùy codec), chi phí tuần tự chấp nhận được hơn nhiều; vì vậy tự hồi quy vẫn sống khỏe ở tầng mô hình âm học và mô hình ngôn ngữ.

### 7.3.3. Exposure bias

Teacher forcing tạo ra một độ lệch tinh vi. Khi huấn luyện, mạng luôn thấy tiền tố **thật** $\mathbf{s}_{<t} \sim p_{\text{data}}$; khi suy luận, nó thấy tiền tố **do chính nó sinh ra**, $\hat{\mathbf{s}}_{<t} \sim p_\theta$. Nếu một bước sinh hơi lệch, các bước sau nhận một đầu vào chưa từng gặp trong huấn luyện, có thể lệch thêm, và sai số **tích lũy** theo thời gian. Hiện tượng này được gọi là **exposure bias** (Ranzato và cộng sự, 2016). Trong TTS tự hồi quy dựa trên attention, nó biểu hiện thành lặp từ, bỏ từ, lảm nhảm hoặc không chịu dừng ([Chương 20](../phan-5-mo-hinh-am-hoc/ch20-tacotron.md)), và nặng hơn với câu dài.

Các biện pháp giảm nhẹ gồm:

- **Scheduled sampling** (Bengio và cộng sự, 2015): trong huấn luyện, thỉnh thoảng thay tiền tố thật bằng mẫu của mô hình, với xác suất tăng dần. Cách này cải thiện thực nghiệm trong nhiều tác vụ, nhưng Huszár (2015) chỉ ra rằng nó không phải là một mục tiêu huấn luyện nhất quán theo nghĩa thống kê; hãy coi nó là một mẹo, không phải lời giải.
- **Làm nhiễu tiền tố**: Tacotron giữ dropout ở pre-net ngay cả khi suy luận và dự đoán nhiều khung mỗi bước (hệ số rút gọn), khiến mạng bớt phụ thuộc vào chi tiết của khung trước ([Chương 20](../phan-5-mo-hinh-am-hoc/ch20-tacotron.md)).
- **Mục tiêu cấp chuỗi** như huấn luyện đối kháng hoặc học tăng cường, đánh giá cả chuỗi sinh ra thay vì từng bước.
- **Bỏ hẳn tự hồi quy**: các mô hình không tự hồi quy (FastSpeech, Glow-TTS, VITS, flow matching) không có exposure bias theo thời gian, đổi lại phải giải bài toán căn chỉnh và trường độ một cách tường minh ([Chương 21](../phan-5-mo-hinh-am-hoc/ch21-alignment.md), [Chương 22](../phan-5-mo-hinh-am-hoc/ch22-fastspeech.md)).

## 7.4. TTS là bài toán sinh có điều kiện một–nhiều

Đây là mục quan trọng nhất của chương. Từ đây đến hết mục, ta dùng ký hiệu TTS: $\mathbf{x}$ là văn bản (chuỗi âm vị hoặc ký tự), $\mathbf{Y} \in \mathbb{R}^{T \times F}$ là mel-spectrogram. Để các phép tính gọn, ta coi $\mathbf{Y}$ như một véc-tơ dài $TF$ chiều, chuẩn $\lVert \cdot \rVert$ là chuẩn Euclid của véc-tơ đó (tức chuẩn Frobenius của ma trận), và tạm giả định số khung $T$ đã được cố định (ví dụ biết trước trường độ). Ta sẽ quay lại giả định này ở cuối mục.

### 7.4.1. Văn bản không quyết định tiếng nói

Hãy nghĩ về những gì một bản ghi âm chứa mà văn bản **không** nói ra:

- **Trường độ và nhịp**: mỗi âm tiết dài bao nhiêu khung, ngắt hơi ở đâu, ngắt bao lâu.
- **Cao độ và ngữ điệu**: đường F0 cụ thể, độ rộng quãng cao độ, cách lên giọng cuối câu hỏi.
- **Năng lượng và nhấn mạnh**: từ nào được nhấn, nói to hay nhỏ.
- **Người nói và chất giọng**: nếu mô hình không được cho biết người nói.
- **Cảm xúc, phong cách**: đọc tin tức, kể chuyện, trò chuyện.
- **Kênh thu**: micro, phòng, nhiễu nền, tiếng vang.
- **Pha** của tín hiệu, nếu đầu ra là dạng sóng.

Về mặt lý thuyết thông tin, entropy của tiếng nói tách thành hai phần:

$$
H(\mathbf{Y}) = I(\mathbf{Y}; \mathbf{x}) + H(\mathbf{Y} \mid \mathbf{x}).
$$

Văn bản chỉ giải thích được phần thông tin tương hỗ $I(\mathbf{Y}; \mathbf{x})$. Phần còn lại, entropy có điều kiện $H(\mathbf{Y} \mid \mathbf{x})$, là "lượng tự do" của tiếng nói khi đã biết văn bản, và nó rất lớn. Đối tượng đúng mà mô hình TTS cần học vì thế không phải một hàm $f(\mathbf{x})$ mà là **phân phối có điều kiện $p(\mathbf{Y} \mid \mathbf{x})$**, và phân phối này **đa mode**: "ngắt hơi sau chữ thứ ba" và "không ngắt hơi" là hai mode tách biệt, không có cách đọc hợp lệ nào nằm "nửa chừng" giữa chúng. Một mô hình TTS tốt phải *lấy mẫu* trong lượng tự do đó, không được *lấy trung bình* qua nó. FastSpeech 2 gọi đây là **bài toán ánh xạ một–nhiều** (*one-to-many mapping problem*) của TTS (Ren và cộng sự, 2021).

### 7.4.2. Định lý: hồi quy L2 học trung bình có điều kiện

Giả sử ta làm điều tự nhiên nhất: huấn luyện một mạng $f_\theta(\mathbf{x})$ dự đoán $\mathbf{Y}$ bằng sai số bình phương. Rủi ro (*risk*) trên toàn phân phối là

$$
\mathcal{R}(f) = \mathbb{E}_{(\mathbf{x}, \mathbf{Y}) \sim p_{\text{data}}} \big[ \lVert \mathbf{Y} - f(\mathbf{x}) \rVert^2 \big].
$$

**Mệnh đề 7.1.** Gọi $\mathbf{m}(\mathbf{x}) = \mathbb{E}[\mathbf{Y} \mid \mathbf{x}]$ là trung bình có điều kiện. Với mọi hàm $f$,

$$
\mathcal{R}(f) = \underbrace{\mathbb{E}_{\mathbf{x}}\Big[ \operatorname{tr} \operatorname{Cov}(\mathbf{Y} \mid \mathbf{x}) \Big]}_{\text{không thể giảm}} + \underbrace{\mathbb{E}_{\mathbf{x}} \big[ \lVert \mathbf{m}(\mathbf{x}) - f(\mathbf{x}) \rVert^2 \big]}_{\text{sai số của } f},
$$

do đó hàm tối ưu là $f^\ast(\mathbf{x}) = \mathbf{m}(\mathbf{x}) = \mathbb{E}[\mathbf{Y} \mid \mathbf{x}]$.

*Chứng minh.* Cố định một $\mathbf{x}$ và đặt $\mathbf{c} = f(\mathbf{x})$. Thêm bớt $\mathbf{m}(\mathbf{x})$ trong chuẩn:

$$
\lVert \mathbf{Y} - \mathbf{c} \rVert^2
= \lVert \mathbf{Y} - \mathbf{m} \rVert^2 + 2 (\mathbf{Y} - \mathbf{m})^\top (\mathbf{m} - \mathbf{c}) + \lVert \mathbf{m} - \mathbf{c} \rVert^2 .
$$

Lấy kỳ vọng theo $p(\mathbf{Y} \mid \mathbf{x})$. Vì $\mathbf{m} - \mathbf{c}$ không phụ thuộc $\mathbf{Y}$ và $\mathbb{E}[\mathbf{Y} - \mathbf{m} \mid \mathbf{x}] = \mathbf{0}$, số hạng chéo triệt tiêu. Số hạng đầu là $\mathbb{E}[\lVert \mathbf{Y} - \mathbf{m} \rVert^2 \mid \mathbf{x}] = \sum_j \operatorname{Var}(Y_j \mid \mathbf{x}) = \operatorname{tr}\operatorname{Cov}(\mathbf{Y} \mid \mathbf{x})$. Vậy

$$
\mathbb{E}\big[\lVert \mathbf{Y} - \mathbf{c} \rVert^2 \,\big\vert\, \mathbf{x}\big] = \operatorname{tr}\operatorname{Cov}(\mathbf{Y} \mid \mathbf{x}) + \lVert \mathbf{m}(\mathbf{x}) - \mathbf{c} \rVert^2 .
$$

Lấy tiếp kỳ vọng theo $\mathbf{x}$ được đẳng thức cần chứng minh. Vì $f$ được chọn tự do tại từng $\mathbf{x}$, cực tiểu đạt được khi $\mathbf{c} = \mathbf{m}(\mathbf{x})$ với mọi $\mathbf{x}$. $\square$

Mệnh đề nói hai điều. Thứ nhất, **mạng hồi quy L2 hoàn hảo nhất cũng chỉ xuất ra trung bình của mọi cách đọc**. Thứ hai, ngay cả mạng hoàn hảo đó vẫn có loss bằng $\mathbb{E}_{\mathbf{x}}[\operatorname{tr}\operatorname{Cov}(\mathbf{Y} \mid \mathbf{x})]$: tổng phương sai có điều kiện, chính là "độ một–nhiều" của dữ liệu. Đường cong loss mel L2 của bạn chạm một cái sàn không phải vì mô hình yếu mà vì dữ liệu *vốn dĩ* không xác định.

**Mệnh đề 7.2 (L1 học trung vị).** Với một chiều $Y_j$ có phân phối có điều kiện liên tục, hàm $c \mapsto \mathbb{E}[\,\lvert Y_j - c \rvert \mid \mathbf{x}]$ đạt cực tiểu tại trung vị có điều kiện của $Y_j$.

*Chứng minh.* Viết $\mathbb{E}\lvert Y - c\rvert = \int_{-\infty}^{c} (c - y)\, p(y)\, dy + \int_{c}^{\infty} (y - c)\, p(y)\, dy$. Đạo hàm theo $c$ (quy tắc Leibniz, các số hạng biên bằng 0) cho $P(Y < c) - P(Y > c)$. Đạo hàm bằng 0 khi và chỉ khi $P(Y < c) = P(Y > c) = 1/2$, tức $c$ là trung vị. $\square$

Hai mệnh đề trên có một cách đọc xác suất mà [Chương 2](../phan-1-nen-tang/ch02-hoc-may-va-mang-no-ron.md) đã giới thiệu: cực tiểu L2 tương đương MLE với mô hình $p_\theta(\mathbf{Y} \mid \mathbf{x}) = \mathcal{N}(f_\theta(\mathbf{x}), \sigma^2 \mathbf{I})$, cực tiểu L1 tương đương MLE với phân phối Laplace. Như vậy "hồi quy mel bằng L1/L2" thực chất là **MLE với một họ mô hình có ba giả định ngầm**:

1. **Đơn mode**: mỗi văn bản chỉ có một "cách đọc trung tâm".
2. **Độc lập giữa các chiều**: $TF$ ô của mel-spectrogram độc lập với nhau khi biết $\mathbf{x}$ (hiệp phương sai đường chéo).
3. **Phương sai cố định**: mọi ô, mọi văn bản đều có cùng mức bất định.

Tiếng nói vi phạm cả ba. Ta đã gặp đúng tình huống này ở mục 7.2.2: MLE với họ quá nghèo đặt điểm trung tâm vào giữa các mode.

### 7.4.3. Ba biểu hiện của hiện tượng trơn quá mức

**(a) Trung bình rơi vào vùng mật độ thấp.** Trung bình của "ngắt hơi" và "không ngắt hơi" là một khoảng "nửa im lặng" mà không người thật nào tạo ra. Trung bình của một câu lên giọng và một câu xuống giọng là một câu đều đều. Vì $p(\mathbf{Y} \mid \mathbf{x})$ đa mode, $\mathbb{E}[\mathbf{Y} \mid \mathbf{x}]$ có thể có mật độ rất thấp dưới chính phân phối dữ liệu: nó là một điểm "không có thật".

**(b) Lệch thời gian và tần số làm nhòe cấu trúc.** Phổ của một khung hữu thanh có các đỉnh hài tại bội số của F0. Khi các cách đọc khác nhau có F0 khác nhau, đỉnh hài thứ $h$ nằm ở vị trí $h \cdot f_0$, và độ phân tán của vị trí này **tăng tỉ lệ với $h$**: nếu $f_0$ có độ lệch chuẩn $s$ thì hài thứ $h$ có độ lệch chuẩn $hs$. Lấy trung bình, các hài thấp còn lờ mờ, các hài cao nhòe hẳn thành một dải phẳng (Bài tập 4 tính chính xác điều này). Âm thanh tổng hợp từ phổ như vậy nghe "đục", "rè", thiếu độ trong của giọng thật. Hiện tượng tương tự xảy ra theo trục thời gian: các ranh giới âm vị lệch nhau vài khung giữa các lần đọc, nên trung bình làm chuyển tiếp bị mềm nhũn.

**(c) Co rút phương sai.** Áp dụng định luật phương sai toàn phần cho từng chiều $j$:

$$
\operatorname{Var}(Y_j) = \underbrace{\mathbb{E}_{\mathbf{x}}\big[\operatorname{Var}(Y_j \mid \mathbf{x})\big]}_{\text{phần mô hình L2 bỏ mất}} + \underbrace{\operatorname{Var}_{\mathbf{x}}\big(\mathbb{E}[Y_j \mid \mathbf{x}]\big)}_{\text{phần mô hình L2 giữ được}} .
$$

Đầu ra của bộ hồi quy tối ưu, $\mathbb{E}[Y_j \mid \mathbf{x}]$, chỉ có phần phương sai thứ hai. Do đó **phương sai của đặc trưng sinh ra luôn nhỏ hơn phương sai của đặc trưng thật**, một cách có hệ thống. Cộng đồng tổng hợp tiếng nói thống kê đã nhận ra điều này từ thời mô hình HMM, gọi đại lượng liên quan là *phương sai toàn cục* (*global variance*, GV) và đề xuất thuật toán sinh tham số có bù GV (Toda và Tokuda, 2007). Với F0, co rút phương sai nghe như giọng đều đều, "robot"; với phổ, nó làm mất độ tương phản giữa đỉnh và thung lũng formant.

**Trường hợp cực đoan: pha.** Nếu đầu ra là dạng sóng và các bản ghi chỉ khác nhau ở pha, trung bình của chúng triệt tiêu gần như hoàn toàn:

```python
import numpy as np

rng = np.random.default_rng(0)
fs, f, n = 16000, 220.0, 400                         # 400 mẫu = 25 ms
t = np.arange(n) / fs
# 1000 "bản ghi" cùng một nội dung (cùng tần số, cùng biên độ) nhưng pha ngẫu nhiên
phases = rng.uniform(0, 2 * np.pi, size=(1000, 1))
waves = np.sin(2 * np.pi * f * t[None, :] + phases)    # (1000, L)
mean_wave = waves.mean(0)                              # nghiệm tối ưu của hồi quy L2

rms = lambda s: np.sqrt(np.mean(s ** 2))
print(f"RMS một bản ghi: {rms(waves[0]):.3f}   RMS trung bình L2: {rms(mean_wave):.3f}")
assert rms(mean_wave) < 0.1 * rms(waves[0])            # tín hiệu "trung bình" gần như im lặng
```

RMS của tín hiệu "trung bình" chỉ còn khoảng 0,03 so với 0,71 của một bản ghi. Đây là lý do không vocoder hiện đại nào hồi quy dạng sóng bằng L2 thuần: chúng dùng mô hình likelihood (WaveNet), flow (WaveGlow), diffusion (DiffWave) hoặc loss đối kháng (HiFi-GAN), còn các loss phụ trợ thì đặt trên **biên độ** phổ, đại lượng bất biến với pha ([Chương 25](../phan-6-vocoder/ch25-gan-vocoder.md)).

### 7.4.4. Thí nghiệm: hồi quy một–nhiều trên "mel" giả

Thí nghiệm dưới đây dựng một tập dữ liệu giả lập đúng cấu trúc vấn đề. Mỗi "token" $\mathbf{x}$ quyết định đường bao phổ; F0 thì có hai mode (tượng trưng cho hai cách đọc của cùng một nội dung). Ta huấn luyện cùng một mạng với L2 và L1 rồi đo: (i) dự đoán L2 có đúng là trung bình có điều kiện không, (ii) độ "sắc nét" của phổ, (iii) tỉ lệ phương sai mà mô hình tái tạo được.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
F, n_tok, n_per_tok = 64, 4, 512          # F kênh "mel", 4 token, 512 mẫu/token
k = torch.arange(F, dtype=torch.float32)  # chỉ số bin tần số

def frame(f0, tilt):
    """Một khung log-phổ giả: các đỉnh hài tại bội số của f0 (đơn vị: bin)."""
    h = torch.arange(1, 12, dtype=torch.float32)                 # 11 hài
    peaks = torch.exp(-0.5 * ((k[None, :] - h[:, None] * f0) / 0.6) ** 2)
    env = torch.exp(-k / tilt)                                   # đường bao phổ
    return torch.log(1e-3 + env * peaks.sum(0))                  # (F,)

# Dữ liệu một-nhiều: cùng một token, F0 có hai "mode" (ví dụ hai cách đọc)
X, Y = [], []
for tok in range(n_tok):
    tilt = 6.0 * 2 ** tok                                         # token quyết định đường bao
    for _ in range(n_per_tok):
        f0 = (5.0 if torch.rand(()) < 0.5 else 7.0) + 0.2 * torch.randn(())
        X.append(tok)
        Y.append(frame(f0, tilt))
X = nn.functional.one_hot(torch.tensor(X), n_tok).float()       # (n, 4)
Y = torch.stack(Y)                                               # (n, F)

def fit(loss_fn, steps=600):
    net = nn.Sequential(nn.Linear(n_tok, 128), nn.ReLU(), nn.Linear(128, F))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(steps):
        opt.zero_grad()
        loss_fn(net(X), Y).backward()
        opt.step()
    return net(X).detach()                                       # (n, F)

pred_l2 = fit(nn.functional.mse_loss)
pred_l1 = fit(nn.functional.l1_loss)

# Kiểm chứng: dự đoán L2 hội tụ về trung bình có điều kiện E[Y | x]
cond_mean = torch.stack([Y[X[:, t] == 1].mean(0) for t in range(n_tok)])  # (4, F)
gap = (pred_l2 - cond_mean[X.argmax(1)]).abs().max().item()
print(f"max |f_L2(x) - E[Y|x]| = {gap:.3f}")

def contrast(Z):
    """Độ 'sắc nét' của phổ: độ lệch chuẩn theo trục tần số, lấy trung bình."""
    return Z.std(dim=1).mean().item()

print(f"độ tương phản phổ  dữ liệu: {contrast(Y):.2f}  L2: {contrast(pred_l2):.2f}"
      f"  L1: {contrast(pred_l1):.2f}")

# Định luật phương sai toàn phần: Var(Y) = E[Var(Y|x)] + Var(E[Y|x])
total = Y.var(0, unbiased=False).sum()
explained = pred_l2.var(0, unbiased=False).sum()
print(f"tỉ lệ phương sai mà hồi quy L2 tái tạo được: {(explained / total).item():.2%}")
assert gap < 0.1 and contrast(pred_l2) < contrast(Y)
```

Trên máy của người biên soạn, kết quả là: dự đoán L2 trùng với trung bình có điều kiện đến ba chữ số thập phân; độ tương phản phổ giảm từ khoảng 2,0 (dữ liệu) xuống khoảng 1,2 (L2) và 1,65 (L1); và mô hình L2 chỉ tái tạo được khoảng 20% tổng phương sai của dữ liệu. Phần 80% còn lại chính là $\mathbb{E}_{\mathbf{x}}[\operatorname{Var}(\mathbf{Y} \mid \mathbf{x})]$, phần mà văn bản (token) không quyết định. Con số cụ thể có thể dao động nhẹ theo phiên bản thư viện, nhưng các quan hệ thứ tự thì không.

### 7.4.5. Vì sao L1 không cứu được

Thực tế nhiều hệ thống dùng L1 cho mel và thấy kết quả bớt nhòe hơn L2, như thí nghiệm trên cũng cho thấy. Lý do là trung vị bền vững hơn trung bình với các mẫu ngoại lai, và với phân phối lệch, trung vị nằm gần phần "đông dân" hơn. Nhưng L1 không giải quyết tận gốc:

- Trung vị được chọn **độc lập cho từng chiều**. Ghép trung vị của 80 kênh mel lại không cho một khung phổ nào có thật: kênh 10 có thể lấy giá trị từ "mode F0 thấp", kênh 11 từ "mode F0 cao".
- Với phân phối hai mode đối xứng, trung vị có thể là **bất kỳ điểm nào** trong khoảng trống giữa hai mode, vẫn là vùng không có dữ liệu.
- L1 vẫn là MLE với một họ đơn mode, độc lập giữa các chiều. Ba giả định ở mục 7.4.2 vẫn nguyên đó.

### 7.4.6. Hai lối thoát

Phân tích của Ren và cộng sự (2022) về hiện tượng trơn quá mức trong TTS đi đến một kết luận tương tự với phân tích trên: có hai hướng bổ sung cho nhau, *đơn giản hóa phân phối dữ liệu* và *tăng sức biểu đạt của mô hình*.

**Lối A: giảm bất định bằng cách bổ sung điều kiện.** Điều kiện hóa không bao giờ làm tăng entropy: $H(\mathbf{Y} \mid \mathbf{x}, \mathbf{c}) \le H(\mathbf{Y} \mid \mathbf{x})$, và tương tự phần phương sai không thể giảm trong Mệnh đề 7.1 cũng không tăng khi thêm điều kiện (Bài tập 6). Nếu ta cung cấp thêm $\mathbf{c}$ chứa đúng những thông tin văn bản còn thiếu, phân phối $p(\mathbf{Y} \mid \mathbf{x}, \mathbf{c})$ hẹp lại và gần đơn mode, khi đó hồi quy bớt có hại. Đây là triết lý của:

- **Trường độ, cao độ, năng lượng tường minh** trong FastSpeech 2 (bộ thích nghi phương sai, *variance adaptor*), lấy giá trị thật khi huấn luyện ([Chương 22](../phan-5-mo-hinh-am-hoc/ch22-fastspeech.md)).
- **Embedding người nói** và **embedding phong cách/tham chiếu** ([Chương 28](../phan-7-e2e-bieu-cam/ch28-da-nguoi-noi-voice-cloning.md), [Chương 29](../phan-7-e2e-bieu-cam/ch29-prosody-cam-xuc-phong-cach.md)).
- **Các khung trước đó** trong mô hình tự hồi quy như Tacotron: biết $\mathbf{y}_{<t}$ thì khung $\mathbf{y}_t$ bớt bất định đi rất nhiều. Đó là một lý do Tacotron 2 cho chất lượng tốt dù vẫn dùng loss hồi quy cho từng khung.

Nhưng lối A có một cái giá: khi suy luận, ta phải *dự đoán* $\mathbf{c}$ từ văn bản, và bộ dự đoán $\mathbf{c}$ (ví dụ bộ dự đoán cao độ) lại đối mặt với chính bài toán một–nhiều. Nếu nó cũng được huấn luyện bằng hồi quy, ngôn điệu lại bị "trung bình hóa": câu đọc rõ ràng, phổ sắc nét hơn, nhưng ngữ điệu đều đều. Vấn đề chỉ bị đẩy sang chỗ khác, dù sang một không gian ít chiều hơn và dễ xử lý hơn.

**Lối B: mô hình hóa toàn bộ phân phối có điều kiện.** Thay họ "Gauss đơn mode, đường chéo, phương sai cố định" bằng một họ đủ giàu, rồi **lấy mẫu** từ nó. Đó là nội dung của Chương 8–12, và là cách hầu hết hệ thống TTS hiện đại vận hành:

```mermaid
flowchart LR
    P["Bài toán một–nhiều: p(Y∣x) đa mode"] --> R["Hồi quy L1/L2: ra trung bình, phổ nhòe"]
    P --> A["Lối A: thêm điều kiện (trường độ, F0, người nói, tham chiếu)"]
    P --> B["Lối B: mô hình sinh đầy đủ"]
    B --> B1["Tự hồi quy / token rời rạc"]
    B --> B2["VAE, CVAE"]
    B --> B3["GAN"]
    B --> B4["Flow, diffusion, flow matching"]
    A --> H["Hệ thống thực tế: kết hợp cả hai"]
    B1 --> H
    B2 --> H
    B3 --> H
    B4 --> H
```

Trên thực tế, các hệ thống tốt kết hợp cả hai lối. VITS dùng biến ẩn (lối B) cùng bộ dự đoán trường độ ngẫu nhiên; Matcha-TTS dùng trường độ tường minh (lối A) cùng bộ giải mã flow matching (lối B); các hệ thống codec LM điều kiện hóa trên đoạn âm thanh mẫu (lối A) và lấy mẫu token (lối B).

**Về giả định $T$ cố định.** Thực tế độ dài $T$ cũng là một biến ngẫu nhiên: cùng câu có thể dài 2,1 giây hoặc 2,6 giây. Khi $T$ khác nhau, sai số L2 giữa hai chuỗi thậm chí *không xác định* nếu không có căn chỉnh. Đây là lý do các mô hình không tự hồi quy phải xử lý trường độ riêng (trường độ thật khi huấn luyện, bộ dự đoán khi suy luận), và là lý do bài toán căn chỉnh xứng đáng có hẳn một chương ([Chương 21](../phan-5-mo-hinh-am-hoc/ch21-alignment.md)). Phân tích trong mục này áp dụng *sau khi* đã cố định căn chỉnh; trường độ chỉ là thêm một tầng một–nhiều nữa.

!!! note "Góc tiếng Việt"
    - **Thanh điệu là "phần chắc chắn", cách thể hiện thanh điệu thì không.** Chữ viết tiếng Việt ghi rõ thanh điệu, nên *loại* thanh gần như được văn bản quyết định. Nhưng đường F0 cụ thể của thanh hỏi (vị trí điểm trũng, độ sâu), hiện tượng thanh quản hóa của thanh ngã và thanh nặng, độ rộng quãng của thanh sắc khi ở cuối câu hỏi đều thay đổi theo người nói, tốc độ và ngữ điệu câu ([Chương 15](../phan-3-tin-hieu-ngu-am/ch15-ngu-am-tieng-viet.md), [Chương 16](../phan-3-tin-hieu-ngu-am/ch16-ngon-dieu.md)). Hồi quy làm trung bình hóa thời điểm của điểm trũng, khiến đường hỏi bị "là" phẳng ra, dễ nghe nhầm với thanh khác; trung bình hóa các xung thanh môn không đều của giọng thanh quản hóa sinh ra một phổ nhòe không giống giọng thật nào.
    - **Dữ liệu trộn phương ngữ làm bài toán một–nhiều nặng thêm.** Nếu tập dữ liệu trộn giọng Bắc và giọng Nam mà không có nhãn, cùng một chữ có thể có hai cách phát âm (ví dụ thanh hỏi và thanh ngã nhập làm một ở nhiều giọng miền Nam). Bộ hồi quy sẽ sinh ra một giọng "lai" không thuộc vùng nào. Cách chữa thuộc lối A: thêm nhãn phương ngữ hoặc người nói, hoặc chọn một phương ngữ mục tiêu ngay từ khâu dữ liệu ([Chương 36](../phan-9-thuc-tien/ch36-du-lieu-tts.md)).
    - **Dữ liệu ít.** Với tiếng Việt, tập dữ liệu một người nói chất lượng phòng thu thường chỉ vài giờ đến vài chục giờ. Khi đó việc cung cấp điều kiện tường minh (trường độ, F0) giúp mô hình học nhanh và ổn định hơn; nhưng nếu muốn ngôn điệu tự nhiên, vẫn cần một thành phần sinh thực sự cho các đại lượng đó.

## 7.5. Mạng mật độ hỗn hợp (MDN)

### 7.5.1. Trực giác

Cách sửa trực tiếp nhất cho giả định "đơn mode" là cho mạng xuất ra **một hỗn hợp** các phân phối thay vì một điểm. Đây là ý tưởng của **mạng mật độ hỗn hợp** (*mixture density network*, MDN) do Bishop (1994) đề xuất: một mạng nơ-ron bình thường, nhưng lớp đầu ra dự đoán *tham số của một mô hình hỗn hợp Gauss* có điều kiện trên đầu vào.

### 7.5.2. Hình thức hóa

Với $K$ thành phần và đầu ra $\mathbf{y} \in \mathbb{R}^d$:

$$
p_\theta(\mathbf{y} \mid \mathbf{x}) = \sum_{k=1}^{K} \pi_k(\mathbf{x})\, \mathcal{N}\big(\mathbf{y};\, \boldsymbol{\mu}_k(\mathbf{x}),\, \operatorname{diag}(\boldsymbol{\sigma}_k^2(\mathbf{x}))\big),
$$

trong đó mạng xuất ra $K$ logit $\to$ softmax để có trọng số $\pi_k \ge 0$, $\sum_k \pi_k = 1$; $K$ véc-tơ trung bình $\boldsymbol{\mu}_k$; và $K$ véc-tơ $\log \boldsymbol{\sigma}_k$ (dự đoán log để bảo đảm dương). Hàm mất mát là log-likelihood âm:

$$
\mathcal{L}(\theta) = -\log \sum_{k=1}^{K} \exp\Big( \log \pi_k(\mathbf{x}) + \log \mathcal{N}\big(\mathbf{y};\, \boldsymbol{\mu}_k, \boldsymbol{\sigma}_k^2\big) \Big),
$$

viết dưới dạng $-\operatorname{logsumexp}$ để ổn định số học. Gradient của $\mathcal{L}$ theo $\boldsymbol{\mu}_k$ có trọng số là **độ trách nhiệm** (*responsibility*) $\gamma_k = \pi_k \mathcal{N}_k / \sum_j \pi_j \mathcal{N}_j$, xác suất hậu nghiệm rằng $\mathbf{y}$ đến từ thành phần $k$. Mỗi thành phần vì thế chỉ "kéo" về phía những mẫu nó chịu trách nhiệm, thay vì mọi thành phần cùng bị kéo về trung bình chung. Để lấy mẫu: chọn $k \sim \operatorname{Categorical}(\boldsymbol{\pi})$, rồi lấy $\mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\sigma}_k^2)$.

### 7.5.3. Mã minh họa

Ví dụ kinh điển của Bishop là một **bài toán ngược**: biến $y$ sinh ra $x = y + 0.3 \sin(2\pi y) + \text{nhiễu}$; ta cần dự đoán $y$ từ $x$. Với một số giá trị $x$, có tới ba giá trị $y$ hợp lệ. Đây là một phiên bản thu nhỏ của TTS: từ "kết quả" (văn bản) suy ngược ra "nguyên nhân" (tiếng nói) không duy nhất.

```python
import math
import torch
import torch.nn as nn

torch.manual_seed(0)

# Bài toán ngược kinh điển (Bishop, 1994): ánh xạ x -> y đa trị
y = torch.rand(2000, 1)                                       # (n, 1)
x = y + 0.3 * torch.sin(2 * math.pi * y) + 0.1 * (2 * torch.rand_like(y) - 1)

class MDN(nn.Module):
    """Mạng mật độ hỗn hợp: x -> (pi_k, mu_k, sigma_k), k = 1..K."""
    def __init__(self, n_comp=3, hidden=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.head = nn.Linear(hidden, 3 * n_comp)

    def forward(self, x):                                     # x: (B, 1)
        logit_pi, mu, log_sigma = self.head(self.body(x)).chunk(3, dim=-1)  # mỗi cái (B, K)
        return logit_pi, mu, log_sigma.clamp(-7.0, 3.0)

def mdn_nll(logit_pi, mu, log_sigma, y):
    """-log sum_k pi_k N(y; mu_k, sigma_k^2), tính ổn định bằng logsumexp."""
    log_pi = torch.log_softmax(logit_pi, dim=-1)                          # (B, K)
    log_n = (-0.5 * ((y - mu) / log_sigma.exp()) ** 2
             - log_sigma - 0.5 * math.log(2 * math.pi))                   # (B, K)
    return -torch.logsumexp(log_pi + log_n, dim=-1).mean()

mdn = MDN()
reg = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
opt = torch.optim.Adam(list(mdn.parameters()) + list(reg.parameters()), lr=3e-3)
for step in range(3000):
    opt.zero_grad()
    loss_mdn = mdn_nll(*mdn(x), y)
    loss_reg = nn.functional.mse_loss(reg(x), y)
    (loss_mdn + loss_reg).backward()
    opt.step()

x0 = torch.tensor([[0.5]])                     # tại x = 0.5 có ba nhánh y hợp lệ
with torch.no_grad():
    logit_pi, mu, log_sigma = mdn(x0)
    pi = torch.softmax(logit_pi, -1)
    print("hồi quy L2  :", f"{reg(x0).item():.3f}")
    for p_k, m_k, s_k in zip(pi[0], mu[0], log_sigma[0].exp()):
        print(f"thành phần  : pi={p_k:.2f}  mu={m_k:.3f}  sigma={s_k:.3f}")
    # Lấy mẫu từ MDN: chọn thành phần theo pi rồi lấy mẫu Gauss
    comp = torch.multinomial(pi.expand(5, -1), 1)                       # (5, 1)
    samples = mu[0][comp] + log_sigma[0].exp()[comp] * torch.randn(5, 1)
    print("mẫu MDN     :", [round(v, 3) for v in samples.squeeze(1).tolist()])
print(f"NLL MDN cuối: {loss_mdn.item():.3f}")
```

Tại $x = 0.5$, phương trình $y + 0.3\sin(2\pi y) = 0.5$ có ba nghiệm xấp xỉ $0.22$, $0.5$ và $0.78$. Bộ hồi quy L2 trả về một con số gần $0.48$, xấp xỉ trung bình có trọng số của ba nhánh. MDN học được ba thành phần với trung bình gần $0.22$, $0.49$, $0.77$; trọng số của chúng phản ánh mật độ dữ liệu trên từng nhánh (ở nhánh giữa, $x$ thay đổi chậm hơn theo $y$ nên có nhiều giá trị $y$ cho ra $x \approx 0.5$ hơn), và mẫu lấy từ MDN rơi vào cả ba nhánh.

### 7.5.4. MDN trong TTS và giới hạn của nó

MDN và các biến thể hỗn hợp từng là lựa chọn hấp dẫn cho tổng hợp tiếng nói. Zen và Senior (2014) dùng MDN sâu làm mô hình âm học trong tổng hợp tham số thống kê; các vocoder họ WaveNet về sau dùng hỗn hợp logistic rời rạc hóa cho từng mẫu âm thanh. Nhưng MDN không mở rộng tốt lên một khung mel 80 chiều, chứ chưa nói cả mel-spectrogram:

- Với các thành phần có hiệp phương sai đường chéo, một hỗn hợp $K$ thành phần chỉ biểu diễn được $K$ "khung nguyên mẫu" cộng nhiễu độc lập. Số mode thực của không gian tiếng nói tăng **tổ hợp** theo số yếu tố biến thiên (F0 × trường độ × năng lượng × chất giọng...), nên cần $K$ lớn theo cấp số mũ.
- Mẫu lấy từ một thành phần đường chéo có nhiễu **độc lập giữa các kênh**, nghe như tiếng xì chồng lên phổ; trong khi biến thiên thật của tiếng nói có tương quan mạnh giữa các kênh và các khung.
- Huấn luyện dễ gặp sự cố: một thành phần co $\sigma \to 0$ quanh một điểm dữ liệu làm likelihood tiến tới vô cùng; một số thành phần "chết" (trọng số $\pi_k \approx 0$).

MDN vì vậy hợp với đầu ra **ít chiều** (một mẫu âm thanh, một giá trị F0, một trường độ). Để mô hình hóa các đối tượng nhiều chiều có tương quan phức tạp, ta cần những ý tưởng mạnh hơn: biến ẩn liên tục ([Chương 8](ch08-vae.md)), biến đổi nhiễu bằng mạng ([Chương 9](ch09-gan.md), [Chương 10](ch10-normalizing-flows.md)), hoặc khử nhiễu lặp ([Chương 11](ch11-diffusion.md), [Chương 12](ch12-flow-matching.md)).

## 7.6. Đánh giá mô hình sinh

Đánh giá một bộ phân loại thì dễ: so nhãn dự đoán với nhãn thật. Đánh giá một mô hình sinh khó hơn nhiều, vì không có "đáp án đúng" cho một mẫu sinh ra: ta cần biết cả phân phối mẫu có giống phân phối dữ liệu không. Mỗi thước đo dưới đây chỉ nhìn được một mặt của vấn đề.

### 7.6.1. Log-likelihood trên tập kiểm tra

Với các mô hình tính được mật độ, thước đo tự nhiên là log-likelihood âm (NLL) trung bình trên dữ liệu chưa thấy. Với dữ liệu rời rạc, người ta thường quy về **bit trên mỗi chiều** (*bits per dimension*): $\text{bpd} = \text{NLL} / (d \ln 2)$ với NLL tính bằng nat và $d$ là số chiều. Có ba cái bẫy:

- **Mật độ liên tục phụ thuộc phép đổi thang.** Nếu $\mathbf{y}' = a\mathbf{y}$ với $\mathbf{y} \in \mathbb{R}^d$ thì $\log p(\mathbf{y}') = \log p(\mathbf{y}) - d \log \lvert a \rvert$. Đổi cách chuẩn hóa log-mel (cơ số log, trừ trung bình, chia độ lệch chuẩn) làm NLL dịch đi một hằng số. Chỉ so NLL khi tiền xử lý giống hệt nhau.
- **Dữ liệu thực ra là rời rạc.** Âm thanh 16 bit là rời rạc; đặt mật độ liên tục lên đó có thể cho likelihood lớn vô hạn bằng cách co mật độ quanh các giá trị nguyên. Kỹ thuật *dequantization* xử lý vấn đề này ([Chương 10](ch10-normalizing-flows.md)).
- **Likelihood và chất lượng mẫu có thể gần như độc lập** (Theis và cộng sự, 2016). Xét mô hình hỗn hợp $p_{\text{mix}} = \alpha\, p_{\text{tốt}} + (1-\alpha)\, q_{\text{rác}}$. Vì $\log p_{\text{mix}}(\mathbf{x}) \ge \log \alpha + \log p_{\text{tốt}}(\mathbf{x})$, NLL của nó kém mô hình tốt **tối đa** $-\log \alpha$ nat mỗi mẫu. Với $\alpha = 0.01$ đó là khoảng 4,6 nat. Chia cho số chiều của một mel-spectrogram cỡ $200 \times 80 = 16\,000$, mức chênh chỉ khoảng $3 \times 10^{-4}$ nat mỗi chiều, gần như không nhìn thấy được, trong khi **99% mẫu sinh ra là rác**. Ngược lại, một mô hình cho mẫu tuyệt đẹp nhưng bỏ sót một mode sẽ có NLL rất tệ (thậm chí vô cùng) trên những mẫu thuộc mode bị bỏ sót.

### 7.6.2. Khoảng cách giữa các phân phối embedding: FID và FAD

Với các mô hình không tính được mật độ (GAN) hoặc khi quan tâm đến chất lượng mẫu, người ta so sánh **tập mẫu sinh** với **tập mẫu thật** trong một không gian embedding do một mạng được huấn luyện trước cung cấp. Khớp mỗi tập một phân phối Gauss $\mathcal{N}(\boldsymbol{\mu}_r, \boldsymbol{\Sigma}_r)$ và $\mathcal{N}(\boldsymbol{\mu}_g, \boldsymbol{\Sigma}_g)$, rồi tính **khoảng cách Fréchet** (khoảng cách Wasserstein-2 giữa hai Gauss):

$$
\operatorname{FD} = \lVert \boldsymbol{\mu}_r - \boldsymbol{\mu}_g \rVert^2 + \operatorname{tr}\Big( \boldsymbol{\Sigma}_r + \boldsymbol{\Sigma}_g - 2\big(\boldsymbol{\Sigma}_r \boldsymbol{\Sigma}_g\big)^{1/2} \Big).
$$

Với ảnh, embedding lấy từ mạng Inception và thước đo có tên FID (Heusel và cộng sự, 2017); với âm thanh, phiên bản tương ứng là FAD (*Fréchet Audio Distance*), ban đầu dùng embedding VGGish (Kilgour và cộng sự, 2019). Số hạng vết của công thức phạt đúng hiện tượng co rút phương sai ở mục 7.4.3:

```python
import numpy as np
from scipy import linalg

def frechet_distance(feat_a, feat_b):
    """Khoảng cách Fréchet giữa hai Gauss khớp với hai tập embedding (n, d)."""
    mu_a, mu_b = feat_a.mean(0), feat_b.mean(0)
    cov_a = np.cov(feat_a, rowvar=False)
    cov_b = np.cov(feat_b, rowvar=False)
    covmean = linalg.sqrtm(cov_a @ cov_b)
    covmean = covmean.real                       # bỏ phần ảo rất nhỏ do sai số số học
    return float(np.sum((mu_a - mu_b) ** 2) + np.trace(cov_a + cov_b - 2 * covmean))

rng = np.random.default_rng(0)
real = rng.normal(size=(2000, 8))                # embedding của âm thanh thật (giả lập)
diverse = rng.normal(size=(2000, 8))             # mô hình tốt: cùng phân phối
smooth = 0.4 * rng.normal(size=(2000, 8))        # mô hình "trơn": đúng trung bình, thiếu phương sai
print(f"FD(thật, đa dạng) = {frechet_distance(real, diverse):.3f}")
print(f"FD(thật, trơn)    = {frechet_distance(real, smooth):.3f}")
assert frechet_distance(real, smooth) > 10 * frechet_distance(real, diverse)
```

Mô hình "trơn" có trung bình đúng nhưng độ lệch chuẩn chỉ bằng 0,4 lần; FD của nó xấp xỉ $8 \times (1 + 0.16 - 2 \times 0.4) = 2.88$, gấp hơn trăm lần mô hình tốt. Nhưng FD/FAD cũng có điểm mù: giá trị phụ thuộc mạnh vào mạng embedding; ước lượng bị chệch khi số mẫu ít, nên chỉ so sánh khi dùng cùng số mẫu; giả định Gauss trong không gian embedding chỉ là xấp xỉ. Quan trọng nhất với TTS: **FAD không kiểm tra tính đúng điều kiện**. Một mô hình phớt lờ văn bản và chỉ sinh ra tiếng nói tự nhiên ngẫu nhiên vẫn có thể có FAD rất tốt.

### 7.6.3. Precision và recall cho mô hình sinh

FID/FAD gộp hai loại lỗi vào một con số. Sajjadi và cộng sự (2018), rồi Kynkäänniemi và cộng sự (2019), đề xuất tách chúng ra: **precision** là tỉ lệ mẫu sinh ra nằm trong vùng của dữ liệu thật (đo *chất lượng*), **recall** là tỉ lệ dữ liệu thật nằm trong vùng mà mô hình sinh ra được (đo *độ phủ*, *đa dạng*). Theo ngôn ngữ của mục 7.2.2: mô hình tìm mode có precision cao, recall thấp; mô hình phủ mode nhưng họ quá nghèo có recall cao, precision thấp.

### 7.6.4. Đánh giá riêng cho TTS

Với TTS, một mô hình sinh phải được đánh giá trên nhiều trục cùng lúc, chi tiết ở [Chương 38](../phan-9-thuc-tien/ch38-danh-gia-tts.md):

| Thước đo | Đo cái gì | Điểm mù |
|---|---|---|
| MOS, CMOS (người nghe chấm) | Độ tự nhiên cảm nhận | Tốn kém, phương sai cao, phụ thuộc thiết kế thí nghiệm |
| WER/CER qua ASR | Độ dễ hiểu, tính đúng nội dung | Không đo độ tự nhiên; phụ thuộc chất lượng ASR và chuẩn hóa văn bản |
| Độ tương đồng người nói (cosine embedding) | Tính đúng điều kiện người nói | Phụ thuộc mô hình embedding |
| Thống kê ngôn điệu (phương sai F0, phân phối trường độ, GV) | Hiện tượng trơn quá mức | Không nói gì về tính đúng nội dung |
| Độ đa dạng giữa nhiều mẫu của cùng một câu | Khả năng mô hình hóa một–nhiều | Đa dạng cao chưa chắc đã hợp lý |
| NLL (nếu tính được) | Độ khớp phân phối | Không tương quan chặt với chất lượng nghe |
| FAD | Độ giống phân phối tiếng nói thật | Không kiểm tra điều kiện văn bản |
| RTF, độ trễ | Tốc độ | Không nói gì về chất lượng |

Thước đo "độ đa dạng giữa nhiều mẫu" xứng đáng được nhấn mạnh vì nó kiểm tra trực tiếp luận điểm trung tâm của chương: sinh nhiều lần cùng một câu với các hạt giống ngẫu nhiên khác nhau, rồi đo độ biến thiên của tổng trường độ và đường F0. Một mô hình hồi quy thuần cho độ biến thiên bằng 0; một mô hình sinh tốt cho độ biến thiên gần với độ biến thiên giữa các lần đọc thật của cùng một người (nếu có dữ liệu như vậy).

## 7.7. Bản đồ: các họ mô hình sinh trong TTS

Bảng dưới xếp các hệ thống TTS nổi tiếng theo họ mô hình sinh *chủ đạo* của thành phần tương ứng. Hầu hết hệ thống hiện đại là **lai**: VITS kết hợp VAE, flow và GAN; CosyVoice kết hợp mô hình ngôn ngữ trên token, flow matching và một vocoder GAN. Danh sách phản ánh hiểu biết tính đến thời điểm biên soạn (2026); các hệ thống mới xuất hiện rất nhanh.

| Họ | Mục tiêu huấn luyện | Lấy mẫu | Hệ thống TTS tiêu biểu | Chương |
|---|---|---|---|---|
| Tự hồi quy trên giá trị liên tục | NLL chính xác (hoặc hồi quy từng bước) | Tuần tự, chậm với dạng sóng | Vocoder: WaveNet, WaveRNN, LPCNet. Mô hình âm học: Tacotron, Tacotron 2 (loss hồi quy cho từng khung) | [20](../phan-5-mo-hinh-am-hoc/ch20-tacotron.md), [24](../phan-6-vocoder/ch24-vocoder-tu-hoi-quy-va-flow.md) |
| Mô hình ngôn ngữ trên token rời rạc | Cross-entropy trên token | Tuần tự theo token | VALL-E, SPEAR-TTS, XTTS, tầng mô hình ngôn ngữ của CosyVoice | [31](../phan-8-mo-hinh-lon/ch31-audio-codec-speech-token.md)–[34](../phan-8-mo-hinh-lon/ch34-he-thong-lai-hien-dai.md) |
| VAE / CVAE | Cận dưới ELBO | Một lượt | VITS (CVAE kết hợp flow và GAN), bộ mã hóa tham chiếu dạng VAE cho ngôn điệu, NaturalSpeech | [8](ch08-vae.md), [27](../phan-7-e2e-bieu-cam/ch27-vits.md), [29](../phan-7-e2e-bieu-cam/ch29-prosody-cam-xuc-phong-cach.md) |
| GAN | Trò chơi đối kháng, thường kèm loss phổ phụ trợ | Một lượt, rất nhanh | Vocoder: MelGAN, Parallel WaveGAN, HiFi-GAN, BigVGAN, Vocos. Loss đối kháng trong VITS, StyleTTS 2 | [9](ch09-gan.md), [25](../phan-6-vocoder/ch25-gan-vocoder.md) |
| Normalizing flow | NLL chính xác qua đổi biến | Một lượt (coupling) hoặc tuần tự (một số flow tự hồi quy) | WaveGlow, FloWaveNet, Parallel WaveNet (IAF, chưng cất), Glow-TTS, Flowtron, khối flow trong VITS | [10](ch10-normalizing-flows.md), [23](../phan-5-mo-hinh-am-hoc/ch23-glow-grad-matcha.md) |
| Diffusion | ELBO đơn giản hóa / khớp score | Nhiều bước lặp | Vocoder: WaveGrad, DiffWave. Mô hình âm học: Grad-TTS, DiffSinger. NaturalSpeech 2 (diffusion trên latent của codec) | [11](ch11-diffusion.md), [23](../phan-5-mo-hinh-am-hoc/ch23-glow-grad-matcha.md), [26](../phan-6-vocoder/ch26-diffusion-vocoder-va-so-sanh.md) |
| Flow matching | Hồi quy trường vận tốc | Giải ODE, thường ít bước hơn diffusion | Matcha-TTS, Voicebox, E2 TTS, F5-TTS, bộ giải mã flow matching của CosyVoice | [12](ch12-flow-matching.md), [33](../phan-8-mo-hinh-lon/ch33-nar-quy-mo-lon.md) |

Một bảng thứ hai so sánh định tính các họ theo các trục của "bộ ba bất khả". Đây là khái quát hóa thô, mọi ô đều có ngoại lệ; bạn nên đọc nó như một tấm bản đồ để định hướng, không phải như một kết luận thực nghiệm:

| Họ | Mật độ | Tốc độ lấy mẫu | Chất lượng mẫu | Độ phủ mode | Độ ổn định huấn luyện |
|---|---|---|---|---|---|
| Tự hồi quy | Chính xác | Chậm | Cao | Tốt | Ổn định; exposure bias khi suy luận |
| VAE | Cận dưới | Nhanh | Dễ nhòe nếu bộ giải mã quá đơn giản | Tốt | Ổn định; coi chừng sụp hậu nghiệm |
| GAN | Không có | Rất nhanh | Rất cao | Có nguy cơ sụp mode | Khó |
| Normalizing flow | Chính xác | Nhanh | Khá | Tốt | Ổn định; tốn tham số |
| Diffusion | Cận dưới | Chậm (nhiều bước) | Rất cao | Tốt | Ổn định |
| Flow matching | Tính được qua ODE (tốn kém) | Trung bình (ít bước) | Rất cao | Tốt | Ổn định |

## 7.8. Nhìn trước Chương 8–12

- **[Chương 8 – VAE](ch08-vae.md)** đưa vào biến ẩn $\mathbf{z}$ để "hấp thụ" phần thông tin mà điều kiện không quyết định. Ta sẽ suy ra cận dưới ELBO, mẹo tái tham số hóa, KL giữa hai Gauss, hiện tượng sụp hậu nghiệm, CVAE (nền tảng của VITS) và VQ-VAE (tổ tiên của các codec âm thanh).
- **[Chương 9 – GAN](ch09-gan.md)** thay hàm mất mát cố định bằng một bộ phân biệt được học: thay vì hỏi "mẫu này có gần mục tiêu không" (câu hỏi dẫn đến trung bình hóa), GAN hỏi "mẫu này có giống thật không". Đó là chìa khóa của các vocoder nhanh và sắc nét.
- **[Chương 10 – Normalizing flows](ch10-normalizing-flows.md)** xây dựng các biến đổi khả nghịch có Jacobian dễ tính để có likelihood chính xác mà vẫn lấy mẫu song song (Glow-TTS, WaveGlow, khối flow của VITS).
- **[Chương 11 – Diffusion](ch11-diffusion.md)** học cách khử nhiễu từng bước, biến bài toán sinh khó thành chuỗi các bài toán hồi quy dễ, mỗi bài *đơn mode* một cách cục bộ, một cách khéo léo để né đúng cái bẫy của mục 7.4.
- **[Chương 12 – Flow matching](ch12-flow-matching.md)** đơn giản hóa diffusion thành bài toán hồi quy một trường vận tốc dọc theo các đường đi xác suất, nền tảng của Matcha-TTS, F5-TTS và bộ giải mã của nhiều hệ thống lai hiện đại.

Một sợi chỉ đỏ nối các chương: **mỗi họ là một cách khác nhau để biến một nguồn ngẫu nhiên đơn giản thành mẫu của một phân phối phức tạp, đa mode**, sao cho mẫu nằm *trong* một mode thay vì nằm *giữa* các mode.

## Lỗi thường gặp và mẹo thực hành

- **Cố ép loss mel L1/L2 xuống thấp mãi.** Theo Mệnh đề 7.1, loss có một cái sàn bằng phần phương sai có điều kiện mà văn bản không quyết định. Khi loss validation đã chững lại, tăng mô hình hay huấn luyện lâu thêm thường chỉ dẫn đến quá khớp. Hãy nghe thử mẫu, đo thống kê ngôn điệu, thay vì nhìn mỗi con số loss.
- **Đánh giá mô hình sinh bằng đúng một mẫu cho mỗi câu.** Sinh ít nhất vài mẫu với các hạt giống khác nhau. Nếu các mẫu giống hệt nhau, mô hình của bạn thực chất đã thoái hóa thành một bộ hồi quy (hoặc một mode duy nhất).
- **Rò rỉ tương lai trong mô hình tự hồi quy.** Đệm sai phía trong tích chập, quên mặt nạ nhân quả, hoặc dùng chuẩn hóa theo cả chuỗi (ví dụ BatchNorm trên trục thời gian khi huấn luyện) đều có thể làm rò rỉ. Dấu hiệu: loss huấn luyện thấp bất thường, suy luận ra rác. Hãy viết phép kiểm tra "song song bằng tuần tự" như ở mục 7.3.2.
- **So sánh NLL giữa các cấu hình tiền xử lý khác nhau.** Đổi cách chuẩn hóa log-mel, đổi tần số lấy mẫu hay số kênh mel đều làm NLL không còn so sánh được.
- **Lạm dụng nhiệt độ lấy mẫu.** Hạ nhiệt độ làm mẫu "sạch" hơn nhưng đơn điệu hơn, che giấu khuyết điểm của mô hình. Luôn báo cáo nhiệt độ và các tham số lấy mẫu (top-k, top-p) khi so sánh hệ thống.
- **MDN cho ra NaN.** Nguyên nhân thường gặp: không giới hạn $\log \sigma$, tính log của tổng thay vì dùng `logsumexp`, tốc độ học quá lớn khiến một thành phần co $\sigma$ về 0. Kẹp $\log\sigma$ trong một khoảng hợp lý và khởi tạo các trung bình tách xa nhau.
- **Nhãn bẩn làm bài toán một–nhiều nặng thêm.** Phiên âm sai, chuẩn hóa văn bản không nhất quán (cùng "2024" lúc đọc "hai nghìn không trăm hai mươi tư", lúc đọc "hai không hai bốn") khiến cùng một văn bản đi với nhiều tiếng nói hoàn toàn khác nhau, mô hình sẽ "trung bình hóa" giữa chúng. Làm sạch dữ liệu ([Chương 17](../phan-4-frontend/ch17-chuan-hoa-van-ban.md), [Chương 36](../phan-9-thuc-tien/ch36-du-lieu-tts.md)) thường rẻ hơn mọi cải tiến mô hình.
- **Tin FAD một mình.** FAD tốt không bảo đảm mô hình đọc đúng văn bản; luôn đi kèm WER/CER và nghe thử.

## Tóm tắt

- Mô hình sinh học phân phối $p_\theta(\mathbf{x})$ hoặc $p_\theta(\mathbf{x} \mid \mathbf{c})$; ba việc chính là ước lượng mật độ, lấy mẫu và học biểu diễn. Các họ mô hình khác nhau ở cách chúng xử lý likelihood: chính xác (tự hồi quy, flow), xấp xỉ (VAE, diffusion), ẩn (GAN), hoặc huấn luyện bằng hồi quy trường vận tốc (flow matching).
- MLE tương đương cực tiểu hóa $D_{\mathrm{KL}}(p_{\text{data}} \Vert p_\theta)$, có xu hướng phủ mode; với một họ mô hình quá nghèo, MLE đặt khối xác suất vào giữa các mode.
- Mô hình tự hồi quy phân rã mật độ chính xác theo quy tắc chuỗi, huấn luyện song song nhờ teacher forcing nhưng suy luận tuần tự, và chịu exposure bias.
- TTS là bài toán sinh có điều kiện **một–nhiều**: $H(\mathbf{Y} \mid \mathbf{x})$ lớn vì văn bản không quyết định trường độ, F0, năng lượng, phong cách, kênh thu, pha.
- Hồi quy L2 hội tụ về $\mathbb{E}[\mathbf{Y} \mid \mathbf{x}]$, L1 hội tụ về trung vị từng chiều. Cả hai là MLE với họ đơn mode, độc lập giữa các chiều, phương sai cố định.
- Hệ quả là hiện tượng trơn quá mức: trung bình rơi vào vùng mật độ thấp, cấu trúc hài và chuyển tiếp bị nhòe, phương sai bị co rút một cách có hệ thống (định luật phương sai toàn phần). Với dạng sóng, trung bình qua pha gần như triệt tiêu tín hiệu.
- Hai lối thoát bổ sung cho nhau: thêm điều kiện để giảm bất định (trường độ, F0, người nói, tham chiếu) và dùng mô hình sinh đầy đủ để lấy mẫu từ phần bất định còn lại.
- MDN là cách sửa đơn giản nhất cho tính đơn mode, tốt cho đầu ra ít chiều nhưng không mở rộng được cho mel-spectrogram.
- Không thước đo đơn lẻ nào đủ để đánh giá mô hình sinh: likelihood, FAD, precision/recall, MOS, WER, độ tương đồng người nói và độ đa dạng mẫu mỗi thứ nhìn một mặt.

## Bài tập

**Bài 1 [Cơ bản].** Cho $Y$ nhận giá trị $-1$ và $+1$, mỗi giá trị với xác suất $1/2$ (hãy hình dung $Y$ là "độ dài khoảng ngắt" đã chuẩn hóa: hoặc ngắt dài, hoặc không ngắt). Tìm *mọi* nghiệm của $\min_c \mathbb{E}[(Y - c)^2]$ và $\min_c \mathbb{E}\lvert Y - c \rvert$, cùng giá trị loss tối ưu. Nghiệm nào trong số đó là một giá trị $Y$ có thể xảy ra?

??? success "Lời giải"
    $\mathbb{E}[(Y-c)^2] = 1 + c^2$ (vì $\mathbb{E}[Y] = 0$, $\mathbb{E}[Y^2] = 1$), cực tiểu duy nhất tại $c = 0$, loss bằng 1. Với L1: khi $c \in [-1, 1]$ thì $\mathbb{E}\lvert Y - c\rvert = \tfrac12(1 + c) + \tfrac12(1 - c) = 1$; khi $\lvert c \rvert > 1$ thì giá trị bằng $\lvert c \rvert > 1$. Vậy mọi $c \in [-1, 1]$ đều là nghiệm, loss bằng 1. Nghiệm L2 ($c = 0$, "ngắt nửa vời") không bao giờ xảy ra; với L1, chỉ hai đầu mút $c = \pm 1$ là giá trị có thật, và trình tối ưu không có lý do gì để chọn chúng thay vì một điểm ở giữa.

**Bài 2 [Cơ bản].** Trong lớp `CausalConvAR` ở mục 7.3.2, giả sử ta đệm đối xứng hai phía (`Fn.pad(h, (pad // 2, pad - pad // 2))`) thay vì chỉ đệm bên trái. (a) Phép kiểm tra `assert` sẽ ra sao? (b) Khi huấn luyện, loss sẽ trông thế nào, và vì sao mô hình đó vô dụng khi suy luận?

??? success "Lời giải"
    (a) Đầu ra tại vị trí $t$ giờ phụ thuộc vào đầu vào tại các vị trí sau $t$, mà đầu vào tại $t+1$ chính là $s_t$ (do dịch phải một bước). Khi tính song song, mạng "nhìn thấy" giá trị cần dự đoán; khi tính tuần tự trên tiền tố, phần tương lai không tồn tại (bị đệm bằng 0). Hai cách tính cho kết quả khác nhau nên `assert` thất bại. (b) Khi huấn luyện, mạng học cách sao chép $s_t$ từ đầu vào, loss giảm rất nhanh xuống mức thấp bất thường. Khi suy luận, thông tin tương lai không có, nên mô hình chưa bao giờ học cách dự đoán thực sự và sinh ra rác.

**Bài 3 [Cơ bản].** Chứng minh rằng nếu khớp một phân phối Gauss $\mathcal{N}(\mu, \sigma^2)$ vào một phân phối $p$ bất kỳ (có mô-men bậc hai hữu hạn) bằng cách cực tiểu $D_{\mathrm{KL}}(p \Vert \mathcal{N}(\mu, \sigma^2))$, nghiệm là $\mu = \mathbb{E}_p[y]$, $\sigma^2 = \operatorname{Var}_p(y)$. Áp dụng cho hỗn hợp $p = \sum_k \pi_k \mathcal{N}(\mu_k, \sigma_k^2)$ và kiểm tra lại con số $\sigma \approx 2.06$ trong mục 7.2.2.

??? success "Lời giải"
    Cực tiểu KL thuận tương đương cực đại $J(\mu, \sigma^2) = \mathbb{E}_p[\log \mathcal{N}(y; \mu, \sigma^2)] = -\tfrac12 \log(2\pi\sigma^2) - \mathbb{E}_p[(y - \mu)^2] / (2\sigma^2)$. Đạo hàm theo $\mu$: $\mathbb{E}_p[y - \mu] / \sigma^2 = 0 \Rightarrow \mu = \mathbb{E}_p[y]$. Đạo hàm theo $\sigma^2$: $-1/(2\sigma^2) + \mathbb{E}_p[(y-\mu)^2] / (2\sigma^4) = 0 \Rightarrow \sigma^2 = \mathbb{E}_p[(y-\mu)^2]$. Với hỗn hợp: $\mu = \sum_k \pi_k \mu_k$ và $\sigma^2 = \sum_k \pi_k(\sigma_k^2 + \mu_k^2) - \mu^2$. Với $\pi = (0.5, 0.5)$, $\mu_k = \pm 2$, $\sigma_k = 0.5$: $\mu = 0$, $\sigma^2 = 0.25 + 4 = 4.25$, $\sigma \approx 2.06$.

**Bài 4 [Nâng cao].** Mô hình hóa biên độ phổ (thang tuyến tính) của một khung hữu thanh là $Y(k) = \sum_{h \ge 1} g(k - h f_0)$ với $g(u) = \exp(-u^2 / (2w^2))$, trong đó $k$ là chỉ số bin và $f_0 \sim \mathcal{N}(\bar{f}, s^2)$ thay đổi giữa các lần đọc. Chứng minh

$$
\mathbb{E}[Y(k)] = \sum_{h \ge 1} \frac{w}{\sqrt{w^2 + h^2 s^2}} \exp\!\left( -\frac{(k - h\bar{f})^2}{2(w^2 + h^2 s^2)} \right).
$$

Từ đó giải thích vì sao các hài bậc cao bị nhòe nhiều hơn hài bậc thấp. Tính chiều cao đỉnh của hài thứ nhất và thứ mười khi $w = 0.6$ và $s = 0.5$ (đơn vị bin).

??? tip "Gợi ý"
    Vị trí hài thứ $h$ là $u = h f_0 \sim \mathcal{N}(h\bar{f}, h^2 s^2)$. Cần tính tích chập của một hàm Gauss (chưa chuẩn hóa) với một mật độ Gauss.

??? success "Lời giải"
    $\mathbb{E}[g(k - u)] = \int \exp\!\big(-\tfrac{(k-u)^2}{2w^2}\big)\, \mathcal{N}(u; h\bar{f}, h^2 s^2)\, du = \sqrt{2\pi w^2}\, \int \mathcal{N}(k; u, w^2)\, \mathcal{N}(u; h\bar{f}, h^2s^2)\, du = \sqrt{2\pi w^2}\, \mathcal{N}(k; h\bar{f}, w^2 + h^2 s^2)$, vì tích chập hai mật độ Gauss là một mật độ Gauss có phương sai bằng tổng. Rút gọn được biểu thức cần chứng minh. Đỉnh hài thứ $h$ có chiều cao $w / \sqrt{w^2 + h^2 s^2}$ (giảm theo $h$) và độ rộng $\sqrt{w^2 + h^2 s^2}$ (tăng theo $h$). Với $w = 0.6$, $s = 0.5$: hài thứ nhất cao $0.6 / \sqrt{0.61} \approx 0.77$, hài thứ mười chỉ còn $0.6 / \sqrt{25.36} \approx 0.12$. Trong phổ trung bình, vùng tần số cao gần như mất hẳn cấu trúc hài, đó là nguồn gốc của âm sắc "đục" và "rè".

**Bài 5 [Nâng cao].** (a) Chứng minh $\log\big(\alpha p(\mathbf{x}) + (1-\alpha) q(\mathbf{x})\big) \ge \log \alpha + \log p(\mathbf{x})$ với mọi $\alpha \in (0, 1]$. (b) Dùng kết quả này để lập luận rằng một mô hình có 99% mẫu là rác vẫn có thể có NLL trên mỗi chiều gần như bằng một mô hình tốt khi số chiều $d$ lớn. (c) Điều này nói gì về việc dùng NLL để chọn checkpoint cho một mô hình TTS?

??? success "Lời giải"
    (a) Vì $(1-\alpha) q(\mathbf{x}) \ge 0$ và $\log$ đồng biến. (b) NLL của hỗn hợp không vượt quá NLL của $p$ cộng $-\log \alpha$; chia cho $d$, mức chênh lệch trên mỗi chiều tối đa là $-\log(\alpha)/d$, tiến về 0 khi $d$ lớn (ví dụ $4.6 / 16000 \approx 3 \times 10^{-4}$ nat). (c) NLL có thể dùng để phát hiện mô hình hỏng nặng hoặc quá khớp, nhưng không đủ để chọn giữa các checkpoint có NLL gần nhau; cần kết hợp nghe thử, WER và các thước đo theo mẫu.

**Bài 6 [Nâng cao].** Chứng minh rằng thêm điều kiện $\mathbf{c}$ không làm tăng phần lỗi không thể giảm trong Mệnh đề 7.1, tức là $\mathbb{E}\big[\operatorname{tr}\operatorname{Cov}(\mathbf{Y} \mid \mathbf{x}, \mathbf{c})\big] \le \mathbb{E}\big[\operatorname{tr}\operatorname{Cov}(\mathbf{Y} \mid \mathbf{x})\big]$. Khi nào xảy ra dấu bằng? Liên hệ với bộ thích nghi phương sai của FastSpeech 2.

??? success "Lời giải"
    Áp dụng định luật phương sai toàn phần có điều kiện trên $\mathbf{x}$: $\operatorname{Cov}(\mathbf{Y} \mid \mathbf{x}) = \mathbb{E}\big[\operatorname{Cov}(\mathbf{Y} \mid \mathbf{x}, \mathbf{c}) \,\big\vert\, \mathbf{x}\big] + \operatorname{Cov}\big(\mathbb{E}[\mathbf{Y} \mid \mathbf{x}, \mathbf{c}] \,\big\vert\, \mathbf{x}\big)$. Lấy vết: số hạng thứ hai có vết không âm vì là ma trận hiệp phương sai. Lấy kỳ vọng theo $\mathbf{x}$ được bất đẳng thức. Dấu bằng xảy ra khi $\mathbb{E}[\mathbf{Y} \mid \mathbf{x}, \mathbf{c}]$ không phụ thuộc $\mathbf{c}$ (hầu chắc chắn), tức $\mathbf{c}$ không mang thêm thông tin nào về trung bình của $\mathbf{Y}$. FastSpeech 2 cung cấp trường độ, cao độ, năng lượng thật khi huấn luyện, chính là những $\mathbf{c}$ giải thích phần lớn biến thiên của mel, nên bộ giải mã hồi quy bớt nhòe.

**Bài 7 [Lập trình].** Sửa thí nghiệm ở mục 7.4.4 theo "lối A": nối thêm giá trị F0 (đã trừ đi 6) vào đầu vào one-hot, tăng kích thước mạng và số bước huấn luyện. Đo lại độ tương phản phổ và tỉ lệ phương sai được tái tạo. So sánh với kết quả ban đầu và giải thích bằng Bài 6.

??? tip "Gợi ý"
    Một MLP hai lớp ẩn 256 đơn vị, khoảng 1500 bước Adam với tốc độ học $3 \times 10^{-3}$ là đủ. Khi người biên soạn thử, độ tương phản phổ trở về xấp xỉ mức của dữ liệu và tỉ lệ phương sai được tái tạo lên gần 100%: một khi biết F0, phần còn lại của bài toán gần như là một–một.

**Bài 8 [Lập trình].** Với ví dụ MDN ở mục 7.5.3: (a) huấn luyện với $K = 1, 3, 5$ và so sánh NLL cuối cùng; (b) với $K = 3$, sinh 1000 mẫu tại $x = 0.5$ và vẽ histogram; (c) kiểm tra bằng số rằng trung bình của các mẫu MDN xấp xỉ bằng dự đoán của bộ hồi quy L2, rồi giải thích vì sao điều đó phải đúng.

??? tip "Gợi ý"
    (c) Bộ hồi quy L2 tối ưu xuất ra $\mathbb{E}[y \mid x]$; một MDN khớp tốt có trung bình $\sum_k \pi_k \mu_k \approx \mathbb{E}[y \mid x]$. Hai mô hình đồng ý về *trung bình*, chỉ khác nhau ở chỗ MDN biết rằng không mẫu thật nào nằm gần trung bình đó.

**Bài 9 [Lập trình].** Viết hàm `diversity(samples)` nhận một tensor `(S, T)` gồm $S$ đường F0 sinh ra cho cùng một câu (đã căn về cùng độ dài) và trả về độ lệch chuẩn trung bình theo trục mẫu. Dùng nó để so sánh (a) một "mô hình hồi quy" luôn trả về cùng một đường, (b) một "mô hình sinh" cộng thêm biến thiên ngẫu nhiên có tương quan theo thời gian (ví dụ nhiễu Gauss đã làm trơn bằng trung bình trượt). Thảo luận: vì sao nhiễu trắng độc lập theo từng khung là một mô hình *tồi* cho biến thiên ngôn điệu, dù nó cho độ đa dạng cao?

## Tài liệu tham khảo

1. Bishop, C. M. (1994). *Mixture Density Networks*. Technical Report NCRG/94/004, Aston University.
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
3. Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press.
4. Goodfellow, I. (2016). *NIPS 2016 Tutorial: Generative Adversarial Networks*. arXiv.
5. Murphy, K. P. (2023). *Probabilistic Machine Learning: Advanced Topics*. MIT Press.
6. Xiao, Z., Kreis, K., Vahdat, A. (2022). *Tackling the Generative Learning Trilemma with Denoising Diffusion GANs*. ICLR.
7. van den Oord, A. và cộng sự (2016a). *WaveNet: A Generative Model for Raw Audio*. arXiv:1609.03499.
8. van den Oord, A., Kalchbrenner, N., Kavukcuoglu, K. (2016b). *Pixel Recurrent Neural Networks*. ICML.
9. Salimans, T., Karpathy, A., Chen, X., Kingma, D. P. (2017). *PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications*. ICLR.
10. Ranzato, M., Chopra, S., Auli, M., Zaremba, W. (2016). *Sequence Level Training with Recurrent Neural Networks*. ICLR.
11. Bengio, S., Vinyals, O., Jaitly, N., Shazeer, N. (2015). *Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks*. NeurIPS.
12. Huszár, F. (2015). *How (not) to Train your Generative Model: Scheduled Sampling, Likelihood, Adversary?* arXiv.
13. Wang, Y. và cộng sự (2017). *Tacotron: Towards End-to-End Speech Synthesis*. Interspeech.
14. Shen, J. và cộng sự (2018). *Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions*. ICASSP.
15. Ren, Y. và cộng sự (2021). *FastSpeech 2: Fast and High-Quality End-to-End Text to Speech*. ICLR.
16. Ren, Y., Tan, X., Qin, T., Zhao, Z., Liu, T.-Y. (2022). *Revisiting Over-Smoothness in Text to Speech*. ACL.
17. Toda, T., Tokuda, K. (2007). *A Speech Parameter Generation Algorithm Considering Global Variance for HMM-Based Speech Synthesis*. IEICE Transactions on Information and Systems, E90-D(5).
18. Zen, H., Senior, A. (2014). *Deep Mixture Density Networks for Acoustic Modeling in Statistical Parametric Speech Synthesis*. ICASSP.
19. Theis, L., van den Oord, A., Bethge, M. (2016). *A Note on the Evaluation of Generative Models*. ICLR.
20. Heusel, M. và cộng sự (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*. NeurIPS.
21. Kilgour, K., Zuluaga, M., Roblek, D., Sharifi, M. (2019). *Fréchet Audio Distance: A Reference-Free Metric for Evaluating Music Enhancement Algorithms*. Interspeech.
22. Sajjadi, M. S. M. và cộng sự (2018). *Assessing Generative Models via Precision and Recall*. NeurIPS.
23. Kynkäänniemi, T. và cộng sự (2019). *Improved Precision and Recall Metric for Assessing Generative Models*. NeurIPS.
24. Tan, X., Qin, T., Soong, F., Liu, T.-Y. (2021). *A Survey on Neural Speech Synthesis*. arXiv.
25. Kim, J., Kong, J., Son, J. (2021). *Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech*. ICML.
26. Kong, J., Kim, J., Bae, J. (2020). *HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis*. NeurIPS.
27. Kim, J., Kim, S., Kong, J., Yoon, S. (2020). *Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search*. NeurIPS.
28. Popov, V. và cộng sự (2021). *Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech*. ICML.
29. Mehta, S. và cộng sự (2024). *Matcha-TTS: A Fast TTS Architecture with Conditional Flow Matching*. ICASSP.
30. Wang, C. và cộng sự (2023). *Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers*. arXiv:2301.02111.
