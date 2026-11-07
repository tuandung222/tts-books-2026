# Chương 9. Mạng đối sinh (GAN)

!!! abstract "Mục tiêu học tập"
    Sau chương này, bạn có thể:

    - **Giải thích** vì sao học một bộ phân biệt làm "thước đo" giúp tránh hiện tượng trung bình hóa của hồi quy L1/L2.
    - **Suy ra** bộ phân biệt tối ưu $D^\ast = p_{\text{data}} / (p_{\text{data}} + p_g)$ và chứng minh trò chơi minimax tương đương cực tiểu hóa phân kỳ Jensen–Shannon.
    - **Phân tích** hiện tượng bão hòa gradient và lý do dùng mục tiêu không bão hòa; giải thích sụp mode và tính không hội tụ của động lực học trò chơi.
    - **So sánh** các biến thể hàm mất mát: GAN gốc, LSGAN, hinge, WGAN/WGAN-GP; suy ra bộ phân biệt tối ưu của LSGAN.
    - **Cài đặt** các kỹ thuật ổn định hóa: chuẩn hóa phổ, phạt gradient R1, cập nhật hai thang thời gian, feature matching.
    - **Thiết kế** GAN có điều kiện theo công thức "loss tái tạo + loss đối kháng" và giải thích vì sao vocoder GAN cần bộ phân biệt đa thang/đa chu kỳ và loss phổ phụ trợ.

!!! info "Kiến thức cần có"
    - [Chương 1](../phan-1-nen-tang/ch01-toan-nen-tang.md): phân kỳ KL, kỳ vọng, đổi biến trong kỳ vọng.
    - [Chương 3](../phan-1-nen-tang/ch03-toi-uu-va-ky-thuat-huan-luyen.md): Adam, weight normalization, spectral normalization.
    - [Chương 4](../phan-1-nen-tang/ch04-cnn-rnn-mo-hinh-chuoi.md): Conv1d, tích chập chuyển vị.
    - [Chương 7](ch07-tong-quan-mo-hinh-sinh.md): bài toán một–nhiều, KL thuận và KL ngược, FID/FAD.
    - [Chương 8](ch08-vae.md): mô hình biến ẩn, vì sao decoder Gauss cho mẫu mờ.

[Chương 7](ch07-tong-quan-mo-hinh-sinh.md) chỉ ra rằng mọi hàm mất mát dạng "khoảng cách tới mục tiêu" đều bị kéo về trung bình khi bài toán là một–nhiều. [Chương 8](ch08-vae.md) đưa vào biến ẩn để giảm bất định, nhưng decoder Gauss vẫn hồi quy trung bình nên mẫu còn mờ. Mạng đối sinh (*generative adversarial network*, GAN) của Goodfellow và cộng sự (2014) tấn công vấn đề từ một hướng khác hẳn: **thay vì chọn sẵn một hàm mất mát, hãy học nó**. Một mạng thứ hai, bộ phân biệt, được huấn luyện để phân biệt mẫu thật với mẫu giả; bộ sinh được huấn luyện để đánh lừa nó. Câu hỏi mà bộ sinh phải trả lời không còn là "mẫu của tôi có gần đáp án không?" mà là "mẫu của tôi có giống thật không?". Một phổ bị nhòe, một dạng sóng gần như im lặng do trung bình hóa pha, đều trông rất "giả" dưới con mắt của một bộ phân biệt tốt.

Trong TTS, GAN là xương sống của gần như mọi vocoder nhanh hiện đại (MelGAN, Parallel WaveGAN, HiFi-GAN, BigVGAN, Vocos) và là một thành phần của nhiều hệ thống end-to-end (VITS, StyleTTS 2). Chương này xây dựng nền tảng lý thuyết và kỹ thuật; kiến trúc vocoder cụ thể được trình bày ở [Chương 25](../phan-6-vocoder/ch25-gan-vocoder.md).

## 9.1. Ý tưởng: học một thước đo thay vì chọn sẵn

### 9.1.1. Bộ sinh ngầm định

GAN định nghĩa phân phối mô hình **một cách ngầm định** qua một quy trình lấy mẫu: rút nhiễu $\mathbf{z} \sim p(\mathbf{z})$ (thường là Gauss chuẩn hoặc phân phối đều), rồi đưa qua **bộ sinh** (*generator*) $G_\theta$:

$$
\mathbf{x} = G_\theta(\mathbf{z}), \qquad \mathbf{z} \sim p(\mathbf{z}).
$$

Phân phối của $\mathbf{x}$ thu được, ký hiệu $p_g$, là "đẩy tiến" (*pushforward*) của $p(\mathbf{z})$ qua $G_\theta$. Ta không có công thức cho $p_g(\mathbf{x})$: nếu $G_\theta$ không khả nghịch hoặc $\mathbf{z}$ ít chiều hơn $\mathbf{x}$, mật độ này thậm chí không tồn tại theo nghĩa thông thường. Đổi lại, lấy mẫu chỉ tốn **một lượt truyền xuôi**, nhanh hơn rất nhiều so với mô hình tự hồi quy hay khuếch tán.

### 9.1.2. Bộ phân biệt như một hàm mất mát được học

Vì không có $p_g(\mathbf{x})$, ta không thể dùng hợp lý cực đại. GAN thay vào đó dùng một **bộ phân biệt** (*discriminator*) $D_\psi$, một mạng nhận $\mathbf{x}$ và xuất ra điểm số "mức độ thật". Hai mạng chơi một trò chơi: $D$ cố phân biệt, $G$ cố đánh lừa. Hình ảnh quen thuộc là kẻ làm tiền giả và cảnh sát: cảnh sát càng giỏi, kẻ làm giả càng phải tinh vi.

Hãy nghĩ về điều $D$ "nhìn thấy" trong bài toán một–nhiều. Giả sử dữ liệu có hai mode (ngắt hơi và không ngắt hơi). Một bộ sinh xuất ra trung bình của hai mode sẽ cho những mẫu nằm ở vùng không có dữ liệu thật, và $D$ dễ dàng nhận ra chúng là giả. Để đánh lừa $D$, bộ sinh buộc phải đặt mẫu **vào bên trong** các mode. Chính cơ chế này cho phép GAN sinh ra phổ sắc nét và dạng sóng có pha hợp lý. Trước khi trở thành vocoder, GAN đã được dùng trong tổng hợp tham số thống kê với đúng mục đích khắc phục hiện tượng trơn quá mức, ví dụ như một bộ hậu lọc (*postfilter*) cho đặc trưng phổ (Kaneko và cộng sự, 2017) hoặc như một loss phụ trợ cho mô hình âm học (Saito và cộng sự, 2018).

## 9.2. Trò chơi minimax

Trong công thức gốc, $D(\mathbf{x}) \in (0, 1)$ là xác suất $\mathbf{x}$ là thật (đầu ra sigmoid của một logit). Hàm giá trị của trò chơi là

$$
V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}\big[\log D(\mathbf{x})\big] + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})}\big[\log\big(1 - D(G(\mathbf{z}))\big)\big],
$$

và bài toán là

$$
\min_G \max_D V(D, G).
$$

Với $G$ cố định, $\max_D V$ chính là bài toán **phân loại nhị phân** bằng entropy chéo: nhãn 1 cho mẫu thật, nhãn 0 cho mẫu giả. $G$ thì muốn giảm $V$, tức làm cho $D(G(\mathbf{z}))$ lớn. Trong thực tế, hai mạng được cập nhật **luân phiên** bằng gradient ngẫu nhiên:

1. Lấy một minibatch dữ liệu thật và một minibatch nhiễu; sinh mẫu giả $G(\mathbf{z})$ và **ngắt gradient** về $G$.
2. Cập nhật $D$ theo hướng tăng $V$ (giảm loss phân loại).
3. Lấy minibatch nhiễu (có thể dùng lại), cập nhật $G$ theo hướng giảm loss của bộ sinh, với gradient đi xuyên qua $D$ (nhưng chỉ tham số của $G$ được cập nhật).

Bài báo gốc cho phép $k$ bước cập nhật $D$ cho mỗi bước $G$, nhưng chính các tác giả dùng $k = 1$; đó cũng là lựa chọn phổ biến nhất hiện nay.

## 9.3. Bộ phân biệt tối ưu và góc nhìn Jensen–Shannon

### 9.3.1. Bộ phân biệt tối ưu

Cố định $G$. Đổi biến trong kỳ vọng thứ hai: vì $\mathbf{x} = G(\mathbf{z})$ có phân phối $p_g$, ta có $\mathbb{E}_{\mathbf{z}}[f(G(\mathbf{z}))] = \mathbb{E}_{\mathbf{x} \sim p_g}[f(\mathbf{x})]$ với mọi hàm $f$. Do đó

$$
V(D, G) = \int \Big[ p_{\text{data}}(\mathbf{x}) \log D(\mathbf{x}) + p_g(\mathbf{x}) \log\big(1 - D(\mathbf{x})\big) \Big]\, d\mathbf{x}.
$$

(Ở đây ta giả định $p_g$ có mật độ; phân tích là một lý tưởng hóa.) Vì $D$ được chọn tự do tại từng điểm, ta cực đại hóa hàm dưới dấu tích phân **tại từng $\mathbf{x}$**. Với $a = p_{\text{data}}(\mathbf{x}) \ge 0$, $b = p_g(\mathbf{x}) \ge 0$ không đồng thời bằng 0, xét $h(y) = a \log y + b \log(1-y)$ trên $(0, 1)$:

$$
h'(y) = \frac{a}{y} - \frac{b}{1-y} = 0 \;\Longleftrightarrow\; y = \frac{a}{a + b}, \qquad h''(y) = -\frac{a}{y^2} - \frac{b}{(1-y)^2} < 0 .
$$

Vậy **bộ phân biệt tối ưu** là

$$
D^\ast_G(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}.
$$

Ngoài hợp của hai giá (*support*), giá trị của $D$ không ảnh hưởng đến $V$ và có thể tùy ý. Một cách đọc hữu ích: logit của bộ phân biệt tối ưu là $\log D^\ast/(1 - D^\ast) = \log p_{\text{data}}(\mathbf{x}) - \log p_g(\mathbf{x})$. Bộ phân biệt đang **ước lượng tỉ số mật độ** giữa dữ liệu và mô hình, đúng thứ ta cần để so sánh hai phân phối mà không phải tính từng mật độ.

### 9.3.2. Trò chơi tối ưu hóa phân kỳ Jensen–Shannon

Thay $D^\ast$ vào $V$:

$$
C(G) := V(D^\ast_G, G) = \mathbb{E}_{p_{\text{data}}}\!\left[\log \frac{p_{\text{data}}}{p_{\text{data}} + p_g}\right] + \mathbb{E}_{p_g}\!\left[\log \frac{p_g}{p_{\text{data}} + p_g}\right].
$$

Đặt $m = \tfrac12(p_{\text{data}} + p_g)$, tức $p_{\text{data}} + p_g = 2m$. Mỗi số hạng tách thành một KL trừ $\log 2$:

$$
\mathbb{E}_{p_{\text{data}}}\!\left[\log \frac{p_{\text{data}}}{2m}\right] = D_{\mathrm{KL}}(p_{\text{data}} \,\Vert\, m) - \log 2,
\qquad
\mathbb{E}_{p_g}\!\left[\log \frac{p_g}{2m}\right] = D_{\mathrm{KL}}(p_g \,\Vert\, m) - \log 2 .
$$

Do đó

$$
C(G) = D_{\mathrm{KL}}(p_{\text{data}} \,\Vert\, m) + D_{\mathrm{KL}}(p_g \,\Vert\, m) - \log 4 = 2\, \mathrm{JSD}(p_{\text{data}} \,\Vert\, p_g) - \log 4,
$$

trong đó **phân kỳ Jensen–Shannon** được định nghĩa là $\mathrm{JSD}(p \Vert q) = \tfrac12 D_{\mathrm{KL}}(p \Vert m) + \tfrac12 D_{\mathrm{KL}}(q \Vert m)$ với $m = \tfrac12(p+q)$. JSD đối xứng, luôn nằm trong $[0, \log 2]$, và bằng 0 khi và chỉ khi $p = q$. Kết luận (Goodfellow và cộng sự, 2014): **nếu $D$ luôn tối ưu, cực tiểu hóa $V$ theo $G$ tương đương cực tiểu hóa JSD**, và cực tiểu toàn cục $C = -\log 4$ đạt được duy nhất khi $p_g = p_{\text{data}}$; khi đó $D^\ast \equiv \tfrac12$.

Đoạn mã sau kiểm chứng cả hai kết quả trên hai phân phối Gauss một chiều đã biết: huấn luyện một bộ phân biệt logistic (họ này chứa đúng $D^\ast$, vì log-tỉ số của hai Gauss là đa thức bậc hai), rồi so với công thức; sau đó tính $V(D^\ast, G)$ và $-\log 4 + 2\,\mathrm{JSD}$ bằng tích phân số.

```python
import math
import torch
import torch.nn.functional as Fn

torch.manual_seed(0)
# Hai phân phối 1-D đã biết: p_data = N(1, 0.5^2), p_g = N(-0.5, 1)
p_data = torch.distributions.Normal(1.0, 0.5)
p_g = torch.distributions.Normal(-0.5, 1.0)

# Bộ phân biệt: hồi quy logistic trên đặc trưng (1, x, x^2). Log-odds của hai Gauss
# là một đa thức bậc hai, nên họ này chứa đúng D* = p_data / (p_data + p_g).
w = torch.zeros(3, requires_grad=True)
feats = lambda x: torch.stack([torch.ones_like(x), x, x ** 2], dim=-1)   # (n, 3)
opt = torch.optim.Adam([w], lr=0.05)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3000)
for _ in range(3000):
    xr, xf = p_data.sample((4096,)), p_g.sample((4096,))
    # Cực đại V(D, G) = E_data[log D] + E_g[log(1 - D)]  <=>  cực tiểu BCE với nhãn 1/0
    loss = -(Fn.logsigmoid(feats(xr) @ w).mean() + Fn.logsigmoid(-(feats(xf) @ w)).mean())
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()

x = torch.linspace(-3, 3, 601)
log_pd, log_pg = p_data.log_prob(x), p_g.log_prob(x)
d_star = torch.sigmoid(log_pd - log_pg)                  # = p_data / (p_data + p_g)
with torch.no_grad():
    d_learned = torch.sigmoid(feats(x) @ w)
print(f"max |D_học - D*| trên [-3, 3]: {(d_learned - d_star).abs().max():.3f}")

# Kiểm tra V(D*, G) = -log 4 + 2 * JSD(p_data || p_g) bằng tích phân số (float64)
xx = torch.linspace(-8, 8, 20001, dtype=torch.float64); dx = xx[1] - xx[0]
pd = torch.exp(-0.5 * ((xx - 1.0) / 0.5) ** 2) / (0.5 * math.sqrt(2 * math.pi))
pg = torch.exp(-0.5 * (xx + 0.5) ** 2) / math.sqrt(2 * math.pi)
m = 0.5 * (pd + pg)
v_star = ((torch.xlogy(pd, pd / (pd + pg)) + torch.xlogy(pg, pg / (pd + pg))) * dx).sum()
jsd = 0.5 * (torch.xlogy(pd, pd / m) * dx).sum() + 0.5 * (torch.xlogy(pg, pg / m) * dx).sum()
print(f"V(D*, G) = {v_star:.4f}   -log 4 + 2 JSD = {-math.log(4) + 2 * jsd:.4f}   JSD = {jsd:.4f}")
assert abs(v_star - (-math.log(4) + 2 * jsd)) < 1e-6
assert (d_learned - d_star).abs().max() < 0.05
```

Bộ phân biệt học được khớp $D^\ast$ với sai số cỡ $10^{-3}$, và hai vế của đẳng thức JSD trùng nhau đến bốn chữ số thập phân.

### 9.3.3. Giới hạn của phân tích lý tưởng

Kết quả trên đẹp nhưng dựa trên ba giả định hiếm khi đúng:

1. **$D$ luôn tối ưu.** Thực tế $D$ chỉ được cập nhật vài bước gradient cho mỗi bước của $G$.
2. **Tối ưu trong không gian hàm.** Lập luận "$C(G)$ lồi theo $p_g$" nói về không gian phân phối; tối ưu theo tham số $\theta$ của một mạng nơ-ron là bài toán không lồi.
3. **Hai phân phối có mật độ chồng lấn.** Dữ liệu thật như tiếng nói nằm gần một đa tạp (*manifold*) ít chiều trong không gian rất nhiều chiều; mẫu của $G$ nằm trên ảnh của một không gian nhiễu ít chiều. Arjovsky và Bottou (2017) chỉ ra rằng trong tình huống này, hai giá thường gần như không giao nhau. Khi đó tồn tại một bộ phân biệt **hoàn hảo**, JSD đứng yên ở giá trị $\log 2$ bất kể $G$ ở gần hay xa dữ liệu, và gradient mang thông tin hữu ích về $G$ biến mất.

Ví dụ đơn giản nhất: $p_{\text{data}} = \delta_0$ (một khối lượng điểm tại 0) và $p_\theta = \delta_\theta$. Với mọi $\theta \ne 0$, $\mathrm{JSD} = \log 2$: hằng số, gradient bằng 0, không cho biết nên dịch $\theta$ về phía nào. Ta sẽ quay lại ví dụ này với khoảng cách Wasserstein ở mục 9.6.3.

## 9.4. Mục tiêu không bão hòa

Ngay cả khi bỏ qua vấn đề giá không giao nhau, mục tiêu minimax của $G$ có một khuyết điểm thực tế. Viết $D = \sigma(\ell)$ với $\ell$ là logit của bộ phân biệt tại một mẫu giả. Mục tiêu gốc của $G$ là **cực tiểu** $\log(1 - \sigma(\ell))$, có đạo hàm theo logit

$$
\frac{\partial}{\partial \ell} \log\big(1 - \sigma(\ell)\big) = -\sigma(\ell).
$$

Đầu quá trình huấn luyện, mẫu giả rất tệ và $D$ dễ dàng từ chối chúng: $\sigma(\ell) \approx 0$. Khi đó gradient cũng $\approx 0$: **bộ sinh không học được gì đúng vào lúc nó cần học nhất**. Đây là hiện tượng bão hòa (*saturation*).

Goodfellow và cộng sự (2014) đề xuất **mục tiêu không bão hòa** (*non-saturating*, NS): thay vì cực tiểu $\log(1 - D(G(\mathbf{z})))$, $G$ cực tiểu $-\log D(G(\mathbf{z}))$. Đạo hàm theo logit:

$$
\frac{\partial}{\partial \ell}\big[-\log \sigma(\ell)\big] = -\big(1 - \sigma(\ell)\big),
$$

gần $-1$ khi $\sigma(\ell) \approx 0$. Ví dụ với $D(G(\mathbf{z})) = 0.01$: độ lớn gradient theo logit là $0.01$ với mục tiêu gốc và $0.99$ với mục tiêu NS, gấp gần trăm lần. Hai mục tiêu có cùng điểm dừng (đều đẩy $D(G(\mathbf{z}))$ lên), nhưng động lực học khác hẳn. Trong cài đặt, mục tiêu NS chính là entropy chéo với nhãn "thật" cho mẫu giả, viết ổn định bằng `logsigmoid`.

Cái giá của mục tiêu NS là nó không còn tương ứng với JSD. Arjovsky và Bottou (2017) chỉ ra rằng khi $D = D^\ast$ được giữ cố định, kỳ vọng gradient của mục tiêu NS bằng gradient của $D_{\mathrm{KL}}(p_g \Vert p_{\text{data}}) - 2\,\mathrm{JSD}(p_{\text{data}} \Vert p_g)$. Thành phần $D_{\mathrm{KL}}(p_g \Vert p_{\text{data}})$ là **KL ngược**, loại phân kỳ "tìm mode" mà ta đã gặp ở [Chương 7](ch07-tong-quan-mo-hinh-sinh.md): nó phạt nặng mẫu nằm ngoài dữ liệu, nhưng phạt nhẹ việc bỏ sót mode. Điều này giải thích vì sao GAN thường cho **mẫu sắc nét nhưng có xu hướng mất đa dạng**, trái ngược với các mô hình huấn luyện bằng MLE (KL thuận).

## 9.5. Sụp mode và bất ổn định khi huấn luyện

### 9.5.1. Sụp mode

**Sụp mode** (*mode collapse*) là khi bộ sinh ánh xạ nhiều giá trị $\mathbf{z}$ khác nhau về cùng một hoặc vài mẫu, bỏ qua phần lớn các mode của dữ liệu. Một lý do đơn giản: với $D$ cố định, phản ứng tốt nhất của $G$ là đặt **toàn bộ** khối lượng xác suất vào điểm $\arg\max_{\mathbf{x}} D(\mathbf{x})$, tức một phân phối Dirac. Nếu $G$ cập nhật nhanh hơn $D$ "theo kịp", nó sẽ trượt về những điểm $D$ đang ưa thích; $D$ học cách từ chối những điểm đó; $G$ nhảy sang chỗ khác. Kết quả có thể là sụp hoàn toàn, sụp một phần, hoặc dao động giữa các mode (Goodfellow, 2016).

Ở mục 9.7.2 bạn sẽ thấy hiện tượng này xảy ra ngay trên một bài toán một chiều có hai mode. Các biện pháp đã được đề xuất gồm: cho $D$ nhìn thống kê của cả minibatch (*minibatch discrimination*, Salimans và cộng sự, 2016) hoặc nhìn nhiều mẫu cùng lúc (PacGAN, Lin và cộng sự, 2018); "trải" vài bước tối ưu của $D$ khi tính gradient cho $G$ (Metz và cộng sự, 2017); và các phép điều chuẩn bộ phân biệt như phạt gradient (mục 9.7). Trong vocoder GAN có điều kiện, sụp mode ít nghiêm trọng hơn vì mel-spectrogram điều kiện đã quyết định gần hết đầu ra (mục 9.8).

### 9.5.2. Trò chơi không hội tụ

Huấn luyện GAN là tìm **điểm cân bằng** của một trò chơi, không phải cực tiểu của một hàm. Gradient descent làm việc tốt với bài toán cực tiểu, nhưng có thể thất bại với trò chơi ngay cả trong trường hợp đơn giản nhất. Xét trò chơi song tuyến tính $\min_x \max_y f(x, y) = xy$, có cân bằng duy nhất tại $(0, 0)$:

```python
import numpy as np

# Trò chơi song tuyến tính min_x max_y f(x, y) = x * y; cân bằng Nash duy nhất tại (0, 0)
eta, steps = 0.1, 200
x, y = 1.0, 1.0
for _ in range(steps):                       # cập nhật ĐỒNG THỜI
    x, y = x - eta * y, y + eta * x          # x giảm theo df/dx = y, y tăng theo df/dy = x
r_sim = np.hypot(x, y)

x, y = 1.0, 1.0
for _ in range(steps):                       # cập nhật LUÂN PHIÊN (y dùng x mới)
    x = x - eta * y
    y = y + eta * x
r_alt = np.hypot(x, y)

print(f"bán kính ban đầu: {np.hypot(1, 1):.3f}")
print(f"đồng thời : {r_sim:.3f}  (lý thuyết: {np.hypot(1, 1) * (1 + eta**2) ** (steps / 2):.3f})")
print(f"luân phiên: {r_alt:.3f}  (bị chặn, quay vòng quanh cân bằng)")
assert r_sim > 2 * np.hypot(1, 1) and r_alt < 2 * np.hypot(1, 1)
```

Với cập nhật đồng thời, mỗi bước nhân véc-tơ $(x, y)$ với ma trận $\begin{pmatrix} 1 & -\eta \\ \eta & 1 \end{pmatrix}$ có trị riêng $1 \pm i\eta$, mô-đun $\sqrt{1 + \eta^2} > 1$: quỹ đạo **xoắn ốc ra xa** cân bằng. Với cập nhật luân phiên, quỹ đạo bị chặn nhưng **quay vòng mãi** quanh cân bằng mà không hội tụ (Bài tập 6). Phân tích của Mescheder và cộng sự (2018) trên một GAN tối giản ("Dirac-GAN") cho thấy đúng hành vi này xuất hiện ở GAN thực, và các phép điều chuẩn như phạt gradient làm hệ hội tụ cục bộ.

Hệ quả thực tế: **đường cong loss của GAN không phải là thước đo tiến độ**. Loss của $D$ và $G$ dao động, có thể đi lên trong khi chất lượng mẫu đi lên. Hãy theo dõi chất lượng bằng mẫu (nghe thử, FAD, các loss phụ trợ như mel L1 ở vocoder) thay vì nhìn loss đối kháng.

### 9.5.3. Cân bằng giữa hai người chơi

Nếu $D$ quá mạnh so với $G$, nó phân biệt hoàn hảo và gradient cho $G$ trở nên vô dụng (đặc biệt với mục tiêu minimax gốc). Nếu $D$ quá yếu, nó không cung cấp tín hiệu gì có ích. Phần lớn các kỹ thuật ở mục 9.6 và 9.7 có thể hiểu là những cách giữ cho $D$ **mượt** và **có thông tin** ở mọi nơi, kể cả xa dữ liệu thật.

## 9.6. Các biến thể hàm mất mát

### 9.6.1. LSGAN

Mao và cộng sự (2017) thay entropy chéo bằng **bình phương sai số**: $D$ xuất ra một số thực (không sigmoid), được hồi quy về nhãn $b$ cho mẫu thật và $a$ cho mẫu giả, còn $G$ muốn $D$ cho mẫu giả giá trị $c$:

$$
\min_D \tfrac12 \mathbb{E}_{p_{\text{data}}}\big[(D(\mathbf{x}) - b)^2\big] + \tfrac12 \mathbb{E}_{p_g}\big[(D(\mathbf{x}) - a)^2\big],
\qquad
\min_G \tfrac12 \mathbb{E}_{p_g}\big[(D(\mathbf{x}) - c)^2\big].
$$

**Bộ phân biệt tối ưu.** Cực tiểu tại từng điểm $p_{\text{data}}(y - b)^2 + p_g(y - a)^2$ theo $y$ cho

$$
D^\ast(\mathbf{x}) = \frac{b\, p_{\text{data}}(\mathbf{x}) + a\, p_g(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}.
$$

**Phân kỳ được tối ưu.** Thêm vào mục tiêu của $G$ số hạng $\tfrac12 \mathbb{E}_{p_{\text{data}}}[(D(\mathbf{x}) - c)^2]$, không phụ thuộc $G$ nên không đổi nghiệm, rồi thay $D^\ast$:

$$
2C(G) = \int \frac{\big( (b - c)\, p_{\text{data}} + (a - c)\, p_g \big)^2}{p_{\text{data}} + p_g}\, d\mathbf{x}.
$$

Nếu chọn $b - c = 1$ và $b - a = 2$ (ví dụ $a = -1, b = 1, c = 0$) thì $a - c = -1$ và tử số là $(p_{\text{data}} - p_g)^2 = \big(2p_g - (p_{\text{data}} + p_g)\big)^2$, nên

$$
2C(G) = \int \frac{\big(2p_g(\mathbf{x}) - (p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x}))\big)^2}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}\, d\mathbf{x},
$$

là phân kỳ **Pearson $\chi^2$** giữa $p_{\text{data}} + p_g$ và $2p_g$. Phân kỳ này bằng 0 khi và chỉ khi $p_g = p_{\text{data}}$. Trong thực hành, cách mã hóa phổ biến hơn là $a = 0$, $b = 1$, $c = 1$ (không thỏa điều kiện trên nhưng hoạt động tốt); HiFi-GAN dùng đúng cách mã hóa này (Kong và cộng sự, 2020).

Vì sao LSGAN ổn định hơn GAN gốc? Hàm sigmoid kết hợp log bão hòa với những mẫu bị phân loại đúng một cách "tự tin", kể cả khi chúng nằm rất xa dữ liệu thật. Hàm bình phương thì không: một mẫu giả có $D(\mathbf{x})$ ở xa $c$ vẫn nhận gradient tỉ lệ với khoảng cách, nên được kéo về phía ranh giới quyết định và về phía dữ liệu.

### 9.6.2. Hinge loss

Mất mát hinge (Lim và Ye, 2017) dùng một lề (*margin*):

$$
\mathcal{L}_D = \mathbb{E}_{p_{\text{data}}}\big[\max(0, 1 - D(\mathbf{x}))\big] + \mathbb{E}_{p_g}\big[\max(0, 1 + D(\mathbf{x}))\big],
\qquad
\mathcal{L}_G = -\mathbb{E}_{p_g}\big[D(\mathbf{x})\big].
$$

$D$ ngừng bị phạt khi mẫu thật có điểm vượt 1 và mẫu giả có điểm dưới $-1$, nên nó không bị thúc đẩy đến mức tự tin cực đoan. Hinge loss được dùng rộng rãi trong GAN ảnh quy mô lớn (Brock và cộng sự, 2019) và trong MelGAN (Kumar và cộng sự, 2019).

### 9.6.3. WGAN và WGAN-GP

Arjovsky, Chintala và Bottou (2017) đề xuất thay JSD bằng **khoảng cách Wasserstein-1** (khoảng cách "di chuyển đất", *earth mover's distance*):

$$
W(p, q) = \inf_{\gamma \in \Pi(p, q)} \mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim \gamma}\big[\lVert \mathbf{x} - \mathbf{y} \rVert\big],
$$

trong đó $\Pi(p, q)$ là tập mọi phân phối đồng thời có biên là $p$ và $q$: chi phí nhỏ nhất để "vận chuyển" khối lượng xác suất từ $p$ sang $q$. Quay lại ví dụ $\delta_0$ và $\delta_\theta$: $W = \lvert\theta\rvert$, có gradient $\operatorname{sign}(\theta)$ ở mọi nơi, luôn chỉ đúng hướng, trong khi JSD đứng yên ở $\log 2$. Khoảng cách Wasserstein phản ánh "xa bao nhiêu" ngay cả khi hai giá không giao nhau.

Tính trực tiếp infimum trên là bất khả thi, nhưng **đối ngẫu Kantorovich–Rubinstein** (Villani, 2008) cho

$$
W(p_{\text{data}}, p_g) = \sup_{\lVert f \rVert_L \le 1} \mathbb{E}_{p_{\text{data}}}[f(\mathbf{x})] - \mathbb{E}_{p_g}[f(\mathbf{x})],
$$

với supremum trên mọi hàm 1-Lipschitz. Bộ phân biệt, nay gọi là **critic**, đóng vai $f$ (không có sigmoid) và cực đại $\mathbb{E}_{p_{\text{data}}}[D] - \mathbb{E}_{p_g}[D]$; bộ sinh cực tiểu $-\mathbb{E}_{p_g}[D]$. Cái khó là ràng buộc Lipschitz. Bài báo gốc **cắt trọng số** (*weight clipping*) vào một khoảng nhỏ, một cách thô và dễ làm critic quá đơn giản. **WGAN-GP** (Gulrajani và cộng sự, 2017) thay bằng một phạt mềm: chuẩn gradient của một critic 1-Lipschitz tối ưu bằng 1 hầu khắp nơi trên các đoạn nối mẫu thật và mẫu giả, nên ta phạt

$$
\lambda\, \mathbb{E}_{\hat{\mathbf{x}}}\Big[ \big( \lVert \nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}}) \rVert_2 - 1 \big)^2 \Big], \qquad \hat{\mathbf{x}} = \epsilon\, \mathbf{x}_{\text{thật}} + (1 - \epsilon)\, \mathbf{x}_{\text{giả}},\; \epsilon \sim \mathcal{U}[0, 1],
$$

với $\lambda = 10$ theo bài báo gốc. Lưu ý: giá trị loss của critic chỉ là ước lượng của $W$ khi critic gần tối ưu; và phạt gradient đòi hỏi lan truyền ngược hai lần, tốn thêm tính toán.

**Góc nhìn tổng quát.** Nowozin và cộng sự (2016) chỉ ra rằng có thể xây dựng GAN cho **bất kỳ $f$-phân kỳ nào** qua biểu diễn biến phân của nó (f-GAN); GAN gốc, KL, KL ngược, Pearson $\chi^2$ đều là trường hợp riêng. WGAN thì thuộc một họ khác (khoảng cách vận chuyển tối ưu). Không có phân kỳ nào "đúng nhất"; trong thực tế, độ ổn định của động lực học quan trọng hơn lựa chọn phân kỳ.

### 9.6.4. Bảng tổng kết và mã cài đặt

Ký hiệu $\ell_r = D(\mathbf{x}_{\text{thật}})$, $\ell_f = D(G(\mathbf{z}))$ là điểm số thô (logit) của bộ phân biệt.

| Biến thể | Loss của $D$ | Loss của $G$ | Dùng trong TTS |
|---|---|---|---|
| GAN gốc (NS) | $-\log\sigma(\ell_r) - \log(1 - \sigma(\ell_f))$ | $-\log\sigma(\ell_f)$ | GAN-TTS, nhiều mô hình nghiên cứu |
| LSGAN (0/1/1) | $(\ell_r - 1)^2 + \ell_f^2$ | $(\ell_f - 1)^2$ | Parallel WaveGAN, HiFi-GAN, VITS |
| Hinge | $\max(0, 1 - \ell_r) + \max(0, 1 + \ell_f)$ | $-\ell_f$ | MelGAN |
| WGAN-GP | $\ell_f - \ell_r + \lambda\,\text{GP}$ | $-\ell_f$ | Ít dùng cho vocoder |

Đoạn mã dưới đây gom các hàm mất mát, hai phép phạt gradient, loss feature matching (mục 9.7.4) và minh họa chuẩn hóa phổ (mục 9.7.1) vào một chỗ, kèm kiểm thử: với một bộ phân biệt tuyến tính $D(\mathbf{x}) = \mathbf{w}^\top\mathbf{x} + b$, gradient theo $\mathbf{x}$ là $\mathbf{w}$, nên giá trị của hai phép phạt biết trước.

```python
import torch
import torch.nn as nn
import torch.nn.functional as Fn

# ---- Các cặp hàm mất mát (D, G); đầu vào là điểm số thô (logit) của bộ phân biệt ----
def d_loss_fn(kind, d_real, d_fake):
    if kind == "ns":        # GAN gốc: BCE, ổn định số học nhờ logsigmoid
        return -(Fn.logsigmoid(d_real).mean() + Fn.logsigmoid(-d_fake).mean())
    if kind == "lsgan":     # a = 0 (giả), b = 1 (thật)
        return ((d_real - 1) ** 2).mean() + (d_fake ** 2).mean()
    if kind == "hinge":
        return Fn.relu(1 - d_real).mean() + Fn.relu(1 + d_fake).mean()
    if kind == "wgan":      # critic: cực đại E[D(x)] - E[D(G(z))]
        return d_fake.mean() - d_real.mean()
    raise ValueError(kind)

def g_loss_fn(kind, d_fake):
    if kind == "ns":        # không bão hòa: -log D(G(z))
        return -Fn.logsigmoid(d_fake).mean()
    if kind == "lsgan":     # c = 1
        return ((d_fake - 1) ** 2).mean()
    if kind in ("hinge", "wgan"):
        return -d_fake.mean()
    raise ValueError(kind)

def r1_penalty(D, x_real, gamma=10.0):
    """R1 (Mescheder và cộng sự, 2018): (gamma/2) E ||grad_x D(x)||^2 trên dữ liệu thật."""
    x_real = x_real.detach().requires_grad_(True)
    grad, = torch.autograd.grad(D(x_real).sum(), x_real, create_graph=True)
    return 0.5 * gamma * grad.flatten(1).pow(2).sum(1).mean()

def gradient_penalty(D, x_real, x_fake, lam=10.0):
    """WGAN-GP (Gulrajani và cộng sự, 2017): lam * E (||grad D(x_hat)|| - 1)^2, x_hat nội suy."""
    eps = torch.rand(x_real.shape[0], *([1] * (x_real.dim() - 1)))
    x_hat = (eps * x_real + (1 - eps) * x_fake.detach()).requires_grad_(True)
    grad, = torch.autograd.grad(D(x_hat).sum(), x_hat, create_graph=True)
    return lam * (grad.flatten(1).norm(dim=1) - 1).pow(2).mean()

def feature_matching_loss(feats_real, feats_fake):
    """Feature matching kiểu MelGAN/HiFi-GAN: L1 giữa feature map thật và giả theo từng lớp của D."""
    return sum(Fn.l1_loss(f_fake, f_real.detach()) for f_real, f_fake in zip(feats_real, feats_fake))

# ---- Kiểm thử nhanh ----
torch.manual_seed(0)
d_real, d_fake = torch.randn(8, 1) + 1, torch.randn(8, 1) - 1    # D đang khá "tự tin"
for kind in ("ns", "lsgan", "hinge", "wgan"):
    print(f"{kind:6s} L_D={d_loss_fn(kind, d_real, d_fake).item():+.3f}"
          f"  L_G={g_loss_fn(kind, d_fake).item():+.3f}")

lin = nn.Linear(4, 1)                                    # D tuyến tính: grad_x D = w, ||w|| biết trước
x_r, x_f = torch.randn(16, 4), torch.randn(16, 4)
w_norm2 = lin.weight.pow(2).sum()
assert torch.allclose(r1_penalty(lin, x_r, gamma=2.0), w_norm2, atol=1e-6)
assert torch.allclose(gradient_penalty(lin, x_r, x_f, lam=1.0), (w_norm2.sqrt() - 1) ** 2, atol=1e-6)

# Feature matching: lấy feature map trung gian của một D tích chập nhỏ, bố cục (B, C, L)
convs = nn.ModuleList([nn.Conv1d(1, 8, 15, padding=7), nn.Conv1d(8, 8, 15, stride=4, padding=7)])
def d_with_feats(wav):                                   # wav: (B, 1, L)
    feats, h = [], wav
    for conv in convs:
        h = Fn.leaky_relu(conv(h), 0.1)
        feats.append(h)                                  # (B, 8, L), (B, 8, L/4)
    return feats
wav_real, wav_fake = torch.randn(2, 1, 256), torch.randn(2, 1, 256, requires_grad=True)
fm = feature_matching_loss(d_with_feats(wav_real), d_with_feats(wav_fake))
fm.backward()
assert wav_fake.grad is not None and fm.item() > 0
print(f"feature matching = {fm.item():.4f}")

# Chuẩn hóa phổ: sau vài lượt power iteration, chuẩn phổ của trọng số hiệu dụng xấp xỉ 1
sn = nn.utils.parametrizations.spectral_norm(nn.Linear(64, 64))
for _ in range(20):
    sn(torch.randn(1, 64))                               # mỗi lần forward ở chế độ train cập nhật u, v
print(f"chuẩn phổ của W/sigma(W): {torch.linalg.matrix_norm(sn.weight.detach(), ord=2):.3f}")
```
