# Chương 3. Tối ưu hóa và kỹ thuật huấn luyện

[Chương 2](ch02-hoc-may-va-mang-no-ron.md) cho ta cách tính gradient của hàm mất mát theo mọi tham số. Chương này trả lời câu hỏi tiếp theo, và trên thực tế là câu hỏi tốn nhiều công sức nhất của người huấn luyện mô hình: **dùng gradient đó như thế nào** để đi đến một bộ tham số tốt, nhanh, ổn định và tái lập được.

Có một khoảng cách lớn giữa "gradient descent" trong sách giáo khoa và vòng lặp huấn luyện của một mô hình TTS thật. Vòng lặp thật dùng AdamW chứ không phải SGD thuần, có lịch tốc độ học với giai đoạn khởi động, cắt gradient, chạy với độ chính xác hỗn hợp, tích lũy gradient khi bộ nhớ không đủ, giữ một bản trung bình động của trọng số, và đặt các lớp chuẩn hóa ở những vị trí được cân nhắc kỹ. Mỗi chi tiết ấy tồn tại vì một lý do cụ thể, và hầu hết các sự cố khi huấn luyện ("loss thành `NaN` sau hai giờ", "mô hình EMA tệ hơn mô hình thường", "huấn luyện tiếp từ checkpoint thì loss vọt lên") đều bắt nguồn từ việc hiểu sai một trong số chúng. Chương này đi qua từng chi tiết theo nhịp quen thuộc: vấn đề là gì, giải pháp nào, vì sao nó hoạt động, và trong TTS nó xuất hiện ở đâu.

!!! abstract "Mục tiêu học tập"
    Sau chương này, bạn có thể:

    - **Phân tích** sự hội tụ của gradient descent trên hàm bậc hai bằng trị riêng của Hessian và **giải thích** vai trò của số điều kiện, momentum và Nesterov.
    - **Suy ra** cơ chế hiệu chỉnh độ chệch của Adam và **giải thích** vì sao AdamW tách weight decay khỏi gradient.
    - **Thiết kế** lịch tốc độ học (warmup tuyến tính, cosine, lịch Noam) và **chọn** ngưỡng cắt gradient phù hợp.
    - **So sánh** BatchNorm, LayerNorm, RMSNorm, GroupNorm, InstanceNorm, weight normalization và spectral normalization theo trục chuẩn hóa và tình huống sử dụng trong TTS.
    - **Áp dụng** dropout, weight decay, early stopping và **đánh giá** rủi ro của tăng cường dữ liệu với tiếng nói có thanh điệu.
    - **Cấu hình** huấn luyện độ chính xác hỗn hợp (fp16/bf16, loss scaling), tích lũy gradient đúng với chuỗi độ dài thay đổi, và EMA trọng số.
    - **Cài đặt** một vòng lặp huấn luyện hoàn chỉnh với AdamW, warmup + cosine, AMP, cắt gradient, EMA và checkpoint đầy đủ.

!!! info "Kiến thức cần có"
    - [Chương 1](ch01-toan-nen-tang.md): trị riêng, SVD và chuẩn phổ (Mục 1.2.5–1.2.6), khai triển Taylor (Mục 1.3.2), ước lượng Monte Carlo (Mục 1.4.6).
    - [Chương 2](ch02-hoc-may-va-mang-no-ron.md): rủi ro thực nghiệm, MAP và điều chuẩn L2 (Mục 2.4), lan truyền ngược và autograd trong PyTorch (Mục 2.6).

## 3.1. Địa hình hàm mất mát: một vài trực giác

Hãy hình dung hàm mất mát $\mathcal{L}(\theta)$ như một **địa hình** trên không gian tham số hàng triệu chiều, và quá trình huấn luyện như một người leo núi trong sương mù, chỉ cảm nhận được độ dốc ngay dưới chân. Với mạng nơ-ron, địa hình này **không lồi**: nó có vô số điểm dừng. Vài trực giác giúp ta không quá bi quan và cũng không quá lạc quan:

- **Điểm yên ngựa nhiều hơn cực tiểu địa phương xấu.** Trong không gian nhiều chiều, một điểm dừng là cực tiểu thực sự chỉ khi độ cong dương theo *mọi* hướng. Phân tích và thực nghiệm (Dauphin và cộng sự, 2014) cho thấy các điểm dừng có loss cao thường là điểm yên ngựa, nơi luôn tồn tại hướng đi xuống; nhiễu của SGD và momentum giúp thoát khỏi chúng.
- **Nhiều cực tiểu tốt ngang nhau.** Vì đối xứng (hoán vị các nơ-ron trong một lớp cho cùng một hàm) và tham số dư thừa, có vô số bộ tham số khác nhau cho cùng chất lượng. Ta không cần tìm cực tiểu toàn cục, chỉ cần một vùng đủ tốt.
- **Cực tiểu "phẳng" và "nhọn".** Có giả thuyết cho rằng các cực tiểu nằm trong thung lũng rộng, phẳng tổng quát hóa tốt hơn cực tiểu hẹp, nhọn (Keskar và cộng sự, 2017). Tuy nhiên, "độ phẳng" phụ thuộc vào cách tham số hóa (Dinh và cộng sự, 2017), nên đây nên được xem là một trực giác hữu ích chứ không phải định luật. Các hình vẽ địa hình hai chiều (Li và cộng sự, 2018) cho thấy kết nối tắt và chuẩn hóa làm địa hình trơn hơn hẳn, phù hợp với kinh nghiệm rằng chúng giúp huấn luyện dễ hơn.
- **Không phải bài toán nào cũng là cực tiểu hóa.** Huấn luyện GAN, và do đó các vocoder như HiFi-GAN, là một **trò chơi** giữa hai mạng chứ không phải đi xuống một địa hình cố định: địa hình của bộ sinh thay đổi mỗi khi bộ phân biệt cập nhật. Nhiều kỹ thuật trong chương này (spectral normalization, tốc độ học riêng, $\beta_1$ thấp) được dùng chính để giữ trò chơi ấy ổn định ([Chương 9](../phan-2-mo-hinh-sinh/ch09-gan.md)).

## 3.2. Gradient descent

### 3.2.1. Quy tắc cập nhật

Gradient descent (GD) cập nhật tham số theo

$$
\theta_{k+1} = \theta_k - \eta\,\nabla_\theta\mathcal{L}(\theta_k),
$$

với $\eta > 0$ là **tốc độ học**. Có một cách nhìn giúp hiểu vai trò của $\eta$: bước GD chính là nghiệm của

$$
\theta_{k+1} = \arg\min_{\theta}\Big[\mathcal{L}(\theta_k) + \nabla\mathcal{L}(\theta_k)^\top(\theta - \theta_k) + \frac{1}{2\eta}\lVert\theta - \theta_k\rVert_2^2\Big],
$$

tức là cực tiểu hóa **xấp xỉ tuyến tính** của hàm mất mát (Mục 1.3.2), cộng với một số hạng phạt việc đi quá xa điểm hiện tại. Xấp xỉ tuyến tính chỉ đáng tin trong một vùng nhỏ, và $\eta$ quyết định ta "tin" nó đến đâu. Lấy gradient biểu thức trong ngoặc theo $\theta$ và cho bằng 0, bạn sẽ thu lại đúng quy tắc cập nhật ở trên.

### 3.2.2. Phân tích trên hàm bậc hai

Gần một cực tiểu $\theta^\star$, mọi hàm trơn đều xấp xỉ được bằng một hàm bậc hai $\mathcal{L}(\theta) \approx \mathcal{L}(\theta^\star) + \frac{1}{2}(\theta - \theta^\star)^\top\mathbf{H}(\theta - \theta^\star)$, với Hessian $\mathbf{H}$ đối xứng xác định dương. Khi đó $\nabla\mathcal{L}(\theta) = \mathbf{H}(\theta - \theta^\star)$, và sai số $\mathbf{e}_k = \theta_k - \theta^\star$ biến đổi theo

$$
\mathbf{e}_{k+1} = (\mathbf{I} - \eta\mathbf{H})\,\mathbf{e}_k .
$$

Phân rã $\mathbf{H} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ (Mục 1.2.5) và chiếu sai số lên các véc-tơ riêng, mỗi thành phần tiến triển **độc lập**: $e_{k,i} = (1 - \eta\lambda_i)^k e_{0,i}$. Từ đây rút ra ba kết luận then chốt.

1. **Điều kiện hội tụ.** Mọi thành phần co lại khi và chỉ khi $\lvert 1 - \eta\lambda_i \rvert < 1$ với mọi $i$, tức

    $$
    0 < \eta < \frac{2}{\lambda_{\max}} .
    $$

    Hướng có **độ cong lớn nhất** quyết định tốc độ học tối đa. Vượt ngưỡng này, thành phần đó dao động với biên độ tăng dần và GD phân kỳ: đây là cơ chế đằng sau hiện tượng "loss bùng nổ khi tốc độ học quá lớn".

2. **Tốc độ hội tụ.** Thành phần chậm nhất co lại với hệ số $\max_i\lvert 1 - \eta\lambda_i \rvert$. Chọn tối ưu $\eta^\star = 2/(\lambda_{\max} + \lambda_{\min})$ cho hệ số

    $$
    \rho_{\text{GD}} = \frac{\kappa - 1}{\kappa + 1}, \qquad \kappa = \frac{\lambda_{\max}}{\lambda_{\min}} .
    $$

    Số bước cần để giảm sai số đi một thừa số cố định tỷ lệ xấp xỉ với $\kappa$.

3. **Hiện tượng zíc-zắc.** Khi $\kappa$ lớn (thung lũng hẹp và dài), tốc độ học bị chặn bởi hướng dốc đứng, nên dọc theo đáy thung lũng thoai thoải ta chỉ nhích từng chút, trong khi theo hướng dốc ta nảy qua nảy lại.

Ta có thể làm $\kappa$ "hiệu dụng" nhỏ đi bằng hai cách: thay đổi **bài toán** (chuẩn hóa đầu vào và kích hoạt, Mục 3.9, làm các hướng có độ cong đồng đều hơn) hoặc thay đổi **thuật toán** (momentum, Mục 3.4; tốc độ học thích nghi cho từng tham số, Mục 3.5). Ta sẽ kiểm chứng bằng số ở cuối Mục 3.4.

## 3.3. SGD và minibatch

Hàm mất mát là trung bình trên $n$ mẫu, $\mathcal{L}(\theta) = \frac{1}{n}\sum_i\ell_i(\theta)$. Tính gradient đầy đủ cần duyệt toàn bộ dữ liệu cho **mỗi** bước, quá đắt với hàng trăm giờ âm thanh. **Gradient descent ngẫu nhiên** (*stochastic gradient descent*, SGD; Robbins và Monro, 1951) thay gradient đầy đủ bằng gradient trên một **minibatch** $\mathcal{B}$ gồm $B$ mẫu chọn ngẫu nhiên:

$$
\mathbf{g} = \frac{1}{B}\sum_{i \in \mathcal{B}}\nabla\ell_i(\theta), \qquad \theta \leftarrow \theta - \eta\,\mathbf{g} .
$$

Đây là một ước lượng Monte Carlo (Mục 1.4.6): **không chệch**, $\mathbb{E}[\mathbf{g}] = \nabla\mathcal{L}$, với hiệp phương sai giảm theo $1/B$. Một lượt duyệt hết dữ liệu gọi là một **epoch**; mỗi lần cập nhật gọi là một **bước** (*step*, *iteration*).

**Kích thước batch và tốc độ học.** Batch lớn hơn cho gradient ít nhiễu hơn, nên cho phép bước lớn hơn. Một quy tắc kinh nghiệm phổ biến cho SGD là **tỷ lệ tuyến tính**: nhân batch lên $k$ lần thì nhân tốc độ học lên $k$ lần, kèm một giai đoạn khởi động (Goyal và cộng sự, 2017). Quy tắc này chỉ đúng trong một phạm vi nhất định, và với Adam mối quan hệ khác hơn (thường gần với căn bậc hai), nên hãy coi nó là điểm xuất phát để dò, không phải công thức.

**Nhiễu không hẳn là xấu.** Nhiễu của SGD giúp thoát điểm yên ngựa và, theo nhiều quan sát, có tác dụng điều chuẩn ngầm. Mặt khác, nhiễu đặt ra một "sàn": với tốc độ học cố định, tham số không hội tụ hẳn mà dao động quanh cực tiểu với biên độ tăng theo $\eta$. Đó là một lý do chính để **giảm tốc độ học** về cuối quá trình huấn luyện (Mục 3.7).

**Trong TTS**, các câu dài ngắn rất khác nhau, nên "một batch $B$ câu" chứa số khung mel thay đổi mạnh từ batch này sang batch khác. Các hệ thống thực tế thường gom batch theo **tổng số khung** (*dynamic batching*) để ổn định cả bộ nhớ lẫn độ nhiễu của gradient ([Chương 6](ch06-pytorch-cho-tieng-noi.md)). Khi đó, cách lấy trung bình hàm mất mát (trên câu hay trên khung) cũng ảnh hưởng tới độ lớn gradient; ta quay lại điểm này ở Mục 3.12.

## 3.4. Momentum và Nesterov

**Trực giác.** Thả một quả bóng nặng lăn xuống thung lũng hẹp: quán tính làm nó tích lũy vận tốc dọc theo đáy thung lũng (nơi gradient luôn cùng hướng) và triệt tiêu dao động qua lại giữa hai sườn (nơi gradient đổi dấu liên tục).

**Momentum kiểu quả cầu nặng** (*heavy ball*, Polyak, 1964):

$$
\mathbf{v}_{k+1} = \beta\,\mathbf{v}_k - \eta\,\nabla\mathcal{L}(\theta_k), \qquad \theta_{k+1} = \theta_k + \mathbf{v}_{k+1},
$$

với $\beta \in [0, 1)$, thường là 0,9. Khai triển truy hồi, $\mathbf{v}_{k+1} = -\eta\sum_{j=0}^{k}\beta^{j}\nabla\mathcal{L}(\theta_{k-j})$: vận tốc là **tổng có trọng số giảm dần theo cấp số nhân** của các gradient quá khứ. Nếu gradient giữ nguyên hướng, bước đi thực tế lớn gấp $\sum_j\beta^j = 1/(1 - \beta)$ lần, tức 10 lần với $\beta = 0{,}9$. Nếu gradient đổi dấu luân phiên, các số hạng triệt tiêu nhau.

Trên hàm bậc hai, với $\eta$ và $\beta$ chọn tối ưu, momentum đạt hệ số co

$$
\rho_{\text{HB}} = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1},
$$

tức số bước tỷ lệ với $\sqrt{\kappa}$ thay vì $\kappa$. Với $\kappa = 10\,000$, đó là khác biệt giữa khoảng 100 lần và khoảng 10 000 lần.

**Momentum Nesterov** (Nesterov, 1983; Sutskever và cộng sự, 2013) tính gradient tại điểm "nhìn trước" $\theta_k + \beta\mathbf{v}_k$, nơi quán tính sắp đưa ta tới, thay vì tại $\theta_k$. Nhờ đó nó "phanh" sớm hơn khi sắp vượt quá đáy thung lũng.

Đoạn mã sau so sánh số bước cần để đưa sai số về dưới $10^{-6}$ trên hàm bậc hai hai chiều với $\kappa = 50$:

```python
import numpy as np

lam = np.array([1.0, 50.0])                 # trị riêng của Hessian: số điều kiện 50
grad = lambda x: lam * x                    # f(x) = 0.5 * sum(lam * x^2)
x0 = np.array([1.0, 1.0])

def run(lr, beta=0.0, nesterov=False, max_steps=5000, tol=1e-6):
    x, v = x0.copy(), np.zeros(2)
    for k in range(max_steps):
        g = grad(x + beta * v) if nesterov else grad(x)   # Nesterov: gradient tại điểm "nhìn trước"
        v = beta * v - lr * g
        x = x + v
        if np.linalg.norm(x) < tol:
            return k + 1
        if np.linalg.norm(x) > 1e6:
            return "phân kỳ"
    return f"> {max_steps}"

L = lam.max()
print("GD, lr = 1/L          :", run(1.0 / L))
print("GD, lr = 2/(L+mu)     :", run(2.0 / (L + lam.min())))
print("GD, lr = 2.05/L       :", run(2.05 / L))
print("Momentum 0.9, lr = 1/L:", run(1.0 / L, beta=0.9))
print("Nesterov 0.9, lr = 1/L:", run(1.0 / L, beta=0.9, nesterov=True))
k = lam.max() / lam.min()
beta_opt = ((np.sqrt(k) - 1) / (np.sqrt(k) + 1)) ** 2
lr_opt = 4.0 / (np.sqrt(L) + np.sqrt(lam.min())) ** 2
print(f"Heavy ball tối ưu (beta={beta_opt:.3f}):", run(lr_opt, beta=beta_opt))
```

| Phương pháp | Số bước |
|---|---|
| GD, $\eta = 1/\lambda_{\max}$ | 684 |
| GD, $\eta^\star = 2/(\lambda_{\max} + \lambda_{\min})$ | 355 |
| GD, $\eta = 2{,}05/\lambda_{\max}$ | phân kỳ |
| Momentum $\beta = 0{,}9$, $\eta = 1/\lambda_{\max}$ | 246 |
| Nesterov $\beta = 0{,}9$, $\eta = 1/\lambda_{\max}$ | 181 |
| Heavy ball với $(\eta, \beta)$ tối ưu | 66 |

Con số 684 khớp lý thuyết: thành phần chậm co với hệ số $1 - 1/50 = 0{,}98$ mỗi bước, và $\ln(10^{-6})/\ln(0{,}98) \approx 684$. Chỉ vượt ngưỡng $2/\lambda_{\max}$ một chút (2,05 thay vì 2) là đủ để phân kỳ.

!!! tip "Hai cách viết momentum"
    `torch.optim.SGD` cài đặt momentum theo dạng $\mathbf{b} \leftarrow \mu\mathbf{b} + \mathbf{g}$, $\theta \leftarrow \theta - \eta\mathbf{b}$, khác dạng ở trên một hệ số $\eta$ trong định nghĩa vận tốc. Hai dạng tương đương khi $\eta$ không đổi, và chỉ khác nhau chút ít khi tốc độ học thay đổi theo lịch. Biến thể Nesterov được bật bằng `nesterov=True`.

## 3.5. Tốc độ học thích nghi: AdaGrad, RMSProp, Adam

Momentum vẫn dùng **một** tốc độ học cho mọi tham số. Nhưng trong một mạng TTS, các tham số rất khác nhau: embedding của một âm vị hiếm nhận gradient khác 0 ở rất ít bước, còn trọng số của lớp chiếu cuối nhận gradient lớn ở mọi bước. Ý tưởng của nhóm phương pháp thích nghi là **chia tốc độ học của mỗi tham số cho một ước lượng độ lớn gradient của chính nó**, một dạng tiền điều kiện hóa (*preconditioning*) theo đường chéo, giúp giảm số điều kiện hiệu dụng.

**AdaGrad** (Duchi và cộng sự, 2011) cộng dồn bình phương gradient: $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$ và $\theta_t = \theta_{t-1} - \eta\,\mathbf{g}_t/(\sqrt{\mathbf{s}_t} + \epsilon)$, với các phép toán từng phần tử. Tham số hiếm khi được cập nhật giữ tốc độ học lớn. Nhược điểm: $\mathbf{s}_t$ chỉ tăng, nên tốc độ học hiệu dụng giảm dần về 0, không phù hợp với huấn luyện dài trên bài toán không lồi.

**RMSProp** (Tieleman và Hinton, 2012) thay tổng bằng **trung bình động** $\mathbf{s}_t = \rho\mathbf{s}_{t-1} + (1 - \rho)\mathbf{g}_t^2$, nên chỉ "nhớ" độ lớn gradient gần đây.

**Adam** (Kingma và Ba, 2015) kết hợp momentum (trung bình động của gradient) với RMSProp (trung bình động của bình phương gradient), cộng thêm một bước hiệu chỉnh độ chệch:

$$
\begin{aligned}
\mathbf{m}_t &= \beta_1\mathbf{m}_{t-1} + (1 - \beta_1)\,\mathbf{g}_t, &
\hat{\mathbf{m}}_t &= \frac{\mathbf{m}_t}{1 - \beta_1^t},\\
\mathbf{v}_t &= \beta_2\mathbf{v}_{t-1} + (1 - \beta_2)\,\mathbf{g}_t^2, &
\hat{\mathbf{v}}_t &= \frac{\mathbf{v}_t}{1 - \beta_2^t},\\
\theta_t &= \theta_{t-1} - \eta\,\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} .
\end{aligned}
$$

**Suy ra hiệu chỉnh độ chệch.** Vì $\mathbf{m}_0 = \mathbf{0}$, khai triển truy hồi cho

$$
\mathbf{m}_t = (1 - \beta_1)\sum_{i=1}^{t}\beta_1^{t-i}\,\mathbf{g}_i .
$$

Giả sử phân phối gradient ổn định, $\mathbb{E}[\mathbf{g}_i] = \mathbb{E}[\mathbf{g}]$ với mọi $i$. Lấy kỳ vọng và dùng tổng cấp số nhân $\sum_{j=0}^{t-1}\beta_1^j = (1 - \beta_1^t)/(1 - \beta_1)$:

$$
\mathbb{E}[\mathbf{m}_t] = (1 - \beta_1)\,\mathbb{E}[\mathbf{g}]\sum_{j=0}^{t-1}\beta_1^{j} = (1 - \beta_1^t)\,\mathbb{E}[\mathbf{g}] .
$$

Vậy $\mathbf{m}_t$ bị **kéo về 0** một thừa số $(1 - \beta_1^t)$, rất đáng kể khi $t$ nhỏ; chia cho thừa số đó cho ước lượng không chệch $\hat{\mathbf{m}}_t$. Lập luận y hệt cho $\mathbf{v}_t$ với $\beta_2$. Vì $\beta_2$ thường rất gần 1 (0,999), độ chệch của $\mathbf{v}_t$ kéo dài cỡ hàng nghìn bước, lâu hơn nhiều so với $\mathbf{m}_t$.

**Điều gì xảy ra nếu bỏ hiệu chỉnh?** Ở bước đầu tiên, $\mathbf{m}_1 = (1 - \beta_1)\mathbf{g}_1$ và $\mathbf{v}_1 = (1 - \beta_2)\mathbf{g}_1^2$, nên (bỏ qua $\epsilon$) bước cập nhật có độ lớn $\eta(1 - \beta_1)/\sqrt{1 - \beta_2} \approx 3{,}16\,\eta$ với các giá trị mặc định. Ngược với trực giác "khởi đầu từ 0 thì bước sẽ nhỏ", bỏ hiệu chỉnh làm các bước đầu **lớn hơn** khoảng ba lần, đúng lúc mô hình còn mong manh nhất. Có hiệu chỉnh, bước đầu có độ lớn đúng bằng $\eta$ (theo mỗi tọa độ). Đoạn mã sau cài đặt Adam, đối chiếu với `torch.optim.Adam`, và đo độ lớn bước đầu tiên:

```python
import torch

def adam_step(p, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, correct=True):
    m.mul_(b1).add_((1 - b1) * g)                  # trung bình động bậc 1
    v.mul_(b2).add_((1 - b2) * g * g)              # trung bình động bậc 2
    if correct:
        m_hat, v_hat = m / (1 - b1 ** t), v / (1 - b2 ** t)
    else:
        m_hat, v_hat = m, v
    p.sub_(lr * m_hat / (v_hat.sqrt() + eps))

torch.manual_seed(0)
target = torch.randn(10)
p_ours = torch.zeros(10)
p_ref = torch.zeros(10, requires_grad=True)
opt = torch.optim.Adam([p_ref], lr=1e-2)
m, v = torch.zeros(10), torch.zeros(10)
for t in range(1, 101):
    g = 2 * (p_ours - target)                      # gradient của ||p - target||^2
    adam_step(p_ours, g, m, v, t, lr=1e-2)
    opt.zero_grad()
    ((p_ref - target) ** 2).sum().backward()
    opt.step()
assert torch.allclose(p_ours, p_ref.detach(), atol=1e-6)
print("Adam tự viết khớp torch.optim.Adam sau 100 bước.")

# Bước đầu tiên có độ lớn bao nhiêu? (gradient rất nhỏ: 1e-4)
for correct in [True, False]:
    p, m, v = torch.zeros(1), torch.zeros(1), torch.zeros(1)
    adam_step(p, torch.tensor([1e-4]), m, v, t=1, lr=1e-3, correct=correct)
    print(f"hiệu chỉnh độ chệch={correct}: |bước 1| = {p.abs().item():.2e}")
```

Kết quả: $1{,}00 \times 10^{-3}$ khi có hiệu chỉnh và $3{,}15 \times 10^{-3}$ khi không. Chú ý gradient chỉ là $10^{-4}$, nhưng bước đi vẫn là $10^{-3}$: một đặc điểm quan trọng của Adam.

**Hiểu Adam như thế nào?**

- **Bước đi gần như bất biến với thang đo gradient.** Nhân hàm mất mát với 100 thì $\hat{\mathbf{m}}$ tăng 100 lần, $\sqrt{\hat{\mathbf{v}}}$ cũng tăng 100 lần, và bước cập nhật hầu như không đổi (trừ ảnh hưởng của $\epsilon$). Vì vậy với Adam, tốc độ học $\eta$ gần với **"độ thay đổi tối đa của mỗi tham số sau một bước"** hơn là một hệ số nhân gradient. Điều này giải thích vì sao $\eta = 10^{-3}$ hay $2 \times 10^{-4}$ dùng được cho rất nhiều mô hình khác nhau, và vì sao đổi trọng số giữa các số hạng mất mát ảnh hưởng tới *hướng* đi nhiều hơn là tới *độ dài* bước.
- **Tỷ số tín hiệu trên nhiễu.** $\hat{\mathbf{m}}/\sqrt{\hat{\mathbf{v}}}$ gần $\pm 1$ khi gradient của tham số đó ổn định về dấu, và gần 0 khi gradient dao động quanh 0. Adam tự động đi chậm theo những hướng "nhiễu".
- **$\epsilon$ không chỉ để tránh chia cho 0.** Khi $\sqrt{\hat{v}}$ nhỏ hơn nhiều so với $\epsilon$, Adam hành xử giống SGD với tốc độ học $\eta/\epsilon$. Các giá trị dùng trong thực tế dao động từ $10^{-9}$ (Transformer) đến $10^{-6}$ (Tacotron 2).
- **$\beta_2$ nhỏ hơn cho phản ứng nhanh hơn.** $\beta_2 = 0{,}999$ "nhớ" độ lớn gradient của khoảng 1 000 bước gần nhất. Với những bài toán mà thang đo gradient thay đổi đột ngột (Transformer giai đoạn đầu, GAN), người ta thường hạ $\beta_2$ xuống 0,98 hoặc 0,99.

**Cấu hình thường gặp trong TTS** (theo các bài báo gốc; luôn kiểm tra lại cấu hình của bản cài đặt bạn dùng):

| Mô hình | Bộ tối ưu | $(\beta_1, \beta_2)$, $\epsilon$ | Tốc độ học và lịch |
|---|---|---|---|
| Tacotron 2 (Shen và cộng sự, 2018) | Adam, L2 $10^{-6}$ | $(0{,}9;\, 0{,}999)$, $10^{-6}$ | $10^{-3}$, giảm theo hàm mũ về $10^{-5}$ bắt đầu từ bước 50 000 |
| FastSpeech 2 (Ren và cộng sự, 2021) | Adam | $(0{,}9;\, 0{,}98)$, $10^{-9}$ | Lịch Noam của Transformer (Mục 3.7) |
| HiFi-GAN (Kong và cộng sự, 2020) | AdamW, weight decay 0,01 | $(0{,}8;\, 0{,}99)$ | $2 \times 10^{-4}$, nhân 0,999 sau mỗi epoch |
| VITS (Kim và cộng sự, 2021) | AdamW | $(0{,}8;\, 0{,}99)$ | $2 \times 10^{-4}$, giảm theo hàm mũ mỗi epoch |

Để ý $\beta_1 = 0{,}8$ của các mô hình có thành phần GAN: momentum yếu hơn giúp bộ sinh và bộ phân biệt phản ứng nhanh hơn với nhau, tránh "đuổi theo" một mục tiêu đã cũ.

## 3.6. AdamW: weight decay tách rời

Ở [Chương 2](ch02-hoc-may-va-mang-no-ron.md) ta thấy điều chuẩn L2 tương ứng với tiên nghiệm Gauss. Có hai cách cài đặt "kéo trọng số về 0":

- **Điều chuẩn L2:** thêm $\frac{\lambda}{2}\lVert\theta\rVert_2^2$ vào hàm mất mát, tức thêm $\lambda\theta$ vào gradient.
- **Weight decay:** sau mỗi bước, co trọng số lại một tỷ lệ cố định, $\theta \leftarrow (1 - \eta\lambda)\theta$.

**Với SGD, hai cách là một.** Bước SGD với L2 là $\theta \leftarrow \theta - \eta(\mathbf{g} + \lambda\theta) = (1 - \eta\lambda)\theta - \eta\mathbf{g}$: đúng là weight decay.

**Với Adam, chúng khác nhau.** Khi thêm $\lambda\theta$ vào gradient, số hạng này đi qua $\mathbf{m}$ và $\mathbf{v}$ và bị **chia cho** $\sqrt{\hat{\mathbf{v}}}$. Tham số nào có lịch sử gradient lớn (tức $\hat{v}$ lớn) thì bị điều chuẩn **yếu** hơn, đúng ngược với mong muốn "phạt đều mọi trọng số". Loshchilov và Hutter (2019) đề xuất **tách rời** (*decouple*) weight decay khỏi bước thích nghi:

$$
\theta_t = \theta_{t-1} - \eta\Big(\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} + \lambda\,\theta_{t-1}\Big),
$$

trong đó $\hat{\mathbf{m}}_t$, $\hat{\mathbf{v}}_t$ chỉ được tính từ gradient của hàm mất mát "sạch". Đây là **AdamW**, bộ tối ưu mặc định của phần lớn mô hình TTS hiện đại.

Ba lưu ý thực hành:

1. Trong `torch.optim.AdamW`, weight decay được **nhân với tốc độ học**: mỗi bước làm $\theta \leftarrow \theta(1 - \eta\lambda)$. Khi tốc độ học giảm theo lịch, weight decay hiệu dụng cũng giảm theo. Giá trị mặc định của PyTorch là $\lambda = 0{,}01$.
2. `torch.optim.Adam(..., weight_decay=λ)` cài đặt **L2 kiểu cũ**, không phải AdamW. Nhầm hai thứ này là một lỗi phổ biến khi chép cấu hình từ bài báo.
3. Theo thông lệ, **không** áp weight decay cho độ lệch (bias) và tham số của các lớp chuẩn hóa (hệ số $\gamma$, $\beta$): kéo hệ số co giãn $\gamma$ về 0 không có ý nghĩa điều chuẩn mà chỉ làm hỏng chuẩn hóa. Với embedding thì ý kiến còn khác nhau. Cách làm chuẩn là chia tham số thành hai **nhóm** (*parameter groups*), như trong vòng lặp ở Mục 3.15.

## 3.7. Lịch tốc độ học

Một tốc độ học cố định hiếm khi là tối ưu: đầu quá trình ta muốn đi nhanh (nhưng không được làm mạng "vỡ"), cuối quá trình ta muốn đi chậm để vượt qua "sàn nhiễu" của SGD (Mục 3.3) và lắng xuống đáy thung lũng. **Lịch tốc độ học** (*learning-rate schedule*) mô tả $\eta$ như hàm của số bước $k$.

- **Giảm theo bậc thang** (*step decay*): $\eta_k = \eta_0\gamma^{\lfloor k/s \rfloor}$, giảm $\gamma$ lần (ví dụ 10 lần) sau mỗi $s$ bước.
- **Giảm theo hàm mũ**: $\eta_k = \eta_0\gamma^{k}$, tính theo bước hoặc theo epoch. HiFi-GAN nhân tốc độ học với 0,999 sau mỗi epoch; Tacotron 2 giảm theo hàm mũ từ $10^{-3}$ về $10^{-5}$.
- **Cosine** (Loshchilov và Hutter, 2017): $\eta_k = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\big(1 + \cos(\pi k/K)\big)$. Giảm chậm ở đầu, nhanh ở giữa, rồi lại chậm ở cuối khi tiến về $\eta_{\min}$. Ưu điểm lớn là chỉ có một tham số thực sự cần chọn: tổng số bước $K$.
- **Khởi động tuyến tính** (*linear warmup*): $\eta_k = \eta_{\max}\,k/W$ trong $W$ bước đầu, sau đó chuyển sang một lịch giảm. Tổ hợp "warmup + cosine" là lựa chọn mặc định rất phổ biến hiện nay.

**Vì sao cần warmup?** Có ít nhất ba lý do. (1) Ở những bước đầu, ước lượng $\hat{\mathbf{v}}$ của Adam được tính từ rất ít gradient nên có phương sai lớn, khiến bước cập nhật thất thường; Liu và cộng sự (2020) phân tích hiện tượng này và đề xuất RAdam như một cách sửa tự động. (2) Lúc khởi tạo, mạng có thể nằm ở vùng có độ cong lớn; một tốc độ học lớn ngay từ đầu vi phạm điều kiện $\eta < 2/\lambda_{\max}$ của Mục 3.2.2. (3) Với Transformer đặt LayerNorm *sau* nhánh dư (Post-LN), gradient ở các lớp gần đầu ra lúc khởi tạo rất lớn, và warmup gần như bắt buộc; đặt LayerNorm *trước* (Pre-LN) giảm đáng kể sự phụ thuộc này (Xiong và cộng sự, 2020; [Chương 5](ch05-transformer.md)).

**Lịch Noam** (Vaswani và cộng sự, 2017), đặt tên theo một tác giả của bài báo Transformer, kết hợp warmup tuyến tính với suy giảm theo căn bậc hai nghịch đảo:

$$
\eta_k = d_{\text{model}}^{-0.5}\cdot\min\big(k^{-0.5},\; k\cdot W^{-1.5}\big).
$$

Hai nhánh của $\min$ bằng nhau tại $k = W$, nên tốc độ học đạt đỉnh $\eta_{\text{peak}} = (d_{\text{model}}W)^{-0.5}$ tại cuối giai đoạn khởi động, rồi giảm theo $1/\sqrt{k}$. Thừa số $d_{\text{model}}^{-0.5}$ làm mô hình rộng hơn tự động dùng tốc độ học nhỏ hơn. Lịch này được nhiều mô hình TTS dựa trên Transformer dùng lại, chẳng hạn FastSpeech 2 (Ren và cộng sự, 2021). Trong thực hành người ta thường nhân thêm một hệ số tùy chỉnh để điều chỉnh đỉnh.

```python
import math
import torch

def warmup_cosine(step, warmup, total, min_ratio=0.0):
    """Hệ số nhân với tốc độ học gốc: tăng tuyến tính rồi giảm theo cosine."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

def noam(step, d_model=512, warmup=4000):
    """Lịch của Vaswani và cộng sự (2017); trả về tốc độ học tuyệt đối."""
    step = max(step, 1)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)

for s in [0, 1000, 3999, 4000, 8000, 16000, 100000]:
    print(f"step {s:>6}: warmup+cosine x {warmup_cosine(s, 4000, 100000):.3f} | "
          f"noam lr = {noam(s):.2e}")

# Dùng với PyTorch: LambdaLR nhân lr gốc với hệ số trả về
model = torch.nn.Linear(4, 4)
opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, 100, 1000))
lrs = []
for step in range(1000):
    lrs.append(opt.param_groups[0]["lr"])          # lr dùng cho bước này
    opt.step()                                     # (bỏ qua forward/backward cho gọn)
    sched.step()                                   # gọi SAU optimizer.step(), mỗi bước
print(f"lr cực đại = {max(lrs):.2e} tại bước {lrs.index(max(lrs))}, lr cuối = {lrs[-1]:.2e}")
```

Với $d_{\text{model}} = 512$ và $W = 4\,000$, lịch Noam đạt đỉnh khoảng $7{,}0 \times 10^{-4}$, còn khoảng $4{,}9 \times 10^{-4}$ ở bước 8 000 và $1{,}4 \times 10^{-4}$ ở bước 100 000. Lưu ý rằng lịch Noam không bao giờ về 0, khác với cosine.

!!! tip "Chọn tốc độ học đỉnh như thế nào?"
    Một cách nhanh là **thử nghiệm quét tốc độ học** (*LR range test*): huấn luyện vài trăm bước trong khi tăng $\eta$ theo cấp số nhân, rồi chọn giá trị nhỏ hơn vài lần so với điểm mà loss bắt đầu tăng vọt. Khi fine-tune một mô hình đã huấn luyện trước (rất phổ biến với tiếng Việt, xem hộp "Góc tiếng Việt" ở Mục 3.10), tốc độ học đỉnh thường được chọn nhỏ hơn so với lúc huấn luyện từ đầu, kèm một giai đoạn warmup ngắn.

## 3.8. Cắt gradient

Thỉnh thoảng một batch "xấu" (một câu cực dài, một bản ghi lỗi, một điểm mà attention vừa đổi hướng) sinh ra gradient lớn gấp hàng trăm lần bình thường. Một bước cập nhật như vậy có thể phá hủy nhiều giờ huấn luyện. Ở các mạng hồi quy, hiện tượng **gradient bùng nổ** xảy ra một cách có hệ thống vì gradient là tích của nhiều Jacobian theo thời gian (Pascanu và cộng sự, 2013; [Chương 4](ch04-cnn-rnn-mo-hinh-chuoi.md)).

**Cắt theo chuẩn** (*clip by norm*) co véc-tơ gradient **toàn cục** (ghép mọi tham số lại) khi chuẩn của nó vượt ngưỡng $c$:

$$
\mathbf{g} \leftarrow \mathbf{g}\cdot\min\Big(1,\; \frac{c}{\lVert\mathbf{g}\rVert_2}\Big).
$$

Phép này **giữ nguyên hướng** của gradient và chỉ giới hạn độ dài bước. Ngược lại, **cắt theo giá trị** (kẹp từng phần tử vào $[-c, c]$) làm thay đổi hướng và ít được dùng hơn. Trong PyTorch, `torch.nn.utils.clip_grad_norm_(params, max_norm)` thực hiện cắt theo chuẩn và **trả về chuẩn trước khi cắt**.

Hai lời khuyên thực hành:

- **Hãy ghi nhật ký chuẩn gradient** ở mỗi bước. Đây là một trong những tín hiệu chẩn đoán giá trị nhất: chuẩn gradient tăng vọt thường báo trước loss `NaN` vài trăm bước, và một chuẩn gradient luôn bị cắt nghĩa là ngưỡng quá thấp hoặc tốc độ học quá cao ([Chương 37](../phan-9-thuc-tien/ch37-huan-luyen-va-debug.md)).
- **Ngưỡng điển hình.** Nhiều bản cài đặt Tacotron 2 và các mô hình Transformer dùng ngưỡng cỡ 1; các mô hình khác dùng từ vài đơn vị trở lên. Hãy xem nó là "lưới an toàn" chống bước cập nhật bất thường, chứ không phải công cụ điều chỉnh tốc độ học thường xuyên.
