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

## 3.9. Chuẩn hóa

### 3.9.1. Vì sao chuẩn hóa giúp huấn luyện

Mục 3.2.2 cho thấy tốc độ hội tụ phụ thuộc vào số điều kiện: địa hình càng "méo" (độ cong rất khác nhau theo các hướng), GD càng chậm. Chuẩn hóa đầu vào của mỗi lớp về trung bình 0, phương sai 1 làm các hướng có thang đo tương đương, nhờ đó địa hình tròn trịa hơn. Động cơ ban đầu của BatchNorm được phát biểu là giảm "dịch chuyển hiệp biến nội tại" (*internal covariate shift*) (Ioffe và Szegedy, 2015); về sau, Santurkar và cộng sự (2018) lập luận rằng tác dụng chính là làm địa hình hàm mất mát trơn hơn. Dù giải thích nào đúng hơn, hiệu quả thực nghiệm là rõ ràng: chuẩn hóa cho phép tốc độ học lớn hơn, ít nhạy với khởi tạo hơn, và ổn định các mạng rất sâu.

Mọi phép chuẩn hóa theo kích hoạt đều có cùng dạng:

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad y = \gamma\,\hat{x} + \beta,
$$

với $\gamma$, $\beta$ là tham số học được (khôi phục khả năng biểu diễn nếu chuẩn hóa "quá tay"). Chúng chỉ khác nhau ở **trục tính thống kê** $\mu$, $\sigma^2$. Với tensor bố cục `(B, C, T)` của `Conv1d`:

| Phép chuẩn hóa | Thống kê tính trên | Phụ thuộc batch? | Hay gặp ở |
|---|---|---|---|
| BatchNorm | $(B, T)$, riêng từng kênh | Có | Encoder và post-net của Tacotron 2 |
| LayerNorm | $C$, riêng từng (mẫu, khung) | Không | Transformer, FastSpeech, bộ mã hóa văn bản của VITS |
| RMSNorm | $C$ (chỉ căn trung bình bình phương) | Không | Các LLM như LLaMA và hệ TTS dựa trên LLM |
| InstanceNorm | $T$, riêng từng (mẫu, kênh) | Không | Chuyển đổi giọng nói, tách thông tin người nói |
| GroupNorm | $(C/G, T)$, riêng từng (mẫu, nhóm) | Không | U-Net của mô hình khuếch tán |

### 3.9.2. BatchNorm

**BatchNorm** tính $\mu_c$, $\sigma_c^2$ cho mỗi kênh $c$ trên toàn bộ batch và trục thời gian. Khi huấn luyện, nó dùng thống kê của batch hiện tại và đồng thời cập nhật một **trung bình chạy** (*running mean/var*, mặc định với hệ số động lượng 0,1 trong PyTorch); khi suy luận (`model.eval()`), nó dùng trung bình chạy. Ba hệ quả quan trọng cho TTS:

1. **Phụ thuộc vào thành phần batch.** Đầu ra của một câu phụ thuộc vào những câu khác cùng batch; với batch nhỏ, thống kê nhiễu.
2. **Khung đệm làm sai thống kê.** Các câu trong batch được đệm (*padding*) tới cùng độ dài. Nếu không che (*mask*), các khung đệm (thường bằng 0) bị tính vào trung bình và phương sai. Log-mel thường mang giá trị âm (cỡ $-5$ đến $-10$ tùy cách chuẩn hóa), nên các số 0 đệm vào làm thống kê sai lệch nghiêm trọng, như đoạn mã bên dưới cho thấy.
3. **Khác biệt giữa huấn luyện và suy luận.** Nếu phân phối kích hoạt lúc suy luận khác lúc huấn luyện (ví dụ mô hình tự hồi quy chạy tự do thay vì teacher forcing), thống kê chạy có thể không còn phù hợp.

Dù vậy, BatchNorm vẫn hoạt động tốt trong các khối tích chập của Tacotron 2 (ba lớp tích chập của encoder và năm lớp của post-net đều có BatchNorm), nơi batch đủ lớn và thống kê được trung bình trên rất nhiều khung.

### 3.9.3. LayerNorm và RMSNorm

**LayerNorm** (Ba và cộng sự, 2016) chuẩn hóa mỗi véc-tơ đặc trưng $\mathbf{h}_t \in \mathbb{R}^{C}$ **một cách độc lập**: thống kê tính trên $C$ kênh của chính khung đó. Do đó nó không phụ thuộc batch, không bị ảnh hưởng bởi khung đệm của câu khác, và hành xử giống hệt nhau khi huấn luyện và suy luận. Đó là lý do LayerNorm là lựa chọn mặc định trong Transformer và hầu hết mô hình TTS không tự hồi quy.

**RMSNorm** (Zhang và Sennrich, 2019) bỏ bước trừ trung bình và bỏ $\beta$: $y = \gamma \odot \mathbf{h}/\sqrt{\frac{1}{C}\sum_c h_c^2 + \epsilon}$. Nó rẻ hơn một chút và, theo báo cáo gốc, cho chất lượng tương đương LayerNorm trong nhiều tác vụ. Các LLM hiện đại như LLaMA dùng RMSNorm, và các hệ TTS dựng trên những LLM này thừa hưởng nó ([Chương 34](../phan-8-mo-hinh-lon/ch34-he-thong-lai-hien-dai.md)).

!!! warning "LayerNorm với bố cục `(B, C, T)`"
    `nn.LayerNorm(C)` chuẩn hóa **chiều cuối cùng**. Với tensor `(B, C, T)` của `Conv1d`, gọi trực tiếp `nn.LayerNorm(C)` sẽ báo lỗi hình dạng, còn `nn.LayerNorm(T)` sẽ chạy nhưng chuẩn hóa nhầm theo thời gian. Cách đúng là chuyển vị sang `(B, T, C)`, chuẩn hóa, rồi chuyển vị lại; nhiều bản cài đặt (ví dụ VITS) đóng gói thao tác này trong một lớp LayerNorm riêng cho bố cục kênh-trước.

### 3.9.4. InstanceNorm và GroupNorm

**InstanceNorm** (Ulyanov và cộng sự, 2016) tính thống kê trên trục thời gian cho từng (mẫu, kênh). Với đặc trưng tiếng nói, trung bình và phương sai theo thời gian của mỗi kênh mang nhiều thông tin **toàn cục** của câu: âm sắc người nói, kênh thu âm. InstanceNorm xóa đi những thông tin đó, nên nó được dùng để tách nội dung khỏi người nói trong chuyển đổi giọng nói (Chou và cộng sự, 2019). Chiều ngược lại, **AdaIN** (Huang và Belongie, 2017) chuẩn hóa rồi "tiêm" lại phong cách bằng các $\gamma$, $\beta$ dự đoán từ một véc-tơ điều kiện; đây là họ hàng gần của LayerNorm có điều kiện dùng trong thích nghi người nói ([Chương 28](../phan-7-e2e-bieu-cam/ch28-da-nguoi-noi-voice-cloning.md)) và AdaLN trong DiT ([Chương 5](ch05-transformer.md)).

**GroupNorm** (Wu và He, 2018) chia $C$ kênh thành $G$ nhóm và chuẩn hóa trong từng nhóm, nằm giữa LayerNorm ($G = 1$) và InstanceNorm ($G = C$). Nó không phụ thuộc batch nên phù hợp khi batch nhỏ, và là lựa chọn phổ biến trong các U-Net của mô hình khuếch tán (Ho và cộng sự, 2020).

Đoạn mã sau cài đặt thủ công năm phép chuẩn hóa, đối chiếu với các mô-đun PyTorch, rồi minh họa tác hại của khung đệm với BatchNorm:

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
B, C, T = 4, 8, 50
x = torch.randn(B, C, T)                                   # (B, C, T) bố cục của Conv1d
eps = 1e-5

def standardize(x, dims):
    mu = x.mean(dim=dims, keepdim=True)
    var = x.var(dim=dims, keepdim=True, unbiased=False)
    return (x - mu) / torch.sqrt(var + eps)

# BatchNorm: thống kê trên (B, T) cho mỗi kênh
bn = nn.BatchNorm1d(C).train()
assert torch.allclose(bn(x), standardize(x, (0, 2)), atol=1e-5)
# LayerNorm theo kênh tại từng khung: chuyển về (B, T, C) rồi chuẩn hóa chiều cuối
ln = nn.LayerNorm(C)
y_ln = ln(x.transpose(1, 2)).transpose(1, 2)
assert torch.allclose(y_ln, standardize(x, (1,)), atol=1e-5)
# InstanceNorm: thống kê trên T cho mỗi (mẫu, kênh)
inorm = nn.InstanceNorm1d(C)
assert torch.allclose(inorm(x), standardize(x, (2,)), atol=1e-5)
# GroupNorm với 2 nhóm: thống kê trên (C/2, T) cho mỗi (mẫu, nhóm)
gn = nn.GroupNorm(2, C)
ref = standardize(x.view(B, 2, C // 2, T), (2, 3)).view(B, C, T)
assert torch.allclose(gn(x), ref, atol=1e-5)
# RMSNorm: không trừ trung bình, chỉ chia cho căn trung bình bình phương
rms = nn.RMSNorm(C, eps=eps)
xt = x.transpose(1, 2)                                     # (B, T, C)
assert torch.allclose(rms(xt), xt / torch.sqrt(xt.pow(2).mean(-1, keepdim=True) + eps), atol=1e-5)
print("Năm phép chuẩn hóa khớp với cài đặt thủ công.")

# Đệm (padding) làm sai lệch thống kê của BatchNorm
logmel = torch.randn(B, C, T) - 5.0                                 # log-mel thường âm, quanh -5
lengths = torch.tensor([50, 30, 20, 10])
mask = (torch.arange(T)[None, :] < lengths[:, None]).unsqueeze(1)   # (B, 1, T)
x_pad = logmel * mask                                               # khung đệm = 0
naive_mean = x_pad.mean(dim=(0, 2))                                 # BatchNorm "ngây thơ"
naive_std = x_pad.std(dim=(0, 2), unbiased=False)
valid = logmel.transpose(1, 2)[mask.squeeze(1)]                     # (số khung hợp lệ, C)
print(f"kênh 0 - có đệm: mean {naive_mean[0]:.2f}, std {naive_std[0]:.2f} | "
      f"chỉ khung hợp lệ: mean {valid[:, 0].mean():.2f}, std {valid[:, 0].std():.2f}")
```

Chỉ 110 trên 200 khung là hợp lệ. BatchNorm "ngây thơ" ước lượng trung bình khoảng $-2{,}8$ và độ lệch chuẩn khoảng $2{,}6$, trong khi giá trị đúng là khoảng $-5{,}1$ và $0{,}95$: sau chuẩn hóa, các khung thật bị dời và co lại sai hẳn. Nếu buộc phải dùng BatchNorm với chuỗi có đệm, hãy tính thống kê có mặt nạ (Bài tập 7).

### 3.9.5. Weight normalization

Các phép ở trên chuẩn hóa **kích hoạt**. **Weight normalization** (Salimans và Kingma, 2016) thay vào đó tham số hóa lại **trọng số**: mỗi véc-tơ trọng số (ứng với một kênh đầu ra) được viết thành

$$
\mathbf{w} = g\,\frac{\mathbf{v}}{\lVert\mathbf{v}\rVert_2},
$$

với $g$ vô hướng điều khiển **độ lớn** và $\mathbf{v}$ điều khiển **hướng**. Tách hai thành phần này giúp gradient descent ổn định hơn: cập nhật theo $\mathbf{v}$ chỉ xoay hướng mà không vô tình thay đổi độ lớn. Ưu điểm quyết định với vocoder là weight normalization **không dùng thống kê của batch hay của tín hiệu**, nên không gây phụ thuộc giữa các mẫu và không làm méo tín hiệu âm thanh theo kiểu chuẩn hóa kích hoạt có thể gây ra. MelGAN, HiFi-GAN và bộ giải mã của VITS dùng weight normalization trong hầu hết các lớp tích chập ([Chương 25](../phan-6-vocoder/ch25-gan-vocoder.md)).

Khi suy luận, phép tham số hóa lại chỉ tốn thêm tính toán, nên người ta **gộp** $g$ và $\mathbf{v}$ thành một trọng số thường trước khi xuất mô hình (hàm `remove_weight_norm` trong mã HiFi-GAN). Trong các phiên bản PyTorch gần đây, API được khuyến nghị là `torch.nn.utils.parametrizations.weight_norm`; hàm cũ `torch.nn.utils.weight_norm` đã bị đánh dấu lỗi thời, và mã nguồn cũ của nhiều dự án TTS vẫn dùng nó. Hai API lưu tham số dưới tên khác nhau trong `state_dict` (`weight_g`/`weight_v` so với `parametrizations.weight.original0`/`original1`), một nguồn lỗi thường gặp khi nạp checkpoint cũ.

### 3.9.6. Spectral normalization

**Spectral normalization** (Miyato và cộng sự, 2018) chia ma trận trọng số cho giá trị kỳ dị lớn nhất của nó, $\mathbf{W}_{\text{SN}} = \mathbf{W}/\sigma_{\max}(\mathbf{W})$. Theo Mục 1.2.6, $\sigma_{\max}$ là hằng số Lipschitz của phép biến đổi tuyến tính, nên mỗi lớp sau chuẩn hóa có hằng số Lipschitz không quá 1; ghép với các hàm kích hoạt 1-Lipschitz như Leaky ReLU, cả mạng bị chặn Lipschitz. Với **bộ phân biệt** của GAN, điều này ngăn nó trở nên quá "sắc", vốn là nguyên nhân gradient cho bộ sinh trở nên vô dụng hoặc bất ổn ([Chương 9](../phan-2-mo-hinh-sinh/ch09-gan.md)). $\sigma_{\max}$ được ước lượng rẻ bằng phép lặp lũy thừa (Mục 1.2.6): PyTorch chạy một vòng lặp mỗi lần forward ở chế độ huấn luyện và giữ lại véc-tơ ước lượng giữa các lần gọi. Với lớp tích chập, trọng số được trải thành ma trận `(C_out, C_in * k)` trước khi tính; chuẩn phổ của ma trận này chỉ là một xấp xỉ của hằng số Lipschitz thực sự của phép tích chập, nhưng đủ tốt trong thực hành. HiFi-GAN dùng spectral normalization cho bộ phân biệt con đầu tiên của bộ phân biệt đa thang (MSD), còn các bộ phân biệt con khác dùng weight normalization ([Chương 25](../phan-6-vocoder/ch25-gan-vocoder.md)).

```python
import torch
import torch.nn as nn
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

torch.manual_seed(0)
# Weight normalization: w = g * v / ||v||, chuẩn tính riêng cho từng kênh ra (dim=0)
conv = weight_norm(nn.Conv1d(16, 32, kernel_size=7, padding=3))
g = conv.parametrizations.weight.original0                # (32, 1, 1)  độ lớn
v = conv.parametrizations.weight.original1                # (32, 16, 7) hướng
w = g * v / v.norm(dim=(1, 2), keepdim=True)
assert torch.allclose(conv.weight, w, atol=1e-6)
# Trước khi xuất mô hình để suy luận: gộp lại thành một tensor trọng số thường
parametrize.remove_parametrizations(conv, "weight")
assert isinstance(conv.weight, nn.Parameter)

# Spectral normalization: chia trọng số cho giá trị kỳ dị lớn nhất (ước lượng bằng power iteration)
lin = spectral_norm(nn.Linear(64, 64))
x = torch.randn(8, 64)
for _ in range(20):                                      # mỗi forward ở chế độ train làm 1 vòng lặp
    lin(x)
W = lin.weight.detach()                                  # trọng số đã chuẩn hóa
print(f"giá trị kỳ dị lớn nhất sau spectral norm: {torch.linalg.matrix_norm(W, ord=2):.4f}")
```

Sau 20 lần forward, giá trị kỳ dị lớn nhất của trọng số đã chuẩn hóa xấp xỉ 1,000.

## 3.10. Điều chuẩn

Điều chuẩn (*regularization*) là mọi thứ ta làm để giảm quá khớp mà không giảm năng lực biểu diễn một cách thô bạo.

**Weight decay.** Đã bàn ở Mục 3.6 (góc nhìn MAP ở [Chương 2](ch02-hoc-may-va-mang-no-ron.md)). Giá trị 0,01 với AdamW là điểm xuất phát phổ biến.

**Dropout** (Srivastava và cộng sự, 2014). Khi huấn luyện, mỗi kích hoạt bị đặt về 0 với xác suất $p$; các kích hoạt còn lại được nhân với $1/(1 - p)$ (*inverted dropout*) để kỳ vọng không đổi:

$$
\tilde{h} = \frac{m\,h}{1 - p}, \quad m \sim \operatorname{Bernoulli}(1 - p) \quad\Longrightarrow\quad \mathbb{E}[\tilde{h}] = h .
$$

Khi suy luận, dropout tắt và không cần co giãn gì thêm. Dropout buộc mạng không dựa dẫm vào một vài nơ-ron cụ thể và có thể hiểu như huấn luyện đồng thời một tập hợp lớn các mạng con dùng chung trọng số. Trong TTS, các mô hình Transformer thường dùng dropout cỡ 0,1–0,2. Hai trường hợp đặc biệt đáng nhớ của Tacotron 2: pre-net dùng dropout 0,5 **kể cả khi suy luận**, để tạo biến thiên và vì pre-net đóng vai trò nút thắt thông tin giúp attention học được; các lớp LSTM dùng **zoneout** (Krueger và cộng sự, 2017), một biến thể cho mạng hồi quy giữ nguyên ngẫu nhiên một phần trạng thái ẩn của bước trước thay vì đặt về 0 ([Chương 20](../phan-5-mo-hinh-am-hoc/ch20-tacotron.md)).

**Tăng cường dữ liệu** (*data augmentation*). Trong nhận dạng tiếng nói, tăng cường rất hiệu quả (thêm nhiễu, vang, nhiễu loạn tốc độ, che dải tần và đoạn thời gian kiểu SpecAugment; Park và cộng sự, 2019). Trong TTS, bức tranh khác hẳn: **đầu ra** của mô hình là âm thanh sạch của một giọng cụ thể, nên mọi biến đổi lên mục tiêu đều có nguy cơ dạy mô hình sinh ra chính những biến dạng đó. Tăng cường vì vậy thường được áp dụng cho **đầu vào** của các thành phần phụ (bộ mã hóa người nói, bộ mã hóa tham chiếu), cho việc cắt đoạn ngẫu nhiên khi huấn luyện vocoder, hoặc ở phía văn bản (nhiều cách viết khác nhau của cùng một cách đọc), thay vì cho mel mục tiêu.

!!! note "Góc tiếng Việt"
    - **Tăng cường cao độ và thanh điệu.** Dịch cao độ đồng đều (nhân F0 với một hằng số) tương ứng với cộng một hằng số trên thang log/bán cung, nên *hình dạng* đường nét thanh điệu được giữ nguyên; đây là biến đổi tương đối an toàn nếu biên độ nhỏ (dù có thể gây méo âm sắc). Ngược lại, mọi biến đổi làm phẳng, kéo giãn không đều hoặc làm mượt mạnh đường F0 có thể phá hủy thông tin thanh điệu, thứ mang nghĩa từ vựng trong tiếng Việt.
    - **Dữ liệu ít, fine-tune nhiều.** Với vài giờ dữ liệu một giọng, quá khớp đến nhanh. Cách phổ biến là fine-tune từ một checkpoint đã huấn luyện trên dữ liệu lớn (đa ngôn ngữ hoặc tiếng Việt nhiều người nói) với tốc độ học nhỏ hơn lúc huấn luyện từ đầu, warmup ngắn, theo dõi chặt tập kiểm định và **nghe** mẫu thường xuyên. Đừng quên kiểm tra rằng bộ ký hiệu đầu vào của checkpoint gốc bao phủ đầy đủ các ký tự và thanh điệu tiếng Việt ([Chương 18](../phan-4-frontend/ch18-g2p-bieu-dien-dau-vao.md), [Chương 37](../phan-9-thuc-tien/ch37-huan-luyen-va-debug.md)).

**Dừng sớm** (*early stopping*). Theo dõi một thước đo trên tập kiểm định và giữ lại checkpoint tốt nhất. Với TTS, loss kiểm định của mô hình huấn luyện bằng teacher forcing hoặc của GAN thường phản ánh chất lượng nghe rất kém ([Chương 2](ch02-hoc-may-va-mang-no-ron.md)), nên hãy dùng thêm các thước đo trên âm thanh sinh ra thật sự (tỷ lệ lỗi ký tự qua ASR, độ giống giọng, điểm MOS dự đoán) và lưu checkpoint định kỳ để so sánh bằng tai ([Chương 38](../phan-9-thuc-tien/ch38-danh-gia-tts.md)).

**Làm mượt nhãn** (*label smoothing*; Szegedy và cộng sự, 2016) thay nhãn one-hot bằng $(1 - \alpha)\mathbf{y} + \alpha/K$, ngăn mô hình phân loại trở nên quá tự tin. Kỹ thuật này đôi khi được dùng khi huấn luyện các mô hình ngôn ngữ trên token âm thanh ([Chương 32](../phan-8-mo-hinh-lon/ch32-codec-language-model.md)).

## 3.11. Huấn luyện với độ chính xác hỗn hợp

### 3.11.1. Các định dạng số thực

| Định dạng | Bit (dấu, mũ, phần định trị) | Giá trị lớn nhất | Số chuẩn nhỏ nhất | Độ phân giải tương đối $\epsilon$ |
|---|---|---|---|---|
| fp32 | 1, 8, 23 | $\approx 3{,}4 \times 10^{38}$ | $\approx 1{,}2 \times 10^{-38}$ | $\approx 1{,}2 \times 10^{-7}$ |
| fp16 | 1, 5, 10 | $65\,504$ | $\approx 6{,}1 \times 10^{-5}$ | $\approx 9{,}8 \times 10^{-4}$ |
| bf16 | 1, 8, 7 | $\approx 3{,}4 \times 10^{38}$ | $\approx 1{,}2 \times 10^{-38}$ | $\approx 7{,}8 \times 10^{-3}$ |

Tính toán ở 16 bit nhanh hơn nhiều trên GPU có lõi chuyên dụng (*tensor core*) và tiết kiệm một nửa bộ nhớ cho kích hoạt. Nhưng hai định dạng 16 bit có điểm yếu ngược nhau:

- **fp16** có **dải giá trị hẹp**: giá trị trên 65 504 thành `inf` (tràn trên), và gradient nhỏ hơn khoảng $6 \times 10^{-8}$ (ngưỡng của số không chuẩn) thành 0 (tràn dưới). Gradient của các mô hình sâu thường rất nhỏ, nên tràn dưới là vấn đề thực sự.
- **bf16** có cùng dải giá trị với fp32 nên hầu như không tràn, nhưng chỉ có 7 bit định trị: $1 + 1/256$ làm tròn thành đúng $1$. Phép cộng dồn nhiều số hạng nhỏ vào một số lớn dễ mất chính xác.

### 3.11.2. Autocast và loss scaling

**Độ chính xác hỗn hợp** (*mixed precision*; Micikevicius và cộng sự, 2018) giữ **bản gốc của trọng số ở fp32** và chỉ thực hiện các phép toán đắt (nhân ma trận, tích chập) ở 16 bit. Trong PyTorch, `torch.autocast` tự động chọn kiểu dữ liệu cho từng phép toán: nhân ma trận và tích chập chạy ở 16 bit, còn các phép nhạy cảm về số học (softmax, log, các phép rút gọn như tổng, các hàm mất mát, chuẩn hóa) được giữ ở fp32.

Với fp16, ta còn cần **loss scaling**: nhân hàm mất mát với một hệ số lớn $S$ trước khi lan truyền ngược, để các gradient nhỏ được đẩy lên vùng biểu diễn được; trước khi cập nhật, chia gradient lại cho $S$ (ở fp32). `torch.amp.GradScaler` chọn $S$ **động**: nếu phát hiện gradient chứa `inf`/`NaN`, nó **bỏ qua bước cập nhật** đó và giảm $S$ một nửa; sau một số bước liên tiếp không có sự cố (mặc định 2 000), nó tăng $S$ gấp đôi. Giá trị khởi đầu mặc định là $2^{16} = 65\,536$. Với bf16, loss scaling thường không cần thiết.

```python
import torch

for dt in [torch.float32, torch.float16, torch.bfloat16]:
    fi = torch.finfo(dt)
    print(f"{str(dt):15s} max={fi.max:.3e}  tiny(normal)={fi.tiny:.3e}  eps={fi.eps:.3e}")

g = torch.tensor(1e-8)                              # một gradient rất nhỏ (fp32)
print("fp16 của 1e-8:", g.half().item())             # tràn dưới thành 0
scale = 2.0 ** 16                                    # hệ số nhân loss (loss scaling)
print("fp16 của 1e-8 * 2^16:", (g * scale).half().item(),
      "-> chia lại:", ((g * scale).half().float() / scale).item())
print("fp16 của 70000:", torch.tensor(70000.0).half().item())   # tràn trên thành inf
print("bf16: 1 + 1/256 =", (torch.tensor(1.0, dtype=torch.bfloat16) + 1 / 256).item())
```

Gradient $10^{-8}$ biến mất khi đổi sang fp16, nhưng sau khi nhân với $2^{16}$ nó trở thành khoảng $6{,}6 \times 10^{-4}$, biểu diễn được, và chia lại cho kết quả $\approx 10^{-8}$ với sai số nhỏ.

**Những bẫy riêng của TTS khi dùng độ chính xác hỗn hợp.**

- **Trích xuất đặc trưng ở fp32.** STFT, mel và đặc biệt phép `log` trên năng lượng rất nhỏ nên được tính ở fp32, bên ngoài vùng autocast (hoặc trong `torch.autocast(..., enabled=False)` với đầu vào đã `.float()`). Khi mel được tính *bên trong* vòng lặp huấn luyện (ví dụ mất mát mel của vocoder GAN), đây là nguồn `NaN` rất phổ biến.
- **Tính hàm mất mát ở fp32.** Ép đầu ra về `.float()` trước khi tính loss, như trong vòng lặp ở Mục 3.15.
- **Log-likelihood và log-det.** Các số hạng của normalizing flow và VAE (tổng của rất nhiều log mật độ) nên được cộng dồn ở fp32.
- **Theo dõi hệ số scale.** Nếu `scaler.get_scale()` liên tục giảm, mô hình đang sinh `inf`/`NaN` thường xuyên và nhiều bước bị bỏ qua một cách âm thầm: hãy tìm nguyên nhân thay vì để nó tự xoay xở.

## 3.12. Tích lũy gradient

Khi một batch đủ lớn không vừa bộ nhớ GPU, ta chia nó thành $K$ **batch con** (*micro-batch*), gọi `backward()` cho từng batch con mà **không** xóa gradient (PyTorch tự cộng dồn, Mục 2.6.5), rồi mới cập nhật một lần. Nếu hàm mất mát của batch lớn là trung bình trên $K$ batch con có **cùng kích thước**, ta chỉ cần chia mỗi loss con cho $K$:

$$
\nabla\Big(\frac{1}{K}\sum_{k=1}^{K}\mathcal{L}_k\Big) = \frac{1}{K}\sum_{k=1}^{K}\nabla\mathcal{L}_k .
$$

**Bẫy với chuỗi độ dài thay đổi.** Trong TTS, hàm mất mát mel thường được lấy trung bình trên **các khung hợp lệ**. Nếu các batch con có số khung rất khác nhau, "trung bình của các trung bình chia cho $K$" **không** bằng trung bình trên toàn bộ khung: các khung thuộc batch con ngắn bị đặt trọng số quá cao. Cách đúng là cộng **tổng** mất mát của từng batch con rồi chia cho **tổng số phần tử hợp lệ** của cả batch lớn (cần biết trước, hoặc tính trong một lượt duyệt nhẹ):

```python
import torch

torch.manual_seed(0)
F_mel = 4
model = torch.nn.Linear(F_mel, F_mel)
# Hai batch con có số khung hợp lệ rất khác nhau: 100 khung và 10 khung
x1, y1 = torch.randn(100, F_mel), torch.randn(100, F_mel)
x2, y2 = torch.randn(10, F_mel), torch.randn(10, F_mel)

def grad_of(loss):
    model.zero_grad()
    loss.backward()
    return model.weight.grad.clone()

l1 = lambda x, y: (model(x) - y).abs()                     # L1 từng phần tử
# Mục tiêu: trung bình trên TẤT CẢ các khung của cả "batch lớn"
g_full = grad_of(torch.cat([l1(x1, y1), l1(x2, y2)]).mean())
# Cách sai: trung bình trong từng batch con rồi chia cho số batch con
g_naive = grad_of(l1(x1, y1).mean() / 2 + l1(x2, y2).mean() / 2)
# Cách đúng: cộng tổng mất mát rồi chia cho tổng số phần tử hợp lệ
n_total = l1(x1, y1).numel() + l1(x2, y2).numel()
g_right = grad_of(l1(x1, y1).sum() / n_total + l1(x2, y2).sum() / n_total)

print("sai khác (cách ngây thơ):", (g_naive - g_full).norm().item())
print("sai khác (cách đúng)    :", (g_right - g_full).norm().item())
assert torch.allclose(g_right, g_full, atol=1e-6)
```

Ba lưu ý thêm: (1) lịch tốc độ học, warmup và EMA phải đếm theo **bước cập nhật** chứ không theo batch con; (2) BatchNorm vẫn tính thống kê trên từng batch con, nên tích lũy gradient không tương đương hoàn toàn với batch lớn nếu mạng có BatchNorm; (3) khi huấn luyện phân tán, nên tắt đồng bộ gradient ở các batch con trung gian (`no_sync()` của DDP) để tránh truyền thông thừa ([Chương 6](ch06-pytorch-cho-tieng-noi.md)).

## 3.13. Trung bình động của trọng số (EMA)

Với tốc độ học khác 0, tham số SGD không đứng yên mà "nhảy" quanh vùng cực tiểu. **Lấy trung bình** các điểm trên quỹ đạo cho một ước lượng ổn định hơn bất kỳ điểm riêng lẻ nào, một ý tưởng có từ lâu trong lý thuyết xấp xỉ ngẫu nhiên (Polyak và Juditsky, 1992). Trong học sâu, ta dùng **trung bình động hàm mũ** (*exponential moving average*, EMA) của trọng số:

$$
\theta^{\text{EMA}}_{k} = \beta\,\theta^{\text{EMA}}_{k-1} + (1 - \beta)\,\theta_k .
$$

Giống như ở momentum, trọng số của $\theta_{k-j}$ là $(1 - \beta)\beta^{j}$, nên EMA "nhớ" khoảng $1/(1 - \beta)$ bước gần nhất: 1 000 bước với $\beta = 0{,}999$, 10 000 bước với $\beta = 0{,}9999$. Bản EMA **không** tham gia huấn luyện; nó chỉ được dùng để đánh giá và suy luận.

EMA đặc biệt phổ biến với các mô hình **khuếch tán và flow matching** (DDPM dùng $\beta = 0{,}9999$; Ho và cộng sự, 2020), nơi hàm mất mát rất nhiễu (Mục 1.4.6) và chất lượng mẫu từ trọng số EMA thường tốt hơn rõ rệt so với trọng số "thô" ([Chương 11](../phan-2-mo-hinh-sinh/ch11-diffusion.md), [Chương 12](../phan-2-mo-hinh-sinh/ch12-flow-matching.md)).

Ba chi tiết cài đặt hay bị bỏ qua:

1. **Độ chệch ban đầu.** Bản EMA khởi tạo bằng trọng số ngẫu nhiên ban đầu. Sau $k$ bước, trọng số ngẫu nhiên đó vẫn chiếm tỷ lệ $\beta^{k}$ trong bản EMA, tức khoảng 5% sau 300 bước với $\beta = 0{,}99$, và khoảng 37% sau 1 000 bước với $\beta = 0{,}999$. Cách sửa thông dụng là **khởi động hệ số**: dùng $\beta_k = \min\big(\beta, \frac{1 + k}{10 + k}\big)$, để EMA bám sát mô hình ở giai đoạn đầu.
2. **Buffer.** Các buffer như thống kê chạy của BatchNorm không phải tham số; thường người ta sao chép thẳng chúng từ mô hình đang huấn luyện.
3. **Lưu cả EMA vào checkpoint** và nạp đúng bản EMA khi suy luận. PyTorch có sẵn `torch.optim.swa_utils.AveragedModel` kết hợp với `get_ema_multi_avg_fn(decay)` nếu bạn không muốn tự cài.

## 3.14. Tái lập kết quả

Một thí nghiệm không tái lập được thì không gỡ lỗi được và không so sánh được. Các bước tối thiểu:

- **Gieo hạt cho mọi nguồn ngẫu nhiên**: `random.seed`, `numpy.random.seed` (hoặc truyền `Generator` tường minh), `torch.manual_seed` (đã bao gồm mọi GPU). Với `DataLoader` nhiều tiến trình con, truyền một `generator` và `worker_init_fn` để mỗi tiến trình có hạt riêng nhưng xác định.
- **Thuật toán tất định**: `torch.use_deterministic_algorithms(True)` buộc PyTorch dùng cài đặt tất định hoặc báo lỗi nếu không có (một số phép toán cuBLAS còn yêu cầu đặt biến môi trường `CUBLAS_WORKSPACE_CONFIG=:4096:8`). Đặt `torch.backends.cudnn.benchmark = False` để cuDNN không tự chọn thuật toán khác nhau giữa các lần chạy. Cái giá là tốc độ chậm hơn, nên thường chỉ bật khi gỡ lỗi.
- **Lưu checkpoint đầy đủ**: không chỉ trọng số mô hình mà cả trạng thái của bộ tối ưu (các mô-men của Adam), bộ lập lịch, `GradScaler`, bản EMA, số bước, và lý tưởng là trạng thái của các bộ sinh số ngẫu nhiên. Nạp lại thiếu trạng thái Adam làm các bước đầu sau khi huấn luyện tiếp hành xử như lúc mới khởi tạo (Mục 3.5), thường thấy rõ thành một "cú nhảy" trên đường loss.
- **Chấp nhận giới hạn.** Kết quả vẫn có thể khác nhau giữa các loại GPU, phiên bản CUDA hay PyTorch. Khi so sánh hai cấu hình, hãy chạy nhiều hạt ngẫu nhiên và báo cáo độ dao động, thay vì kết luận từ một lần chạy ([Chương 37](../phan-9-thuc-tien/ch37-huan-luyen-va-debug.md)).

## 3.15. Ghép tất cả: một vòng lặp huấn luyện hoàn chỉnh

Ta gom mọi kỹ thuật của chương vào một vòng lặp huấn luyện có cấu trúc giống hệt (ở quy mô thu nhỏ) vòng lặp của các dự án TTS ở Phần X. Bài toán đồ chơi: một "decoder" nhỏ ánh xạ chuỗi đặc trưng văn bản *đã căn chỉnh* `(B, T, N_IN)` thành chuỗi "mel" `(B, T, F_MEL)`, với độ dài câu thay đổi và mất mát L1 có mặt nạ. Dữ liệu giả được sinh từ một "giáo viên" cố định cộng nhiễu, để ta biết bài toán có lời giải.

**Phần 1: mô hình, dữ liệu, bộ tối ưu, lịch tốc độ học.**

```python
import copy
import math
import torch
import torch.nn as nn

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# fp16 + GradScaler trên GPU; bf16 (không cần GradScaler) trên CPU
amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
B, N_IN, F_MEL, T_MAX = 16, 32, 20, 60

class TinyDecoder(nn.Module):
    """Mạng đồ chơi: đặc trưng 'văn bản' đã căn chỉnh (B, T, N_IN) -> 'mel' (B, T, F_MEL)."""
    def __init__(self, c=128):
        super().__init__()
        self.inp = nn.Linear(N_IN, c)
        self.conv = nn.Conv1d(c, c, kernel_size=5, padding=2)
        self.norm = nn.LayerNorm(c)
        self.drop = nn.Dropout(0.1)
        self.out = nn.Linear(c, F_MEL)

    def forward(self, x):                                   # x: (B, T, N_IN)
        h = self.inp(x)                                     # (B, T, C)
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)  # khối dư; Conv1d cần (B, C, T)
        h = self.drop(torch.relu(self.norm(h)))
        return self.out(h)                                  # (B, T, F_MEL)

def make_batch():
    """Dữ liệu giả có độ dài thay đổi; mục tiêu sinh từ một 'giáo viên' cố định + nhiễu."""
    lengths = torch.randint(T_MAX // 3, T_MAX + 1, (B,))
    x = torch.randn(B, T_MAX, N_IN)
    y = torch.tanh(x @ TEACHER) + 0.05 * torch.randn(B, T_MAX, F_MEL)
    mask = (torch.arange(T_MAX)[None, :] < lengths[:, None]).unsqueeze(-1)  # (B, T, 1)
    return x.to(device), y.to(device), mask.to(device)

TEACHER = torch.randn(N_IN, F_MEL) / math.sqrt(N_IN)
model = TinyDecoder().to(device)
ema_model = copy.deepcopy(model).eval().requires_grad_(False)   # bản sao trung bình động

# AdamW với hai nhóm tham số: không áp weight decay cho bias và tham số chuẩn hóa
decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
    lr=3e-3, betas=(0.9, 0.98), eps=1e-9)

TOTAL_STEPS, WARMUP, ACCUM, CLIP, EMA_DECAY = 300, 30, 2, 1.0, 0.99

def lr_lambda(step):                                        # warmup tuyến tính + cosine
    if step < WARMUP:
        return (step + 1) / WARMUP
    progress = (step - WARMUP) / max(1, TOTAL_STEPS - WARMUP)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler = torch.amp.GradScaler(device.type, enabled=(amp_dtype == torch.float16))
```

**Phần 2: vòng lặp.** Chạy tiếp trong cùng phiên. Thứ tự các lệnh trong mỗi bước cập nhật là điều quan trọng nhất cần ghi nhớ: *xóa gradient → (forward trong autocast → loss fp32 → backward có scale) × số batch con → unscale → cắt gradient → bước tối ưu → cập nhật scaler → bước lập lịch → cập nhật EMA*.

```python
def masked_l1(pred, target, mask):
    """L1 trung bình trên các phần tử HỢP LỆ (bỏ khung đệm)."""
    err = (pred.float() - target).abs() * mask
    return err.sum() / (mask.sum() * target.shape[-1])

@torch.no_grad()
def update_ema(ema, online, decay):
    for p_ema, p in zip(ema.parameters(), online.parameters()):
        p_ema.lerp_(p, 1.0 - decay)                         # p_ema = decay*p_ema + (1-decay)*p
    for b_ema, b in zip(ema.buffers(), online.buffers()):
        b_ema.copy_(b)                                      # buffer (nếu có) sao chép thẳng

@torch.no_grad()
def evaluate(net, n_batches=4):
    net.eval()
    g = torch.Generator().manual_seed(123)                  # tập kiểm định cố định
    total = 0.0
    for _ in range(n_batches):
        x = torch.randn(B, T_MAX, N_IN, generator=g).to(device)
        y = torch.tanh(x @ TEACHER.to(device))
        total += masked_l1(net(x), y, torch.ones_like(y[..., :1])).item()
    return total / n_batches

model.train()
for step in range(TOTAL_STEPS):
    optimizer.zero_grad(set_to_none=True)
    for _ in range(ACCUM):                                  # tích lũy gradient qua ACCUM batch con
        x, y, mask = make_batch()
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            pred = model(x)
        loss = masked_l1(pred, y, mask) / ACCUM             # loss tính bằng fp32
        scaler.scale(loss).backward()
    scaler.unscale_(optimizer)                              # PHẢI unscale trước khi cắt gradient
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    scaler.step(optimizer)                                  # tự bỏ qua bước nếu gradient có inf/NaN
    scaler.update()
    scheduler.step()
    update_ema(ema_model, model, min(EMA_DECAY, (1 + step) / (10 + step)))  # EMA có khởi động
    if step % 100 == 99:
        print(f"step {step + 1}: loss {loss.item() * ACCUM:.4f}  grad_norm {grad_norm:.3f}  "
              f"lr {scheduler.get_last_lr()[0]:.2e}")

val_online, val_ema = evaluate(model), evaluate(ema_model)
print(f"L1 kiểm định - mô hình thường: {val_online:.4f} | mô hình EMA: {val_ema:.4f}")
checkpoint = {"model": model.state_dict(), "ema": ema_model.state_dict(),
              "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
              "scaler": scaler.state_dict(), "step": TOTAL_STEPS}
torch.save(checkpoint, "checkpoint.pt")                    # đủ để huấn luyện tiếp chính xác
```

Chạy trên CPU (bf16, không cần `GradScaler`), toàn bộ vòng lặp mất vài giây. Trong một lần chạy, loss huấn luyện giảm từ khoảng 0,16 ở bước 100 xuống khoảng 0,15 ở bước 300, tốc độ học đi theo đúng hình dạng warmup + cosine (đạt đỉnh $3 \times 10^{-3}$ ở bước 30, về 0 ở bước 300), và L1 kiểm định của mô hình thường và mô hình EMA lần lượt khoảng 0,118 và 0,118. Chuẩn gradient (khoảng 0,06–0,11) luôn dưới ngưỡng 1,0, nên phép cắt chỉ đóng vai trò lưới an toàn. Trên GPU, cùng đoạn mã tự chuyển sang fp16 với `GradScaler` (nhánh này chưa được chạy trong môi trường biên soạn, nhưng chỉ dùng các API chuẩn của `torch.amp`).

Hai quan sát đáng giá từ thí nghiệm nhỏ này:

- **EMA không phải lúc nào cũng thắng.** Ở đây lịch cosine đã đưa tốc độ học về 0, bản thân quá trình đó đã "lắng" tham số, nên EMA và mô hình thường gần như bằng nhau. EMA phát huy tác dụng rõ nhất khi huấn luyện dài với gradient nhiễu và tốc độ học còn đáng kể, như với mô hình khuếch tán.
- **Khởi động hệ số EMA là quan trọng.** Nếu bỏ phần `min(EMA_DECAY, (1 + step) / (10 + step))` và dùng thẳng $\beta = 0{,}99$, trong một lần chạy của chúng tôi L1 kiểm định của bản EMA là khoảng 0,125 so với 0,118 của mô hình thường, vì bản EMA vẫn còn mang khoảng 5% "dấu vết" của trọng số khởi tạo ngẫu nhiên (Mục 3.13).

Một vòng lặp TTS thật sẽ có thêm các thành phần mà [Chương 6](ch06-pytorch-cho-tieng-noi.md) trình bày: `DataLoader` với hàm gom batch có đệm và mặt nạ, lấy mẫu theo nhóm độ dài, ghi nhật ký lên TensorBoard (loss, chuẩn gradient, tốc độ học, hình mel và mẫu âm thanh), lưu và nạp checkpoint định kỳ, cùng huấn luyện phân tán.

## Lỗi thường gặp và mẹo thực hành

- **Gọi `scheduler.step()` sai chỗ hoặc sai tần suất.** Lịch theo *bước* phải được gọi sau mỗi `optimizer.step()`; lịch theo *epoch* chỉ gọi một lần mỗi epoch. Nhầm hai thứ này làm warmup kéo dài hàng nghìn lần hoặc tốc độ học về 0 sau vài epoch. PyTorch cảnh báo nếu `scheduler.step()` được gọi trước `optimizer.step()`.
- **Cắt gradient trước khi unscale.** Với `GradScaler`, gradient đang bị nhân với $S \approx 65\,536$; cắt ở ngưỡng 1,0 lúc này làm mọi bước cập nhật bé đi hàng nghìn lần. Luôn gọi `scaler.unscale_(optimizer)` trước `clip_grad_norm_`.
- **Dùng `Adam(weight_decay=...)` khi định dùng AdamW**, hoặc áp weight decay cho bias và tham số chuẩn hóa (Mục 3.6).
- **Huấn luyện tiếp mà không nạp trạng thái bộ tối ưu, bộ lập lịch, scaler.** Hệ quả: warmup chạy lại từ đầu, các mô-men Adam bị đặt lại về 0, loss vọt lên. Hãy lưu và nạp *toàn bộ* như trong vòng lặp ở Mục 3.15.
- **Tốc độ học quá lớn, không warmup, với Transformer.** Triệu chứng điển hình: loss giảm nhanh vài trăm bước rồi thành `NaN`, hoặc attention không bao giờ hội tụ. Thử giảm tốc độ học đỉnh, kéo dài warmup, kiểm tra vị trí LayerNorm.
- **`NaN` khi dùng fp16.** Kiểm tra theo thứ tự: log của số 0 trong mel, phép chia cho chuẩn hoặc độ dài bằng 0, softmax với mặt nạ toàn `-inf` (một hàng bị che hoàn toàn), giá trị kích hoạt vượt 65 504. Chuyển phần nghi vấn sang fp32, hoặc thử bf16 nếu phần cứng hỗ trợ.
- **BatchNorm với batch nhỏ hoặc chuỗi có đệm**, và **quên `model.eval()`** khi đánh giá (dropout và BatchNorm hành xử sai). Ngược lại, nhớ gọi lại `model.train()` sau khi đánh giá giữa chừng.
- **So sánh loss giữa các cấu hình có quy ước khác nhau.** Loss trung bình trên khung và trung bình trên câu, có hay không có mặt nạ, batch động hay batch cố định cho các con số không so sánh được trực tiếp. Ghi rõ quy ước trong nhật ký thí nghiệm.
- **Quên dùng (hoặc quên lưu) trọng số EMA khi suy luận**, hoặc đánh giá bản EMA ngay từ đầu mà không khởi động hệ số.
- **Huấn luyện thuần fp16 (không autocast) với $\epsilon = 10^{-8}$.** Giá trị này nằm dưới ngưỡng biểu diễn của fp16, nên mẫu số của Adam có thể bằng 0. Với autocast, trạng thái của bộ tối ưu vẫn ở fp32 nên không gặp vấn đề này.

## Tóm tắt

- Trên hàm bậc hai, GD hội tụ khi và chỉ khi $\eta < 2/\lambda_{\max}$, và số bước tỷ lệ với số điều kiện $\kappa$; momentum (heavy ball, Nesterov) giảm con số này xuống cỡ $\sqrt{\kappa}$.
- SGD dùng gradient minibatch, một ước lượng không chệch có phương sai giảm theo $1/B$; nhiễu của nó giúp thoát điểm yên ngựa nhưng tạo "sàn" buộc ta giảm tốc độ học về cuối.
- Adam chuẩn hóa bước đi theo từng tham số bằng $\hat{\mathbf{m}}/\sqrt{\hat{\mathbf{v}}}$; hiệu chỉnh độ chệch bù cho việc khởi tạo các mô-men bằng 0 (không có nó, bước đầu lớn hơn khoảng 3 lần). Tốc độ học của Adam gần với "độ thay đổi tối đa mỗi bước" của từng tham số.
- AdamW tách weight decay khỏi gradient thích nghi; không áp weight decay cho bias và tham số chuẩn hóa. Các mô hình có thành phần GAN thường dùng $\beta_1 = 0{,}8$, $\beta_2 = 0{,}99$.
- Warmup + cosine là lịch mặc định phổ biến; lịch Noam đạt đỉnh $(d_{\text{model}}W)^{-0.5}$ rồi giảm theo $1/\sqrt{k}$. Cắt gradient theo chuẩn là lưới an toàn, và chuẩn gradient là tín hiệu chẩn đoán quý giá.
- BatchNorm phụ thuộc batch và bị khung đệm làm sai lệch; LayerNorm/RMSNorm là mặc định cho mô hình chuỗi; InstanceNorm xóa thông tin toàn cục của câu (hữu ích cho tách người nói); GroupNorm cho batch nhỏ. Weight normalization và spectral normalization (chia cho $\sigma_{\max}$) là trụ cột của vocoder GAN.
- Dropout, weight decay và dừng sớm là các công cụ điều chuẩn chính; tăng cường dữ liệu cho mục tiêu TTS phải rất thận trọng, đặc biệt với thanh điệu tiếng Việt.
- Độ chính xác hỗn hợp: fp16 cần loss scaling động, bf16 có dải rộng nhưng kém chính xác; luôn tính đặc trưng âm thanh, log và hàm mất mát ở fp32. Tích lũy gradient với chuỗi độ dài thay đổi phải chuẩn hóa theo tổng số phần tử hợp lệ. EMA cần khởi động hệ số và phải được lưu cùng checkpoint.

## Bài tập

**Bài 1 [Cơ bản].** Hessian của một hàm bậc hai có trị riêng nhỏ nhất 0,5 và lớn nhất 200. (a) Tốc độ học lớn nhất để GD hội tụ là bao nhiêu? (b) Với tốc độ học tối ưu, hệ số co mỗi bước là bao nhiêu và cần khoảng bao nhiêu bước để giảm sai số đi $10^{6}$ lần? (c) Lặp lại câu (b) cho heavy ball với tham số tối ưu.

??? success "Lời giải"
    (a) $\eta < 2/200 = 0{,}01$. (b) $\kappa = 400$, $\eta^\star = 2/200{,}5 \approx 0{,}00998$, $\rho = 399/401 \approx 0{,}995$; số bước $\approx \ln(10^{6})/(-\ln 0{,}995) \approx 13{,}8/0{,}005 \approx 2\,760$. (c) $\rho = (\sqrt{400} - 1)/(\sqrt{400} + 1) = 19/21 \approx 0{,}905$; số bước $\approx 13{,}8/0{,}100 \approx 138$, nhanh hơn khoảng 20 lần (đúng bằng $\sqrt{\kappa}$).

**Bài 2 [Cơ bản].** Với lịch Noam, $d_{\text{model}} = 256$ và $W = 4\,000$: tốc độ học đỉnh là bao nhiêu? Tốc độ học ở bước 16 000 bằng bao nhiêu phần trăm đỉnh? Nếu tăng $d_{\text{model}}$ lên 1 024 mà giữ nguyên $W$, đỉnh thay đổi thế nào?

??? success "Lời giải"
    Đỉnh $= (256 \times 4\,000)^{-0.5} \approx 9{,}9 \times 10^{-4}$. Sau warmup, $\eta_k = d^{-0.5}k^{-0.5}$, nên $\eta_{16000}/\eta_{\text{peak}} = \sqrt{4\,000/16\,000} = 0{,}5$, tức 50% (khoảng $4{,}9 \times 10^{-4}$). Với $d_{\text{model}} = 1\,024$, đỉnh giảm $\sqrt{4} = 2$ lần, còn khoảng $4{,}9 \times 10^{-4}$.

**Bài 3 [Cơ bản].** Chứng minh rằng với SGD (không momentum), điều chuẩn L2 với hệ số $\lambda$ tương đương weight decay. Sau đó xét Adam với một tham số duy nhất, $\epsilon = 0$, và giả sử gradient của hàm mất mát luôn bằng hằng số $c \ne 0$ (sau khi đã ổn định, bỏ qua hiệu chỉnh độ chệch). So sánh độ lớn của "lực kéo về 0" trong hai cách cài đặt.

??? tip "Gợi ý"
    Với L2 trong Adam, gradient tổng là $c + \lambda\theta$; khi $\lambda\theta \ll c$, bước đi xấp xỉ $\eta\,\operatorname{sign}(c + \lambda\theta)$, và phần đóng góp của $\lambda\theta$ vào bước đi xấp xỉ $\eta\lambda\theta/\lvert c \rvert$: nó bị chia cho độ lớn gradient. Với AdamW, lực kéo luôn là $\eta\lambda\theta$, không phụ thuộc $c$.

**Bài 4 [Nâng cao].** (a) Lặp lại phép suy ra hiệu chỉnh độ chệch cho $\mathbf{v}_t$ của Adam. (b) Chứng minh rằng nếu bỏ hiệu chỉnh, bước đầu tiên có độ lớn $\eta(1 - \beta_1)/\sqrt{1 - \beta_2}$ (bỏ qua $\epsilon$). Tính giá trị này cho $(\beta_1, \beta_2) = (0{,}9;\, 0{,}999)$ và $(0{,}8;\, 0{,}99)$. (c) Với $t$ lớn bao nhiêu thì $1 - \beta_2^t > 0{,}99$ khi $\beta_2 = 0{,}999$?

??? success "Lời giải"
    (a) $\mathbf{v}_t = (1 - \beta_2)\sum_{i}\beta_2^{t-i}\mathbf{g}_i^2$, nên $\mathbb{E}[\mathbf{v}_t] = (1 - \beta_2^t)\,\mathbb{E}[\mathbf{g}^2]$ dưới giả định ổn định. (b) $\mathbf{m}_1 = (1 - \beta_1)g$, $\mathbf{v}_1 = (1 - \beta_2)g^2$, tỷ số là $(1 - \beta_1)/\sqrt{1 - \beta_2}$ nhân dấu của $g$. Với $(0{,}9;\, 0{,}999)$: $0{,}1/0{,}0316 \approx 3{,}16$; với $(0{,}8;\, 0{,}99)$: $0{,}2/0{,}1 = 2$. (c) Cần $0{,}999^t < 0{,}01$, tức $t > \ln 100/(-\ln 0{,}999) \approx 4\,603$ bước.

**Bài 5 [Nâng cao].** Với EMA hệ số $\beta$ (bắt đầu từ $\theta^{\text{EMA}}_0 = \theta_0$), (a) viết $\theta^{\text{EMA}}_k$ dưới dạng tổng có trọng số của $\theta_0, \ldots, \theta_k$ và kiểm tra tổng các trọng số bằng 1; (b) tính "độ trễ trung bình" $\sum_j j\,w_j$ khi $k \to \infty$; (c) với $\beta = 0{,}999$, sau bao nhiêu bước thì trọng số của $\theta_0$ giảm xuống dưới 1%?

??? success "Lời giải"
    (a) $\theta^{\text{EMA}}_k = \beta^k\theta_0 + \sum_{j=0}^{k-1}(1 - \beta)\beta^{j}\theta_{k-j}$; tổng trọng số là $\beta^k + (1 - \beta)\frac{1 - \beta^k}{1 - \beta} = 1$. (b) $\sum_{j \ge 0} j(1 - \beta)\beta^j = \beta/(1 - \beta)$, khoảng 999 bước với $\beta = 0{,}999$: bản EMA phản ánh mô hình của khoảng một nghìn bước trước. (c) $0{,}999^k < 0{,}01 \Rightarrow k > 4\,603$.

**Bài 6 [Nâng cao].** Với inverted dropout tỷ lệ $p$, chứng minh $\mathbb{E}[\tilde{h}] = h$ và tính $\operatorname{Var}(\tilde{h})$. Giải thích vì sao, ngay cả khi kỳ vọng được bảo toàn, việc **giữ dropout bật khi suy luận** (như pre-net của Tacotron 2) làm đầu ra của mô hình thay đổi giữa các lần chạy, và vì sao điều này có thể là mong muốn.

??? tip "Gợi ý"
    $\tilde{h} = mh/(1 - p)$ với $m \sim \operatorname{Bernoulli}(1 - p)$: $\mathbb{E}[\tilde{h}] = h$, $\mathbb{E}[\tilde{h}^2] = h^2/(1 - p)$, nên $\operatorname{Var}(\tilde{h}) = h^2 p/(1 - p)$, bằng $h^2$ khi $p = 0{,}5$. Phương sai này là một nguồn ngẫu nhiên "rẻ" giúp tạo biến thiên ngôn điệu giữa các lần tổng hợp.

**Bài 7 [Lập trình].** Viết lớp `MaskedBatchNorm1d` cho tensor `(B, C, T)` kèm mặt nạ `(B, 1, T)`: khi huấn luyện, tính trung bình và phương sai **chỉ trên khung hợp lệ**, cập nhật thống kê chạy, và khi suy luận dùng thống kê chạy. Kiểm tra rằng kết quả trên các khung hợp lệ trùng với `nn.BatchNorm1d` áp lên tensor ghép tất cả các khung hợp lệ lại (không đệm).

??? tip "Gợi ý"
    `n = mask.sum()`; `mean = (x * mask).sum(dim=(0, 2)) / n`; `var = (((x - mean[None, :, None]) * mask) ** 2).sum(dim=(0, 2)) / n`. Để so sánh, tạo `x_valid = x.transpose(1, 2)[mask.squeeze(1)]` hình dạng `(n, C)` rồi áp `nn.BatchNorm1d(C)` lên nó (BatchNorm1d chấp nhận đầu vào 2 chiều `(N, C)`). Chú ý PyTorch cập nhật trung bình chạy của phương sai bằng ước lượng **không chệch** (chia cho $n - 1$).

**Bài 8 [Lập trình].** Dùng vòng lặp ở Mục 3.15 làm khung thí nghiệm. (a) Đặt tốc độ học đỉnh $10^{-2}$ và so sánh có warmup (30 bước) với không warmup (`WARMUP = 0`, nhớ sửa hàm `lr_lambda` cho trường hợp này). (b) Thay AdamW bằng `torch.optim.Adam(..., weight_decay=0.01)` và so sánh độ lớn trọng số cuối cùng (chuẩn Frobenius của từng ma trận) giữa hai bộ tối ưu. (c) Ghi lại chuẩn gradient ở mỗi bước và vẽ (hoặc in) phân phối của nó; đề xuất một ngưỡng cắt hợp lý dựa trên phân vị 99%.

??? tip "Gợi ý"
    Với bài toán đồ chơi này, khác biệt có thể nhỏ; hãy chạy vài hạt ngẫu nhiên và nhìn cả đường loss những bước đầu, không chỉ kết quả cuối. Ở câu (b), weight decay kiểu L2 trong Adam bị chia cho $\sqrt{\hat{v}}$, nên với các tham số có gradient lớn, lực kéo về 0 yếu hơn hẳn so với AdamW.

## Tài liệu tham khảo

1. Robbins, H., Monro, S. (1951). *A stochastic approximation method*. Annals of Mathematical Statistics.
2. Polyak, B. T. (1964). *Some methods of speeding up the convergence of iteration methods*. USSR Computational Mathematics and Mathematical Physics.
3. Nesterov, Y. (1983). *A method for solving the convex programming problem with convergence rate O(1/k²)*. Soviet Mathematics Doklady.
4. Polyak, B. T., Juditsky, A. B. (1992). *Acceleration of stochastic approximation by averaging*. SIAM Journal on Control and Optimization.
5. Sutskever, I., Martens, J., Dahl, G., Hinton, G. (2013). *On the importance of initialization and momentum in deep learning*. ICML.
6. Duchi, J., Hazan, E., Singer, Y. (2011). *Adaptive subgradient methods for online learning and stochastic optimization*. Journal of Machine Learning Research.
7. Tieleman, T., Hinton, G. (2012). *Lecture 6.5 – RMSProp: Divide the gradient by a running average of its recent magnitude*. COURSERA: Neural Networks for Machine Learning.
8. Kingma, D. P., Ba, J. (2015). *Adam: A method for stochastic optimization*. ICLR.
9. Loshchilov, I., Hutter, F. (2019). *Decoupled weight decay regularization*. ICLR.
10. Loshchilov, I., Hutter, F. (2017). *SGDR: Stochastic gradient descent with warm restarts*. ICLR.
11. Liu, L. và cộng sự (2020). *On the variance of the adaptive learning rate and beyond*. ICLR.
12. Vaswani, A. và cộng sự (2017). *Attention is all you need*. NeurIPS.
13. Xiong, R. và cộng sự (2020). *On layer normalization in the Transformer architecture*. ICML.
14. Goyal, P. và cộng sự (2017). *Accurate, large minibatch SGD: Training ImageNet in 1 hour*. arXiv:1706.02677.
15. Pascanu, R., Mikolov, T., Bengio, Y. (2013). *On the difficulty of training recurrent neural networks*. ICML.
16. Dauphin, Y. N. và cộng sự (2014). *Identifying and attacking the saddle point problem in high-dimensional non-convex optimization*. NeurIPS.
17. Keskar, N. S. và cộng sự (2017). *On large-batch training for deep learning: Generalization gap and sharp minima*. ICLR.
18. Dinh, L., Pascanu, R., Bengio, S., Bengio, Y. (2017). *Sharp minima can generalize for deep nets*. ICML.
19. Li, H. và cộng sự (2018). *Visualizing the loss landscape of neural nets*. NeurIPS.
20. Ioffe, S., Szegedy, C. (2015). *Batch normalization: Accelerating deep network training by reducing internal covariate shift*. ICML.
21. Santurkar, S., Tsipras, D., Ilyas, A., Madry, A. (2018). *How does batch normalization help optimization?* NeurIPS.
22. Ba, J. L., Kiros, J. R., Hinton, G. E. (2016). *Layer normalization*. arXiv:1607.06450.
23. Zhang, B., Sennrich, R. (2019). *Root mean square layer normalization*. NeurIPS.
24. Ulyanov, D., Vedaldi, A., Lempitsky, V. (2016). *Instance normalization: The missing ingredient for fast stylization*. arXiv:1607.08022.
25. Wu, Y., He, K. (2018). *Group normalization*. ECCV.
26. Huang, X., Belongie, S. (2017). *Arbitrary style transfer in real-time with adaptive instance normalization*. ICCV.
27. Chou, J.-c., Yeh, C.-c., Lee, H.-y. (2019). *One-shot voice conversion by separating speaker and content representations with instance normalization*. Interspeech.
28. Salimans, T., Kingma, D. P. (2016). *Weight normalization: A simple reparameterization to accelerate training of deep neural networks*. NeurIPS.
29. Miyato, T., Kataoka, T., Koyama, M., Yoshida, Y. (2018). *Spectral normalization for generative adversarial networks*. ICLR.
30. Srivastava, N. và cộng sự (2014). *Dropout: A simple way to prevent neural networks from overfitting*. Journal of Machine Learning Research.
31. Krueger, D. và cộng sự (2017). *Zoneout: Regularizing RNNs by randomly preserving hidden activations*. ICLR.
32. Park, D. S. và cộng sự (2019). *SpecAugment: A simple data augmentation method for automatic speech recognition*. Interspeech.
33. Szegedy, C. và cộng sự (2016). *Rethinking the Inception architecture for computer vision*. CVPR.
34. Micikevicius, P. và cộng sự (2018). *Mixed precision training*. ICLR.
35. Ho, J., Jain, A., Abbeel, P. (2020). *Denoising diffusion probabilistic models*. NeurIPS.
36. Shen, J. và cộng sự (2018). *Natural TTS synthesis by conditioning WaveNet on mel spectrogram predictions*. ICASSP.
37. Ren, Y. và cộng sự (2021). *FastSpeech 2: Fast and high-quality end-to-end text to speech*. ICLR.
38. Kong, J., Kim, J., Bae, J. (2020). *HiFi-GAN: Generative adversarial networks for efficient and high fidelity speech synthesis*. NeurIPS.
39. Kim, J., Kong, J., Son, J. (2021). *Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech*. ICML.
