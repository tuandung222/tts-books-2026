# Chương 2. Học máy và mạng nơ-ron

Chương 1 đã chuẩn bị các công cụ toán học. Chương này dùng chúng để trả lời câu hỏi trung tâm của học máy: **làm thế nào để một chương trình học được một hàm từ ví dụ**, và làm sao biết hàm đó sẽ hoạt động tốt trên dữ liệu chưa từng thấy? Ta sẽ đi từ khung lý thuyết (học có giám sát, cực tiểu hóa rủi ro thực nghiệm, độ chệch và phương sai), qua các mô hình tuyến tính, đến mạng nơ-ron nhiều lớp và thuật toán làm cho chúng huấn luyện được: **lan truyền ngược**.

Một điểm sẽ lặp lại suốt chương: **hàm mất mát không phải là lựa chọn tùy tiện**. Mỗi hàm mất mát quen thuộc (MSE, L1, entropy chéo) là hệ quả của một giả định xác suất về dữ liệu, và mỗi số hạng điều chuẩn là hệ quả của một niềm tin tiên nghiệm về tham số. Hiểu điều này giúp bạn đọc được *vì sao* các bài báo TTS chọn hàm mất mát như họ đã chọn, và khi nào những lựa chọn đó bắt đầu thất bại.

!!! abstract "Mục tiêu học tập"
    Sau chương này, bạn có thể:

    - **Phát biểu** bài toán học có giám sát dưới dạng cực tiểu hóa rủi ro thực nghiệm và **thiết kế** cách chia dữ liệu huấn luyện/kiểm định/kiểm tra phù hợp cho TTS.
    - **Suy ra** phân rã độ chệch–phương sai và **giải thích** vì sao số hạng nhiễu tương ứng với bản chất một-nhiều của TTS.
    - **Suy ra** MSE, L1, entropy chéo từ nguyên lý hợp lý cực đại, và các số hạng điều chuẩn L2/L1 từ ước lượng hậu nghiệm cực đại.
    - **Tính** gradient của hồi quy logistic và softmax, và **so sánh** các hàm kích hoạt sigmoid, tanh, ReLU, GELU, SiLU.
    - **Suy ra** từng bước thuật toán lan truyền ngược cho MLP và **cài đặt** một bộ vi phân tự động chế độ ngược thu nhỏ.
    - **Giải thích** khởi tạo Xavier/Glorot và He bằng lập luận bảo toàn phương sai.
    - **Cài đặt** một MLP từ đầu bằng numpy, **kiểm tra** gradient bằng sai phân hữu hạn và đối chiếu với `torch.autograd`.

!!! info "Kiến thức cần có"
    - [Chương 1](ch01-toan-nen-tang.md): tích ma trận, gradient, Jacobian và quy tắc dây chuyền cho véc-tơ (Mục 1.3), phân phối Gauss/Laplace/Categorical (Mục 1.4), entropy chéo và KL (Mục 1.5).
    - Python và numpy cơ bản; PyTorch ở mức tạo tensor và gọi `.backward()` là đủ (chi tiết hơn ở [Chương 6](ch06-pytorch-cho-tieng-noi.md)).

## 2.1. Bài toán học có giám sát

### 2.1.1. Bốn thành phần của một hệ học máy

Trong **học có giám sát** (*supervised learning*), ta có một tập ví dụ gồm các cặp (đầu vào, đầu ra mong muốn) và muốn tìm một hàm ánh xạ đầu vào thành đầu ra, sao cho hàm đó cũng đúng với những đầu vào mới. Mọi hệ học máy có giám sát, từ hồi quy tuyến tính đến một mô hình TTS hàng tỷ tham số, đều gồm bốn thành phần:

1. **Dữ liệu** $\mathcal{D} = \{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^{n}$, giả định được lấy mẫu độc lập từ một phân phối chưa biết $p_{\text{data}}(\mathbf{x}, \mathbf{y})$.
2. **Mô hình** (hay *lớp giả thuyết*) $\{f_\theta\}$: một họ hàm được tham số hóa bởi $\theta$, ví dụ mọi mạng nơ-ron có một kiến trúc cố định.
3. **Hàm mất mát** $\ell(\hat{\mathbf{y}}, \mathbf{y})$ đo mức "sai" của dự đoán $\hat{\mathbf{y}} = f_\theta(\mathbf{x})$ so với đáp án $\mathbf{y}$.
4. **Thuật toán tối ưu** tìm $\theta$ làm hàm mất mát nhỏ, thường là các biến thể của gradient descent ([Chương 3](ch03-toi-uu-va-ky-thuat-huan-luyen.md)).

Khi một mô hình TTS hoạt động kém, lỗi gần như luôn nằm ở một trong bốn thành phần này: dữ liệu bẩn hoặc lệch, kiến trúc thiếu năng lực hoặc sai thiên kiến quy nạp, hàm mất mát không phản ánh điều ta thực sự muốn, hay quá trình tối ưu không hội tụ. Tách bạch bốn thành phần là bước đầu tiên của mọi quy trình gỡ lỗi ([Chương 37](../phan-9-thuc-tien/ch37-huan-luyen-va-debug.md)).

### 2.1.2. Rủi ro kỳ vọng và cực tiểu hóa rủi ro thực nghiệm

Điều ta thực sự muốn là hàm $f$ có **rủi ro kỳ vọng** (*expected risk*) nhỏ, tức mất mát trung bình trên *toàn bộ* phân phối dữ liệu:

$$
R(f) = \mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim p_{\text{data}}}\big[\ell(f(\mathbf{x}), \mathbf{y})\big] .
$$

Nhưng ta không biết $p_{\text{data}}$; ta chỉ có $n$ mẫu. Vì vậy ta thay kỳ vọng bằng trung bình mẫu (một ước lượng Monte Carlo, Mục 1.4.6) và được **rủi ro thực nghiệm**:

$$
\hat{R}_n(f) = \frac{1}{n}\sum_{i=1}^{n} \ell\big(f(\mathbf{x}_i), \mathbf{y}_i\big),
\qquad
\hat{f} = \arg\min_{f \in \{f_\theta\}} \hat{R}_n(f) .
$$

Nguyên lý chọn $\hat{f}$ như trên gọi là **cực tiểu hóa rủi ro thực nghiệm** (*empirical risk minimization*, ERM). Với một hàm $f$ *cố định*, luật số lớn bảo đảm $\hat{R}_n(f) \to R(f)$. Nhưng $\hat{f}$ không cố định: nó được chọn *chính vì* có $\hat{R}_n$ nhỏ trên đúng tập dữ liệu này, nên $\hat{R}_n(\hat{f})$ thường **lạc quan** hơn $R(\hat{f})$. Khoảng cách $R(\hat{f}) - \hat{R}_n(\hat{f})$ gọi là **khoảng cách tổng quát hóa** (*generalization gap*). Toàn bộ Mục 2.2 bàn về cách kiểm soát nó.

### 2.1.3. TTS nhìn như một bài toán học có giám sát

Một hệ TTS gồm nhiều mô-đun, và mỗi mô-đun là một bài toán học có giám sát với loại đầu ra riêng:

| Mô-đun | Đầu vào $\mathbf{x}$ | Đầu ra $\mathbf{y}$ | Loại bài toán | Hàm mất mát điển hình |
|---|---|---|---|---|
| Mô hình âm học hồi quy (Tacotron 2, FastSpeech 2) | Chuỗi âm vị | Log-mel $\mathbf{Y} \in \mathbb{R}^{T \times F}$ | Hồi quy | L1 hoặc L2 |
| Bộ dự đoán trường độ | Chuỗi âm vị | $\log d_i$ (số khung) | Hồi quy | L2 trên log trường độ |
| Bộ dự đoán cao độ/năng lượng | Chuỗi âm vị | F0, năng lượng | Hồi quy | L2 hoặc L1 |
| Cờ dừng (Tacotron 2) | Trạng thái decoder | Dừng/chưa dừng | Phân loại nhị phân | Entropy chéo nhị phân |
| Codec language model (VALL-E) | Văn bản + token trước đó | Token âm thanh kế tiếp | Phân loại nhiều lớp | Entropy chéo |
| Vocoder GAN (HiFi-GAN) | Mel | Dạng sóng | Sinh có điều kiện | Đối kháng + L1 trên mel + feature matching |

Hai dòng cuối báo trước một điều: với những đầu ra phức tạp, "đáp án đúng" không còn duy nhất, và khung học có giám sát thuần túy phải được mở rộng thành **mô hình sinh có điều kiện** ([Phần II](../phan-2-mo-hinh-sinh/ch07-tong-quan-mo-hinh-sinh.md)). Nhưng các khái niệm của chương này (hàm mất mát, tổng quát hóa, lan truyền ngược) vẫn là nền tảng.

## 2.2. Tổng quát hóa

### 2.2.1. Tập huấn luyện, kiểm định và kiểm tra

Vì $\hat{R}_n(\hat{f})$ lạc quan, ta cần dữ liệu **chưa dùng để huấn luyện** để ước lượng $R(\hat{f})$ một cách trung thực. Quy trình chuẩn chia dữ liệu thành ba phần:

- **Tập huấn luyện** (*training set*): dùng để tối ưu tham số $\theta$.
- **Tập kiểm định** (*validation set*): dùng để chọn siêu tham số (tốc độ học, kích thước mô hình, số bước huấn luyện, checkpoint tốt nhất). Vì ta "chọn" dựa trên nó, tập này cũng dần bị khớp theo.
- **Tập kiểm tra** (*test set*): chỉ dùng **một lần** ở cuối để báo cáo kết quả. Nếu bạn nhìn kết quả trên tập kiểm tra rồi quay lại chỉnh mô hình, nó đã trở thành tập kiểm định.

Với TTS, việc chia dữ liệu có những điểm tinh tế riêng ([Chương 36](../phan-9-thuc-tien/ch36-du-lieu-tts.md), [Chương 38](../phan-9-thuc-tien/ch38-danh-gia-tts.md)):

- **Rò rỉ văn bản.** Nếu cùng một câu văn (hoặc những câu gần như trùng lặp, rất hay gặp trong sách nói và tin tức) xuất hiện ở cả tập huấn luyện lẫn tập kiểm tra, kết quả đánh giá sẽ đẹp một cách giả tạo.
- **Chia theo người nói.** Muốn đánh giá khả năng nhân bản giọng *zero-shot*, người nói trong tập kiểm tra phải **hoàn toàn không** xuất hiện trong tập huấn luyện.
- **Loss kiểm định không phải là chất lượng.** Với mô hình tự hồi quy huấn luyện bằng teacher forcing, loss kiểm định được tính khi mô hình được "nhắc" đáp án đúng ở mỗi bước, khác hẳn lúc suy luận thật. Với GAN, loss của bộ sinh gần như không nói lên chất lượng âm thanh. Hãy luôn **nghe** mẫu sinh ra và dùng các thước đo khách quan phù hợp ([Chương 38](../phan-9-thuc-tien/ch38-danh-gia-tts.md)).

### 2.2.2. Quá khớp và chưa khớp

**Năng lực** (*capacity*) của một lớp mô hình là mức độ phong phú của các hàm mà nó biểu diễn được. Mô hình quá đơn giản sẽ **chưa khớp** (*underfitting*): lỗi cao ngay trên tập huấn luyện. Mô hình quá phức tạp so với lượng dữ liệu sẽ **quá khớp** (*overfitting*): lỗi huấn luyện rất thấp nhưng lỗi kiểm định cao, vì nó học thuộc cả nhiễu. Dấu hiệu quá khớp kinh điển là đường loss huấn luyện tiếp tục giảm trong khi đường loss kiểm định chững lại rồi đi lên.

Bức tranh cổ điển này cần một lời bình: các mạng nơ-ron hiện đại thường có số tham số vượt xa số mẫu mà vẫn tổng quát hóa tốt, và lỗi kiểm định đôi khi giảm *trở lại* khi tiếp tục tăng kích thước mô hình, hiện tượng được gọi là **double descent** (Belkin và cộng sự, 2019). Thiên kiến quy nạp của kiến trúc, của thuật toán tối ưu và các kỹ thuật điều chuẩn ([Chương 3](ch03-toi-uu-va-ky-thuat-huan-luyen.md)) đều đóng vai trò. Dù vậy, với lượng dữ liệu TTS khiêm tốn, đặc biệt là tiếng Việt (vài giờ đến vài chục giờ cho một giọng), quá khớp vẫn là mối lo rất thực tế.

### 2.2.3. Phân rã độ chệch – phương sai

Phân rã độ chệch–phương sai (*bias–variance decomposition*) cho ta một cách định lượng để nghĩ về năng lực và quá khớp.

**Thiết lập.** Giả sử $y = f(\mathbf{x}) + \varepsilon$, với $f$ là hàm thật, nhiễu $\varepsilon$ có $\mathbb{E}[\varepsilon] = 0$, $\operatorname{Var}(\varepsilon) = \sigma^2$ và độc lập với mọi thứ khác. Huấn luyện trên một tập dữ liệu ngẫu nhiên $\mathcal{D}$ cho ta hàm $\hat{f}_{\mathcal{D}}$. Vì $\mathcal{D}$ ngẫu nhiên nên $\hat{f}_{\mathcal{D}}(\mathbf{x})$ cũng ngẫu nhiên; gọi $\bar{f}(\mathbf{x}) = \mathbb{E}_{\mathcal{D}}[\hat{f}_{\mathcal{D}}(\mathbf{x})]$ là dự đoán trung bình trên mọi tập huấn luyện có thể có.

**Suy ra.** Cố định một điểm $\mathbf{x}$ và viết tắt $f = f(\mathbf{x})$, $\hat{f} = \hat{f}_{\mathcal{D}}(\mathbf{x})$, $\bar{f} = \bar{f}(\mathbf{x})$. Sai số bình phương kỳ vọng trên một điểm mới là

$$
\mathbb{E}_{\mathcal{D}, \varepsilon}\big[(y - \hat{f})^2\big] = \mathbb{E}\big[(f - \hat{f} + \varepsilon)^2\big] = \mathbb{E}_{\mathcal{D}}\big[(f - \hat{f})^2\big] + 2\,\mathbb{E}\big[(f - \hat{f})\,\varepsilon\big] + \mathbb{E}[\varepsilon^2] .
$$

Số hạng giữa bằng 0 vì $\varepsilon$ độc lập với $\hat{f}$ và có kỳ vọng 0. Số hạng cuối là $\sigma^2$. Với số hạng đầu, cộng và trừ $\bar{f}$:

$$
\mathbb{E}_{\mathcal{D}}\big[(f - \hat{f})^2\big] = (f - \bar{f})^2 + 2(f - \bar{f})\,\mathbb{E}_{\mathcal{D}}\big[\bar{f} - \hat{f}\big] + \mathbb{E}_{\mathcal{D}}\big[(\hat{f} - \bar{f})^2\big] .
$$

Số hạng giữa lại bằng 0 vì $\mathbb{E}_{\mathcal{D}}[\hat{f}] = \bar{f}$. Kết hợp lại:

$$
\mathbb{E}\big[(y - \hat{f}(\mathbf{x}))^2\big] = \underbrace{\big(f(\mathbf{x}) - \bar{f}(\mathbf{x})\big)^2}_{\text{độ chệch}^2} + \underbrace{\mathbb{E}_{\mathcal{D}}\big[(\hat{f}(\mathbf{x}) - \bar{f}(\mathbf{x}))^2\big]}_{\text{phương sai}} + \underbrace{\sigma^2}_{\text{nhiễu không thể giảm}} .
$$

- **Độ chệch** đo mức "sai có hệ thống": ngay cả khi lấy trung bình trên vô số tập huấn luyện, mô hình vẫn lệch khỏi hàm thật. Mô hình quá đơn giản có độ chệch lớn.
- **Phương sai** đo độ nhạy của mô hình với tập huấn luyện cụ thể. Mô hình quá phức tạp có phương sai lớn.
- **Nhiễu không thể giảm** là giới hạn dưới của *mọi* dự đoán tất định.

Thí nghiệm sau khớp đa thức bậc 1, 3 và 12 vào 20 điểm nhiễu của hàm $\sin(\pi x)$, lặp lại trên 500 tập huấn luyện khác nhau:

```python
import numpy as np

rng = np.random.default_rng(0)
f_true = lambda x: np.sin(np.pi * x)            # hàm "thật" mà ta không biết
noise_std, n_train, n_repeat = 0.3, 20, 500
x_train = np.linspace(-1, 1, n_train)           # vị trí đầu vào cố định
x_test = np.linspace(-0.95, 0.95, 101)

for degree in [1, 3, 12]:
    preds = np.empty((n_repeat, x_test.size))
    for r in range(n_repeat):                    # nhiều tập huấn luyện khác nhau
        y = f_true(x_train) + noise_std * rng.standard_normal(n_train)
        poly = np.polynomial.Polynomial.fit(x_train, y, degree)
        preds[r] = poly(x_test)
    mean_pred = preds.mean(axis=0)               # "dự đoán trung bình" f_bar(x)
    bias2 = np.mean((mean_pred - f_true(x_test)) ** 2)
    var = np.mean(preds.var(axis=0))
    print(f"bậc {degree:2d}: bias^2 = {bias2:.4f}  variance = {var:.4f}  "
          f"sai số kỳ vọng = {bias2 + var + noise_std ** 2:.4f}")
```

Kết quả: bậc 1 có độ chệch bình phương khoảng 0,19 (đường thẳng không thể uốn theo hình sin); bậc 12 có độ chệch gần như bằng 0 nhưng phương sai khoảng 0,065, gấp hơn bốn lần bậc 3; bậc 3 cân bằng tốt nhất với sai số kỳ vọng khoảng 0,11. Tất cả đều không xuống dưới được $\sigma^2 = 0{,}09$.

**Liên hệ với TTS: nhiễu không thể giảm chính là tính một-nhiều.** Trong TTS, "nhiễu" $\varepsilon$ không chỉ là tạp âm thu âm. Nó bao gồm mọi biến thiên của tiếng nói mà văn bản không quyết định: người đọc ngắt nghỉ ở đâu, nhấn vào từ nào, tốc độ ra sao, hơi thở, cảm xúc. Với một bộ hồi quy tất định, phần biến thiên này là "không thể giảm", và cách tốt nhất mà hồi quy L2 làm được là dự đoán **trung bình** của nó (Mục 1.4.7), dẫn tới giọng đều đều, phẳng. Muốn vượt qua giới hạn này, ta phải ngừng dự đoán một giá trị và bắt đầu **mô hình hóa cả phân phối**: đó là chủ đề của [Phần II](../phan-2-mo-hinh-sinh/ch07-tong-quan-mo-hinh-sinh.md).

## 2.3. Mô hình tuyến tính

Mô hình tuyến tính đáng học kỹ vì hai lý do. Thứ nhất, lớp cuối của gần như mọi mạng nơ-ron là một mô hình tuyến tính đặt trên các đặc trưng học được. Thứ hai, chúng là nơi ta thấy rõ nhất mối liên hệ giữa hàm mất mát, xác suất và gradient.

### 2.3.1. Hồi quy tuyến tính

Mô hình $\hat{y} = \mathbf{w}^\top\mathbf{x} + b$. Gộp độ lệch vào trọng số bằng cách thêm một thành phần hằng 1 vào $\mathbf{x}$, rồi xếp $n$ mẫu thành ma trận thiết kế $\mathbf{X} \in \mathbb{R}^{n \times d}$ (mỗi hàng là một mẫu) và véc-tơ đích $\mathbf{y} \in \mathbb{R}^{n}$. Hàm mất mát MSE là

$$
\mathcal{L}(\mathbf{w}) = \frac{1}{n}\lVert \mathbf{X}\mathbf{w} - \mathbf{y} \rVert_2^2 ,
\qquad
\nabla_{\mathbf{w}}\mathcal{L} = \frac{2}{n}\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y}) .
$$

Đây đúng là bài toán bình phương tối thiểu ở Mục 1.2.4, với nghiệm dạng đóng $\mathbf{w}^\star = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ khi $\mathbf{X}^\top\mathbf{X}$ khả nghịch. Với mạng nơ-ron ta không còn nghiệm dạng đóng, nhưng dạng gradient "**ma trận đầu vào chuyển vị nhân phần dư**" sẽ xuất hiện lại ở mọi lớp tuyến tính.

### 2.3.2. Hồi quy logistic

Với nhãn nhị phân $y \in \{0, 1\}$, ta muốn mô hình xuất ra một **xác suất**. Hồi quy logistic tính một điểm số tuyến tính $z = \mathbf{w}^\top\mathbf{x} + b$, gọi là **logit**, rồi ép nó vào khoảng $(0, 1)$ bằng hàm sigmoid:

$$
\hat{\pi} = p(y = 1 \mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}} .
$$

Logit có một ý nghĩa đẹp: $z = \log\frac{\hat{\pi}}{1 - \hat{\pi}}$, tức logarit của tỷ lệ cược. Hàm mất mát là negative log-likelihood của phân phối Bernoulli, còn gọi là **entropy chéo nhị phân** (BCE):

$$
\ell(z, y) = -\big[y\log\sigma(z) + (1 - y)\log(1 - \sigma(z))\big] .
$$

**Gradient.** Dùng $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ (bạn hãy tự kiểm tra), ta có $\frac{d}{dz}\log\sigma(z) = 1 - \sigma(z)$ và $\frac{d}{dz}\log(1 - \sigma(z)) = -\sigma(z)$. Do đó

$$
\frac{\partial \ell}{\partial z} = -\big[y(1 - \sigma(z)) - (1 - y)\sigma(z)\big] = \sigma(z) - y ,
\qquad
\nabla_{\mathbf{w}}\ell = (\sigma(z) - y)\,\mathbf{x} .
$$

Gradient theo logit đơn giản là **dự đoán trừ nhãn**. Không có thừa số $\sigma'(z)$ nào làm gradient tiêu biến khi mô hình sai một cách tự tin, và đó là lý do ta ghép sigmoid với BCE thay vì với MSE.

**Trong TTS**, Tacotron 2 dự đoán ở mỗi bước decoder một xác suất "dừng" bằng đúng cơ chế này ([Chương 20](../phan-5-mo-hinh-am-hoc/ch20-tacotron.md)). Một câu dài vài trăm khung chỉ có **một** khung dương, nên dữ liệu mất cân bằng nghiêm trọng; một số bản cài đặt tăng trọng số cho lớp dương (tham số `pos_weight` của `BCEWithLogitsLoss`) để mô hình không học cách "không bao giờ dừng".

### 2.3.3. Hồi quy softmax

Với $K$ lớp, mô hình tính $K$ logit $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b} \in \mathbb{R}^{K}$ và biến chúng thành phân phối Categorical bằng softmax, $p_k = e^{z_k}/\sum_j e^{z_j}$. Hàm mất mát là entropy chéo với nhãn one-hot $\mathbf{y}$ (Mục 1.5.2):

$$
\ell(\mathbf{z}, y) = -\log p_y = -z_y + \log\sum_{j} e^{z_j} .
$$

**Gradient, cách 1** (dùng đồng nhất thức log-sum-exp ở Mục 1.3.4): $\nabla_{\mathbf{z}}\big(\log\sum_j e^{z_j}\big) = \mathbf{p}$ và $\nabla_{\mathbf{z}}(-z_y) = -\mathbf{y}$, nên

$$
\nabla_{\mathbf{z}}\ell = \mathbf{p} - \mathbf{y} .
$$

**Cách 2** (dùng Jacobian của softmax): $\partial\ell/\partial p_k = -y_k/p_k$. Nhân với Jacobian chuyển vị $\operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$ (Mục 1.3.4): thành phần thứ $j$ là $\sum_k (p_k\mathbb{1}[k = j] - p_k p_j)(-y_k/p_k) = -y_j + p_j\sum_k y_k = p_j - y_j$. Hai cách cho cùng kết quả, và lại là "**dự đoán trừ nhãn**". Gradient theo trọng số là tích ngoài $(\mathbf{p} - \mathbf{y})\mathbf{x}^\top$.

Đoạn mã sau kiểm chứng cả hai công thức bằng autograd:

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, K = 4, 6                                   # 4 mẫu, 6 lớp (ví dụ: 6 thanh điệu)
z = torch.randn(B, K, dtype=torch.float64, requires_grad=True)   # (B, K) logit
y = torch.tensor([0, 3, 5, 1])                # (B,) nhãn đúng

loss = F.cross_entropy(z, y)                  # nhận LOGIT, tự làm log-softmax
loss.backward()
p = torch.softmax(z.detach(), dim=1)
y_onehot = F.one_hot(y, K).double()
assert torch.allclose(z.grad, (p - y_onehot) / B)   # dL/dz = (p - y) / B

# Trường hợp nhị phân: cờ dừng (stop token) với BCE trên logit
s = torch.randn(B, dtype=torch.float64, requires_grad=True)       # (B,) logit
t = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)      # (B,) nhãn
F.binary_cross_entropy_with_logits(s, t).backward()
assert torch.allclose(s.grad, (torch.sigmoid(s.detach()) - t) / B)
print("Gradient của CE và BCE theo logit đều có dạng (dự đoán - nhãn).")
```

Thừa số $1/B$ xuất hiện vì `F.cross_entropy` mặc định lấy **trung bình** trên batch. Chi tiết nhỏ này trở nên quan trọng khi ta tích lũy gradient qua nhiều batch con ([Chương 3](ch03-toi-uu-va-ky-thuat-huan-luyen.md)).

!!! warning "Truyền logit, đừng truyền xác suất"
    `F.cross_entropy` nhận **logit** và tự tính `log_softmax` theo cách ổn định số học; `F.binary_cross_entropy_with_logits` cũng vậy. Nếu bạn áp `softmax` (hoặc `sigmoid`) trước rồi mới đưa vào, mô hình vẫn chạy nhưng học chậm và sai lệch, vì softmax bị áp hai lần. Đây là một trong những lỗi phổ biến nhất của người mới.

## 2.4. Hàm mất mát từ góc nhìn xác suất: MLE và MAP

### 2.4.1. Ước lượng hợp lý cực đại

Thay vì tự chọn một hàm mất mát, ta chọn một **mô hình xác suất có điều kiện** $p_\theta(\mathbf{y} \mid \mathbf{x})$ và tìm tham số làm dữ liệu quan sát được "hợp lý nhất". Với dữ liệu độc lập cùng phân phối, hàm hợp lý (*likelihood*) là tích $\prod_i p_\theta(\mathbf{y}_i \mid \mathbf{x}_i)$; lấy logarit biến tích thành tổng (và tránh tràn số), nên **ước lượng hợp lý cực đại** (*maximum likelihood estimation*, MLE) là

$$
\theta_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^{n}\log p_\theta(\mathbf{y}_i \mid \mathbf{x}_i) = \arg\min_\theta \frac{1}{n}\sum_{i=1}^{n}\big[-\log p_\theta(\mathbf{y}_i \mid \mathbf{x}_i)\big] .
$$

Vế phải chính là ERM với hàm mất mát $\ell = -\log p_\theta$, gọi là **negative log-likelihood** (NLL). Theo Mục 1.5.3, đây cũng là cực tiểu hóa (ước lượng thực nghiệm của) $D_{\mathrm{KL}}(p_{\text{data}} \,\Vert\, p_\theta)$. Chọn họ phân phối nào, ta được hàm mất mát tương ứng:

- **Gauss với phương sai cố định** cho MSE, và mô hình học trung bình có điều kiện (Mục 1.4.7).
- **Laplace với thang đo cố định** cho L1, và mô hình học trung vị có điều kiện.
- **Bernoulli** cho entropy chéo nhị phân (Mục 2.3.2).
- **Categorical** cho entropy chéo (Mục 2.3.3).

**Mở rộng: học cả phương sai.** Nếu mạng xuất cả trung bình $\mu_\theta(\mathbf{x})$ lẫn phương sai $\sigma^2_\theta(\mathbf{x})$ thì NLL Gauss (bỏ hằng số) là

$$
\ell = \frac{(y - \mu_\theta(\mathbf{x}))^2}{2\sigma_\theta^2(\mathbf{x})} + \tfrac{1}{2}\log\sigma_\theta^2(\mathbf{x}) .
$$

Mô hình giờ có thể "thừa nhận" những vùng khó đoán bằng cách tăng $\sigma$, đổi lại bị phạt bởi số hạng $\log\sigma^2$. Đây là ý tưởng của các mạng mật độ hỗn hợp (*mixture density network*) và của tiên nghiệm có phương sai học được trong một số mô hình dựa trên flow ([Chương 7](../phan-2-mo-hinh-sinh/ch07-tong-quan-mo-hinh-sinh.md), [Chương 23](../phan-5-mo-hinh-am-hoc/ch23-glow-grad-matcha.md)). Trong thực hành, nên tham số hóa $\log\sigma^2$ thay vì $\sigma^2$ và đặt một ngưỡng dưới cho nó, nếu không mô hình có thể đẩy $\sigma \to 0$ trên các mẫu dễ và làm loss tiến về $-\infty$.

### 2.4.2. Hậu nghiệm cực đại và điều chuẩn

MLE chỉ nhìn vào dữ liệu. Nếu ta có **niềm tin tiên nghiệm** $p(\theta)$ về tham số (ví dụ: trọng số không nên quá lớn), quy tắc Bayes cho hậu nghiệm $p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta)\,p(\theta)$, và **ước lượng hậu nghiệm cực đại** (MAP) là

$$
\theta_{\text{MAP}} = \arg\min_\theta \Big[\sum_{i=1}^{n} -\log p_\theta(\mathbf{y}_i \mid \mathbf{x}_i) \;-\; \log p(\theta)\Big] .
$$

Số hạng $-\log p(\theta)$ chính là một **số hạng điều chuẩn** (*regularizer*):

- **Tiên nghiệm Gauss** $\theta \sim \mathcal{N}(\mathbf{0}, \tau^2\mathbf{I})$: $-\log p(\theta) = \frac{1}{2\tau^2}\lVert\theta\rVert_2^2 + \text{hằng số}$. Đây là **điều chuẩn L2** (*weight decay*, ridge). Tiên nghiệm càng hẹp ($\tau$ nhỏ) thì phạt càng mạnh.
- **Tiên nghiệm Laplace** $p(\theta_j) \propto e^{-\lvert\theta_j\rvert/b}$: $-\log p(\theta) = \frac{1}{b}\lVert\theta\rVert_1 + \text{hằng số}$. Đây là **điều chuẩn L1** (lasso), có xu hướng đẩy nhiều tham số về **đúng bằng 0** (thưa).

Với hồi quy tuyến tính, nhiễu Gauss phương sai $\sigma^2$ và tiên nghiệm Gauss phương sai $\tau^2$, MAP có nghiệm dạng đóng $\mathbf{w}^\star = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$ với $\lambda = \sigma^2/\tau^2$ (Bài tập 3). Số hạng $\lambda\mathbf{I}$ còn có lợi ích số học: nó làm ma trận luôn khả nghịch và giảm số điều kiện. [Chương 3](ch03-toi-uu-va-ky-thuat-huan-luyen.md) sẽ chỉ ra một điểm tinh tế: với các bộ tối ưu thích nghi như Adam, "thêm L2 vào hàm mất mát" và "weight decay" **không còn tương đương**, và đó là lý do AdamW ra đời.

### 2.4.3. Bảng tổng hợp

| Giả định xác suất | Hàm mất mát / điều chuẩn | Mô hình học được | Ví dụ trong TTS |
|---|---|---|---|
| $y \sim \mathcal{N}(\hat{y}, \sigma^2)$ | MSE (L2) | Trung bình có điều kiện | Mel của Tacotron 2; trường độ, F0, năng lượng của FastSpeech 2 |
| $y \sim \operatorname{Laplace}(\hat{y}, b)$ | L1 (MAE) | Trung vị có điều kiện | Mất mát mel phụ trợ của HiFi-GAN, VITS |
| $y \sim \operatorname{Bernoulli}(\sigma(z))$ | Entropy chéo nhị phân | Xác suất lớp dương | Cờ dừng |
| $y \sim \operatorname{Cat}(\operatorname{softmax}(\mathbf{z}))$ | Entropy chéo | Phân phối trên các lớp | Token codec, mẫu μ-law của WaveNet |
| $\theta \sim \mathcal{N}(\mathbf{0}, \tau^2\mathbf{I})$ | Phạt $\lVert\theta\rVert_2^2$ | Trọng số nhỏ | Weight decay trong AdamW |
| $\theta_j \sim \operatorname{Laplace}(0, b)$ | Phạt $\lVert\theta\rVert_1$ | Trọng số thưa | Cắt tỉa (*pruning*) mô hình |

## 2.5. Mạng nơ-ron nhiều lớp

### 2.5.1. Vì sao cần phi tuyến: bài toán XOR

Mô hình tuyến tính chỉ vẽ được **siêu phẳng** làm biên quyết định. Bài toán XOR (bốn điểm $(0,0), (1,1)$ thuộc lớp 0 và $(0,1), (1,0)$ thuộc lớp 1) không thể tách bằng một đường thẳng. Nhưng nếu ta trước hết biến đổi đầu vào bằng hai đặc trưng phi tuyến $h_1 = \operatorname{ReLU}(x_1 + x_2)$ và $h_2 = \operatorname{ReLU}(x_1 + x_2 - 1)$, thì $h_1 - 2h_2$ cho đúng $0, 1, 1, 0$ trên bốn điểm. Bài học: **một tầng ẩn phi tuyến biến bài toán không tuyến tính thành bài toán tuyến tính trong không gian đặc trưng mới.** Mạng nơ-ron là cách *học* những đặc trưng như vậy thay vì tự thiết kế.

Như Mục 1.2.1 đã nhấn mạnh, hàm phi tuyến là bắt buộc: không có nó, mọi tầng tuyến tính chồng lên nhau sụp đổ về một tầng.

### 2.5.2. Kiến trúc MLP

**Perceptron nhiều lớp** (*multilayer perceptron*, MLP) gồm $L$ lớp, mỗi lớp là một phép biến đổi affine theo sau bởi một hàm kích hoạt $\phi$ áp dụng từng phần tử:

$$
\mathbf{h}^{(0)} = \mathbf{x}, \qquad
\mathbf{z}^{(l)} = \mathbf{W}^{(l)}\mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}, \qquad
\mathbf{h}^{(l)} = \phi\big(\mathbf{z}^{(l)}\big), \qquad l = 1, \ldots, L ,
$$

với $\mathbf{W}^{(l)} \in \mathbb{R}^{d_l \times d_{l-1}}$. Lớp cuối thường không có hàm kích hoạt (hồi quy) hoặc xuất logit cho softmax (phân loại). Ta gọi $\mathbf{z}^{(l)}$ là **tiền kích hoạt** (*pre-activation*) và $\mathbf{h}^{(l)}$ là **kích hoạt**.

MLP xuất hiện ở khắp nơi trong TTS, thường dưới những cái tên khác: **pre-net** của Tacotron 2 là một MLP hai lớp 256 đơn vị với ReLU và dropout; **mạng truyền thẳng** (*feed-forward network*, FFN) trong mỗi khối Transformer là một MLP hai lớp áp dụng độc lập cho từng vị trí ([Chương 5](ch05-transformer.md)); các phép chiếu embedding người nói hay embedding thời gian $t$ trong diffusion cũng thường là MLP nhỏ.

### 2.5.3. Các hàm kích hoạt

| Tên | Công thức $\phi(z)$ | Đạo hàm $\phi'(z)$ | Ưu điểm | Nhược điểm | Hay gặp ở |
|---|---|---|---|---|---|
| Sigmoid | $\sigma(z) = \frac{1}{1 + e^{-z}}$ | $\sigma(z)(1 - \sigma(z)) \le \frac{1}{4}$ | Đầu ra trong $(0, 1)$, hiểu như xác suất hoặc cổng | Bão hòa, gradient tiêu biến; không đối xứng quanh 0 | Cổng của LSTM, cờ dừng |
| tanh | $\frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$ | $1 - \tanh^2(z) \le 1$ | Đối xứng quanh 0 | Vẫn bão hòa ở hai đầu | Post-net Tacotron 2, đơn vị cổng của WaveNet |
| ReLU | $\max(0, z)$ | $\mathbb{1}[z > 0]$ | Rẻ, không bão hòa với $z > 0$ | "Chết" nếu $z < 0$ mãi mãi | Pre-net Tacotron 2, FFN của Transformer gốc, FastSpeech |
| Leaky ReLU | $\max(az, z)$, $0 < a < 1$ | $1$ hoặc $a$ | Không bao giờ "chết" hẳn | Thêm siêu tham số $a$ | HiFi-GAN, MelGAN (với $a = 0{,}1$ hoặc $0{,}2$) |
| GELU | $z\,\Phi(z)$ | $\Phi(z) + z\,\varphi(z)$ | Trơn, hoạt động tốt trong Transformer | Đắt hơn ReLU một chút | BERT, GPT, nhiều Transformer hiện đại |
| SiLU/Swish | $z\,\sigma(z)$ | $\sigma(z)\big(1 + z(1 - \sigma(z))\big)$ | Trơn, không đơn điệu | Đắt hơn ReLU một chút | Conformer, các khối gated như SwiGLU |

Ở đây $\Phi$ và $\varphi$ là hàm phân phối tích lũy và hàm mật độ của $\mathcal{N}(0, 1)$. Một vài nhận xét quan trọng:

- **Bão hòa và gradient tiêu biến.** Đạo hàm của sigmoid không vượt quá $1/4$. Qua 10 lớp sigmoid, gradient bị nhân với một thừa số tối đa $(1/4)^{10} \approx 10^{-6}$ (chưa kể trọng số). Đây là lý do mạng sâu chỉ thực sự huấn luyện được sau khi ReLU trở nên phổ biến (Nair và Hinton, 2010; Glorot và cộng sự, 2011), cùng với khởi tạo tốt (Mục 2.7), chuẩn hóa và kết nối tắt ([Chương 3](ch03-toi-uu-va-ky-thuat-huan-luyen.md), [Chương 4](ch04-cnn-rnn-mo-hinh-chuoi.md)).
- **ReLU "chết".** Nếu một nơ-ron ReLU có tiền kích hoạt âm với mọi đầu vào (ví dụ sau một bước cập nhật quá lớn), gradient qua nó bằng 0 vĩnh viễn. Leaky ReLU (Maas và cộng sự, 2013) để lại một độ dốc nhỏ $a$ cho phần âm. Các vocoder GAN như HiFi-GAN dùng Leaky ReLU với độ dốc 0,1 xen giữa các lớp tích chập.
- **GELU** (Hendrycks và Gimpel, 2016) có thể hiểu là "nhân đầu vào với xác suất nó dương" dưới phân phối chuẩn. Dạng xấp xỉ $0{,}5z\big(1 + \tanh\big[\sqrt{2/\pi}(z + 0{,}044715z^3)\big]\big)$ được dùng rộng rãi; trong PyTorch đó là `F.gelu(x, approximate="tanh")`, với sai số tuyệt đối so với dạng chính xác dưới $5 \times 10^{-4}$.
- **SiLU/Swish** (Elfwing và cộng sự, 2018; Ramachandran và cộng sự, 2017) có giá trị nhỏ nhất xấp xỉ $-0{,}28$ tại $z \approx -1{,}28$: hơi âm với đầu vào âm vừa phải, rồi tiến về 0. Nó là thành phần của SwiGLU, khối FFN có cổng trong nhiều mô hình ngôn ngữ lớn mà các hệ TTS dựa trên LLM thừa hưởng ([Chương 5](ch05-transformer.md), [Chương 34](../phan-8-mo-hinh-lon/ch34-he-thong-lai-hien-dai.md)).
- **Snake**, $z + \frac{1}{\alpha}\sin^2(\alpha z)$ với $\alpha$ học được (Ziyin và cộng sự, 2020), đưa một thiên kiến quy nạp **tuần hoàn** vào mạng. BigVGAN dùng nó để vocoder tái tạo tốt hơn các thành phần điều hòa của tiếng nói ([Chương 25](../phan-6-vocoder/ch25-gan-vocoder.md)). Đây là ví dụ đẹp về việc chọn hàm kích hoạt theo *bản chất của tín hiệu*.

### 2.5.4. Định lý xấp xỉ vạn năng

Mạng nơ-ron biểu diễn được những hàm nào? **Định lý xấp xỉ vạn năng** (Cybenko, 1989; Hornik và cộng sự, 1989; Leshno và cộng sự, 1993) khẳng định: một MLP với **một** tầng ẩn đủ rộng và hàm kích hoạt không phải đa thức (sigmoid, ReLU, ...) có thể xấp xỉ bất kỳ hàm liên tục nào trên một tập compact với sai số tùy ý.

**Trực giác với ReLU.** Hiệu của hai ReLU lệch nhau, $\operatorname{ReLU}(z - a) - \operatorname{ReLU}(z - b)$, là một "đoạn dốc" lên rồi đi ngang. Tổ hợp hai đoạn dốc ngược chiều cho ta một "cái bướu" hình thang. Cộng nhiều cái bướu có chiều cao khác nhau, ta xấp xỉ được bất kỳ đường cong liên tục nào bằng một hàm tuyến tính từng khúc, càng nhiều nơ-ron càng mịn. Trong nhiều chiều, ý tưởng tương tự (dù kỹ thuật phức tạp hơn) vẫn đúng.

**Những gì định lý *không* nói.** (1) Số nơ-ron cần thiết có thể lớn theo cấp số mũ với số chiều đầu vào. (2) Định lý chỉ nói *tồn tại* bộ tham số tốt, không nói gradient descent sẽ *tìm* được nó. (3) Nó không nói gì về tổng quát hóa từ dữ liệu hữu hạn. Trong thực tế, mạng **sâu** thường hiệu quả hơn hẳn mạng nông rộng với cùng số tham số, vì chúng biểu diễn được các hàm có cấu trúc phân cấp: âm vị ghép thành âm tiết, âm tiết thành từ, từ thành ngữ đoạn. Và quan trọng không kém là **thiên kiến quy nạp**: tích chập cho tín hiệu cục bộ và bất biến tịnh tiến, attention cho các phụ thuộc xa ([Chương 4](ch04-cnn-rnn-mo-hinh-chuoi.md), [Chương 5](ch05-transformer.md)).
