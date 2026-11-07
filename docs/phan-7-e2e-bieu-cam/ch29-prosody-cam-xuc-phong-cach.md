# Chương 29. Ngôn điệu, cảm xúc và phong cách

!!! abstract "Mục tiêu học tập"
    Sau chương này, bạn có thể:

    - **Giải thích** bằng định luật phương sai toàn phần vì sao điều kiện hóa thêm một biến ngôn điệu làm giảm hiện tượng quá trơn, và vì sao lúc suy luận lại cần một cách chọn biến đó.
    - **Thao tác** cao độ, trường độ, năng lượng một cách tường minh mà **không phá** đường nét thanh điệu tiếng Việt (làm việc trong miền log-F0, tách thành phần câu và thành phần thanh).
    - **Cài đặt** bộ mã hóa tham chiếu (*reference encoder*) và Global Style Tokens, kiểm chứng rằng véc-tơ phong cách là tổ hợp lồi của các token.
    - **So sánh** các họ mô hình phong cách: bộ mã hóa tham chiếu, GST, VAE, biến ẩn chi tiết theo âm vị/âm tiết, dự đoán từ mô hình ngôn ngữ, StyleTTS/StyleTTS 2, điều khiển bằng mô tả ngôn ngữ tự nhiên.
    - **Thiết kế** dữ liệu và quy trình cho TTS cảm xúc và đa phong cách đọc tiếng Việt (tin tức, sách nói, kể chuyện, chăm sóc khách hàng).
    - **Chẩn đoán** các lỗi điển hình: rò rỉ nội dung qua bộ mã hóa tham chiếu, sụp đổ hậu nghiệm, phong cách "ăn" mất thanh điệu.

!!! info "Kiến thức cần có"
    - Ngôn điệu, quan hệ thanh điệu–ngữ điệu trong tiếng Việt, tiểu từ cuối câu: [Chương 16](../phan-3-tin-hieu-ngu-am/ch16-ngon-dieu.md); thanh điệu: [Chương 15](../phan-3-tin-hieu-ngu-am/ch15-ngu-am-tieng-viet.md).
    - Bài toán một–nhiều và hiện tượng quá trơn: [Chương 7](../phan-2-mo-hinh-sinh/ch07-tong-quan-mo-hinh-sinh.md).
    - VAE, sụp đổ hậu nghiệm: [Chương 8](../phan-2-mo-hinh-sinh/ch08-vae.md).
    - Multi-head attention: [Chương 5](../phan-1-nen-tang/ch05-transformer.md).
    - FastSpeech 2 và bộ điều chỉnh phương sai: [Chương 22](../phan-5-mo-hinh-am-hoc/ch22-fastspeech.md); điều kiện hóa người nói: [Chương 28](ch28-da-nguoi-noi-voice-cloning.md).

## 29.1. Bài toán một–nhiều của ngôn điệu

### 29.1.1. Trực giác

Đưa cho mười người cùng một câu "Mai em về quê." và bạn sẽ nghe mười cách đọc: người đọc chậm và đều như bản tin, người hào hứng, người buồn bã, người nhấn vào "mai", người nhấn vào "quê". Văn bản **không chứa** đủ thông tin để xác định cách đọc. Phần thông tin thiếu đó (cao độ ở mức câu, nhịp, độ nhấn, cảm xúc, phong cách) gọi chung là **ngôn điệu và phong cách**. Nó là nguồn chính của tính "một–nhiều" trong TTS sau khi đã cố định người nói.

Ở [Chương 7](../phan-2-mo-hinh-sinh/ch07-tong-quan-mo-hinh-sinh.md) chúng ta đã thấy: một mô hình hồi quy với L2 sẽ học trung bình có điều kiện $\mathbb{E}[\mathbf{Y} \mid \mathbf{x}]$, và trung bình của nhiều cách đọc là một cách đọc "nhạt": cao độ phẳng, nhịp đều, phổ mờ. Chương này hỏi: làm sao để mô hình **biết**, **học** và **cho phép ta chọn** phần thông tin không có trong văn bản?

### 29.1.2. Hình thức hóa: định luật phương sai toàn phần

Giả sử có một biến ngôn điệu $\mathbf{p}$ (ta chưa nói nó là gì: một véc-tơ phong cách, một chuỗi biến ẩn, một nhãn cảm xúc). Định luật phương sai toàn phần cho (xét từng thành phần của $\mathbf{Y}$):

$$
\operatorname{Var}(\mathbf{Y} \mid \mathbf{x}) = \underbrace{\mathbb{E}_{\mathbf{p} \mid \mathbf{x}}\big[\operatorname{Var}(\mathbf{Y} \mid \mathbf{x}, \mathbf{p})\big]}_{\text{phần còn lại khi đã biết } \mathbf{p}} + \underbrace{\operatorname{Var}_{\mathbf{p} \mid \mathbf{x}}\big(\mathbb{E}[\mathbf{Y} \mid \mathbf{x}, \mathbf{p}]\big)}_{\text{phần do ngôn điệu giải thích}}.
$$

Sai số bình phương trung bình của bộ dự đoán tốt nhất chỉ nhìn văn bản là vế trái; của bộ dự đoán nhìn cả văn bản lẫn $\mathbf{p}$ là số hạng thứ nhất ở vế phải. Hiệu của chúng, số hạng thứ hai, chính là lượng "quá trơn" được loại bỏ nhờ biết $\mathbf{p}$. Phân tích này cho hai bài học:

1. **Khi huấn luyện**, nếu ta trích $\mathbf{p}$ từ chính âm thanh đích (qua một bộ mã hóa tham chiếu), mô hình dự đoán sắc nét hơn hẳn. Mọi phương pháp trong chương này đều bắt đầu bằng việc học một $\mathbf{p}$ như thế.
2. **Khi suy luận**, không có âm thanh đích, nên ta cần một trong ba cách để có $\mathbf{p}$: (a) lấy từ một âm thanh **tham chiếu** khác (chuyển ngôn điệu), (b) do **người dùng chỉ định** (điều khiển), hoặc (c) **lấy mẫu/dự đoán** từ văn bản bằng một mô hình $p(\mathbf{p} \mid \mathbf{x})$. Phần lớn sự khác biệt giữa các phương pháp nằm ở cách (c).

Có một sự đánh đổi ẩn: $\mathbf{p}$ càng giàu thông tin, số hạng thứ nhất càng nhỏ (tái tạo càng tốt), nhưng $\mathbf{p}$ càng dễ "lẫn" cả nội dung lẫn người nói vào, và càng khó dự đoán từ văn bản. Thiết kế một hệ thống ngôn điệu tốt là chọn đúng **độ rộng nút thắt** cho $\mathbf{p}$.

### 29.1.3. Bản đồ các cách tiếp cận

| Cách tiếp cận | Biến ngôn điệu $\mathbf{p}$ | Có giám sát? | Mức chi tiết | Lúc suy luận |
|---|---|---|---|---|
| Điều khiển tường minh (FastSpeech 2, FastPitch) | Trường độ, F0, năng lượng | Có (trích bằng DSP) | Theo âm vị/khung | Dự đoán, rồi người dùng sửa |
| Bộ mã hóa tham chiếu | Véc-tơ toàn câu | Không | Toàn câu | Âm thanh tham chiếu |
| GST | Tổ hợp của $K$ token | Không | Toàn câu | Tham chiếu, chọn token, hoặc dự đoán từ văn bản |
| VAE phong cách | Biến ẩn liên tục (có prior) | Không | Toàn câu | Lấy mẫu prior, tham chiếu |
| Biến ẩn chi tiết | Chuỗi biến ẩn | Không | Âm vị/âm tiết | Prior tự hồi quy hoặc dự đoán |
| Đặc trưng từ mô hình ngôn ngữ | Biểu diễn ngữ cảnh của văn bản | Không (tiền huấn luyện) | Từ/âm tiết | Tính từ văn bản |
| StyleTTS 2 | Véc-tơ phong cách | Không | Toàn câu | Lấy mẫu bằng khuếch tán theo văn bản |
| Nhãn/mô tả (cảm xúc, prompt) | Nhãn hoặc câu mô tả | Có | Toàn câu | Người dùng viết |

## 29.2. Điều khiển tường minh

### 29.2.1. Trường độ, cao độ, năng lượng

Các mô hình có bộ điều chỉnh phương sai như FastSpeech 2 (Ren và cộng sự, 2021; [Chương 22](../phan-5-mo-hinh-am-hoc/ch22-fastspeech.md)) dự đoán tường minh trường độ $\mathbf{d}$, cao độ và năng lượng, rồi điều kiện hóa bộ giải mã trên chúng. Vì các đại lượng này có ý nghĩa vật lý, ta có thể sửa chúng trước khi giải mã:

- **Trường độ**: $d_i' = \operatorname{round}(\alpha_i d_i)$, với $\alpha_i$ toàn cục (tốc độ đọc) hoặc riêng cho một từ/cụm (kéo dài từ cần nhấn, thêm khoảng ngừng).
- **Cao độ**: làm việc trong miền **log**. Đặt $\ell = \log_2 F_0$; dịch $k$ nửa cung là $\ell' = \ell + k/12$, tức là nhân $F_0$ với $2^{k/12}$; mở rộng hoặc thu hẹp biên độ là $\ell' = \bar{\ell} + r(\ell - \bar{\ell})$.
- **Năng lượng**: nhân hệ số, thường cho hiệu quả kém tự nhiên hơn hai đại lượng trên vì năng lượng tương quan chặt với chất giọng và cao độ.

**FastPitch** (Łańcucki, 2021) đẩy ý tưởng này đi xa hơn: mô hình dự đoán **cao độ trung bình cho mỗi ký hiệu đầu vào** (lấy trung bình F0 trên các khung được căn chỉnh với ký hiệu đó), nên người dùng có thể sửa cao độ theo từng âm vị, ví dụ nâng cao một từ để nhấn mạnh, làm phẳng, hoặc đảo đường nét, rồi bộ giải mã sinh lại phổ phù hợp.

Vì sao làm trong miền log? Cảm nhận cao độ gần với thang log (một quãng tám là gấp đôi tần số ở mọi tầng cao độ). Nhân $F_0$ với một hằng số giữ nguyên mọi tỉ số, tức là giữ hình dạng đường nét như tai người nghe; cộng một hằng số theo Hz thì nén các dao động tương đối, làm đường nét "phẳng" đi.

### 29.2.2. Thanh điệu tiếng Việt: thay đổi ngữ điệu mà không phá thanh

Với tiếng Anh, đường F0 chủ yếu mang ngữ điệu; làm phẳng nó làm giọng nghe chán nhưng không đổi nghĩa từ. Với tiếng Việt, F0 mang đồng thời **ngữ điệu** (mức câu) và **thanh điệu** (mức âm tiết, khu biệt nghĩa: *ma, mà, má, mả, mã, mạ*). Một mô hình đơn giản là cộng hai thành phần trong miền log:

$$
\ell(t) = \underbrace{\ell_{\text{câu}}(t)}_{\text{mức nền, độ hạ dần, biên độ}} + \underbrace{\ell_{\text{thanh}}(t)}_{\text{đường nét từng âm tiết}}.
$$

Đây là một xấp xỉ (thực tế hai thành phần tương tác, và thanh ngã, nặng còn có đặc trưng chất giọng như thanh quản hóa, không chỉ là F0; xem [Chương 15](../phan-3-tin-hieu-ngu-am/ch15-ngu-am-tieng-viet.md), [Chương 16](../phan-3-tin-hieu-ngu-am/ch16-ngon-dieu.md)), nhưng đủ để rút ra quy tắc thực hành: **mọi thao tác "làm phẳng" hay "thu hẹp biên độ" chỉ nên tác động lên $\ell_{\text{câu}}$**. Nếu áp lên cả $\ell$, độ dốc của thanh sắc, độ trũng của thanh hỏi cũng bị thu nhỏ, và thanh bắt đầu lẫn với nhau.

Mã dưới đây minh họa với các đường nét thanh **phác họa** (không phải số đo thực nghiệm): dịch theo tỉ lệ Hz giữ nguyên hình dạng thanh; cộng theo Hz hoặc làm phẳng toàn cục làm giảm độ dốc của mọi thanh; còn tách thành phần câu (khớp một đường thẳng qua trung bình các âm tiết) rồi chỉ sửa thành phần đó thì giữ gần như nguyên hình dạng thanh.

```python
import numpy as np

# Đường nét thanh điệu PHÁC HỌA (đơn vị nửa cung, so với mức nền của người nói); chỉ để minh họa
TONES = {
    "ngang": lambda u: 0.0 - 0.5 * u,
    "huyền": lambda u: -2.0 - 2.0 * u,
    "sắc":   lambda u: -0.5 + 4.5 * u,
    "hỏi":   lambda u: -1.0 - 6.0 * u + 5.0 * u ** 2,
    "ngã":   lambda u: 0.5 + 5.0 * u,
    "nặng":  lambda u: -2.0 - 3.5 * u,
}
seq = ["ngang", "sắc", "huyền", "hỏi", "nặng", "ngã", "ngang", "sắc"]
n = 20                                               # số khung mỗi âm tiết
u = np.linspace(0, 1, n)
tone = np.concatenate([TONES[t](u) for t in seq])    # thành phần thanh điệu
t = np.linspace(0, 1, len(tone))
phrase = 1.0 - 3.0 * t                               # thành phần câu: hạ dần (declination)
st = phrase + tone                                   # log-F0 theo nửa cung
f0_hz = 120.0 * 2 ** (st / 12)                        # đổi sang Hz quanh mức nền 120 Hz

def to_st(f, ref=120.0):
    return 12 * np.log2(f / ref)

def syllable_slopes(st_curve):
    """Độ dốc (cuối - đầu) của từng âm tiết: đặc trưng thô của hình dạng thanh."""
    s = st_curve.reshape(len(seq), n)
    return s[:, -1] - s[:, 0]

base = syllable_slopes(st)
# (a) Nhân F0 (Hz) với hằng số = dịch đều trong miền log: hình dạng thanh giữ nguyên
assert np.allclose(syllable_slopes(to_st(f0_hz * 1.25)), base)
# (b) Cộng hằng số theo Hz: đường nét bị nén lại (thanh nghe "phẳng" hơn)
print("cộng 60 Hz, tỉ lệ độ dốc:", np.round(syllable_slopes(to_st(f0_hz + 60)) / base, 2))
# (c) "Làm phẳng" ngây thơ: kéo toàn bộ log-F0 về trung bình với hệ số 0.3
flat = st.mean() + 0.3 * (st - st.mean())
assert np.allclose(syllable_slopes(flat), 0.3 * base)       # độ dốc mọi thanh giảm 70%
# (d) Tách thành phần câu (khớp đường thẳng trên trung bình âm tiết) rồi chỉ sửa thành phần đó
centers = (np.arange(len(seq)) + 0.5) / len(seq)
coef = np.polyfit(centers, st.reshape(len(seq), n).mean(1), deg=1)
phrase_hat = np.polyval(coef, t)
residual = st - phrase_hat                                  # ước lượng thành phần thanh điệu
new_phrase = phrase_hat.mean() + 0.3 * (phrase_hat - phrase_hat.mean()) + 2.0  # ngữ điệu phẳng hơn, cao hơn
edited = new_phrase + residual
ratio = syllable_slopes(edited) / base
print("tách thành phần, tỉ lệ độ dốc:", np.round(ratio, 2))
assert np.all(np.abs(ratio - 1) < 0.25)                     # hình dạng thanh gần như được giữ
```

Tỉ lệ ở trường hợp (d) không đúng bằng 1 vì độ hạ dần của câu cũng đóng góp một phần nhỏ vào độ dốc trong mỗi âm tiết; khi ta làm phẳng thành phần câu, phần đóng góp đó thay đổi theo. Đó là hành vi mong muốn: âm tiết vẫn giữ hình dạng thanh, chỉ "khung" ngữ điệu thay đổi.

!!! note "Góc tiếng Việt: biểu diễn cao độ cho mô hình kiểu FastPitch"
    FastPitch lấy **trung bình** F0 trên mỗi ký hiệu. Với tiếng Việt, trung bình một nguyên âm làm mất hình dạng thanh (sắc và ngang có thể có cùng trung bình nhưng khác độ dốc); mô hình phải tự sinh lại đường nét từ nhãn thanh trong văn bản, và người dùng chỉ sửa được mức chứ không sửa được hình dạng. Một lựa chọn hợp lý hơn là tham số hóa cao độ **theo âm tiết** bằng vài hệ số (ví dụ trung bình, độ dốc, độ cong, tương đương ba hệ số đầu của khai triển đa thức Legendre hoặc DCT trên khoảng thời gian của vần), rồi chỉ cho phép điều khiển hệ số trung bình ở mức câu. Đây là gợi ý thiết kế để thử nghiệm, không phải một phương pháp chuẩn đã được công bố rộng rãi cho tiếng Việt.

## 29.3. Bộ mã hóa tham chiếu

### 29.3.1. Kiến trúc

Skerry-Ryan và cộng sự (2018) đề xuất cách đơn giản nhất để học $\mathbf{p}$ không cần nhãn: thêm vào Tacotron một **bộ mã hóa tham chiếu** nén mel của chính câu đích thành một véc-tơ có độ dài cố định, rồi phát rộng véc-tơ này và nối vào đầu ra bộ mã hóa văn bản. Kiến trúc gốc: 6 lớp tích chập 2-D (nhân $3 \times 3$, bước 2, BatchNorm, ReLU, số kênh 32, 32, 64, 64, 128, 128) quét trên mel như một ảnh, rồi một GRU 128 chiều đọc chuỗi khung đã giảm mẫu; trạng thái cuối của GRU là "embedding ngôn điệu". Các bước giảm mẫu theo thời gian và tần số đóng vai trò **nút thắt**: chúng giữ lại đường nét thô (cao độ, nhịp, năng lượng) và bỏ chi tiết ngữ âm.

Khi huấn luyện, mel tham chiếu chính là mel đích (tự mã hóa). Khi suy luận, đưa vào mel của một câu khác: mô hình đọc văn bản mới với ngôn điệu "chép" từ câu tham chiếu.

### 29.3.2. Giới hạn: rò rỉ nội dung và ràng buộc độ dài

Bài báo gốc và các công trình sau ghi nhận những vấn đề sau:

- **Rò rỉ nội dung (*content leakage*).** Vì tham chiếu lúc huấn luyện là chính câu đích, bộ mã hóa có động cơ mã hóa cả *nội dung* để giảm mất mát tái tạo. Lúc suy luận với văn bản khác, phần nội dung "lậu" này xung đột với văn bản thật, gây đọc lẫn hoặc méo. Nút thắt càng rộng, rò rỉ càng nặng; quá hẹp thì không đủ ngôn điệu.
- **Chuyển ngôn điệu hoạt động tốt nhất khi văn bản tương tự.** Một véc-tơ toàn câu không biết "nhấn ở đâu" khi số âm tiết và cấu trúc cú pháp khác hẳn.
- **Lẫn người nói.** Trong dữ liệu nhiều người nói, véc-tơ ngôn điệu cũng mang thông tin giọng; chép ngôn điệu từ giọng nam sang giọng nữ có thể kéo theo cao độ nam. Các biện pháp thường dùng: chuẩn hóa F0 theo người nói trước khi vào bộ mã hóa, hoặc thêm mục tiêu đối kháng để bộ mã hóa không đoán được người nói (kỹ thuật đảo gradient ở [Chương 30](ch30-da-ngon-ngu-code-switching.md)).

## 29.4. Global Style Tokens

### 29.4.1. Ý tưởng

Global Style Tokens (GST; Wang và cộng sự, 2018) thêm một tầng cấu trúc lên bộ mã hóa tham chiếu: thay vì dùng thẳng véc-tơ tham chiếu $\mathbf{r}$, ta học một **ngân hàng $K$ token phong cách** $\{\mathbf{t}_1, \ldots, \mathbf{t}_K\}$ (bài báo dùng $K = 10$) và biểu diễn phong cách như một **tổ hợp có trọng số** của chúng, với trọng số tính bằng attention:

$$
\alpha_k = \operatorname{softmax}_k\!\left(\frac{(\mathbf{W}_q\mathbf{r})^\top \mathbf{W}_k \tanh(\mathbf{t}_k)}{\sqrt{d}}\right), \qquad
\mathbf{p} = \mathbf{W}_o\sum_{k=1}^{K}\alpha_k\,\mathbf{W}_v\tanh(\mathbf{t}_k).
$$

Công thức trên là cho một đầu attention; bài báo dùng attention nhiều đầu (mỗi đầu có bộ trọng số $\alpha$ riêng), các đầu được nối rồi chiếu bởi $\mathbf{W}_o$. Véc-tơ $\mathbf{p}$ được phát rộng theo thời gian và cộng (hoặc nối) vào đầu ra bộ mã hóa văn bản. Toàn bộ được học **không giám sát**, chỉ từ mất mát tái tạo của TTS.

Vì sao cấu trúc này có ích? Vì $\mathbf{p}$ bị ép nằm trong bao lồi của $K$ véc-tơ (sau phép chiếu), một nút thắt mạnh hơn và có cấu trúc hơn véc-tơ tham chiếu tự do. Trong thực nghiệm của bài báo, các token tự phân hóa thành những "trục" có thể diễn giải được như tốc độ đọc, mức cao độ, độ sôi nổi; với dữ liệu nhiễu, một số token còn tự học tách nhiễu nền hay đặc điểm người nói. Đó là lý do GST trở thành công cụ quen thuộc để điều khiển phong cách.

### 29.4.2. Ba chế độ suy luận

1. **Theo tham chiếu**: đưa mel tham chiếu qua bộ mã hóa và attention như khi huấn luyện.
2. **Chọn token thủ công**: đặt trọng số bằng tay (ví dụ dồn vào một token, hoặc tăng trọng số để "phóng đại" phong cách), bỏ qua bộ mã hóa tham chiếu. Đây là cách điều khiển không cần âm thanh.
3. **Dự đoán từ văn bản** (Text-Predicted GST; Stanton và cộng sự, 2018): huấn luyện thêm một mạng nhỏ dự đoán trọng số token (hoặc trực tiếp $\mathbf{p}$) từ đầu ra bộ mã hóa văn bản, để mô hình tự chọn phong cách hợp với nội dung câu.

Mã dưới đây cài đặt bộ mã hóa tham chiếu (có xử lý độ dài biến thiên bằng `pack_padded_sequence`) và GST với hai chế độ: theo tham chiếu và theo trọng số cho trước. Phép kiểm tra trung tâm là: đưa đúng trọng số attention đã tính vào chế độ thủ công phải cho lại đúng véc-tơ phong cách, tức là $\mathbf{p}$ thật sự là tổ hợp của các token.

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class ReferenceEncoder(nn.Module):
    """Skerry-Ryan và cộng sự (2018): 6 lớp Conv2d 3x3 stride 2 + BN + ReLU, rồi GRU."""
    def __init__(self, n_mels=80, channels=(32, 32, 64, 64, 128, 128), gru_dim=128):
        super().__init__()
        layers, in_ch = [], 1
        for ch in channels:
            layers += [nn.Conv2d(in_ch, ch, 3, stride=2, padding=1), nn.BatchNorm2d(ch), nn.ReLU()]
            in_ch = ch
        self.convs = nn.Sequential(*layers)
        self.n_down = len(channels)
        f = n_mels
        for _ in channels:
            f = (f + 1) // 2                                  # mỗi lớp stride 2 chia đôi (làm tròn lên)
        self.gru = nn.GRU(channels[-1] * f, gru_dim, batch_first=True)

    def forward(self, mel, lengths):                          # mel: (B, T, F), lengths: (B,)
        x = self.convs(mel.unsqueeze(1))                      # (B, C', T', F')
        B, C, T, F = x.shape
        x = x.transpose(1, 2).reshape(B, T, C * F)            # (B, T', C'*F')
        t_lens = lengths.clone()
        for _ in range(self.n_down):
            t_lens = (t_lens + 1) // 2                        # độ dài sau mỗi lần giảm mẫu
        packed = pack_padded_sequence(x, t_lens.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)                               # h: (1, B, gru_dim)
        return h.squeeze(0)                                   # (B, gru_dim) "embedding ngôn điệu"

class GST(nn.Module):
    """Global Style Tokens: attention đa đầu từ embedding tham chiếu tới K token học được."""
    def __init__(self, ref_dim=128, n_tokens=10, style_dim=256, n_heads=4):
        super().__init__()
        self.tokens = nn.Parameter(0.3 * torch.randn(n_tokens, style_dim))   # (K, D)
        self.query = nn.Linear(ref_dim, style_dim)
        self.attn = nn.MultiheadAttention(style_dim, n_heads, batch_first=True)

    def keys(self, B):
        return torch.tanh(self.tokens).unsqueeze(0).expand(B, -1, -1)         # (B, K, D)

    def forward(self, ref_emb):                               # ref_emb: (B, ref_dim)
        q = self.query(ref_emb).unsqueeze(1)                  # (B, 1, D)
        k = self.keys(ref_emb.size(0))
        style, w = self.attn(q, k, k, average_attn_weights=False)   # (B, 1, D), (B, H, 1, K)
        return style.squeeze(1), w.squeeze(2)                 # (B, D), (B, H, K)

    def from_weights(self, w):
        """Chế độ điều khiển tay: w (B, K) dùng chung cho mọi đầu, hoặc (B, H, K)."""
        H, D = self.attn.num_heads, self.attn.embed_dim
        if w.dim() == 2:
            w = w.unsqueeze(1).expand(-1, H, -1)
        k = self.keys(w.size(0))
        W_v, b_v = self.attn.in_proj_weight[2 * D:], self.attn.in_proj_bias[2 * D:]
        v = (k @ W_v.T + b_v).view(w.size(0), -1, H, D // H)      # (B, K, H, D/H)
        mixed = torch.einsum("bhk,bkhd->bhd", w, v).reshape(w.size(0), D)
        return self.attn.out_proj(mixed)                      # (B, D)

torch.manual_seed(0)
B, T, F = 3, 200, 80
mel = torch.randn(B, T, F)
lengths = torch.tensor([200, 150, 64])
ref_enc, gst = ReferenceEncoder(F).eval(), GST().eval()
with torch.no_grad():
    style, w = gst(ref_enc(mel, lengths))
    assert style.shape == (B, 256) and w.shape == (B, 4, 10)
    assert torch.allclose(w.sum(-1), torch.ones(B, 4), atol=1e-5)          # trọng số là phân phối
    assert torch.allclose(gst.from_weights(w), style, atol=1e-5)           # style = tổ hợp lồi của token
    token3 = gst.from_weights(torch.eye(10)[3:4])                          # "giọng" của riêng token 3
    enc_out = torch.randn(B, 37, 256)                                      # đầu ra bộ mã hóa văn bản (B, N, C)
    cond = enc_out + style.unsqueeze(1)                                    # phát rộng (broadcast) theo N
print("style:", tuple(style.shape), "| token 3:", tuple(token3.shape), "| cond:", tuple(cond.shape))
```

Hai chi tiết thực hành: (1) khi batch có câu dài ngắn khác nhau, hãy đặt các khung đệm của mel về giá trị nền (giá trị log-mel của im lặng) trước khi qua tích chập, vì các lớp tích chập ở rìa vẫn "nhìn thấy" vùng đệm dù GRU đã được đóng gói theo độ dài; (2) trong chế độ thủ công, trọng số không bắt buộc phải là phân phối xác suất. Nhân trọng số của một token lên lớn hơn 1 thường "phóng đại" phong cách tương ứng, nhưng quá tay sẽ đưa $\mathbf{p}$ ra ngoài vùng mô hình đã thấy và gây lỗi âm thanh.

## 29.5. Mô hình hóa phong cách bằng VAE

Bộ mã hóa tham chiếu và GST học $\mathbf{p}$ một cách tất định, nên không có cách *có nguyên tắc* để lấy mẫu $\mathbf{p}$ khi không có tham chiếu. Thay bộ mã hóa tham chiếu bằng bộ mã hóa của một **VAE** ([Chương 8](../phan-2-mo-hinh-sinh/ch08-vae.md)) giải quyết đúng điểm này: $q_\phi(\mathbf{p} \mid \mathbf{Y})$ là Gauss, prior $p(\mathbf{p}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$, và mất mát thêm số hạng KL:

$$
\mathcal{L} = \mathbb{E}_{q_\phi(\mathbf{p} \mid \mathbf{Y})}\big[-\log p_\theta(\mathbf{Y} \mid \mathbf{x}, \mathbf{p})\big] + \beta\, D_{\mathrm{KL}}\big(q_\phi(\mathbf{p} \mid \mathbf{Y}) \,\Vert\, \mathcal{N}(\mathbf{0}, \mathbf{I})\big).
$$

Số hạng KL đóng hai vai: nó là nút thắt có thể điều chỉnh (qua $\beta$, chống rò rỉ nội dung), và nó làm không gian $\mathbf{p}$ "liền mạch" để ta có thể lấy mẫu, nội suy giữa hai phong cách, hoặc đi dọc một chiều để điều khiển. Zhang và cộng sự (2019) áp dụng cách này cho Tacotron 2 (thường gọi là VAE-Tacotron).

Hsu và cộng sự (2019) đề xuất mô hình **phân cấp** (GMVAE-Tacotron): một biến rời rạc chọn "cụm" (thành phần của hỗn hợp Gauss) và một biến liên tục bên trong cụm, đồng thời tách hai nhóm biến ẩn: một nhóm cho thuộc tính có nhãn (ví dụ người nói) và một nhóm cho thuộc tính không nhãn (phong cách, mức nhiễu nền, điều kiện thu). Prior hỗn hợp cho phép các cụm phong cách tách biệt và có thể diễn giải, ví dụ một cụm ứng với các câu thu trong môi trường ồn.

Vấn đề kinh điển là **sụp đổ hậu nghiệm**: bộ giải mã TTS (nhất là loại tự hồi quy) đủ mạnh để bỏ qua $\mathbf{p}$, KL về 0 và $\mathbf{p}$ vô dụng. Các biện pháp từ [Chương 8](../phan-2-mo-hinh-sinh/ch08-vae.md) áp dụng trực tiếp: tăng dần $\beta$ (*KL annealing*), "free bits" (không phạt KL dưới một ngưỡng mỗi chiều), và giữ chiều của $\mathbf{p}$ nhỏ.

## 29.6. Ngôn điệu chi tiết theo âm vị và âm tiết

Một véc-tơ toàn câu không mô tả được *chỗ nào* được nhấn hay *chỗ nào* kéo dài. Các mô hình ngôn điệu chi tiết (*fine-grained*) dùng **một biến ẩn cho mỗi âm vị** (hoặc mỗi từ, mỗi âm tiết): bộ mã hóa tham chiếu tính một chuỗi véc-tơ, rồi dùng căn chỉnh để gộp các khung về từng đơn vị (Lee và Kim, 2019). Sun và cộng sự (2020a) xây cấu trúc **phân cấp** (biến ẩn mức từ điều kiện hóa biến ẩn mức âm vị) để các biến có thể diễn giải được.

Biến ẩn chi tiết tái tạo rất tốt nhưng đặt ra câu hỏi khó hơn lúc suy luận: phải sinh cả một **chuỗi** $\mathbf{p}_1, \ldots, \mathbf{p}_N$ phù hợp với văn bản. Lời giải tiêu biểu (Sun và cộng sự, 2020b): **lượng tử hóa** biến ẩn (một dạng VQ-VAE) rồi học một **prior tự hồi quy** trên chuỗi chỉ số rời rạc, có điều kiện theo văn bản. Lấy mẫu từ prior này cho các cách đọc đa dạng mà vẫn tự nhiên. Có thể thấy đây là tổ tiên trực tiếp của ý tưởng "mô hình ngôn ngữ trên token âm thanh" ở [Chương 32](../phan-8-mo-hinh-lon/ch32-codec-language-model.md).

!!! note "Góc tiếng Việt: âm tiết là đơn vị ngôn điệu tự nhiên"
    Tiếng Việt là ngôn ngữ đơn lập, có nhịp theo âm tiết, và mỗi âm tiết mang một thanh. Đặt biến ẩn ngôn điệu ở **mức âm tiết** (thay vì âm vị hay khung) vừa khớp cấu trúc ngôn ngữ, vừa giảm độ dài chuỗi mà prior phải mô hình hóa (mỗi âm tiết thường có 2–4 âm vị). Bộ phân tích âm tiết ở [Chương 15](../phan-3-tin-hieu-ngu-am/ch15-ngu-am-tieng-viet.md) cho biên giới âm tiết trực tiếp từ chữ viết, nên việc gộp theo âm tiết không cần bộ tách từ.

## 29.7. Dự đoán ngôn điệu từ văn bản bằng mô hình ngôn ngữ tiền huấn luyện

Nhiều quyết định ngôn điệu (ngắt ở đâu, nhấn từ nào, câu hỏi hay câu kể) phụ thuộc **cú pháp và ngữ nghĩa**, những thứ mà bộ mã hóa văn bản của TTS, chỉ học từ vài chục giờ dữ liệu có ghi âm, học rất kém. Mô hình ngôn ngữ tiền huấn luyện trên hàng tỉ từ văn bản (BERT và các họ sau) thì đã có sẵn tri thức đó. Ý tưởng chung (Hayashi và cộng sự, 2019; Kenter và cộng sự, 2020): chạy mô hình ngôn ngữ trên văn bản, lấy biểu diễn ngữ cảnh, **căn chỉnh** về đơn vị đầu vào của TTS rồi ghép (hoặc cộng) vào bộ mã hóa. PnG BERT (Jia và cộng sự, 2021) đi xa hơn: tiền huấn luyện một BERT nhận đồng thời âm vị và chữ viết, rồi tinh chỉnh nó làm bộ mã hóa của TTS.

Khó khăn kỹ thuật chính là **căn chỉnh đơn vị**: mô hình ngôn ngữ dùng token con (*subword*, BPE), TTS dùng âm vị hoặc âm tiết. Quy trình chuẩn: gộp các subword về từ (trung bình hoặc lấy subword đầu), rồi lặp đặc trưng của từ cho mọi âm vị/âm tiết thuộc từ đó, kèm vị trí trong từ.

!!! note "Góc tiếng Việt: từ, âm tiết và mô hình ngôn ngữ"
    Một từ tiếng Việt có thể gồm nhiều âm tiết viết rời ("học sinh", "Trần Đại Nghĩa"). Nhiều mô hình ngôn ngữ cho tiếng Việt, tiêu biểu là PhoBERT (Nguyen và Nguyen, 2020), theo hướng dẫn của tác giả cần đầu vào **đã tách từ** (các âm tiết của một từ nối bằng dấu gạch dưới); khi đó mỗi token tương ứng với một từ hoặc một phần của từ, không phải một âm tiết. Bạn cần chạy cùng bộ tách từ mà mô hình được huấn luyện, và nhất quán quy trình này giữa huấn luyện và suy luận. Các mô hình đa ngôn ngữ dùng BPE trên văn bản thô thì không cần tách từ, nhưng token có thể cắt ngang âm tiết. Trong cả hai trường hợp, hãy đưa đặc trưng về **mức âm tiết** như mã dưới đây.

```python
import torch

def subword_to_word(sub_feats, word_ids):
    """Gộp trung bình các subword thuộc cùng một từ. sub_feats: (P, C); word_ids: (P,) -> (W, C)."""
    W = int(word_ids.max()) + 1
    sums = torch.zeros(W, sub_feats.size(1)).index_add_(0, word_ids, sub_feats)
    cnt = torch.zeros(W).index_add_(0, word_ids, torch.ones(len(word_ids)))
    return sums / cnt[:, None]

def word_to_syllable(word_feats, words):
    """Lặp đặc trưng của từ cho mọi âm tiết của nó. words: từ đã tách, âm tiết nối bằng '_'.
    Trả về (S, C) và vị trí của âm tiết trong từ (0 = âm tiết đầu)."""
    counts = torch.tensor([len(w.split("_")) for w in words])
    syl_feats = torch.repeat_interleave(word_feats, counts, dim=0)       # (S, C)
    pos_in_word = torch.cat([torch.arange(c) for c in counts])          # (S,)
    return syl_feats, pos_in_word

words = ["Học_sinh", "trường", "Trần_Đại_Nghĩa", "đạt", "giải", "nhất"]  # đầu ra bộ tách từ
# Giả lập đầu ra BPE của một mô hình ngôn ngữ: từ dài có thể bị cắt thành nhiều subword
word_ids = torch.tensor([0, 1, 2, 2, 3, 4, 5])                           # subword -> từ
torch.manual_seed(0)
sub_feats = torch.randn(len(word_ids), 768)                              # (P, C) đặc trưng ngữ cảnh
word_feats = subword_to_word(sub_feats, word_ids)                        # (W, C)
assert torch.allclose(word_feats[2], sub_feats[2:4].mean(0))
syl_feats, pos = word_to_syllable(word_feats, words)                     # (S, C)
n_syl = sum(len(w.split("_")) for w in words)
assert syl_feats.shape == (n_syl, 768) and n_syl == 9
print("vị trí âm tiết trong từ:", pos.tolist())
# Ghép với embedding âm tiết/âm vị của bộ mã hóa TTS (sau khi chiếu xuống) trước khi mã hóa
proj = torch.nn.Linear(768, 64)
tts_in = torch.cat([torch.randn(n_syl, 192), proj(syl_feats)], dim=-1)   # (S, 256)
assert tts_in.shape == (9, 256)
```

Thông tin "vị trí trong từ" rất hữu ích cho tiếng Việt: các âm tiết trong cùng một từ thường liền mạch hơn (ít khả năng có ngắt giữa chúng) so với ranh giới giữa hai từ.

## 29.8. StyleTTS và StyleTTS 2

**StyleTTS** (Li và cộng sự, 2022) đặt phong cách vào trung tâm kiến trúc: một bộ mã hóa phong cách biến mel tham chiếu thành véc-tơ phong cách, véc-tơ này điều kiện hóa bộ giải mã qua **AdaIN** ([Chương 28](ch28-da-nguoi-noi-voice-cloning.md)) và điều kiện hóa bộ dự đoán trường độ, cao độ, năng lượng. Mô hình dùng một bộ căn chỉnh đơn điệu có thể chuyển giao và một kỹ thuật tăng cường dữ liệu giúp phong cách không phụ thuộc trường độ. Hạn chế chính: vẫn cần âm thanh tham chiếu để có phong cách.

**StyleTTS 2** (Li và cộng sự, 2023) giải đúng hạn chế đó và kết hợp vài ý tưởng mạnh:

- **Khuếch tán phong cách** (*style diffusion*): coi véc-tơ phong cách là một biến ngẫu nhiên và học một mô hình khuếch tán nhỏ sinh véc-tơ này **từ văn bản**. Vì chỉ cần khuếch tán trên một véc-tơ ngắn chứ không trên cả phổ, bước lấy mẫu rẻ hơn nhiều so với TTS khuếch tán trên mel, mà vẫn có được sự đa dạng của mô hình khuếch tán ([Chương 11](../phan-2-mo-hinh-sinh/ch11-diffusion.md)). Đây chính là cách (c) ở Mục 29.1.2, làm bằng một mô hình sinh.
- **Huấn luyện đối kháng với mô hình ngôn ngữ tiếng nói lớn** (*SLM*): dùng một mô hình tự giám sát đã huấn luyện trước như WavLM làm nền cho bộ phân biệt, để "người chấm" có hiểu biết sâu về tiếng nói tự nhiên.
- **Mô hình hóa trường độ khả vi**, cho phép huấn luyện đầu–cuối cùng bộ phân biệt SLM.

Theo báo cáo gốc, StyleTTS 2 được người nghe bản ngữ đánh giá vượt ghi âm thật trên LJSpeech (một người nói) và ngang ghi âm trên VCTK (nhiều người nói). Cần đọc kết quả này một cách thận trọng: LJSpeech có ghi âm không hoàn hảo, và "vượt ghi âm" theo CMOS không có nghĩa là tốt hơn con người trong mọi tình huống. Dù vậy, StyleTTS 2 là một kiến trúc nhẹ, chất lượng cao, và một số mô hình nhỏ phổ biến về sau được xây trên nền kiến trúc này ([Chương 34](../phan-8-mo-hinh-lon/ch34-he-thong-lai-hien-dai.md)).

## 29.9. TTS cảm xúc

### 29.9.1. Biểu diễn cảm xúc

Hai cách mô tả chính:

- **Phân loại**: một tập nhãn rời rạc (trung tính, vui, buồn, giận, ngạc nhiên…). Đơn giản, dễ gán nhãn, nhưng thô.
- **Chiều liên tục**: ví dụ *valence* (tích cực/tiêu cực), *arousal* (mức kích hoạt), *dominance*. Mịn hơn nhưng khó gán nhãn nhất quán.

Về mặt mô hình, cảm xúc thường được đưa vào như một điều kiện toàn cục $\mathbf{g}_{\text{emo}}$ (embedding nhãn, hoặc véc-tơ từ bộ mã hóa cảm xúc huấn luyện trên dữ liệu nhận dạng cảm xúc), dùng đúng các cơ chế điều kiện hóa của [Chương 28](ch28-da-nguoi-noi-voice-cloning.md). Bộ dữ liệu thường được dùng là ESD (Zhou và cộng sự, 2021): theo mô tả của bộ dữ liệu, mỗi người nói (tiếng Anh và tiếng Quan Thoại) đọc cùng một tập câu ở năm trạng thái cảm xúc, nên có dữ liệu **song song**, rất có giá trị để tách cảm xúc khỏi nội dung.

### 29.9.2. Điều khiển cường độ cảm xúc

Người dùng thường muốn "vui một chút" chứ không chỉ "vui". Các cách phổ biến:

- **Nội suy embedding**: $\mathbf{g} = (1 - \lambda)\,\mathbf{g}_{\text{trung tính}} + \lambda\,\mathbf{g}_{\text{vui}}$, $\lambda \in [0, 1]$. Đơn giản và thường hiệu quả nếu không gian embedding đủ trơn (VAE giúp điều này).
- **Thuộc tính tương đối** (*relative attributes*; Zhu và cộng sự, 2019): học một hàm xếp hạng cho điểm "mức độ cảm xúc" của mỗi câu từ các cặp so sánh (câu cảm xúc mạnh hơn câu trung tính), rồi dùng điểm này làm điều kiện liên tục khi huấn luyện TTS. Lúc suy luận, đặt điểm mong muốn.
- **Phóng đại trọng số GST** (Mục 29.4.2), nếu một token tình cờ học được trục cảm xúc.

### 29.9.3. Những khó khăn thực tế

Cảm xúc **dính** vào người nói (mỗi người biểu đạt khác nhau) và vào nội dung (câu "Tôi rất vui!" đọc buồn nghe kỳ lạ). Dữ liệu cảm xúc diễn xuất thường cường điệu hơn đời thực. Đánh giá cũng khó: ngoài MOS độ tự nhiên, cần thêm độ chính xác cảm xúc theo người nghe (chọn nhãn cho câu tổng hợp) hoặc theo một bộ phân loại cảm xúc, với các hạn chế tương tự SECS ở [Chương 28](ch28-da-nguoi-noi-voice-cloning.md).

## 29.10. Điều khiển bằng mô tả ngôn ngữ tự nhiên

Thay vì nhãn rời rạc hay âm thanh tham chiếu, người dùng viết một câu mô tả: "giọng nữ trầm, nói chậm, hơi buồn". Hướng này phát triển nhanh từ 2023; các chi tiết dưới đây dựa trên các bài báo gốc, nên kiểm tra lại khi áp dụng:

- **PromptTTS** (Guo và cộng sự, 2023): một bộ mã hóa phong cách dựa trên BERT biến câu mô tả (*style prompt*) thành véc-tơ phong cách, điều kiện hóa một mô hình TTS. Các mô tả bao phủ những thuộc tính như giới tính, cao độ, tốc độ, âm lượng và cảm xúc; nhóm tác giả xây bộ dữ liệu có cặp (giọng nói, mô tả) cho mục đích này.
- **InstructTTS** (Yang và cộng sự, 2023): mô hình hóa tiếng nói trong một không gian ẩn rời rạc, học biểu diễn của câu mô tả phong cách theo nhiều giai đoạn, và dùng kỹ thuật giảm thông tin tương hỗ để biểu diễn phong cách không mang theo nội dung.
- **Chú thích tổng hợp ở quy mô lớn** (Lyth và King, 2024): thay vì để người viết mô tả, họ tự động đo các thuộc tính (giới tính, cao độ, tốc độ, mức nhiễu, độ vang…) trên một tập khoảng 45 nghìn giờ rồi dùng mô hình ngôn ngữ lớn viết thành câu mô tả tự nhiên, cuối cùng huấn luyện một mô hình ngôn ngữ tiếng nói có điều kiện theo mô tả. **Parler-TTS** của Hugging Face là một dự án mã nguồn mở tái hiện hướng tiếp cận này; tính đến thời điểm biên soạn (2026), chi tiết kiến trúc và dữ liệu của các phiên bản cần được kiểm tra trực tiếp trên trang dự án.

Điểm mạnh của hướng này là giao diện tự nhiên và **không cần giọng thật** của ai (giảm rủi ro nhân bản trái phép). Điểm yếu: chất lượng điều khiển phụ thuộc vào độ phủ và độ chính xác của mô tả trong dữ liệu; các thuộc tính hiếm hoặc khó đo tự động (mỉa mai, ân cần) vẫn khó điều khiển. Với tiếng Việt, gần như chưa có dữ liệu mô tả phong cách công khai; hướng chú thích tổng hợp (đo thuộc tính tự động rồi sinh mô tả) là cách khả thi nhất để xây dữ liệu như vậy.

## 29.11. Phong cách đọc tiếng Việt

Trong ứng dụng thực tế, "phong cách" thường không phải cảm xúc mà là **thể loại đọc**. Bảng dưới đây tổng hợp các đặc điểm định tính mà các nhóm làm TTS tiếng Việt thường mô tả; đây là quan sát chung để định hướng thu dữ liệu và đánh giá, không phải số đo.

| Phong cách | Tốc độ | Cao độ, biên độ | Ngắt nghỉ | Đặc điểm cần giữ |
|---|---|---|---|---|
| Tin tức | Nhanh vừa, đều | Biên độ vừa, hạ dần rõ cuối câu | Theo cụm cú pháp, ngắn | Rõ số liệu, tên riêng; trung tính |
| Sách nói | Chậm hơn tin tức | Biên độ vừa, ổn định qua nhiều giờ | Dài ở ranh giới đoạn | Phân biệt lời dẫn và lời thoại; không mệt tai |
| Kể chuyện (thiếu nhi) | Thay đổi nhiều | Biên độ rộng, nhiều nhấn | Ngắt kịch tính | Biểu cảm nhưng không méo thanh |
| Chăm sóc khách hàng | Vừa phải, rõ | Hơi cao, ấm | Ngắt sau mỗi ý | Tiểu từ lịch sự ("ạ", "nhé") tự nhiên |

Ba khuyến nghị khi xây hệ thống đa phong cách cho tiếng Việt:

1. **Thu dữ liệu song song theo phong cách** khi có thể: cùng một người đọc, cùng (một phần) kịch bản, ở nhiều phong cách. Dữ liệu song song giúp mô hình tách phong cách khỏi nội dung và người nói, giống vai trò của ESD.
2. **Dùng nhãn phong cách mức câu** (embedding phong cách) cộng với GST hoặc VAE để nắm biến thiên trong cùng phong cách. Nhãn rõ ràng giúp điều khiển ổn định; biến ẩn giúp tránh đơn điệu.
3. **Kiểm tra thanh điệu ở mọi phong cách.** Phong cách biểu cảm mạnh (kể chuyện) là nơi lỗi thanh xuất hiện nhiều nhất, vì mô hình kéo giãn biên độ cao độ. Dùng bộ câu cặp tối thiểu về thanh ([Chương 38](../phan-9-thuc-tien/ch38-danh-gia-tts.md)) cho từng phong cách.

!!! note "Góc tiếng Việt: tiểu từ cuối câu mang ngữ dụng"
    Trong tiếng Việt, rất nhiều sắc thái mà tiếng Anh diễn đạt bằng ngữ điệu (hỏi lại, xác nhận, mềm hóa, thân mật) được gánh bởi **tiểu từ tình thái** cuối câu: "à", "ạ", "nhé", "nhỉ", "chứ", "hả", "nha", "đấy" ([Chương 16](../phan-3-tin-hieu-ngu-am/ch16-ngon-dieu.md)). Hệ quả cho TTS: (1) tiểu từ thường được đọc với trường độ, cường độ và mức cao độ đặc thù, và mô hình chỉ học được nếu dữ liệu có đủ câu hội thoại chứa chúng (dữ liệu đọc tin tức gần như không có); (2) ngữ điệu câu hỏi tác động lên âm tiết cuối nhưng **không được** biến thanh của tiểu từ thành thanh khác (ví dụ "à" vẫn phải nghe là thanh huyền); (3) với hệ thống chăm sóc khách hàng, nên thu riêng một lượng câu có tiểu từ lịch sự đọc theo đúng phong cách mong muốn.

## Lỗi thường gặp và mẹo thực hành

- **Rò rỉ nội dung qua bộ mã hóa tham chiếu.** Triệu chứng: tổng hợp với tham chiếu khác văn bản thì nghe lẫn từ của tham chiếu, hoặc đọc sai. Kiểm tra: dùng tham chiếu là câu khác hẳn nội dung và đo CER. Cách sửa: thu hẹp nút thắt (chiều nhỏ hơn, GST ít token, tăng $\beta$ của KL), dùng tham chiếu là một câu *khác* của cùng người và cùng phong cách khi huấn luyện, thay vì chính câu đích.
- **Biến phong cách bị bỏ qua.** Ngược lại với rò rỉ: mọi trọng số GST gần như đều, hoặc KL của VAE về 0. Kiểm tra bằng cách tổng hợp cùng câu với các phong cách rất khác nhau. Cách sửa: KL annealing, free bits, bỏ bớt thông tin "dễ" khỏi các nhánh khác (ví dụ không đưa F0 thật vào bộ giải mã khi muốn biến ẩn học cao độ).
- **Phong cách làm méo thanh.** Biên độ cao độ được phóng đại làm thanh sắc nghe như hỏi, thanh nặng bị mất độ ngắt. Giới hạn biên độ ở mức câu và kiểm tra cặp tối thiểu (Mục 29.2.2).
- **Lẫn người nói và phong cách.** Nếu mỗi phong cách chỉ do một người đọc, mô hình không thể tách hai yếu tố; chuyển "phong cách kể chuyện" sang người khác sẽ chuyển luôn giọng. Thu ít nhất một phần dữ liệu đa phong cách từ cùng người nói.
- **Nhãn cảm xúc không nhất quán.** Gán nhãn cảm xúc bởi nhiều người cho độ đồng thuận thấp; nên dùng nhãn từ kịch bản (yêu cầu người đọc diễn theo nhãn) hoặc lấy nhãn đa số và loại câu có độ đồng thuận thấp.
- **Đánh giá chỉ bằng MOS độ tự nhiên.** Một hệ thống phong cách có thể rất tự nhiên mà không theo đúng phong cách yêu cầu. Thêm phép thử phân loại phong cách bởi người nghe (chọn phong cách cho từng câu tổng hợp) và đo độ tương quan giữa mức điều khiển và đại lượng âm học tương ứng (tốc độ âm tiết, biên độ F0).
- **Chuẩn hóa F0 sai miền.** Chuẩn hóa F0 theo Hz rồi điều khiển tuyến tính sẽ cho kết quả khác nhau giữa giọng nam và nữ. Luôn chuẩn hóa trong miền log (hoặc nửa cung) và theo từng người nói.

## Tóm tắt

- Ngôn điệu và phong cách là phần thông tin không có trong văn bản; điều kiện hóa thêm biến ngôn điệu $\mathbf{p}$ loại bỏ phần phương sai $\operatorname{Var}(\mathbb{E}[\mathbf{Y} \mid \mathbf{x}, \mathbf{p}])$ gây quá trơn, nhưng đòi hỏi một cách chọn $\mathbf{p}$ lúc suy luận.
- Điều khiển tường minh (trường độ, cao độ, năng lượng) nên làm trong miền log-F0; với tiếng Việt, chỉ sửa thành phần ngữ điệu mức câu, giữ thành phần thanh điệu mức âm tiết.
- Bộ mã hóa tham chiếu nén mel thành véc-tơ ngôn điệu; GST biểu diễn phong cách như tổ hợp lồi của $K$ token học được, cho phép điều khiển bằng tham chiếu, bằng trọng số tay, hoặc dự đoán từ văn bản.
- VAE cho phép lấy mẫu và nội suy phong cách; cần chống sụp đổ hậu nghiệm. Biến ẩn chi tiết (âm vị/âm tiết) cần một prior tự hồi quy hoặc mô hình dự đoán khi suy luận.
- Đặc trưng từ mô hình ngôn ngữ tiền huấn luyện mang tri thức cú pháp–ngữ nghĩa cho ngắt nghỉ và nhấn; với tiếng Việt cần xử lý tách từ và đưa đặc trưng về mức âm tiết.
- StyleTTS 2 sinh véc-tơ phong cách từ văn bản bằng khuếch tán và huấn luyện đối kháng với bộ phân biệt dựa trên mô hình tiếng nói tự giám sát.
- TTS cảm xúc dùng nhãn hoặc chiều liên tục, điều khiển cường độ bằng nội suy hoặc thuộc tính tương đối; điều khiển bằng mô tả ngôn ngữ tự nhiên đang phát triển nhanh nhờ chú thích tổng hợp.
- Với tiếng Việt: phong cách đọc (tin tức, sách nói, kể chuyện, chăm sóc khách hàng), tiểu từ cuối câu và việc bảo toàn thanh điệu là ba mối quan tâm trung tâm.

## Bài tập

**Bài 1 [Cơ bản].** Giải thích bằng lời và bằng định luật phương sai toàn phần: vì sao một mô hình FastSpeech 2 *không* có bộ điều chỉnh phương sai sinh mel mờ hơn mô hình có bộ điều chỉnh phương sai dùng F0 thật khi huấn luyện? Vì sao lợi thế đó giảm đi phần nào lúc suy luận?

??? tip "Gợi ý"
    Đặt $\mathbf{p}$ là (trường độ, F0, năng lượng). Khi huấn luyện, $\mathbf{p}$ lấy từ âm thanh thật; khi suy luận, $\mathbf{p}$ do bộ dự đoán hồi quy sinh ra, và chính bộ dự đoán này lại gặp bài toán một–nhiều.

**Bài 2 [Cơ bản].** Một giọng nữ có F0 trung bình 220 Hz và một giọng nam 110 Hz cùng đọc thanh sắc với độ tăng 4 nửa cung. Tính độ tăng tính theo Hz của mỗi giọng. Từ đó giải thích vì sao điều khiển cao độ phải làm trong miền log.

??? success "Lời giải"
    Tăng 4 nửa cung là nhân $2^{4/12} \approx 1{,}26$. Giọng nữ: $220 \to \approx 277{,}2$ Hz, tăng khoảng 57 Hz; giọng nam: $110 \to \approx 138{,}6$ Hz, tăng khoảng 29 Hz. Cùng một thanh nhưng độ tăng theo Hz khác gấp đôi; một quy tắc "cộng $x$ Hz" sẽ làm thanh của một trong hai giọng bị sai hình dạng. Trong miền log, cả hai là $+4/12$ quãng tám.

**Bài 3 [Nâng cao].** Chứng minh rằng véc-tơ phong cách GST với một đầu attention nằm trong ảnh qua phép affine $\mathbf{W}_o(\cdot)$ của bao lồi các véc-tơ $\{\mathbf{W}_v\tanh(\mathbf{t}_k)\}$. Với $H$ đầu, tập các véc-tơ phong cách có thể đạt được lớn hơn hay nhỏ hơn? Ý nghĩa với việc điều khiển bằng tay?

??? tip "Gợi ý"
    Với $H$ đầu, mỗi đầu có một tổ hợp lồi riêng trên phần véc-tơ con của nó; tập đạt được là tích Descartes của $H$ bao lồi (qua phép chiếu ra), lớn hơn bao lồi chung. Chế độ "một bộ trọng số cho mọi đầu" trong mã chỉ là một tập con.

**Bài 4 [Nâng cao].** Trong mô hình $\ell = \ell_{\text{câu}} + \ell_{\text{thanh}}$, phép ước lượng $\ell_{\text{câu}}$ bằng đường thẳng qua trung bình các âm tiết bị thiên lệch khi câu có nhiều thanh thấp (huyền, nặng) dồn ở cuối. Giải thích vì sao, và đề xuất một cách ước lượng ít thiên lệch hơn khi biết nhãn thanh của từng âm tiết.

??? success "Lời giải"
    Trung bình âm tiết bao gồm cả mức trung bình của thanh; nếu thanh thấp tập trung cuối câu, đường thẳng khớp được sẽ dốc hơn độ hạ dần thật, "nuốt" một phần thông tin thanh vào thành phần câu. Khi biết nhãn thanh, có thể khớp mô hình hồi quy $\bar{\ell}_j = a + b\,c_j + \delta_{\text{thanh}(j)}$, với $c_j$ là vị trí âm tiết $j$ và $\delta$ là độ lệch mức riêng của từng thanh (ràng buộc tổng bằng 0), rồi lấy $a + b\,c$ làm thành phần câu.

**Bài 5 [Nâng cao].** Thiết kế một thí nghiệm định lượng để đo mức rò rỉ nội dung của một bộ mã hóa tham chiếu, dùng một mô hình ASR và một tập câu tham chiếu có nội dung khác văn bản cần đọc. Nêu chỉ số, nhóm đối chứng và cách diễn giải.

**Bài 6 [Lập trình].** Mở rộng lớp `GST` với phương thức `text_predicted(enc_out, mask)`: một mạng nhỏ (GRU hoặc trung bình có mặt nạ + MLP) dự đoán trọng số token từ đầu ra bộ mã hóa văn bản, huấn luyện bằng entropy chéo với trọng số attention (đã tách gradient) từ chế độ tham chiếu làm mục tiêu mềm. Viết kiểm thử hình dạng và kiểm tra rằng mất mát giảm trên một batch giả cố định.

**Bài 7 [Lập trình].** Viết hàm `syllable_pitch_coeffs(f0, syl_bounds, order=2)` tính, cho mỗi âm tiết, các hệ số khai triển đa thức Legendre bậc 0..`order` của log-F0 (bỏ qua khung vô thanh). Kiểm tra trên dữ liệu phác họa ở Mục 29.2.2 rằng hệ số bậc 1 phân biệt được thanh sắc và ngang, và rằng dịch toàn câu $k$ nửa cung chỉ thay đổi hệ số bậc 0.

## Tài liệu tham khảo

1. Skerry-Ryan, R. J. và cộng sự (2018). *Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron*. ICML 2018.
2. Wang, Y. và cộng sự (2018). *Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis*. ICML 2018. arXiv:1803.09017.
3. Stanton, D., Wang, Y., và Skerry-Ryan, R. J. (2018). *Predicting Expressive Speaking Style from Text in End-to-End Speech Synthesis*. IEEE SLT 2018.
4. Zhang, Y.-J., Pan, S., He, L., và Ling, Z.-H. (2019). *Learning Latent Representations for Style Control and Transfer in End-to-End Speech Synthesis*. ICASSP 2019.
5. Hsu, W.-N. và cộng sự (2019). *Hierarchical Generative Modeling for Controllable Speech Synthesis*. ICLR 2019.
6. Lee, Y. và Kim, T. (2019). *Robust and Fine-Grained Prosody Control of End-to-End Speech Synthesis*. ICASSP 2019.
7. Sun, G. và cộng sự (2020a). *Fully-Hierarchical Fine-Grained Prosody Modeling for Interpretable Speech Synthesis*. ICASSP 2020.
8. Sun, G. và cộng sự (2020b). *Generating Diverse and Natural Text-to-Speech Samples Using a Quantized Fine-Grained VAE and Autoregressive Prosody Prior*. ICASSP 2020.
9. Łańcucki, A. (2021). *FastPitch: Parallel Text-to-Speech with Pitch Prediction*. ICASSP 2021. arXiv:2006.06873.
10. Ren, Y. và cộng sự (2021). *FastSpeech 2: Fast and High-Quality End-to-End Text to Speech*. ICLR 2021.
11. Hayashi, T. và cộng sự (2019). *Pre-Trained Text Embeddings for Enhanced Text-to-Speech Synthesis*. Interspeech 2019.
12. Kenter, T., Sharma, M., và Clark, R. (2020). *Improving the Prosody of RNN-Based English Text-To-Speech Synthesis by Incorporating a BERT Model*. Interspeech 2020.
13. Jia, Y., Zen, H., Shen, J., Zhang, Y., và Wu, Y. (2021). *PnG BERT: Augmented BERT on Phonemes and Graphemes for Neural TTS*. Interspeech 2021.
14. Nguyen, D. Q. và Nguyen, A. T. (2020). *PhoBERT: Pre-trained Language Models for Vietnamese*. Findings of EMNLP 2020.
15. Li, Y. A., Han, C., và Mesgarani, N. (2022). *StyleTTS: A Style-Based Generative Model for Natural and Diverse Text-to-Speech Synthesis*. arXiv:2205.15439.
16. Li, Y. A., Han, C., Raghavan, V. S., Mischler, G., và Mesgarani, N. (2023). *StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models*. NeurIPS 2023. arXiv:2306.07691.
17. Zhou, K., Sisman, B., Liu, R., và Li, H. (2021). *Seen and Unseen Emotional Style Transfer for Voice Conversion with a New Emotional Speech Dataset*. ICASSP 2021.
18. Zhu, X., Yang, S., Yang, G., và Xie, L. (2019). *Controlling Emotion Strength with Relative Attribute for End-to-End Speech Synthesis*. IEEE ASRU 2019.
19. Guo, Z., Leng, Y., Wu, Y., Zhao, S., và Tan, X. (2023). *PromptTTS: Controllable Text-to-Speech with Text Descriptions*. ICASSP 2023. arXiv:2211.12171.
20. Yang, D. và cộng sự (2023). *InstructTTS: Modelling Expressive TTS in Discrete Latent Space with Natural Language Style Prompt*. arXiv preprint.
21. Lyth, D. và King, S. (2024). *Natural Language Guidance of High-Fidelity Text-to-Speech with Synthetic Annotations*. arXiv:2402.01912.
