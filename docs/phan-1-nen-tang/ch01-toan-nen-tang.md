# Chương 1. Toán nền tảng cho học sâu

Một hệ thống tổng hợp tiếng nói hiện đại, dù là Tacotron 2, VITS hay một mô hình ngôn ngữ trên token âm thanh, rốt cuộc chỉ làm ba việc: **biểu diễn** văn bản và âm thanh bằng các mảng số, **biến đổi** các mảng số đó qua những hàm có tham số, và **điều chỉnh** tham số để đầu ra giống tiếng nói thật. Ba việc ấy tương ứng với ba mảng toán học: đại số tuyến tính, giải tích và xác suất thống kê. Thêm vào đó, lý thuyết thông tin cho ta ngôn ngữ để đo "hai phân phối khác nhau bao nhiêu", còn số phức là cửa ngõ dẫn tới phân tích Fourier, nền móng của mọi đặc trưng phổ trong xử lý tiếng nói.

Chương này không nhằm thay thế một giáo trình toán. Mục đích của nó là trang bị **đúng những công cụ** mà các chương sau sẽ dùng, trình bày theo cách người làm học sâu thực sự dùng chúng, và với mỗi công cụ, chỉ ra nó xuất hiện ở đâu trong TTS. Nếu bạn đã quen một phần nào đó, hãy lướt nhanh nhưng đừng bỏ qua các mục "liên hệ với TTS" và các đoạn mã: chúng là sợi chỉ nối toàn bộ cuốn sách.

!!! abstract "Mục tiêu học tập"
    Sau chương này, bạn có thể:

    - **Giải thích** vì sao bộ lọc mel là một phép chiếu tuyến tính làm mất thông tin, dựa trên khái niệm hạng và không gian rỗng.
    - **Tính** gradient và Jacobian của các biểu thức véc-tơ/ma trận thường gặp, và **kiểm chứng** chúng bằng `torch.autograd`.
    - **Sử dụng** SVD để xấp xỉ hạng thấp và ước lượng chuẩn phổ bằng phép lặp lũy thừa.
    - **Suy ra** hàm mất mát L2 từ phân phối Gauss và L1 từ phân phối Laplace, đồng thời **giải thích** vì sao chúng lần lượt dự đoán trung bình và trung vị.
    - **Áp dụng** công thức đổi biến cho hàm mật độ và ước lượng Monte Carlo.
    - **So sánh** entropy, entropy chéo, phân kỳ KL thuận và nghịch, thông tin tương hỗ, và **chỉ ra** vai trò của chúng trong các mô hình TTS.
    - **Biểu diễn** một tín hiệu hình sin bằng số phức và **cài đặt** DFT như một phép nhân ma trận.

!!! info "Kiến thức cần có"
    - Toán phổ thông và giải tích một biến (đạo hàm, tích phân cơ bản), làm quen với ký hiệu tổng $\sum$.
    - Python cơ bản; biết dùng `numpy` ở mức tạo mảng và nhân ma trận là đủ.
    - Đọc lướt trang [Ký hiệu toán học](../gioi-thieu/ky-hieu.md): chương này dùng đúng các ký hiệu đó.

## 1.1. Bức tranh tổng thể: toán học trong một hệ TTS

Trước khi đi vào chi tiết, hãy nhìn một hệ TTS hai giai đoạn kinh điển qua "lăng kính toán học". Sơ đồ dưới đây ghi rõ mỗi bước dùng công cụ nào của chương này.

```mermaid
flowchart LR
    A["Văn bản tiếng Việt"] --> B["Chuỗi token x (rời rạc)"]
    B --> C["Mạng nơ-ron: nhân ma trận + phi tuyến"]
    C --> D["Mel dự đoán"]
    E["Dạng sóng s(n)"] --> F["DFT/STFT: số phức"]
    F --> G["Phổ công suất"]
    G --> H["Bộ lọc mel: chiếu tuyến tính"]
    H --> I["log-mel Y (mục tiêu)"]
    D --> J["Hàm mất mát: xác suất"]
    I --> J
    J --> K["Gradient: giải tích ma trận"]
    K --> C
```

- **Nhánh dưới** biến dạng sóng thành log-mel-spectrogram. Nó dùng số phức (Mục 1.6), phép chiếu tuyến tính (Mục 1.2) và phép lấy logarit (một phép đổi biến, Mục 1.4.5). Chi tiết xử lý tín hiệu nằm ở [Chương 14](../phan-3-tin-hieu-ngu-am/ch14-xu-ly-tin-hieu-so.md).
- **Nhánh trên** là mô hình âm học: một chuỗi phép nhân ma trận xen kẽ hàm phi tuyến ([Chương 2](ch02-hoc-may-va-mang-no-ron.md), [Chương 4](ch04-cnn-rnn-mo-hinh-chuoi.md), [Chương 5](ch05-transformer.md)).
- **Hàm mất mát** so sánh dự đoán với mục tiêu. Việc chọn L1, L2 hay entropy chéo không phải ngẫu nhiên: mỗi lựa chọn tương ứng với một giả định xác suất (Mục 1.4.7 và 1.5).
- **Gradient** của hàm mất mát theo tham số được tính bằng quy tắc dây chuyền cho véc-tơ (Mục 1.3), là nền tảng của lan truyền ngược ([Chương 2](ch02-hoc-may-va-mang-no-ron.md)) và các thuật toán tối ưu ([Chương 3](ch03-toi-uu-va-ky-thuat-huan-luyen.md)).

## 1.2. Đại số tuyến tính

### 1.2.1. Véc-tơ, ma trận và tensor

**Trực giác.** Trong học sâu, mọi thứ đều là mảng số. Một khung mel-spectrogram với 80 kênh là một điểm trong không gian 80 chiều; cả một câu nói dài $T$ khung là một ma trận; một batch nhiều câu là một mảng ba chiều.

**Hình thức hóa.** Theo quy ước của sách, véc-tơ là **véc-tơ cột** viết đậm chữ thường: $\mathbf{x} \in \mathbb{R}^{d}$. Ma trận viết đậm chữ hoa: $\mathbf{W} \in \mathbb{R}^{m \times n}$ có $m$ hàng, $n$ cột. Một log-mel-spectrogram được ký hiệu

$$
\mathbf{Y} = (\mathbf{y}_1, \ldots, \mathbf{y}_T) \in \mathbb{R}^{T \times F},
$$

trong đó mỗi $\mathbf{y}_t \in \mathbb{R}^{F}$ là một khung (thường $F = 80$ hoặc $100$). Khi gom $B$ câu thành một batch, ta có một **tensor** hình dạng `(B, T, F)`. "Tensor" trong học sâu đơn giản là mảng nhiều chiều; ta không cần đến nghĩa hình học vi phân của từ này.

**Tích ma trận–véc-tơ** có hai cách đọc, và cả hai đều hữu ích:

$$
\mathbf{W}\mathbf{x} = \sum_{j=1}^{n} x_j \, \mathbf{w}_{:,j}
\qquad\text{và}\qquad
(\mathbf{W}\mathbf{x})_i = \mathbf{w}_{i,:}\,\mathbf{x} = \sum_{j=1}^{n} W_{ij} x_j .
$$

Cách đọc thứ nhất (**theo cột**) nói rằng $\mathbf{W}\mathbf{x}$ là một *tổ hợp tuyến tính* các cột của $\mathbf{W}$, với hệ số là các thành phần của $\mathbf{x}$. Cách đọc thứ hai (**theo hàng**) nói rằng mỗi đầu ra là *tích vô hướng* của một hàng với $\mathbf{x}$, tức là một "bộ dò" đo xem $\mathbf{x}$ giống hàng đó đến đâu. Một lớp tuyến tính trong mạng nơ-ron là một ngân hàng các bộ dò như vậy.

Hàm $f(\mathbf{x}) = \mathbf{W}\mathbf{x}$ là **tuyến tính**: $f(a\mathbf{x} + b\mathbf{x}') = a f(\mathbf{x}) + b f(\mathbf{x}')$. Thêm độ lệch $\mathbf{b}$ ta được hàm **affine** $\mathbf{W}\mathbf{x} + \mathbf{b}$. Tích hai ma trận $\mathbf{A}\mathbf{B}$ biểu diễn **hợp** của hai phép biến đổi tuyến tính: áp dụng $\mathbf{B}$ trước, rồi đến $\mathbf{A}$. Hệ quả quan trọng: chồng nhiều lớp tuyến tính mà không có hàm phi tuyến xen giữa thì vẫn chỉ là *một* phép biến đổi tuyến tính, vì $\mathbf{W}_2(\mathbf{W}_1\mathbf{x}) = (\mathbf{W}_2\mathbf{W}_1)\mathbf{x}$. Đây là lý do mạng nơ-ron cần hàm kích hoạt ([Chương 2](ch02-hoc-may-va-mang-no-ron.md)).

Ngoài tích ma trận thông thường, ta thường dùng **tích Hadamard** $\mathbf{a} \odot \mathbf{b}$ (nhân từng phần tử) và **phép ghép** $[\mathbf{a}; \mathbf{b}]$ (nối hai véc-tơ thành một véc-tơ dài hơn), chẳng hạn khi ghép embedding người nói vào đầu ra của bộ mã hóa văn bản ([Chương 28](../phan-7-e2e-bieu-cam/ch28-da-nguoi-noi-voice-cloning.md)).

!!! warning "Hàng hay cột? Quy ước trong PyTorch"
    Sách viết véc-tơ là cột và lớp tuyến tính là $\mathbf{W}\mathbf{x} + \mathbf{b}$. Nhưng `torch.nn.Linear` nhận dữ liệu dạng hàng, hình dạng `(B, d_in)`, và tính `x @ W.T + b` với `W` có hình dạng `(d_out, d_in)`. Hai cách viết tương đương nhau, chỉ khác chuyển vị. Khi đọc bài báo và mã nguồn song song, hãy luôn kiểm tra quy ước nào đang được dùng.

### 1.2.2. Tích vô hướng, chuẩn và độ tương đồng cosine

**Tích vô hướng** (*inner product*) của hai véc-tơ cùng chiều là $\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{x}^\top \mathbf{y} = \sum_i x_i y_i$. Về mặt hình học,

$$
\mathbf{x}^\top \mathbf{y} = \lVert \mathbf{x} \rVert_2 \, \lVert \mathbf{y} \rVert_2 \cos\alpha ,
$$

với $\alpha$ là góc giữa hai véc-tơ. Từ đó suy ra **bất đẳng thức Cauchy–Schwarz** $\lvert \mathbf{x}^\top \mathbf{y} \rvert \le \lVert \mathbf{x} \rVert_2 \lVert \mathbf{y} \rVert_2$, dấu bằng khi hai véc-tơ cùng phương. Ta sẽ dùng bất đẳng thức này ngay ở Mục 1.3 để chứng minh gradient là hướng tăng nhanh nhất.

**Chuẩn** (*norm*) đo "độ lớn" của véc-tơ. Ba chuẩn hay gặp nhất:

$$
\lVert \mathbf{x} \rVert_1 = \sum_i \lvert x_i \rvert, \qquad
\lVert \mathbf{x} \rVert_2 = \Big(\sum_i x_i^2\Big)^{1/2}, \qquad
\lVert \mathbf{x} \rVert_\infty = \max_i \lvert x_i \rvert .
$$

Với ma trận, **chuẩn Frobenius** $\lVert \mathbf{A} \rVert_F = (\sum_{ij} A_{ij}^2)^{1/2}$ coi ma trận như một véc-tơ dài; **chuẩn phổ** $\lVert \mathbf{A} \rVert_2$ (Mục 1.2.6) đo mức khuếch đại lớn nhất mà ma trận gây ra. Khoảng cách $\lVert \mathbf{y} - \hat{\mathbf{y}} \rVert_1$ và $\lVert \mathbf{y} - \hat{\mathbf{y}} \rVert_2^2$ chính là hai hàm mất mát L1 và L2 dùng để so sánh mel dự đoán với mel thật; Mục 1.4.7 giải thích vì sao chúng hành xử khác nhau.

**Độ tương đồng cosine** $\cos\alpha = \mathbf{x}^\top\mathbf{y} / (\lVert \mathbf{x} \rVert_2 \lVert \mathbf{y} \rVert_2)$ chỉ quan tâm đến *hướng*, bỏ qua độ lớn. Trong TTS, đây là thước đo chuẩn để so sánh hai embedding người nói: độ giống giọng của câu tổng hợp với giọng tham chiếu thường được báo cáo bằng cosine giữa hai embedding do một mô hình xác minh người nói trích ra ([Chương 28](../phan-7-e2e-bieu-cam/ch28-da-nguoi-noi-voice-cloning.md), [Chương 38](../phan-9-thuc-tien/ch38-danh-gia-tts.md)).

### 1.2.3. Độc lập tuyến tính, hạng và không gian rỗng

Các véc-tơ $\mathbf{a}_1, \ldots, \mathbf{a}_n$ **độc lập tuyến tính** nếu không véc-tơ nào là tổ hợp tuyến tính của các véc-tơ còn lại. Tập mọi tổ hợp tuyến tính của chúng là **không gian sinh** (*span*). Với ma trận $\mathbf{A} \in \mathbb{R}^{m \times n}$:

- **Không gian cột** $\operatorname{col}(\mathbf{A})$ là span của các cột, tức tập mọi đầu ra $\mathbf{A}\mathbf{x}$ có thể có.
- **Hạng** (*rank*) $\operatorname{rank}(\mathbf{A})$ là số chiều của không gian cột, bằng số cột độc lập tuyến tính tối đa, và cũng bằng số hàng độc lập tối đa. Luôn có $\operatorname{rank}(\mathbf{A}) \le \min(m, n)$.
- **Không gian rỗng** (*null space*, còn gọi là hạt nhân) $\operatorname{null}(\mathbf{A}) = \{\mathbf{x} : \mathbf{A}\mathbf{x} = \mathbf{0}\}$ là tập các đầu vào bị ma trận "xóa sổ".

Hai đại lượng cuối liên hệ qua **định lý hạng–số khuyết** (*rank–nullity*):

$$
\dim \operatorname{null}(\mathbf{A}) + \operatorname{rank}(\mathbf{A}) = n .
$$

Hệ quả thực tế rất đáng nhớ: nếu $\mathbf{A}\mathbf{x} = \mathbf{A}\mathbf{x}'$ thì $\mathbf{x} - \mathbf{x}' \in \operatorname{null}(\mathbf{A})$. Khi không gian rỗng khác $\{\mathbf{0}\}$, **vô số đầu vào khác nhau cho cùng một đầu ra**, và không có cách nào khôi phục chính xác đầu vào từ đầu ra.

**Liên hệ với TTS: bộ lọc mel.** Với `n_fft = 1024`, phổ công suất của một khung có $K = 1024/2 + 1 = 513$ bin tần số. Bộ lọc mel gồm $F = 80$ bộ lọc tam giác, mỗi bộ lọc là một hàng của ma trận $\mathbf{M} \in \mathbb{R}^{80 \times 513}$, và khung mel là $\mathbf{m} = \mathbf{M}\mathbf{p}$ với $\mathbf{p}$ là phổ công suất. Vì $\operatorname{rank}(\mathbf{M}) \le 80$, không gian rỗng có ít nhất $513 - 80 = 433$ chiều. Nói cách khác, mel-spectrogram vứt bỏ rất nhiều chi tiết tần số, nhất là cấu trúc tinh ở dải tần cao, nơi mỗi bộ lọc mel gộp hàng chục bin lại (chưa kể toàn bộ thông tin pha, xem Mục 1.6). Đây là lý do một vocoder ([Phần VI](../phan-6-vocoder/ch24-vocoder-tu-hoi-quy-va-flow.md)) không thể là phép "nghịch đảo" đơn thuần: nó phải *sinh* lại các chi tiết bị mất sao cho nghe tự nhiên.

### 1.2.4. Phép chiếu và bình phương tối thiểu

**Bài toán.** Cho $\mathbf{A} \in \mathbb{R}^{m \times n}$ và $\mathbf{b} \in \mathbb{R}^{m}$, tìm $\mathbf{x}$ sao cho $\mathbf{A}\mathbf{x}$ gần $\mathbf{b}$ nhất:

$$
\min_{\mathbf{x}} \; \lVert \mathbf{A}\mathbf{x} - \mathbf{b} \rVert_2^2 .
$$

**Suy ra nghiệm.** Khai triển $\lVert \mathbf{A}\mathbf{x} - \mathbf{b} \rVert_2^2 = \mathbf{x}^\top \mathbf{A}^\top \mathbf{A} \mathbf{x} - 2\mathbf{b}^\top \mathbf{A}\mathbf{x} + \mathbf{b}^\top\mathbf{b}$. Dùng các đồng nhất thức ở Mục 1.3.4, gradient theo $\mathbf{x}$ là $2\mathbf{A}^\top\mathbf{A}\mathbf{x} - 2\mathbf{A}^\top\mathbf{b}$. Cho bằng không, ta được **phương trình chuẩn** (*normal equations*):

$$
\mathbf{A}^\top\mathbf{A}\,\mathbf{x}^\star = \mathbf{A}^\top \mathbf{b} .
$$

Nếu các cột của $\mathbf{A}$ độc lập tuyến tính thì $\mathbf{A}^\top\mathbf{A}$ khả nghịch và $\mathbf{x}^\star = (\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top\mathbf{b}$. Khi đó $\hat{\mathbf{b}} = \mathbf{A}\mathbf{x}^\star = \mathbf{P}\mathbf{b}$ với

$$
\mathbf{P} = \mathbf{A}(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top ,
$$

là **ma trận chiếu trực giao** lên không gian cột. Bạn có thể kiểm tra $\mathbf{P}^2 = \mathbf{P}$ (chiếu hai lần như chiếu một lần) và $\mathbf{P}^\top = \mathbf{P}$. Ý nghĩa hình học: phần dư $\mathbf{b} - \hat{\mathbf{b}}$ vuông góc với mọi cột của $\mathbf{A}$, đúng như phương trình chuẩn nói: $\mathbf{A}^\top(\mathbf{b} - \mathbf{A}\mathbf{x}^\star) = \mathbf{0}$. Hồi quy tuyến tính ở [Chương 2](ch02-hoc-may-va-mang-no-ron.md) chính là bài toán này.

Khi $\mathbf{A}^\top\mathbf{A}$ không khả nghịch (ví dụ có nhiều ẩn hơn phương trình), bài toán có vô số nghiệm. **Giả nghịch đảo Moore–Penrose** $\mathbf{A}^{+}$ (tính qua SVD, Mục 1.2.6) chọn ra nghiệm có chuẩn nhỏ nhất. Với ma trận "béo" có hạng hàng đầy đủ như $\mathbf{M}$ của bộ lọc mel, $\mathbf{M}^{+} = \mathbf{M}^\top(\mathbf{M}\mathbf{M}^\top)^{-1}$, và $\mathbf{M}^{+}\mathbf{m}$ chính là **hình chiếu của phổ thật lên không gian hàng** của $\mathbf{M}$: mọi thành phần nằm trong không gian rỗng đều mất.

### 1.2.5. Trị riêng và véc-tơ riêng

**Trực giác.** Phần lớn véc-tơ bị ma trận vừa xoay vừa co giãn. Một số hướng đặc biệt chỉ bị co giãn mà không đổi phương: đó là các **véc-tơ riêng**.

**Định nghĩa.** Với ma trận vuông $\mathbf{A}$, nếu $\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$ với $\mathbf{v} \ne \mathbf{0}$ thì $\lambda$ là **trị riêng** và $\mathbf{v}$ là **véc-tơ riêng** tương ứng.

Trường hợp quan trọng nhất trong học máy là **ma trận đối xứng thực** ($\mathbf{A} = \mathbf{A}^\top$). **Định lý phổ** khẳng định mọi ma trận như vậy đều phân rã được thành

$$
\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top = \sum_{i=1}^{n} \lambda_i \mathbf{q}_i \mathbf{q}_i^\top ,
$$

với $\mathbf{Q}$ trực giao ($\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$, các cột $\mathbf{q}_i$ là véc-tơ riêng trực chuẩn) và $\boldsymbol{\Lambda} = \operatorname{diag}(\lambda_1, \ldots, \lambda_n)$ gồm các trị riêng thực. Ma trận đối xứng là **nửa xác định dương** (*positive semi-definite*, PSD) nếu $\mathbf{x}^\top\mathbf{A}\mathbf{x} \ge 0$ với mọi $\mathbf{x}$, tương đương mọi $\lambda_i \ge 0$; **xác định dương** nếu bất đẳng thức là chặt.

Ba nơi trị riêng xuất hiện liên tục trong sách:

1. **Ma trận hiệp phương sai** luôn PSD (Mục 1.4.3). Các véc-tơ riêng ứng với những trị riêng lớn nhất của nó chỉ ra các hướng mà dữ liệu biến thiên nhiều nhất: đây là phân tích thành phần chính (PCA).
2. **Ma trận Hessian** (Mục 1.3.2) đo độ cong của hàm mất mát. Tỷ số giữa trị riêng lớn nhất và nhỏ nhất, gọi là **số điều kiện** (*condition number*) $\kappa = \lambda_{\max}/\lambda_{\min}$, quyết định gradient descent hội tụ nhanh hay chậm và tốc độ học tối đa được phép ([Chương 3](ch03-toi-uu-va-ky-thuat-huan-luyen.md)).
3. **Tích các Jacobian** qua nhiều lớp hoặc nhiều bước thời gian: nếu các trị riêng (chính xác hơn là giá trị kỳ dị) lớn hơn 1 thì gradient bùng nổ, nhỏ hơn 1 thì gradient tiêu biến ([Chương 4](ch04-cnn-rnn-mo-hinh-chuoi.md)).

### 1.2.6. Phân rã giá trị kỳ dị (SVD)

Phân rã riêng chỉ áp dụng cho ma trận vuông (và đẹp nhất với ma trận đối xứng). **Phân rã giá trị kỳ dị** (*singular value decomposition*, SVD, còn gọi là phân tích giá trị suy biến) áp dụng cho **mọi** ma trận $\mathbf{A} \in \mathbb{R}^{m \times n}$:

$$
\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top = \sum_{i=1}^{r} \sigma_i \, \mathbf{u}_i \mathbf{v}_i^\top ,
$$

với $\mathbf{U} \in \mathbb{R}^{m \times m}$ và $\mathbf{V} \in \mathbb{R}^{n \times n}$ trực giao, $\boldsymbol{\Sigma}$ là ma trận "đường chéo" chứa các **giá trị kỳ dị** $\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r > 0$, và $r = \operatorname{rank}(\mathbf{A})$.

**Ý nghĩa hình học.** Mọi phép biến đổi tuyến tính đều là: *xoay* (bởi $\mathbf{V}^\top$), *co giãn dọc các trục* (bởi $\boldsymbol{\Sigma}$), rồi *xoay* tiếp (bởi $\mathbf{U}$). Một hình cầu đơn vị bị biến thành một ellipsoid với các bán trục dài $\sigma_i$.

**Liên hệ với phân rã riêng.** Vì $\mathbf{A}^\top\mathbf{A} = \mathbf{V}\boldsymbol{\Sigma}^\top\boldsymbol{\Sigma}\mathbf{V}^\top$, các $\sigma_i^2$ là trị riêng của $\mathbf{A}^\top\mathbf{A}$ và các cột của $\mathbf{V}$ là véc-tơ riêng tương ứng.

**Chuẩn phổ.** Giá trị kỳ dị lớn nhất bằng mức khuếch đại tối đa:

$$
\lVert \mathbf{A} \rVert_2 = \max_{\mathbf{x} \neq \mathbf{0}} \frac{\lVert \mathbf{A}\mathbf{x} \rVert_2}{\lVert \mathbf{x} \rVert_2} = \sigma_1 .
$$

Đây cũng là **hằng số Lipschitz** của ánh xạ $\mathbf{x} \mapsto \mathbf{A}\mathbf{x}$. Chuẩn hóa phổ (*spectral normalization*), một kỹ thuật ổn định bộ phân biệt trong GAN và vocoder GAN, chia ma trận trọng số cho $\sigma_1$ để khống chế hằng số này ([Chương 3](ch03-toi-uu-va-ky-thuat-huan-luyen.md), [Chương 9](../phan-2-mo-hinh-sinh/ch09-gan.md)). Trong thực tế $\sigma_1$ được ước lượng rẻ bằng **phép lặp lũy thừa** (*power iteration*): lặp $\mathbf{v} \leftarrow \mathbf{A}^\top\mathbf{A}\mathbf{v} / \lVert \mathbf{A}^\top\mathbf{A}\mathbf{v} \rVert$; sai số giảm theo cấp số nhân với tỷ số $(\sigma_2/\sigma_1)^2$ sau mỗi vòng.

**Xấp xỉ hạng thấp.** Giữ lại $k$ thành phần đầu, $\mathbf{A}_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i\mathbf{v}_i^\top$. **Định lý Eckart–Young** (Eckart và Young, 1936) khẳng định $\mathbf{A}_k$ là ma trận hạng $k$ gần $\mathbf{A}$ nhất theo cả chuẩn Frobenius lẫn chuẩn phổ, với sai số

$$
\lVert \mathbf{A} - \mathbf{A}_k \rVert_F = \Big(\sum_{i > k} \sigma_i^2\Big)^{1/2} .
$$

Lưu $\mathbf{A}_k$ chỉ cần $k(m + n)$ số thay vì $mn$. Ý tưởng "cập nhật trọng số có hạng thấp" là nền tảng của LoRA (Hu và cộng sự, 2022), kỹ thuật fine-tune tiết kiệm bộ nhớ mà ta sẽ gặp khi tinh chỉnh các mô hình TTS lớn ([Chương 34](../phan-8-mo-hinh-lon/ch34-he-thong-lai-hien-dai.md), [Chương 37](../phan-9-thuc-tien/ch37-huan-luyen-va-debug.md)).

Đoạn mã sau kiểm chứng định lý Eckart–Young và phép lặp lũy thừa trên một ma trận có phổ giảm dần:

```python
import numpy as np

rng = np.random.default_rng(0)
# Tạo ma trận 64x48 có phổ suy giảm nhanh (dựng có chủ đích để minh họa)
U, _ = np.linalg.qr(rng.standard_normal((64, 48)))       # cột trực chuẩn
V, _ = np.linalg.qr(rng.standard_normal((48, 48)))
sigma = 10.0 * 0.7 ** np.arange(48)                      # giá trị kỳ dị giảm dần
A = U @ np.diag(sigma) @ V.T                             # (64, 48)

U_, S_, Vt_ = np.linalg.svd(A, full_matrices=False)
assert np.allclose(S_, sigma)

for k in [1, 4, 8, 16]:
    A_k = U_[:, :k] @ np.diag(S_[:k]) @ Vt_[:k]          # xấp xỉ hạng k
    err = np.linalg.norm(A - A_k)                        # chuẩn Frobenius
    theory = np.sqrt(np.sum(S_[k:] ** 2))                # định lý Eckart–Young
    print(f"k={k:2d}  sai số={err:.4f}  lý thuyết={theory:.4f}  "
          f"tham số={k * (64 + 48)} / {64 * 48}")
    assert np.isclose(err, theory)

# Lặp lũy thừa (power iteration) ước lượng giá trị kỳ dị lớn nhất
v = rng.standard_normal(48)
for _ in range(50):
    u = A @ v
    u /= np.linalg.norm(u)
    v = A.T @ u
    v /= np.linalg.norm(v)
sigma_max = u @ A @ v
print(f"power iteration: {sigma_max:.6f} | np.linalg.norm(A, 2): {np.linalg.norm(A, 2):.6f}")
assert np.isclose(sigma_max, np.linalg.norm(A, 2), rtol=1e-6)
```

Với $k = 8$, ta chỉ cần 896 trên 3 072 tham số mà sai số Frobenius đã giảm từ khoảng 9,8 (khi $k = 1$) xuống khoảng 0,81. Ma trận trọng số thật không phải lúc nào cũng có phổ giảm nhanh như vậy, nhưng phần *cập nhật* trọng số khi fine-tune thường được giả định là có hạng thấp, và đó chính là giả thuyết mà LoRA khai thác.

### 1.2.7. Ví dụ: bộ lọc mel là một phép chiếu tuyến tính

Ta ghép các ý trên lại trong một ví dụ cụ thể. Đoạn mã dựng bộ lọc mel bằng `librosa`, kiểm tra hạng, chiếu một khung phổ công suất xuống 80 kênh mel, rồi thử khôi phục bằng giả nghịch đảo.

```python
import numpy as np
import librosa

sr, n_fft, n_mels = 22050, 1024, 80
M = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)  # (F, K) = (80, 513)
K = n_fft // 2 + 1
assert M.shape == (n_mels, K)
assert (M >= 0).all()                                        # bộ lọc không âm

# Hạng của ma trận chiếu: tối đa là 80 < 513 -> mất thông tin
rank = np.linalg.matrix_rank(M)
print("hạng của M:", rank, "| số chiều không gian rỗng:", K - rank)

# Một khung tín hiệu tổng hợp: ba thành phần điều hòa của F0 = 220 Hz
n = np.arange(n_fft)
s = sum(a * np.sin(2 * np.pi * f * n / sr) for a, f in [(1.0, 220), (0.5, 440), (0.25, 660)])
window = np.hanning(n_fft)
power = np.abs(np.fft.rfft(s * window)) ** 2               # (K,) phổ công suất
mel = M @ power                                            # (F,) chiếu tuyến tính
print("phổ công suất:", power.shape, "-> mel:", mel.shape)

# Thử "đảo ngược" bằng giả nghịch đảo: chỉ khôi phục được hình chiếu
power_hat = np.linalg.pinv(M) @ mel                        # (K,)
rel_err = np.linalg.norm(power_hat - power) / np.linalg.norm(power)
print(f"sai số tương đối khi khôi phục bằng pinv: {rel_err:.3f}")
# Nhưng chiếu lại thì khớp hoàn toàn: M @ power_hat == mel
assert np.allclose(M @ power_hat, mel, rtol=1e-3, atol=1e-6 * mel.max())
```

Kết quả: hạng của $\mathbf{M}$ là 80, không gian rỗng có 433 chiều, và sai số tương đối khi khôi phục bằng giả nghịch đảo vào khoảng 0,5: chuẩn $\ell_2$ của phần bị mất (thành phần nằm trong không gian rỗng) bằng khoảng một nửa chuẩn của phổ gốc. Điều thú vị là $\mathbf{M}\hat{\mathbf{p}}$ vẫn khớp hoàn toàn với khung mel: hai phổ rất khác nhau cho cùng một mel. Giả nghịch đảo còn có thể cho giá trị *âm* (trong ví dụ này, 59 trên 513 bin bị âm), vô nghĩa với phổ công suất; vì thế `librosa.feature.inverse.mel_to_stft` giải bài toán bình phương tối thiểu **không âm** (NNLS) thay cho giả nghịch đảo. Cách xây dựng bộ lọc mel, các công thức HTK/Slaney và những khác biệt mặc định giữa thư viện được trình bày ở [Chương 14](../phan-3-tin-hieu-ngu-am/ch14-xu-ly-tin-hieu-so.md).

!!! tip "Cùng một ma trận, hai cách sắp xếp"
    `librosa.filters.mel` trả về ma trận hình dạng `(n_mels, 1 + n_fft // 2)`, còn `torchaudio.functional.melscale_fbanks` trả về `(n_freqs, n_mels)`, tức là chuyển vị của nhau. Với mel-spectrogram bố cục `(B, T, K)`, phép chiếu viết là `spec @ M.T` (librosa) hoặc `spec @ fb` (torchaudio). Nhầm chiều ở đây thường không báo lỗi khi kích thước tình cờ tương thích, nhưng cho kết quả sai hoàn toàn.

## 1.3. Giải tích và giải tích ma trận

### 1.3.1. Đạo hàm, đạo hàm riêng và gradient

Đạo hàm của hàm một biến, $f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$, cho biết $f$ thay đổi nhanh đến đâu khi $x$ nhích một chút. Với hàm nhiều biến $f : \mathbb{R}^{d} \to \mathbb{R}$, **đạo hàm riêng** $\partial f / \partial x_i$ đo tốc độ thay đổi khi chỉ nhích $x_i$ và giữ nguyên các biến còn lại. Gom các đạo hàm riêng lại ta được **gradient**:

$$
\nabla_{\mathbf{x}} f(\mathbf{x}) = \left( \frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_d} \right)^{\!\top} \in \mathbb{R}^{d} .
$$

**Vì sao gradient là hướng tăng nhanh nhất?** Đạo hàm theo hướng của véc-tơ đơn vị $\mathbf{u}$ là $D_{\mathbf{u}} f = \nabla f^\top \mathbf{u}$. Theo Cauchy–Schwarz, $\nabla f^\top \mathbf{u} \le \lVert \nabla f \rVert_2 \lVert \mathbf{u} \rVert_2 = \lVert \nabla f \rVert_2$, dấu bằng xảy ra khi $\mathbf{u} = \nabla f / \lVert \nabla f \rVert_2$. Vậy gradient chỉ hướng làm $f$ tăng nhanh nhất, và $-\nabla f$ là hướng **giảm** nhanh nhất. Toàn bộ học sâu dựa trên quan sát giản dị này: muốn giảm hàm mất mát $\mathcal{L}(\theta)$, hãy bước một bước nhỏ ngược chiều $\nabla_\theta \mathcal{L}$.

### 1.3.2. Khai triển Taylor

Quanh một điểm $\mathbf{x}$, hàm trơn được xấp xỉ bởi

$$
f(\mathbf{x} + \boldsymbol{\Delta}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \boldsymbol{\Delta} + \tfrac{1}{2}\boldsymbol{\Delta}^\top \mathbf{H}(\mathbf{x})\,\boldsymbol{\Delta},
$$

với **ma trận Hessian** $H_{ij} = \partial^2 f / \partial x_i \partial x_j$ (đối xứng nếu $f$ đủ trơn). Xấp xỉ bậc một là "mặt phẳng tiếp xúc", bậc hai thêm vào độ cong.

**Hệ quả 1: gradient descent giảm hàm mất mát nếu bước đủ nhỏ.** Đặt $\boldsymbol{\Delta} = -\eta \nabla f$ thì

$$
f(\mathbf{x} - \eta\nabla f) \approx f(\mathbf{x}) - \eta \lVert \nabla f \rVert_2^2 + \tfrac{\eta^2}{2} \nabla f^\top \mathbf{H} \nabla f .
$$

Số hạng bậc một luôn âm. Số hạng bậc hai có thể dương và lấn át khi $\eta$ lớn. Nếu $\lambda_{\max}$ là trị riêng lớn nhất của $\mathbf{H}$ thì $\nabla f^\top \mathbf{H} \nabla f \le \lambda_{\max}\lVert \nabla f \rVert_2^2$, nên chọn $\eta < 2/\lambda_{\max}$ đảm bảo vế phải nhỏ hơn $f(\mathbf{x})$ (trong phạm vi xấp xỉ bậc hai). Đây là nguồn gốc của giới hạn tốc độ học mà [Chương 3](ch03-toi-uu-va-ky-thuat-huan-luyen.md) sẽ phân tích kỹ.

**Hệ quả 2: kiểm tra gradient bằng sai phân trung tâm.** Viết khai triển Taylor cho $f(x + h)$ và $f(x - h)$ rồi trừ vế theo vế, các số hạng bậc chẵn triệt tiêu:

$$
\frac{f(x + h) - f(x - h)}{2h} = f'(x) + \frac{h^2}{6} f'''(x) + O(h^4).
$$

Sai số của sai phân trung tâm là $O(h^2)$, tốt hơn hẳn sai phân tiến $\frac{f(x+h) - f(x)}{h}$ có sai số $O(h)$. [Chương 2](ch02-hoc-may-va-mang-no-ron.md) dùng đúng công thức này để kiểm tra một cài đặt lan truyền ngược viết tay.

### 1.3.3. Jacobian và quy tắc dây chuyền cho véc-tơ

Với hàm véc-tơ $\mathbf{g} : \mathbb{R}^{n} \to \mathbb{R}^{m}$, **ma trận Jacobian** gom mọi đạo hàm riêng:

$$
\mathbf{J}_{\mathbf{g}}(\mathbf{x}) = \frac{\partial \mathbf{g}}{\partial \mathbf{x}} \in \mathbb{R}^{m \times n}, \qquad [\mathbf{J}_{\mathbf{g}}]_{ji} = \frac{\partial g_j}{\partial x_i} .
$$

Hàng $j$ là gradient (chuyển vị) của đầu ra thứ $j$. Jacobian là "xấp xỉ tuyến tính tốt nhất" của $\mathbf{g}$ quanh $\mathbf{x}$: $\mathbf{g}(\mathbf{x} + \boldsymbol{\Delta}) \approx \mathbf{g}(\mathbf{x}) + \mathbf{J}_{\mathbf{g}}\boldsymbol{\Delta}$. Ví dụ: Jacobian của $\mathbf{W}\mathbf{x}$ theo $\mathbf{x}$ chính là $\mathbf{W}$; Jacobian của hàm áp dụng từng phần tử $\phi(\mathbf{x}) = (\phi(x_1), \ldots, \phi(x_n))$ là ma trận đường chéo $\operatorname{diag}(\phi'(x_1), \ldots, \phi'(x_n))$.

**Quy tắc dây chuyền.** Với một biến, $(f \circ g)'(x) = f'(g(x))\,g'(x)$. Với hàm hợp $\mathbf{h} = \mathbf{f} \circ \mathbf{g}$, $\mathbf{x} \in \mathbb{R}^{n} \to \mathbf{y} = \mathbf{g}(\mathbf{x}) \in \mathbb{R}^{m} \to \mathbf{z} = \mathbf{f}(\mathbf{y}) \in \mathbb{R}^{p}$, ta có

$$
\frac{\partial z_k}{\partial x_i} = \sum_{j=1}^{m} \frac{\partial z_k}{\partial y_j} \frac{\partial y_j}{\partial x_i}
\quad\Longleftrightarrow\quad
\mathbf{J}_{\mathbf{f} \circ \mathbf{g}}(\mathbf{x}) = \mathbf{J}_{\mathbf{f}}(\mathbf{y})\,\mathbf{J}_{\mathbf{g}}(\mathbf{x}) .
$$

Tổng theo $j$ cộng dồn mọi "con đường" ảnh hưởng từ $x_i$ tới $z_k$ thông qua các biến trung gian. Jacobian của hàm hợp là **tích các Jacobian**.

**Trường hợp đặc biệt cực kỳ quan trọng: đầu ra là một số.** Trong học sâu, đầu ra cuối cùng là hàm mất mát vô hướng $\mathcal{L}$. Khi đó Jacobian của $\mathcal{L}$ theo $\mathbf{y}$ là một hàng $\nabla_{\mathbf{y}}\mathcal{L}^\top$, và quy tắc dây chuyền trở thành

$$
\nabla_{\mathbf{x}} \mathcal{L} = \mathbf{J}_{\mathbf{g}}(\mathbf{x})^\top \, \nabla_{\mathbf{y}} \mathcal{L} .
$$

Đây là **tích véc-tơ–Jacobian** (*vector–Jacobian product*, VJP). Với một chuỗi hàm $\mathbf{x} \to \mathbf{h}_1 \to \cdots \to \mathbf{h}_L \to \mathcal{L}$, ta nhân lần lượt các Jacobian chuyển vị **từ cuối về đầu**, mỗi bước chỉ là một phép nhân ma trận–véc-tơ. Ta không bao giờ cần dựng tường minh ma trận Jacobian khổng lồ; ví dụ với hàm từng phần tử, VJP chỉ là $\phi'(\mathbf{x}) \odot \nabla_{\mathbf{y}}\mathcal{L}$. Ý tưởng này chính là **lan truyền ngược** (*backpropagation*), được suy ra chi tiết cho mạng nhiều lớp ở [Chương 2](ch02-hoc-may-va-mang-no-ron.md).

### 1.3.4. Quy ước bố cục và các đồng nhất thức hữu ích

Giải tích ma trận nổi tiếng gây rối vì có hai quy ước bố cục. Sách thống nhất như sau:

- **Gradient của một số theo một véc-tơ hoặc ma trận có cùng hình dạng với biến đó.** Ví dụ $\nabla_{\mathbf{W}}\mathcal{L}$ có cùng hình dạng với $\mathbf{W}$. Đây cũng là quy ước của PyTorch: `W.grad.shape == W.shape`.
- **Jacobian** của $\mathbf{g} : \mathbb{R}^{n} \to \mathbb{R}^{m}$ là ma trận $m \times n$ như ở Mục 1.3.3.

Các đồng nhất thức dùng nhiều nhất (chứng minh được bằng cách viết ra từng thành phần):

| Biểu thức | Gradient/Jacobian | Dùng ở đâu |
|---|---|---|
| $\mathbf{a}^\top\mathbf{x}$ | $\nabla_{\mathbf{x}} = \mathbf{a}$ | Lớp tuyến tính |
| $\mathbf{x}^\top\mathbf{A}\mathbf{x}$ | $\nabla_{\mathbf{x}} = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ | Dạng toàn phương, mật độ Gauss |
| $\lVert \mathbf{W}\mathbf{x} - \mathbf{y} \rVert_2^2$ | $\nabla_{\mathbf{W}} = 2(\mathbf{W}\mathbf{x} - \mathbf{y})\mathbf{x}^\top$ | Hồi quy, mất mát L2 |
| $\mathbf{p} = \operatorname{softmax}(\mathbf{z})$ | $\partial\mathbf{p}/\partial\mathbf{z} = \operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$ | Phân loại, attention |
| $\log \sum_j e^{z_j}$ | $\nabla_{\mathbf{z}} = \operatorname{softmax}(\mathbf{z})$ | Entropy chéo |
| $\log \lvert \det \mathbf{A} \rvert$ | $\nabla_{\mathbf{A}} = \mathbf{A}^{-\top}$ | Normalizing flow |

Ta suy ra hai dòng làm ví dụ.

*Dòng thứ ba.* Viết $\mathcal{L} = \sum_i r_i^2$ với phần dư $r_i = \sum_j W_{ij}x_j - y_i$. Khi đó $\partial \mathcal{L}/\partial W_{ij} = 2 r_i \, \partial r_i / \partial W_{ij} = 2 r_i x_j$. Ma trận có phần tử $(i, j)$ bằng $2r_i x_j$ chính là tích ngoài $2\mathbf{r}\mathbf{x}^\top$. Chú ý dạng "**tích ngoài giữa tín hiệu lỗi và đầu vào**": mọi lớp tuyến tính trong mạng nơ-ron đều có gradient trọng số mang dạng này.

*Dòng thứ tư.* Với $p_i = e^{z_i}/S$, $S = \sum_k e^{z_k}$, quy tắc đạo hàm thương cho

$$
\frac{\partial p_i}{\partial z_j} = \frac{\mathbb{1}[i = j]\, e^{z_i} S - e^{z_i} e^{z_j}}{S^2} = p_i\big(\mathbb{1}[i = j] - p_j\big).
$$

Viết dưới dạng ma trận ta được $\operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$. [Chương 2](ch02-hoc-may-va-mang-no-ron.md) sẽ dùng Jacobian này để suy ra gradient gọn gàng $\mathbf{p} - \mathbf{y}$ của hàm mất mát entropy chéo.

Dòng cuối (công thức Jacobi) sẽ quan trọng ở [Chương 10](../phan-2-mo-hinh-sinh/ch10-normalizing-flows.md): các normalizing flow như Glow tối ưu trực tiếp một số hạng $\log\lvert\det\mathbf{W}\rvert$.

Cách tốt nhất để tin một đồng nhất thức là kiểm chứng bằng vi phân tự động. PyTorch tính gradient chính xác (đến sai số dấu phẩy động) cho mọi biểu thức ta viết ra:

```python
import torch

torch.manual_seed(0)
d, m = 5, 3
x = torch.randn(d, dtype=torch.float64, requires_grad=True)
A = torch.randn(d, d, dtype=torch.float64)

# (1) grad_x (x^T A x) = (A + A^T) x
f = x @ A @ x
(g,) = torch.autograd.grad(f, x)
assert torch.allclose(g, (A + A.T) @ x.detach())

# (2) grad_W ||W x - y||^2 = 2 (W x - y) x^T
W = torch.randn(m, d, dtype=torch.float64, requires_grad=True)
y = torch.randn(m, dtype=torch.float64)
loss = ((W @ x.detach() - y) ** 2).sum()
(gW,) = torch.autograd.grad(loss, W)
r = W.detach() @ x.detach() - y
assert torch.allclose(gW, 2 * torch.outer(r, x.detach()))

# (3) Jacobian của softmax = diag(p) - p p^T
z = torch.randn(4, dtype=torch.float64)
J = torch.autograd.functional.jacobian(lambda t: torch.softmax(t, dim=0), z)
p = torch.softmax(z, dim=0)
assert torch.allclose(J, torch.diag(p) - torch.outer(p, p))

# (4) grad_A log|det A| = A^{-T}
A2 = (torch.randn(d, d, dtype=torch.float64) + 3 * torch.eye(d)).requires_grad_()
(gA,) = torch.autograd.grad(torch.linalg.slogdet(A2).logabsdet, A2)
assert torch.allclose(gA, torch.linalg.inv(A2.detach()).T)
print("Cả bốn đồng nhất thức đều khớp với autograd.")
```

Lưu ý ta dùng `float64` khi kiểm chứng: với `float32`, sai số làm tròn có thể làm `allclose` thất bại dù công thức đúng.

## 1.4. Xác suất và thống kê

Vì sao một cuốn sách về TTS cần nhiều xác suất đến vậy? Vì **cùng một câu văn có thể được đọc theo vô số cách**: nhanh hay chậm, ngắt nhịp ở đâu, nhấn vào từ nào, giọng vui hay buồn. Mô hình TTS tốt không học một ánh xạ "văn bản → âm thanh" duy nhất mà học một **phân phối có điều kiện** $p(\mathbf{Y} \mid \mathbf{x})$. Ngôn ngữ xác suất là thứ cho phép ta phát biểu, huấn luyện và đánh giá các mô hình như vậy.

### 1.4.1. Biến ngẫu nhiên và phân phối

Một **biến ngẫu nhiên** là một đại lượng có giá trị không chắc chắn. Với biến **rời rạc** (ví dụ chỉ số token âm thanh trong sổ mã), ta mô tả nó bằng **hàm khối xác suất** $p(x) = P(X = x)$ với $\sum_x p(x) = 1$. Với biến **liên tục** (ví dụ giá trị một kênh mel), ta dùng **hàm mật độ xác suất** $p(x) \ge 0$ với $\int p(x)\,dx = 1$, và

$$
P(a \le X \le b) = \int_a^b p(x)\,dx .
$$

Chú ý: mật độ **không phải** là xác suất và có thể lớn hơn 1 (ví dụ một Gauss rất hẹp). Chỉ tích phân của mật độ trên một khoảng mới là xác suất. Điều này giải thích vì sao log-likelihood của dữ liệu liên tục có thể dương, và hàm mất mát "negative log-likelihood" có thể âm mà không có gì sai.

Với hai biến, ta có **phân phối đồng thời** $p(x, y)$, **phân phối biên** $p(x) = \int p(x, y)\,dy$ (tổng thay cho tích phân nếu rời rạc), và **phân phối có điều kiện**

$$
p(y \mid x) = \frac{p(x, y)}{p(x)}, \qquad\text{tương đương}\qquad p(x, y) = p(y \mid x)\,p(x) .
$$

Đẳng thức thứ hai, **quy tắc nhân**, áp dụng lặp lại cho một chuỗi cho ta $p(y_1, \ldots, y_T) = \prod_{t} p(y_t \mid y_1, \ldots, y_{t-1})$. Đây là nền tảng của mọi mô hình **tự hồi quy**, từ WaveNet sinh từng mẫu âm thanh đến các codec language model sinh từng token ([Chương 7](../phan-2-mo-hinh-sinh/ch07-tong-quan-mo-hinh-sinh.md), [Chương 32](../phan-8-mo-hinh-lon/ch32-codec-language-model.md)). Hai biến **độc lập** nếu $p(x, y) = p(x)\,p(y)$, tức là biết $x$ không cho thêm thông tin gì về $y$.

### 1.4.2. Các phân phối thường gặp

| Phân phối | Tham số | Hàm khối/mật độ | Kỳ vọng, phương sai | Trong TTS |
|---|---|---|---|---|
| Bernoulli | $\pi \in [0, 1]$ | $\pi^{x}(1 - \pi)^{1 - x}$, $x \in \{0, 1\}$ | $\pi$; $\pi(1 - \pi)$ | Cờ dừng (*stop token*) của Tacotron 2, cờ hữu thanh/vô thanh |
| Categorical | $\boldsymbol{\pi}$ trên $K$ lớp | $\prod_k \pi_k^{\mathbb{1}[x = k]}$ | — | Token âm thanh rời rạc, mẫu μ-law 8 bit của WaveNet |
| Gauss $\mathcal{N}(\mu, \sigma^2)$ | $\mu$, $\sigma > 0$ | $\frac{1}{\sqrt{2\pi}\sigma}\exp\!\big(-\frac{(x - \mu)^2}{2\sigma^2}\big)$ | $\mu$; $\sigma^2$ | Nhiễu trong diffusion, tiên nghiệm của VAE, mất mát L2 |
| Laplace | $\mu$, $b > 0$ | $\frac{1}{2b}\exp\!\big(-\frac{\lvert x - \mu \rvert}{b}\big)$ | $\mu$; $2b^2$ | Mất mát L1 cho mel |
| Gauss nhiều chiều | $\boldsymbol{\mu}$, $\boldsymbol{\Sigma}$ | xem bên dưới | $\boldsymbol{\mu}$; $\boldsymbol{\Sigma}$ | Tiên nghiệm của Glow-TTS, VITS; nhiễu $\boldsymbol{\epsilon}$ |

Một vài nhận xét:

- **Gauss** là phân phối "mặc định" vì ba lý do: định lý giới hạn trung tâm (tổng nhiều ảnh hưởng nhỏ độc lập có xu hướng phân phối chuẩn), tính đóng dưới các phép biến đổi affine, và nó có entropy lớn nhất trong các phân phối có cùng phương sai (tức là giả định "ít nhất" ngoài trung bình và phương sai). Khoảng 68%, 95% và 99,7% khối xác suất nằm trong $\mu \pm \sigma$, $\mu \pm 2\sigma$, $\mu \pm 3\sigma$.
- **Laplace** có đuôi dày hơn Gauss: các giá trị lệch xa trung tâm ít "bị phạt" hơn (phạt tuyến tính thay vì bậc hai). Ta sẽ thấy hệ quả của điều này ở Mục 1.4.7.

**Gauss nhiều chiều.** Với $\mathbf{x} \in \mathbb{R}^{d}$,

$$
\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{d/2} \det(\boldsymbol{\Sigma})^{1/2}} \exp\!\Big(-\tfrac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})\Big),
$$

với **ma trận hiệp phương sai** $\boldsymbol{\Sigma}$ đối xứng xác định dương. Lấy logarit:

$$
\log \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) = -\tfrac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu}) - \tfrac{1}{2}\log\det\boldsymbol{\Sigma} - \tfrac{d}{2}\log 2\pi .
$$

Số hạng đầu là một dạng toàn phương (Mục 1.3.4) đo "khoảng cách Mahalanobis" từ $\mathbf{x}$ tới tâm. Ba trường hợp đặc biệt hay gặp: **đường chéo** $\boldsymbol{\Sigma} = \operatorname{diag}(\sigma_1^2, \ldots, \sigma_d^2)$, khi đó mật độ tách thành tích các Gauss một chiều; **đẳng hướng** $\boldsymbol{\Sigma} = \sigma^2\mathbf{I}$; và **chuẩn tắc** $\mathcal{N}(\mathbf{0}, \mathbf{I})$, phân phối của nhiễu $\boldsymbol{\epsilon}$ trong diffusion và flow matching.

Hai tính chất được dùng liên tục ở Phần II:

1. **Biến đổi affine của Gauss vẫn là Gauss**: nếu $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ thì $\mathbf{A}\mathbf{x} + \mathbf{b} \sim \mathcal{N}(\mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top)$.
2. **Lấy mẫu bằng tái tham số hóa**: với $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top$ (phân rã Cholesky), $\mathbf{x} = \boldsymbol{\mu} + \mathbf{L}\boldsymbol{\epsilon}$ với $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ có phân phối $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$. Đây là "mẹo tái tham số hóa" (*reparameterization trick*) của VAE ([Chương 8](../phan-2-mo-hinh-sinh/ch08-vae.md)).

### 1.4.3. Kỳ vọng, phương sai và hiệp phương sai

**Kỳ vọng** của $f(X)$ là trung bình có trọng số theo xác suất: $\mathbb{E}_{p}[f(X)] = \sum_x p(x) f(x)$ hoặc $\int p(x) f(x)\,dx$. Tính chất quan trọng nhất là **tuyến tính**: $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$, đúng **kể cả khi $X$, $Y$ không độc lập**.

**Phương sai** đo độ phân tán: $\operatorname{Var}(X) = \mathbb{E}[(X - \mathbb{E}X)^2] = \mathbb{E}[X^2] - (\mathbb{E}X)^2$, và $\operatorname{Var}(aX + b) = a^2\operatorname{Var}(X)$. **Hiệp phương sai** $\operatorname{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}X)(Y - \mathbb{E}Y)]$ đo mức hai biến biến thiên cùng nhau. Với véc-tơ ngẫu nhiên $\mathbf{x}$ có trung bình $\boldsymbol{\mu}$, **ma trận hiệp phương sai** là

$$
\boldsymbol{\Sigma} = \mathbb{E}\big[(\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^\top\big] .
$$

Nó luôn nửa xác định dương, vì với mọi $\mathbf{u}$, $\mathbf{u}^\top\boldsymbol{\Sigma}\mathbf{u} = \mathbb{E}[(\mathbf{u}^\top(\mathbf{x} - \boldsymbol{\mu}))^2] = \operatorname{Var}(\mathbf{u}^\top\mathbf{x}) \ge 0$.

Ba quy tắc sẽ được dùng ngay ở [Chương 2](ch02-hoc-may-va-mang-no-ron.md) khi phân tích cách khởi tạo trọng số:

- Nếu $X$, $Y$ độc lập thì $\operatorname{Var}(X + Y) = \operatorname{Var}(X) + \operatorname{Var}(Y)$. Tổng $n$ biến độc lập cùng phương sai $\sigma^2$ có phương sai $n\sigma^2$.
- Do đó trung bình mẫu $\bar{X}_n = \frac{1}{n}\sum_i X_i$ có phương sai $\sigma^2/n$: độ lệch chuẩn giảm theo $1/\sqrt{n}$.
- Nếu $X$, $Y$ độc lập và cùng có kỳ vọng bằng 0 thì $\operatorname{Var}(XY) = \operatorname{Var}(X)\operatorname{Var}(Y)$.

### 1.4.4. Quy tắc Bayes

Từ hai cách viết quy tắc nhân $p(x, z) = p(x \mid z)p(z) = p(z \mid x)p(x)$, ta có **quy tắc Bayes**:

$$
\underbrace{p(z \mid x)}_{\text{hậu nghiệm}} = \frac{\overbrace{p(x \mid z)}^{\text{hợp lý}}\;\overbrace{p(z)}^{\text{tiên nghiệm}}}{\underbrace{p(x)}_{\text{bằng chứng}}}, \qquad p(x) = \sum_{z} p(x \mid z)\,p(z) .
$$

Bayes cho ta cách **cập nhật niềm tin** về một đại lượng ẩn $z$ sau khi quan sát $x$.

!!! example "Ví dụ: phát hiện tiếng nói (VAD) trong quy trình làm dữ liệu"
    Khi cắt dữ liệu thu âm thành câu ([Chương 36](../phan-9-thuc-tien/ch36-du-lieu-tts.md)), ta cần quyết định mỗi khung là tiếng nói ($S$) hay im lặng. Giả sử (số liệu minh họa) 60% khung là tiếng nói, một bộ dò năng lượng báo "năng lượng cao" ($E$) cho 90% khung tiếng nói và 20% khung im lặng. Nếu bộ dò báo "cao", xác suất khung đó là tiếng nói là

    $$
    P(S \mid E) = \frac{0{,}9 \times 0{,}6}{0{,}9 \times 0{,}6 + 0{,}2 \times 0{,}4} = \frac{0{,}54}{0{,}62} \approx 0{,}87 .
    $$

    Nếu bộ dò báo "thấp", $P(S \mid \neg E) = \frac{0{,}1 \times 0{,}6}{0{,}1 \times 0{,}6 + 0{,}8 \times 0{,}4} = \frac{0{,}06}{0{,}38} \approx 0{,}16$. Dù bộ dò khá tốt, vẫn còn khoảng 16% khung "năng lượng thấp" thực ra là tiếng nói (ví dụ phụ âm xát vô thanh yếu), nên cắt bỏ im lặng quá mạnh tay có thể làm cụt âm cuối câu.

Trong các chương sau, Bayes xuất hiện dưới nhiều hình thức: hậu nghiệm của biến ẩn $p(\mathbf{z} \mid \mathbf{Y})$ trong VAE và VITS ([Chương 8](../phan-2-mo-hinh-sinh/ch08-vae.md), [Chương 27](../phan-7-e2e-bieu-cam/ch27-vits.md)), hay ước lượng hậu nghiệm cực đại (MAP) dẫn tới các số hạng điều chuẩn ([Chương 2](ch02-hoc-may-va-mang-no-ron.md)).

### 1.4.5. Đổi biến cho hàm mật độ

**Câu hỏi.** Biết $X$ có mật độ $p_X$, và $Y = g(X)$ với $g$ khả nghịch và khả vi. Mật độ của $Y$ là gì?

**Suy ra (một chiều).** Giả sử $g$ đồng biến. Dùng hàm phân phối tích lũy:

$$
F_Y(y) = P(g(X) \le y) = P\big(X \le g^{-1}(y)\big) = F_X\big(g^{-1}(y)\big).
$$

Lấy đạo hàm hai vế theo $y$ bằng quy tắc dây chuyền:

$$
p_Y(y) = p_X\big(g^{-1}(y)\big)\,\frac{d g^{-1}(y)}{dy} .
$$

Nếu $g$ nghịch biến, làm tương tự sẽ xuất hiện dấu trừ, nên công thức chung dùng trị tuyệt đối. Trực giác: xác suất được **bảo toàn**, $p_Y(y)\lvert dy \rvert = p_X(x)\lvert dx \rvert$. Chỗ nào $g$ "kéo giãn" trục số thì mật độ bị loãng ra, chỗ nào $g$ "nén" thì mật độ đặc lại.

**Nhiều chiều.** Với $\mathbf{y} = \mathbf{g}(\mathbf{x})$ khả nghịch trên $\mathbb{R}^{d}$, hệ số co giãn thể tích là trị tuyệt đối của định thức Jacobian:

$$
p_{\mathbf{Y}}(\mathbf{y}) = p_{\mathbf{X}}(\mathbf{x})\,\big\lvert \det \mathbf{J}_{\mathbf{g}}(\mathbf{x}) \big\rvert^{-1},
\qquad\text{hay}\qquad
\log p_{\mathbf{Y}}(\mathbf{y}) = \log p_{\mathbf{X}}(\mathbf{x}) - \log\big\lvert \det \mathbf{J}_{\mathbf{g}}(\mathbf{x}) \big\rvert ,
$$

với $\mathbf{x} = \mathbf{g}^{-1}(\mathbf{y})$. Công thức này là trái tim của **normalizing flow** ([Chương 10](../phan-2-mo-hinh-sinh/ch10-normalizing-flows.md)): ghép nhiều phép biến đổi khả nghịch có định thức Jacobian dễ tính để biến một Gauss đơn giản thành phân phối phức tạp của mel hay dạng sóng, rồi huấn luyện bằng log-likelihood chính xác. Glow-TTS, WaveGlow và phần flow trong VITS đều dựa trên nó ([Chương 23](../phan-5-mo-hinh-am-hoc/ch23-glow-grad-matcha.md), [Chương 24](../phan-6-vocoder/ch24-vocoder-tu-hoi-quy-va-flow.md), [Chương 27](../phan-7-e2e-bieu-cam/ch27-vits.md)).

**Ví dụ: logarit và phân phối log-chuẩn.** Đặt $X \sim \mathcal{N}(0, 1)$ và $Y = e^{X}$. Khi đó $g^{-1}(y) = \log y$ và $dg^{-1}/dy = 1/y$, nên $p_Y(y) = \mathcal{N}(\log y; 0, 1)/y$ với $y > 0$: một phân phối lệch phải, đuôi dài. Chiều ngược lại cũng là một bài học cho TTS: năng lượng phổ trên thang tuyến tính có dải động rất rộng và phân phối lệch mạnh, còn sau khi lấy logarit thì phân bố gọn hơn nhiều. Đó là một lý do (cùng với lý do cảm thụ, [Chương 13](../phan-3-tin-hieu-ngu-am/ch13-am-hoc-tieng-noi.md)) khiến các mô hình hầu như luôn dự đoán **log**-mel chứ không phải mel tuyến tính.

### 1.4.6. Luật số lớn và ước lượng Monte Carlo

**Luật số lớn.** Nếu $X_1, X_2, \ldots$ độc lập, cùng phân phối với kỳ vọng $\mu$ hữu hạn, thì trung bình mẫu $\bar{X}_n$ hội tụ về $\mu$ khi $n \to \infty$. Nếu thêm phương sai $\sigma^2$ hữu hạn, **định lý giới hạn trung tâm** cho biết thêm tốc độ: $\bar{X}_n$ xấp xỉ $\mathcal{N}(\mu, \sigma^2/n)$, tức sai số điển hình tỷ lệ với $\sigma/\sqrt{n}$.

**Ước lượng Monte Carlo.** Nhiều đại lượng trong học máy là kỳ vọng mà ta không tính được dạng đóng. Ta xấp xỉ bằng trung bình trên các mẫu:

$$
\mathbb{E}_{p}[f(X)] \approx \frac{1}{n}\sum_{i=1}^{n} f(x_i), \qquad x_i \sim p .
$$

Ước lượng này **không chệch** (kỳ vọng của nó đúng bằng đại lượng cần tính) và có phương sai $\operatorname{Var}_p[f(X)]/n$. Muốn giảm sai số đi 10 lần, cần tăng số mẫu lên 100 lần. Đoạn mã dưới kiểm chứng cả công thức đổi biến lẫn tốc độ $1/\sqrt{n}$:

```python
import numpy as np

rng = np.random.default_rng(0)

# (a) Đổi biến: X ~ N(0, 1), Y = exp(X)  =>  p_Y(y) = N(log y; 0, 1) / y
x = rng.standard_normal(200_000)
y = np.exp(x)
hist, edges = np.histogram(y, bins=60, range=(0.05, 4.0))
centers = 0.5 * (edges[1:] + edges[:-1])
emp = hist / (len(y) * (edges[1] - edges[0]))           # mật độ thực nghiệm
p_y = np.exp(-0.5 * np.log(centers) ** 2) / np.sqrt(2 * np.pi) / centers
print("sai lệch tuyệt đối lớn nhất của mật độ:", np.abs(emp - p_y).max().round(3))
assert np.abs(emp - p_y).max() < 0.05

# (b) Monte Carlo: E[cos X] với X ~ N(0, 1) có giá trị đúng là exp(-1/2)
true = np.exp(-0.5)
for n in [10, 100, 1_000, 10_000, 100_000]:
    errs = [abs(np.cos(rng.standard_normal(n)).mean() - true) for _ in range(200)]
    print(f"n={n:>6}  sai số trung bình={np.mean(errs):.5f}  "
          f"sai số * sqrt(n)={np.mean(errs) * np.sqrt(n):.3f}")
```

Cột cuối gần như không đổi (khoảng 0,32–0,36) khi $n$ tăng gấp 10 000 lần: đúng tốc độ $1/\sqrt{n}$.

**Monte Carlo ở khắp nơi trong học sâu.**

- **Gradient minibatch** là ước lượng Monte Carlo của gradient trên toàn bộ dữ liệu; kích thước batch điều khiển phương sai của nó ([Chương 3](ch03-toi-uu-va-ky-thuat-huan-luyen.md)).
- **ELBO** của VAE được ước lượng bằng một mẫu $\mathbf{z}$ cho mỗi điểm dữ liệu ([Chương 8](../phan-2-mo-hinh-sinh/ch08-vae.md)).
- Hàm mất mát của **diffusion** và **flow matching** là kỳ vọng theo thời điểm $t$ và nhiễu $\boldsymbol{\epsilon}$; mỗi bước huấn luyện chỉ lấy một mẫu $(t, \boldsymbol{\epsilon})$ cho mỗi câu ([Chương 11](../phan-2-mo-hinh-sinh/ch11-diffusion.md), [Chương 12](../phan-2-mo-hinh-sinh/ch12-flow-matching.md)). Đây là lý do đường loss của các mô hình này rất "nhiễu".
- Khi đánh giá, điểm MOS trung bình cũng là ước lượng từ một mẫu hữu hạn người nghe, nên cần khoảng tin cậy ([Chương 38](../phan-9-thuc-tien/ch38-danh-gia-tts.md)).

### 1.4.7. Từ phân phối đến hàm mất mát: Gauss ↔ L2, Laplace ↔ L1

Đây là mối liên hệ quan trọng nhất của chương đối với người làm TTS. Giả sử mô hình nhận văn bản $\mathbf{x}$ và dự đoán một khung mel $\hat{\mathbf{y}} = f_\theta(\mathbf{x}) \in \mathbb{R}^{F}$. Thay vì nói "dự đoán phải gần mục tiêu", ta phát biểu một **giả định xác suất**: mel thật là dự đoán cộng với nhiễu.

**Giả định Gauss.** Nếu $\mathbf{y} \sim \mathcal{N}(\hat{\mathbf{y}}, \sigma^2\mathbf{I})$ thì

$$
-\log p(\mathbf{y} \mid \mathbf{x}) = \frac{1}{2\sigma^2}\lVert \mathbf{y} - \hat{\mathbf{y}} \rVert_2^2 + \frac{F}{2}\log(2\pi\sigma^2) .
$$

Với $\sigma$ cố định, số hạng thứ hai là hằng số, nên **cực tiểu hóa negative log-likelihood (NLL) tương đương cực tiểu hóa sai số bình phương (L2, MSE)**.

**Giả định Laplace.** Nếu mỗi thành phần $y_f$ độc lập theo phân phối Laplace tâm $\hat{y}_f$, thang đo $b$:

$$
-\log p(\mathbf{y} \mid \mathbf{x}) = \frac{1}{b}\lVert \mathbf{y} - \hat{\mathbf{y}} \rVert_1 + F\log(2b),
$$

tức **cực tiểu hóa NLL tương đương cực tiểu hóa sai số tuyệt đối (L1, MAE)**. Ta kiểm chứng bằng `torch.distributions`:

```python
import math
import torch
from torch.distributions import Laplace, Normal

torch.manual_seed(0)
y = torch.randn(4, 80)          # (T, F) log-mel "thật"
y_hat = torch.randn(4, 80)      # (T, F) log-mel dự đoán (đầu ra mô hình)

# NLL trung bình trên mỗi phần tử, với sigma = 1 và b = 1
nll_gauss = -Normal(y_hat, 1.0).log_prob(y).mean()
nll_laplace = -Laplace(y_hat, 1.0).log_prob(y).mean()

mse = ((y - y_hat) ** 2).mean()
mae = (y - y_hat).abs().mean()
assert torch.allclose(nll_gauss, 0.5 * mse + 0.5 * math.log(2 * math.pi))
assert torch.allclose(nll_laplace, mae + math.log(2.0))
print(f"NLL Gauss = {nll_gauss:.4f} = 0.5*MSE + hằng số")
print(f"NLL Laplace = {nll_laplace:.4f} = MAE + hằng số")
```

**L2 dự đoán trung bình, L1 dự đoán trung vị.** Xét bài toán đơn giản nhất: chọn một hằng số $c$ để dự đoán một biến ngẫu nhiên $Y$.

- Với L2: $\frac{d}{dc}\mathbb{E}[(Y - c)^2] = -2\,\mathbb{E}[Y - c] = 0 \Rightarrow c^\star = \mathbb{E}[Y]$, **trung bình**.
- Với L1: $\frac{d}{dc}\mathbb{E}\lvert Y - c \rvert = P(Y < c) - P(Y > c) = 0 \Rightarrow P(Y < c^\star) = P(Y > c^\star) = \tfrac{1}{2}$, tức $c^\star$ là **trung vị**.

Áp dụng cho từng $\mathbf{x}$: mô hình huấn luyện bằng L2 (với đủ năng lực và dữ liệu) tiến tới **trung bình có điều kiện** $\mathbb{E}[\mathbf{y} \mid \mathbf{x}]$; mô hình huấn luyện bằng L1 tiến tới **trung vị có điều kiện** (theo từng chiều). Khi phân phối $p(\mathbf{y} \mid \mathbf{x})$ có nhiều chế độ (*multimodal*), cả hai đều có thể cho ra một dự đoán **không giống mẫu thật nào**. Hãy xem ví dụ với một "giá trị mel" có hai chế độ:

```python
import numpy as np

rng = np.random.default_rng(0)
# Một "giá trị mel" có hai chế độ: cùng một câu, người nói lúc đọc cao, lúc đọc thấp
y = np.concatenate([rng.normal(-2.0, 0.3, 700), rng.normal(2.0, 0.3, 300)])

c = np.linspace(-3, 3, 6001)                          # các dự đoán hằng số ứng viên
l2 = ((y[None, :] - c[:, None]) ** 2).mean(axis=1)    # MSE  (khả năng Gauss)
l1 = np.abs(y[None, :] - c[:, None]).mean(axis=1)     # MAE  (khả năng Laplace)

c_l2, c_l1 = c[l2.argmin()], c[l1.argmin()]
print(f"cực tiểu L2 = {c_l2:+.3f} | trung bình mẫu = {y.mean():+.3f}")
print(f"cực tiểu L1 = {c_l1:+.3f} | trung vị mẫu  = {np.median(y):+.3f}")
assert abs(c_l2 - y.mean()) < 1e-3
assert abs(c_l1 - np.median(y)) < 0.05
# Tỷ lệ mẫu thật nằm gần dự đoán L2 (trong khoảng ±0.5): gần như không có!
print("tỷ lệ mẫu gần dự đoán L2:", np.mean(np.abs(y - c_l2) < 0.5))
```

Dự đoán L2 rơi vào khoảng $-0{,}81$, đúng chỗ "thung lũng" giữa hai chế độ, nơi chỉ khoảng 0,6% mẫu thật nằm gần. Dự đoán L1 rơi vào khoảng $-1{,}83$, bên trong chế độ chiếm đa số, nên trông "thật" hơn. Nhưng L1 không phải liều thuốc vạn năng: nếu hai chế độ có trọng số bằng nhau, trung vị cũng có thể rơi vào giữa.

**Liên hệ với TTS.** Trên thực tế cả hai lựa chọn đều phổ biến: Tacotron 2 dùng tổng MSE của mel trước và sau post-net (Shen và cộng sự, 2018), còn HiFi-GAN và VITS dùng khoảng cách L1 giữa mel của âm thanh sinh ra và mel thật như một số hạng phụ trợ (Kong và cộng sự, 2020; Kim và cộng sự, 2021). Hiện tượng "lấy trung bình các chế độ" ở trên chính là nguồn gốc của **over-smoothing**: mel dự đoán bằng hồi quy L1/L2 thường mờ, thiếu chi tiết ở dải tần cao, nghe "bẹt" và đều đều. Đây là động lực chính để chuyển sang **mô hình sinh** (GAN, flow, diffusion, flow matching), được phân tích kỹ ở [Chương 7](../phan-2-mo-hinh-sinh/ch07-tong-quan-mo-hinh-sinh.md). [Chương 2](ch02-hoc-may-va-mang-no-ron.md) sẽ khái quát hóa lập luận trên thành nguyên lý **ước lượng hợp lý cực đại** và thêm góc nhìn MAP cho điều chuẩn.

!!! note "Góc tiếng Việt"
    Trộn nhiều phương ngữ trong dữ liệu huấn luyện mà không cho mô hình biết người nói hay vùng miền là một cách chắc chắn để tạo ra phân phối nhiều chế độ. Ví dụ, thanh hỏi và thanh ngã được phân biệt rõ trong giọng Bắc nhưng thường nhập làm một trong giọng Nam ([Chương 15](../phan-3-tin-hieu-ngu-am/ch15-ngu-am-tieng-viet.md)). Nếu dữ liệu chứa cả hai giọng, đường nét F0 "đúng" của một âm tiết mang thanh ngã sẽ có ít nhất hai chế độ. Một mô hình hồi quy L2 khi đó sẽ học một đường nét trung bình không thuộc giọng nào, nghe như thanh điệu bị "nhòe". Cách khắc phục là thêm điều kiện (embedding người nói/phương ngữ, [Chương 28](../phan-7-e2e-bieu-cam/ch28-da-nguoi-noi-voice-cloning.md)) hoặc dùng mô hình sinh.

## 1.5. Lý thuyết thông tin

Lý thuyết thông tin, khởi nguồn từ công trình của Shannon (1948), trả lời hai câu hỏi tưởng như thuộc về viễn thông: một nguồn tin "bất định" đến đâu, và dùng sai mô hình để mã hóa thì tốn thêm bao nhiêu. Hóa ra đó chính là những câu hỏi mà hàm mất mát của mô hình sinh đặt ra.

### 1.5.1. Lượng tin và entropy

**Trực giác.** Một sự kiện càng hiếm thì khi xảy ra càng "mang nhiều tin". Shannon định nghĩa **lượng tin** (*self-information*) của kết quả $x$ là $-\log p(x)$: bằng 0 với sự kiện chắc chắn, tăng dần khi $p(x)$ giảm, và cộng được với các sự kiện độc lập (vì $\log$ biến tích thành tổng).

**Entropy** là lượng tin trung bình:

$$
H(p) = \mathbb{E}_{p}[-\log p(X)] = -\sum_x p(x)\log p(x) .
$$

Với logarit cơ số 2, đơn vị là **bit**; với logarit tự nhiên, đơn vị là **nat** ($1$ nat $= 1/\ln 2 \approx 1{,}443$ bit). PyTorch và hầu hết các bài báo học sâu dùng nat. Với biến rời rạc có $K$ giá trị, $0 \le H(p) \le \log K$; giá trị nhỏ nhất đạt được khi phân phối tất định, lớn nhất khi phân phối đều.

Một đại lượng liên quan rất hay dùng là **perplexity** $\exp(H(p))$ (với $H$ tính bằng nat), có thể hiểu là "số lựa chọn đồng khả năng tương đương". Trong các codec âm thanh ([Chương 31](../phan-8-mo-hinh-lon/ch31-audio-codec-speech-token.md)), perplexity của tần suất sử dụng các mã trong sổ mã là thước đo nhanh cho hiện tượng **sập sổ mã** (*codebook collapse*): một sổ mã 1 024 mục mà perplexity chỉ vài chục nghĩa là phần lớn các mục không được dùng.

```python
import numpy as np

def entropy(p):                                  # đơn vị nat
    p = p[p > 0]
    return -np.sum(p * np.log(p))

V = 1024                                         # kích thước sổ mã (codebook)
uniform = np.full(V, 1.0 / V)
rng = np.random.default_rng(0)
skewed = rng.dirichlet(np.full(V, 0.05))         # chỉ một số ít mã được dùng nhiều
for name, p in [("đều", uniform), ("lệch", skewed)]:
    H = entropy(p)
    print(f"{name:5s}: H = {H:.3f} nat = {H / np.log(2):.2f} bit, "
          f"perplexity = {np.exp(H):7.1f} / {V}")

# Thông tin tương hỗ từ bảng phân phối đồng thời p(x, y)
pxy = np.array([[0.30, 0.05],
                [0.05, 0.60]])
px, py = pxy.sum(1), pxy.sum(0)
mi = np.sum(pxy * np.log(pxy / np.outer(px, py)))
print(f"I(X;Y) = {mi:.4f} nat ; H(X) + H(Y) - H(X,Y) = "
      f"{entropy(px) + entropy(py) - entropy(pxy.ravel()):.4f} nat")
```

Với phân phối đều, entropy đúng bằng $\log_2 1024 = 10$ bit và perplexity bằng 1 024. Với phân phối lệch, perplexity chỉ khoảng 81: sổ mã "hiệu dụng" nhỏ hơn hơn 12 lần so với kích thước danh nghĩa. (Phần thông tin tương hỗ ở cuối đoạn mã được giải thích ở Mục 1.5.4.)

Với biến liên tục, ta có **entropy vi phân** $h(p) = -\int p(x)\log p(x)\,dx$. Ví dụ, Gauss $\mathcal{N}(\mu, \sigma^2)$ có $h = \frac{1}{2}\log(2\pi e\sigma^2)$. Khác với entropy rời rạc, entropy vi phân có thể âm (khi $\sigma$ nhỏ), và nó không bất biến khi đổi biến, một hệ quả trực tiếp của công thức ở Mục 1.4.5.

### 1.5.2. Entropy chéo

Giả sử dữ liệu đến từ phân phối $p$, nhưng ta mô tả chúng bằng mô hình $q$. **Entropy chéo** là lượng tin trung bình khi dùng $q$ để "giải thích" dữ liệu từ $p$:

$$
H(p, q) = \mathbb{E}_{p}[-\log q(X)] = -\sum_x p(x)\log q(x) .
$$

Theo cách hiểu của nén dữ liệu: đó là độ dài mã trung bình (tính bằng bit nếu dùng $\log_2$) khi ta thiết kế bộ mã tối ưu cho $q$ nhưng dữ liệu thực ra theo $p$.

**Đây chính là hàm mất mát phân loại.** Khi nhãn là một lớp cụ thể $y$, ta coi $p$ là phân phối one-hot đặt toàn bộ xác suất vào $y$. Khi đó $H(p, q) = -\log q(y)$: entropy chéo bằng **negative log-likelihood của nhãn đúng**. Trung bình trên tập huấn luyện, nó là hàm mất mát entropy chéo quen thuộc. Trong TTS, nó xuất hiện mỗi khi mô hình dự đoán một đại lượng rời rạc:

- WaveNet lượng tử hóa mẫu âm thanh bằng μ-law thành 256 mức và dự đoán mẫu tiếp theo bằng softmax 256 lớp (van den Oord và cộng sự, 2016; [Chương 24](../phan-6-vocoder/ch24-vocoder-tu-hoi-quy-va-flow.md)).
- Các codec language model như VALL-E dự đoán token của neural audio codec bằng entropy chéo trên sổ mã (Wang và cộng sự, 2023; [Chương 32](../phan-8-mo-hinh-lon/ch32-codec-language-model.md)).
- Cờ dừng của Tacotron 2 dùng entropy chéo nhị phân (phân phối Bernoulli, [Chương 20](../phan-5-mo-hinh-am-hoc/ch20-tacotron.md)).

### 1.5.3. Phân kỳ KL và tính bất đối xứng

**Phân kỳ Kullback–Leibler** đo "chi phí thêm" khi dùng $q$ thay cho $p$:

$$
D_{\mathrm{KL}}(p \,\Vert\, q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = H(p, q) - H(p) .
$$

**KL không âm (bất đẳng thức Gibbs).** Dùng bất đẳng thức Jensen cho hàm lõm $\log$ (tức $\mathbb{E}[\log Z] \le \log\mathbb{E}[Z]$), lấy tổng trên các $x$ có $p(x) > 0$:

$$
-D_{\mathrm{KL}}(p \,\Vert\, q) = \sum_x p(x)\log\frac{q(x)}{p(x)} \le \log\sum_x p(x)\frac{q(x)}{p(x)} = \log\sum_{x:\,p(x) > 0} q(x) \le \log 1 = 0 .
$$

Vậy $D_{\mathrm{KL}}(p \,\Vert\, q) \ge 0$, và dấu bằng xảy ra khi và chỉ khi $p = q$. Từ đó $H(p, q) \ge H(p)$: không mô hình nào mã hóa dữ liệu tốt hơn chính phân phối thật.

**Hệ quả then chốt.** Vì $H(p_{\text{data}})$ không phụ thuộc tham số $\theta$,

$$
\arg\min_\theta H(p_{\text{data}}, q_\theta) = \arg\min_\theta D_{\mathrm{KL}}(p_{\text{data}} \,\Vert\, q_\theta) .
$$

Cực tiểu hóa entropy chéo, cực tiểu hóa KL thuận và cực đại hóa log-likelihood là **cùng một việc**. Mối liên hệ này được dùng lại nhiều lần ở [Chương 7](../phan-2-mo-hinh-sinh/ch07-tong-quan-mo-hinh-sinh.md).

**KL không đối xứng.** Nói chung $D_{\mathrm{KL}}(p \,\Vert\, q) \ne D_{\mathrm{KL}}(q \,\Vert\, p)$, và hai hướng có "tính cách" rất khác nhau:

- **KL thuận** $D_{\mathrm{KL}}(p \,\Vert\, q)$ lấy kỳ vọng theo $p$. Nếu $q(x) \approx 0$ ở nơi $p(x) > 0$ thì $\log(p/q)$ rất lớn: $q$ bị phạt nặng khi **bỏ sót** bất kỳ vùng nào của dữ liệu. Kết quả là $q$ có xu hướng **phủ mọi chế độ** (*mode-covering*), chấp nhận đặt xác suất cả vào vùng "thung lũng" giữa các chế độ.
- **KL nghịch** $D_{\mathrm{KL}}(q \,\Vert\, p)$ lấy kỳ vọng theo $q$. $q$ bị phạt nặng khi đặt xác suất vào nơi $p$ gần bằng 0, nhưng không bị phạt khi bỏ sót. Kết quả là $q$ có xu hướng **bám một chế độ** (*mode-seeking*).

Đoạn mã sau khớp một Gauss $q$ với một hỗn hợp hai Gauss $p$ bằng cách tìm kiếm trên lưới tham số:

```python
import numpy as np

def normal_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

x = np.linspace(-8, 8, 4001)
dx = x[1] - x[0]
# p: hỗn hợp hai Gauss (hai "chế độ", ví dụ hai cách ngắt nhịp một câu)
p = 0.5 * normal_pdf(x, -2.5, 0.6) + 0.5 * normal_pdf(x, 2.5, 0.6)

def kl(a, b):                         # KL(a || b) bằng tích phân số trên lưới
    mask = a > 1e-300
    return np.sum(a[mask] * (np.log(a[mask]) - np.log(b[mask] + 1e-300))) * dx

best_fwd, best_rev = (np.inf, None), (np.inf, None)
for mu in np.linspace(-4, 4, 81):
    for sigma in np.linspace(0.2, 4, 77):
        q = normal_pdf(x, mu, sigma)
        f, r = kl(p, q), kl(q, p)     # thuận KL(p||q), nghịch KL(q||p)
        if f < best_fwd[0]:
            best_fwd = (f, (mu, sigma))
        if r < best_rev[0]:
            best_rev = (r, (mu, sigma))

print("KL thuận  KL(p||q) nhỏ nhất tại (mu, sigma) =", np.round(best_fwd[1], 2))
print("KL nghịch KL(q||p) nhỏ nhất tại (mu, sigma) =", np.round(best_rev[1], 2))
```

KL thuận cho $q$ có $\mu \approx 0$, $\sigma \approx 2{,}55$: một Gauss rộng phủ cả hai chế độ, đặt nhiều xác suất nhất vào đúng chỗ dữ liệu *không bao giờ* xuất hiện (giá trị lý thuyết là $\sigma = \sqrt{0{,}6^2 + 2{,}5^2} \approx 2{,}57$, xem Bài tập 5). KL nghịch cho $\mu = 2{,}5$, $\sigma = 0{,}6$: khớp hoàn hảo một chế độ và bỏ qua chế độ kia (do đối xứng, $\mu = -2{,}5$ cũng tốt như vậy).

**Liên hệ với TTS.**

- Huấn luyện hợp lý cực đại tương đương cực tiểu KL thuận. Khi họ phân phối của mô hình quá đơn giản (ví dụ một Gauss đơn mode cho mỗi khung mel), mô hình bị buộc phải "trải rộng", và mẫu sinh ra trở nên nhòe: đây là cùng một hiện tượng với over-smoothing ở Mục 1.4.7, nhìn từ góc độ thông tin.
- Parallel WaveNet chưng cất một WaveNet tự hồi quy (thầy) sang một flow song song (trò) bằng cách cực tiểu KL **nghịch** từ trò tới thầy ([Chương 24](../phan-6-vocoder/ch24-vocoder-tu-hoi-quy-va-flow.md)), một lựa chọn dẫn đến đặc tính bám chế độ.
- Số hạng KL trong ELBO của VAE và của VITS kéo hậu nghiệm xấp xỉ về gần tiên nghiệm ([Chương 8](../phan-2-mo-hinh-sinh/ch08-vae.md), [Chương 27](../phan-7-e2e-bieu-cam/ch27-vits.md)). Với hai Gauss một chiều, KL có dạng đóng

$$
D_{\mathrm{KL}}\big(\mathcal{N}(\mu_1, \sigma_1^2) \,\Vert\, \mathcal{N}(\mu_2, \sigma_2^2)\big) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2},
$$

mà [Chương 8](../phan-2-mo-hinh-sinh/ch08-vae.md) sẽ suy ra chi tiết.

### 1.5.4. Thông tin tương hỗ

**Thông tin tương hỗ** đo lượng thông tin mà một biến tiết lộ về biến kia:

$$
I(X; Y) = D_{\mathrm{KL}}\big(p(x, y) \,\Vert\, p(x)\,p(y)\big) = H(X) - H(X \mid Y) = H(X) + H(Y) - H(X, Y) .
$$

Định nghĩa đầu tiên nói rằng $I(X; Y)$ là "khoảng cách" giữa phân phối đồng thời và phân phối mà hai biến *sẽ có* nếu chúng độc lập. Vì vậy $I(X; Y) \ge 0$, bằng 0 khi và chỉ khi $X$, $Y$ độc lập, và đối xứng theo $X$, $Y$. Định nghĩa thứ hai nói rằng nó là lượng bất định về $X$ được giảm bớt khi biết $Y$. Phần cuối của đoạn mã ở Mục 1.5.1 tính $I(X; Y) \approx 0{,}328$ nat theo hai cách và cho cùng kết quả.

Trong TTS, thông tin tương hỗ là ngôn ngữ tự nhiên để nói về **tách rời** (*disentanglement*). Ta muốn embedding người nói chứa thông tin về *giọng* nhưng gần như không chứa thông tin về *nội dung*, và ngược lại, biểu diễn nội dung không "rò rỉ" danh tính người nói. Các kỹ thuật như nút thắt thông tin (*information bottleneck*), bộ phân loại đối kháng hay các ràng buộc cực tiểu thông tin tương hỗ đều nhằm mục tiêu đó ([Chương 28](../phan-7-e2e-bieu-cam/ch28-da-nguoi-noi-voice-cloning.md), [Chương 30](../phan-7-e2e-bieu-cam/ch30-da-ngon-ngu-code-switching.md)).

!!! note "Góc tiếng Việt"
    Trong tiếng Việt, đường nét tần số cơ bản (F0) mang **nghĩa từ vựng**: "ma, mà, má, mả, mã, mạ" chỉ khác nhau ở thanh điệu. Nói theo ngôn ngữ của mục này, thông tin tương hỗ giữa F0 và danh tính của từ là lớn hơn hẳn so với tiếng Anh, nơi F0 chủ yếu mang ngữ điệu. Hệ quả thực tế: một kỹ thuật vô tình "xóa" thông tin cao độ, chẳng hạn một nút thắt quá hẹp khi tách nội dung khỏi giọng, hay một phép chuẩn hóa F0 quá mạnh, có thể chấp nhận được với tiếng Anh nhưng sẽ làm hỏng thanh điệu tiếng Việt ([Chương 15](../phan-3-tin-hieu-ngu-am/ch15-ngu-am-tieng-viet.md), [Chương 16](../phan-3-tin-hieu-ngu-am/ch16-ngon-dieu.md)).

## 1.6. Số phức và công thức Euler: cầu nối tới Fourier

Âm thanh là dao động. Công cụ toán học thuận tiện nhất để mô tả dao động là **số phức**, và cây cầu nối hai thứ là công thức Euler. Mục này chỉ trình bày phần tối thiểu để đọc hiểu [Chương 14](../phan-3-tin-hieu-ngu-am/ch14-xu-ly-tin-hieu-so.md).

**Số phức.** Một số phức $z = a + ib$, với $i^2 = -1$, là một điểm $(a, b)$ trên mặt phẳng. Nó có **mô-đun** $\lvert z \rvert = \sqrt{a^2 + b^2}$ (khoảng cách tới gốc) và **argument** $\arg z = \varphi$ (góc so với trục thực). Ta có thể viết $z$ ở **dạng cực** $z = r(\cos\varphi + i\sin\varphi)$ với $r = \lvert z \rvert$.

**Công thức Euler.** Thay $x = i\theta$ vào chuỗi Taylor $e^{x} = \sum_{n \ge 0} x^n/n!$ và dùng $i^2 = -1$, $i^3 = -i$, $i^4 = 1$, ta tách được phần thực và phần ảo:

$$
e^{i\theta} = \Big(1 - \frac{\theta^2}{2!} + \frac{\theta^4}{4!} - \cdots\Big) + i\Big(\theta - \frac{\theta^3}{3!} + \frac{\theta^5}{5!} - \cdots\Big) = \cos\theta + i\sin\theta .
$$

Vậy dạng cực viết gọn thành $z = re^{i\varphi}$. Phép nhân trở nên rất trực quan: $r_1e^{i\varphi_1} \cdot r_2e^{i\varphi_2} = r_1r_2\,e^{i(\varphi_1 + \varphi_2)}$, tức **nhân mô-đun và cộng góc**. Nhân với $e^{i\theta}$ là **phép xoay** một góc $\theta$. Cơ chế mã hóa vị trí RoPE trong Transformer khai thác đúng tính chất này: nó xoay từng cặp chiều của query và key một góc tỷ lệ với vị trí ([Chương 5](ch05-transformer.md)).

**Dao động hình sin là phần thực của một véc-tơ quay.** Một thành phần tần số $\omega$ với biên độ $A$ và pha $\varphi$ viết được thành

$$
A\cos(\omega n + \varphi) = \operatorname{Re}\big\{A e^{i\varphi}\,e^{i\omega n}\big\} .
$$

Số phức $Ae^{i\varphi}$ (gọi là *phasor*) gói **cả biên độ lẫn pha** vào một con số. Đây là lý do mọi phép phân tích tần số đều cho ra số phức.

**DFT là một phép biến đổi tuyến tính.** Biến đổi Fourier rời rạc của $N$ mẫu $s[0], \ldots, s[N-1]$ là

$$
X[k] = \sum_{n=0}^{N-1} s[n]\,e^{-i2\pi kn/N}, \qquad k = 0, \ldots, N - 1 .
$$

Mỗi $X[k]$ là **tích vô hướng** của tín hiệu với một dao động phức tần số $k f_s/N$ Hz: nó cho biết tín hiệu "chứa bao nhiêu" thành phần tần số đó ($\lvert X[k] \rvert$) và với pha nào ($\arg X[k]$). Viết dưới dạng ma trận, $\mathbf{X} = \mathbf{F}\mathbf{s}$ với $F_{kn} = e^{-i2\pi kn/N}$. Các cột của $\mathbf{F}$ trực giao với nhau, $\mathbf{F}^{\mathsf{H}}\mathbf{F} = N\mathbf{I}$ (với $\mathbf{F}^{\mathsf{H}}$ là chuyển vị liên hợp), nên DFT bảo toàn năng lượng đến một hằng số (**định lý Parseval**): $\sum_k \lvert X[k] \rvert^2 = N\sum_n \lvert s[n] \rvert^2$. Như vậy toàn bộ kiến thức đại số tuyến tính ở Mục 1.2 áp dụng được cho phân tích tần số.

```python
import numpy as np

# Công thức Euler: e^{i theta} = cos(theta) + i sin(theta)
theta = np.linspace(0, 2 * np.pi, 7)
assert np.allclose(np.exp(1j * theta), np.cos(theta) + 1j * np.sin(theta))

# DFT là một phép biến đổi tuyến tính: X = F s, với F[k, n] = exp(-2πi k n / N)
N = 16
n = np.arange(N)
F = np.exp(-2j * np.pi * np.outer(n, n) / N)            # (N, N) ma trận DFT
rng = np.random.default_rng(0)
s = rng.standard_normal(N)                               # tín hiệu thực
X = F @ s
assert np.allclose(X, np.fft.fft(s))                     # khớp với FFT
assert np.allclose(F.conj().T @ F, N * np.eye(N))        # các cột trực giao
assert np.allclose(np.sum(np.abs(X) ** 2), N * np.sum(s ** 2))   # Parseval
mag, phase = np.abs(X), np.angle(X)                      # biên độ và pha
assert np.allclose(mag * np.exp(1j * phase), X)          # dạng cực
print("DFT bằng ma trận khớp np.fft.fft; định lý Parseval được nghiệm đúng.")
```

Nhân ma trận trực tiếp tốn $O(N^2)$ phép tính; thuật toán **FFT** tính cùng kết quả trong $O(N\log N)$ bằng cách khai thác cấu trúc đặc biệt của $\mathbf{F}$. Với tín hiệu thực, $X[N - k]$ là liên hợp phức của $X[k]$, nên chỉ cần giữ $N/2 + 1$ bin đầu: đó là lý do `n_fft = 1024` cho 513 bin như ở Mục 1.2.7. Với $f_s = 22\,050$ Hz, các bin cách nhau $22\,050/1024 \approx 21{,}5$ Hz.

**Liên hệ với TTS.** Áp DFT lên từng đoạn ngắn của tín hiệu ta được STFT, một ma trận số phức theo thời gian và tần số. Mel-spectrogram chỉ giữ **biên độ** (qua $\lvert\cdot\rvert^2$) rồi chiếu xuống thang mel, nên **toàn bộ thông tin pha bị bỏ đi**. Khôi phục pha là bài toán khó: Griffin–Lim làm điều đó bằng lặp ([Chương 14](../phan-3-tin-hieu-ngu-am/ch14-xu-ly-tin-hieu-so.md)), các vocoder nơ-ron học cách sinh dạng sóng trực tiếp ([Chương 24](../phan-6-vocoder/ch24-vocoder-tu-hoi-quy-va-flow.md), [Chương 25](../phan-6-vocoder/ch25-gan-vocoder.md)), còn những vocoder như Vocos dự đoán thẳng biên độ và pha rồi dùng STFT ngược.

## Lỗi thường gặp và mẹo thực hành

- **Lấy logarit của số 0.** Mel-spectrogram có những ô năng lượng bằng hoặc gần 0 (đoạn im lặng, dải tần bị lọc). `log(0) = -inf` sẽ lan ra thành `NaN` trong gradient. Luôn kẹp giá trị trước khi lấy log, ví dụ `torch.log(torch.clamp(mel, min=1e-5))` như trong mã nguồn HiFi-GAN. Ngưỡng kẹp và cơ số log là một phần của "hợp đồng" giữa mô hình âm học và vocoder: hai bên phải dùng **cùng** cấu hình ([Chương 14](../phan-3-tin-hieu-ngu-am/ch14-xu-ly-tin-hieu-so.md), [Chương 26](../phan-6-vocoder/ch26-diffusion-vocoder-va-so-sanh.md)).
- **Tràn số trong softmax.** Tính $e^{z}$ với $z = 1000$ cho `inf`. Dùng mẹo log-sum-exp: $\log\sum_j e^{z_j} = m + \log\sum_j e^{z_j - m}$ với $m = \max_j z_j$. Trong PyTorch, hãy dùng `torch.logsumexp`, `F.log_softmax` và truyền **logit** (chưa qua softmax) vào `F.cross_entropy`; đừng tự tính `softmax` rồi lấy `log`.
- **Định thức tràn hoặc mất số.** Định thức của ma trận lớn dễ vượt khỏi dải biểu diễn của số thực dấu phẩy động. Khi cần $\log\lvert\det\mathbf{A}\rvert$, dùng `torch.linalg.slogdet` thay vì `torch.log(torch.det(A))`.
- **Nhầm độ lệch chuẩn với phương sai.** `torch.distributions.Normal(loc, scale)` nhận **độ lệch chuẩn**, còn `torch.nn.GaussianNLLLoss` nhận **phương sai**. Thêm nữa, `torch.var` mặc định dùng ước lượng không chệch (chia cho $n - 1$), còn `numpy.var` mặc định chia cho $n$. Khi so sánh số liệu thống kê giữa hai thư viện, hãy kiểm tra tham số này.
- **Nhầm chiều của KL.** `F.kl_div(input, target)` coi `input` là **log** xác suất của $q$ và `target` là xác suất của $p$, rồi tính $D_{\mathrm{KL}}(p \,\Vert\, q)$. Truyền xác suất thay vì log xác suất, hoặc đảo vai trò hai đối số, đều không báo lỗi nhưng cho kết quả sai. Dùng `reduction="batchmean"` để có đúng giá trị KL trung bình trên mỗi mẫu.
- **Ma trận hiệp phương sai "gần như" PSD.** Do sai số làm tròn, một ma trận hiệp phương sai ước lượng từ dữ liệu có thể có trị riêng hơi âm, khiến phân rã Cholesky thất bại. Mẹo chuẩn: cộng một lượng nhỏ $\varepsilon\mathbf{I}$ (*jitter*) vào đường chéo.
- **Kiểm chứng gradient bằng `float32`.** Sai phân hữu hạn với `float32` rất kém chính xác. Hãy kiểm chứng công thức bằng `float64`, còn huấn luyện thì dùng `float32` hoặc thấp hơn ([Chương 3](ch03-toi-uu-va-ky-thuat-huan-luyen.md)).
- **Lẫn lộn bit và nat.** Bài báo về codec hoặc mô hình ngôn ngữ thường báo cáo theo bit (hoặc bit trên mỗi chiều), còn PyTorch trả về loss theo nat. Chia cho $\ln 2$ để đổi nat sang bit trước khi so sánh.

## Tóm tắt

- Mọi dữ liệu trong TTS là tensor: một khung mel là véc-tơ trong $\mathbb{R}^{F}$, một câu là ma trận $T \times F$, một batch là tensor `(B, T, F)`. Một lớp tuyến tính là một ngân hàng các tích vô hướng.
- Bộ lọc mel là ma trận $80 \times 513$ có hạng tối đa 80; theo định lý hạng–số khuyết, không gian rỗng có ít nhất 433 chiều, nên mel làm mất thông tin và vocoder phải *sinh* lại chi tiết chứ không thể chỉ "nghịch đảo".
- Phân rã riêng mô tả ma trận đối xứng (hiệp phương sai, Hessian); SVD mô tả mọi ma trận. Giá trị kỳ dị lớn nhất là chuẩn phổ (cơ sở của spectral normalization), và SVD rút gọn cho xấp xỉ hạng thấp tối ưu (ý tưởng nền của LoRA).
- Gradient là hướng tăng nhanh nhất; khai triển Taylor giải thích vì sao bước gradient nhỏ làm giảm hàm mất mát và vì sao sai phân trung tâm có sai số $O(h^2)$.
- Jacobian của hàm hợp là tích các Jacobian; với hàm mất mát vô hướng, lan truyền ngược là chuỗi tích véc-tơ–Jacobian tính từ cuối về đầu.
- Giả định nhiễu Gauss dẫn tới mất mát L2 (dự đoán trung bình), nhiễu Laplace dẫn tới L1 (dự đoán trung vị). Khi $p(\mathbf{Y} \mid \mathbf{x})$ nhiều chế độ, cả hai đều gây over-smoothing, và đây là động lực cho các mô hình sinh.
- Công thức đổi biến $\log p_{\mathbf{Y}} = \log p_{\mathbf{X}} - \log\lvert\det\mathbf{J}\rvert$ là nền tảng của normalizing flow; ước lượng Monte Carlo có sai số giảm theo $1/\sqrt{n}$ và có mặt trong mọi gradient minibatch.
- Cực tiểu entropy chéo, cực tiểu KL thuận và cực đại log-likelihood là cùng một việc. KL thuận phủ mọi chế độ, KL nghịch bám một chế độ. Thông tin tương hỗ là ngôn ngữ của bài toán tách rời giọng nói và nội dung.
- Số phức gói biên độ và pha vào một con số; DFT là phép biến đổi tuyến tính trực giao. Mel-spectrogram bỏ đi pha, và khôi phục pha là nhiệm vụ của vocoder.

## Bài tập

**Bài 1 [Cơ bản].** Một cấu hình phổ biến khác dùng `n_fft = 2048` và 128 kênh mel. Ma trận bộ lọc mel khi đó có kích thước bao nhiêu? Không gian rỗng có ít nhất bao nhiêu chiều? Tăng số kênh mel có làm bài toán vocoder "dễ hơn" không, và cái giá phải trả là gì?

??? success "Lời giải"
    Phổ có $2048/2 + 1 = 1025$ bin, nên ma trận có kích thước $128 \times 1025$, hạng tối đa 128, và không gian rỗng có ít nhất $1025 - 128 = 897$ chiều. Thêm kênh mel giữ lại nhiều chi tiết tần số hơn (không gian rỗng nhỏ đi tương đối), nên vocoder phải "đoán" ít hơn. Cái giá là mô hình âm học phải dự đoán một mục tiêu nhiều chiều hơn, và mỗi kênh mel hẹp hơn nên chứa ít bin hơn, có thể nhiễu hơn. Dù vậy, pha vẫn bị bỏ đi hoàn toàn bất kể số kênh mel.

**Bài 2 [Cơ bản].** Tiếng Việt có sáu thanh. (a) Nếu sáu thanh xuất hiện đồng khả năng, entropy của thanh điệu là bao nhiêu bit? (b) Với một phân phối giả định $(0{,}30;\, 0{,}20;\, 0{,}20;\, 0{,}10;\, 0{,}05;\, 0{,}15)$, entropy là bao nhiêu? Vì sao nó nhỏ hơn ở câu (a)?

??? success "Lời giải"
    (a) $\log_2 6 \approx 2{,}585$ bit. (b) $H = -\sum_k p_k\log_2 p_k \approx 0{,}521 + 2 \times 0{,}464 + 0{,}332 + 0{,}216 + 0{,}411 \approx 2{,}41$ bit. Phân phối đều có entropy lớn nhất trong các phân phối trên cùng tập giá trị; mọi độ lệch khỏi phân phối đều đều làm kết quả dễ đoán hơn, tức entropy nhỏ hơn. (Tần suất thanh điệu thật phụ thuộc kho ngữ liệu; con số ở câu (b) chỉ để luyện tập.)

**Bài 3 [Cơ bản].** Chứng minh $\nabla_{\mathbf{x}}(\mathbf{x}^\top\mathbf{A}\mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ bằng cách viết biểu thức dưới dạng tổng theo từng thành phần. Suy ra $\nabla_{\mathbf{x}}\lVert\mathbf{x}\rVert_2^2 = 2\mathbf{x}$.

??? tip "Gợi ý"
    Viết $\mathbf{x}^\top\mathbf{A}\mathbf{x} = \sum_{i}\sum_{j} x_i A_{ij} x_j$. Khi lấy đạo hàm theo $x_k$, có hai nhóm số hạng chứa $x_k$: nhóm $i = k$ cho $\sum_j A_{kj}x_j = (\mathbf{A}\mathbf{x})_k$ và nhóm $j = k$ cho $\sum_i x_i A_{ik} = (\mathbf{A}^\top\mathbf{x})_k$. Với $\mathbf{A} = \mathbf{I}$ ta được kết quả thứ hai.

**Bài 4 [Nâng cao].** Với $\tau \in (0, 1)$, hàm mất mát phân vị (*pinball loss*) là $\rho_\tau(u) = u\,(\tau - \mathbb{1}[u < 0])$. Chứng minh rằng $c^\star = \arg\min_c \mathbb{E}[\rho_\tau(Y - c)]$ thỏa mãn $P(Y \le c^\star) = \tau$, tức $c^\star$ là phân vị mức $\tau$. Kiểm tra rằng $\tau = 0{,}5$ cho lại kết quả "L1 dự đoán trung vị".

??? tip "Gợi ý"
    Giả sử $Y$ có mật độ. Tách kỳ vọng thành hai miền $Y \ge c$ và $Y < c$: $\mathbb{E}[\rho_\tau(Y - c)] = \tau\int_{c}^{\infty}(y - c)p(y)\,dy + (\tau - 1)\int_{-\infty}^{c}(y - c)p(y)\,dy$. Lấy đạo hàm theo $c$ (quy tắc Leibniz; các số hạng biên bằng 0 vì hàm dưới dấu tích phân triệt tiêu tại $y = c$) được $-\tau(1 - F(c)) + (1 - \tau)F(c) = F(c) - \tau$. Với $\tau = 0{,}5$, $\rho_{0.5}(u) = \tfrac{1}{2}\lvert u \rvert$.

**Bài 5 [Nâng cao].** Cho phân phối $p$ bất kỳ trên $\mathbb{R}$ có kỳ vọng $m$ và phương sai $s^2$. Chứng minh rằng trong họ Gauss $q = \mathcal{N}(\mu, \sigma^2)$, KL thuận $D_{\mathrm{KL}}(p \,\Vert\, q)$ nhỏ nhất khi $\mu = m$ và $\sigma^2 = s^2$ (**khớp mô-men**). Áp dụng cho hỗn hợp $\tfrac{1}{2}\mathcal{N}(-2{,}5;\, 0{,}6^2) + \tfrac{1}{2}\mathcal{N}(2{,}5;\, 0{,}6^2)$ và so sánh với kết quả tìm kiếm lưới ở Mục 1.5.3.

??? success "Lời giải"
    $D_{\mathrm{KL}}(p \,\Vert\, q) = -h(p) - \mathbb{E}_p[\log q(X)]$, và số hạng đầu không phụ thuộc $(\mu, \sigma)$. Ta có $-\mathbb{E}_p[\log q(X)] = \tfrac{1}{2}\log(2\pi\sigma^2) + \frac{\mathbb{E}_p[(X - \mu)^2]}{2\sigma^2}$. Vì $\mathbb{E}_p[(X - \mu)^2] = s^2 + (m - \mu)^2$, cực tiểu theo $\mu$ cho $\mu = m$. Thay vào và lấy đạo hàm theo $\sigma^2$: $\frac{1}{2\sigma^2} - \frac{s^2}{2\sigma^4} = 0 \Rightarrow \sigma^2 = s^2$. Với hỗn hợp đã cho, $m = 0$ và $s^2 = 0{,}6^2 + 2{,}5^2 = 6{,}61$ (phương sai trong mỗi thành phần cộng phương sai của các tâm), nên $\sigma \approx 2{,}571$, khớp với giá trị lưới $2{,}55$ trong phạm vi bước lưới $\approx 0{,}05$.

**Bài 6 [Nâng cao].** Cho $\mathbf{x} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ trong $\mathbb{R}^{d}$ và $\mathbf{y} = \mathbf{A}\mathbf{x} + \mathbf{b}$ với $\mathbf{A}$ khả nghịch. Dùng công thức đổi biến để viết $\log p_{\mathbf{Y}}(\mathbf{y})$, rồi chứng tỏ kết quả trùng với $\log\mathcal{N}(\mathbf{y}; \mathbf{b}, \mathbf{A}\mathbf{A}^\top)$.

??? tip "Gợi ý"
    $\mathbf{x} = \mathbf{A}^{-1}(\mathbf{y} - \mathbf{b})$ và $\mathbf{J}_{\mathbf{g}} = \mathbf{A}$. Dùng $\lVert\mathbf{A}^{-1}\mathbf{u}\rVert_2^2 = \mathbf{u}^\top(\mathbf{A}\mathbf{A}^\top)^{-1}\mathbf{u}$ và $\log\lvert\det\mathbf{A}\rvert = \tfrac{1}{2}\log\det(\mathbf{A}\mathbf{A}^\top)$.

**Bài 7 [Lập trình].** Viết hàm `logsumexp(z)` ổn định số học bằng numpy theo mẹo trừ giá trị lớn nhất. So sánh với cách tính ngây thơ `np.log(np.sum(np.exp(z)))` và với `torch.logsumexp` cho `z = [1000.0, 1000.5, 999.0]` và `z = [-1000.0, -1000.5, -999.0]`.

??? success "Lời giải"
    ```python
    import numpy as np
    import torch

    def logsumexp(z):
        m = np.max(z)
        return m + np.log(np.sum(np.exp(z - m)))   # mọi số mũ đều <= 0

    for z in [np.array([1000.0, 1000.5, 999.0]), np.array([-1000.0, -1000.5, -999.0])]:
        with np.errstate(over="ignore", divide="ignore"):
            naive = np.log(np.sum(np.exp(z)))      # inf hoặc -inf
        stable = logsumexp(z)
        ref = torch.logsumexp(torch.tensor(z), dim=0).item()
        print(naive, stable, ref)
        assert np.isclose(stable, ref)
    ```

    Cách ngây thơ cho `inf` ở trường hợp đầu (tràn trên) và `-inf` ở trường hợp sau (tràn dưới), trong khi hàm ổn định cho kết quả đúng ở cả hai.

**Bài 8 [Lập trình].** Tạo 1 024 mẫu của sóng sin 440 Hz với $f_s = 22\,050$ Hz. Dùng `np.fft.rfft` tìm bin có biên độ lớn nhất và đổi chỉ số bin ra Hz. Vì sao kết quả không đúng bằng 440 Hz, và vì sao năng lượng "lan" sang các bin lân cận? (Câu hỏi này dẫn vào khái niệm rò rỉ phổ ở [Chương 14](../phan-3-tin-hieu-ngu-am/ch14-xu-ly-tin-hieu-so.md).)

??? tip "Gợi ý"
    Bin $k$ ứng với tần số $k f_s/N \approx 21{,}5k$ Hz. Ta có $440/21{,}53 \approx 20{,}4$, không phải số nguyên, nên đỉnh nằm ở bin 20 (khoảng 430,7 Hz). Vì 1 024 mẫu không chứa một số nguyên chu kỳ của sóng, khi DFT ngầm coi tín hiệu là tuần hoàn thì xuất hiện chỗ "gãy" ở biên, gây rò rỉ năng lượng sang các bin lân cận.

## Tài liệu tham khảo

1. Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning* (Chương 2–4). MIT Press.
2. Deisenroth, M. P., Faisal, A. A., Ong, C. S. (2020). *Mathematics for Machine Learning*. Cambridge University Press.
3. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
4. Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.
5. Strang, G. (2016). *Introduction to Linear Algebra* (ấn bản 5). Wellesley-Cambridge Press.
6. Petersen, K. B., Pedersen, M. S. (2012). *The Matrix Cookbook*. Technical University of Denmark.
7. Shannon, C. E. (1948). *A mathematical theory of communication*. Bell System Technical Journal.
8. Cover, T. M., Thomas, J. A. (2006). *Elements of Information Theory* (ấn bản 2). Wiley.
9. Eckart, C., Young, G. (1936). *The approximation of one matrix by another of lower rank*. Psychometrika.
10. Oppenheim, A. V., Schafer, R. W. (2010). *Discrete-Time Signal Processing* (ấn bản 3). Pearson.
11. Shen, J. và cộng sự (2018). *Natural TTS synthesis by conditioning WaveNet on mel spectrogram predictions*. ICASSP.
12. Kong, J., Kim, J., Bae, J. (2020). *HiFi-GAN: Generative adversarial networks for efficient and high fidelity speech synthesis*. NeurIPS.
13. Kim, J., Kong, J., Son, J. (2021). *Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech*. ICML.
14. van den Oord, A. và cộng sự (2016). *WaveNet: A generative model for raw audio*. arXiv:1609.03499.
15. van den Oord, A. và cộng sự (2018). *Parallel WaveNet: Fast high-fidelity speech synthesis*. ICML.
16. Wang, C. và cộng sự (2023). *Neural codec language models are zero-shot text to speech synthesizers*. arXiv:2301.02111.
17. Hu, E. J. và cộng sự (2022). *LoRA: Low-rank adaptation of large language models*. ICLR.
