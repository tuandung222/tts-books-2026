# Chương 38. Đánh giá hệ thống TTS

!!! abstract "Mục tiêu học tập"
    Sau chương này, bạn có thể:

    - **Phân biệt** các chiều chất lượng của TTS (độ tự nhiên, độ dễ hiểu, độ giống người nói, ngôn điệu, độ bền, tốc độ) và chọn phép đo phù hợp cho từng chiều.
    - **Thiết kế** bài kiểm tra chủ quan MOS, CMOS, SMOS, AB/ABX và MUSHRA hợp lệ, kể cả khi dùng nền tảng đám đông (*crowdsourcing*).
    - **Tính** khoảng tin cậy cho MOS và **giải thích** vì sao khoảng tin cậy "ngây thơ" quá hẹp; **áp dụng** bootstrap hai chiều, kiểm định Wilcoxon và hiệu chỉnh so sánh bội.
    - **Suy ra** công thức MCD từ định lý Parseval và **cài đặt** MCD với DTW, các chỉ số F0, WER/CER kèm chuẩn hóa văn bản tiếng Việt.
    - **Đánh giá** giới hạn của các chỉ số tự động (UTMOS, DNSMOS, NISQA, PESQ, STOI, FAD).
    - **Xây dựng** bộ kiểm thử đặc thù tiếng Việt: cặp tối thiểu thanh điệu, đọc số và viết tắt, nhất quán phương ngữ.
    - **Báo cáo** kết quả đánh giá đầy đủ, minh bạch và có thể tái lập.

!!! info "Kiến thức cần có"
    - [Chương 1 – Toán nền tảng](../phan-1-nen-tang/ch01-toan-nen-tang.md): kỳ vọng, phương sai, phân phối chuẩn.
    - [Chương 14 – Xử lý tín hiệu số](../phan-3-tin-hieu-ngu-am/ch14-xu-ly-tin-hieu-so.md): mel, cepstrum/MFCC, trích F0.
    - [Chương 17 – Chuẩn hóa văn bản tiếng Việt](../phan-4-frontend/ch17-chuan-hoa-van-ban.md): chuẩn hóa số, ngày tháng, viết tắt.
    - [Chương 26 – Vocoder khuếch tán và so sánh](../phan-6-vocoder/ch26-diffusion-vocoder-va-so-sanh.md): đánh giá vocoder, RTF.
    - [Chương 28 – Đa người nói và nhân bản giọng](../phan-7-e2e-bieu-cam/ch28-da-nguoi-noi-voice-cloning.md): embedding người nói, SECS.

"Hệ thống của chúng tôi tốt hơn" là câu xuất hiện trong gần như mọi bài báo TTS. Chương này dạy bạn đọc câu đó một cách hoài nghi, và viết câu đó một cách trung thực. Đánh giá TTS khó vì chất lượng tiếng nói là khái niệm *cảm nhận*, nhiều chiều và phụ thuộc người nghe. Không có một con số nào nói lên tất cả; nhiệm vụ của người kỹ sư là chọn tập phép đo phù hợp với câu hỏi, thực hiện đúng cách, và báo cáo kèm độ bất định.

## 38.1 Đánh giá cái gì?

### 38.1.1 Các chiều chất lượng

| Chiều | Câu hỏi | Phép đo chủ quan | Đại lượng khách quan thay thế |
|---|---|---|---|
| Độ tự nhiên | Nghe có giống người thật không? | MOS, CMOS, MUSHRA | Bộ dự đoán MOS (UTMOS…), FAD |
| Độ dễ hiểu | Có nghe đúng nội dung không? | Bài chép chính tả, câu không đoán trước được về ngữ nghĩa (SUS, Benoît và cộng sự, 1996) | WER/CER qua ASR |
| Độ giống người nói | Có đúng giọng mục tiêu không? | SMOS, ABX | Cosine embedding người nói (SIM) |
| Ngôn điệu | Ngữ điệu, nhịp, trọng âm có phù hợp? | MOS ngôn điệu, CMOS | Sai số F0, trường độ; tương quan F0 |
| Độ bền | Có bỏ chữ, lặp chữ, ảo giác? | Đếm lỗi thủ công | WER trên câu khó, tỉ lệ câu thất bại |
| Hiệu năng | Có đủ nhanh, đủ nhẹ? | — | RTF, độ trễ đến âm thanh đầu tiên, bộ nhớ |

Hai khái niệm từ đo lường học giúp suy nghĩ rõ ràng: **độ hiệu lực** (*validity*) — phép đo có đo đúng thứ ta quan tâm không, và **độ tin cậy** (*reliability*) — lặp lại phép đo có cho kết quả giống nhau không. WER qua ASR có độ tin cậy cao nhưng độ hiệu lực hạn chế (ASR có thể "đoán đúng" một từ phát âm sai nhờ mô hình ngôn ngữ). MOS có độ hiệu lực cao nhưng độ tin cậy phụ thuộc mạnh vào thiết kế thí nghiệm.

### 38.1.2 Chủ quan và khách quan bổ sung cho nhau

Đánh giá chủ quan (người nghe) là chuẩn vàng nhưng đắt và chậm; đánh giá khách quan rẻ, nhanh, tái lập được, nhưng chỉ là đại lượng thay thế. Quy trình làm việc thực tế:

- **Trong quá trình phát triển**: dùng chỉ số khách quan (CER, SIM, bộ dự đoán MOS) để sàng lọc nhanh hàng chục checkpoint và cấu hình.
- **Trước khi ra quyết định quan trọng** (chọn mô hình triển khai, công bố kết quả): bài nghe được thiết kế cẩn thận trên các ứng viên đã sàng lọc.

## 38.2 Đánh giá chủ quan

### 38.2.1 MOS theo thang ACR

Phương pháp phổ biến nhất là **Mean Opinion Score** với thang phân loại tuyệt đối (*Absolute Category Rating*, ACR) theo khuyến nghị ITU-T P.800. Người nghe nghe từng mẫu và cho điểm từ 1 đến 5:

| Điểm | Nhãn (P.800, tiếng Anh) | Gợi ý nhãn tiếng Việt |
|---|---|---|
| 5 | Excellent | Xuất sắc |
| 4 | Good | Tốt |
| 3 | Fair | Trung bình |
| 2 | Poor | Kém |
| 1 | Bad | Rất kém |

Gọi $r_{ij}$ là điểm của người nghe $i$ cho mẫu $j$ của một hệ thống. MOS là trung bình cộng $\bar{r}$ trên mọi điểm. Một bài MOS tốt có các thành phần:

- **Câu hỏi rõ ràng**: "Mẫu này nghe tự nhiên đến mức nào?" khác với "Chất lượng âm thanh tốt đến mức nào?". Hướng dẫn ảnh hưởng mạnh đến điểm.
- **Mỏ neo** (*anchors*): luôn chèn bản ghi thật (thường được gọi là *ground truth* hoặc *recordings*) làm mỏ neo trên, và nếu có thể, một hệ thống chất lượng thấp đã biết làm mỏ neo dưới. MOS chỉ có nghĩa *tương đối* với các mẫu khác trong cùng bài; cùng một hệ thống có thể nhận MOS khác nhau đáng kể trong hai bài có tập so sánh khác nhau.
- **Ngẫu nhiên hóa thứ tự** trình bày, và cân bằng để mỗi người nghe nghe mỗi hệ thống với số lượng tương đương.
- **Mỗi người nghe chỉ nghe mỗi câu văn một lần** (với một hệ thống), để tránh so sánh trực tiếp và hiệu ứng "đã nghe nội dung này". Thiết kế kiểu ô vuông Latin (*Latin square*) đáp ứng điều này.

Kirkland và cộng sự (2023) phân tích nhiều yếu tố thiết kế MOS trong các bài báo TTS (cách đặt nhãn thang, hướng dẫn, mỏ neo…) và chỉ ra rằng chúng có thể làm điểm thay đổi đáng kể; bài học là luôn báo cáo đầy đủ thiết kế.

### 38.2.2 Khoảng tin cậy: vì sao công thức "sách giáo khoa" sai

Công thức quen thuộc cho khoảng tin cậy 95% của trung bình $n$ quan sát độc lập là

$$
\bar{r} \pm t_{0.975,\, n-1} \frac{s}{\sqrt{n}},
$$

với $s$ là độ lệch chuẩn mẫu. Nhưng các điểm MOS **không độc lập**: cùng một người nghe có xu hướng chấm cao hoặc thấp hơn người khác (độ lệch người nghe), và cùng một câu văn có thể khó hơn câu khác (độ khó câu). Mô hình hóa đơn giản:

$$
r_{ij} = \mu + b_i + c_j + \varepsilon_{ij}, \qquad b_i \sim \mathcal{N}(0, \sigma_b^2),\ c_j \sim \mathcal{N}(0, \sigma_c^2),\ \varepsilon_{ij} \sim \mathcal{N}(0, \sigma_\varepsilon^2).
$$

Ta muốn suy luận về $\mu$ — chất lượng của hệ thống với *người nghe mới và câu văn mới* từ cùng tổng thể — không chỉ với 20 người và 30 câu cụ thể đã dùng. Hãy tính phương sai của trung bình mẫu khi mỗi trong $L$ người nghe chấm tất cả $S$ câu:

$$
\operatorname{Var}(\bar{r}) = \frac{\sigma_b^2}{L} + \frac{\sigma_c^2}{S} + \frac{\sigma_\varepsilon^2}{LS}.
$$

Công thức ngây thơ, coi $n = LS$ điểm là độc lập, ước lượng xấp xỉ $(\sigma_b^2 + \sigma_c^2 + \sigma_\varepsilon^2)/(LS)$ — nhỏ hơn rất nhiều, vì số hạng $\sigma_b^2/L$ chỉ chia cho số *người nghe*, không phải số *điểm*. Một cách nhìn tương đương là **hệ số thiết kế** (*design effect*) của Kish: nếu mỗi người nghe cho $m$ điểm và tương quan nội nhóm giữa các điểm của cùng một người là $\rho$, thì số mẫu hiệu dụng là

$$
n_{\text{hiệu dụng}} = \frac{n}{1 + (m - 1)\rho}.
$$

Ví dụ $m = 30$, $\rho = 0{,}35$ cho hệ số thiết kế khoảng 11: 600 điểm chỉ có giá trị thống kê như khoảng 55 điểm độc lập, và khoảng tin cậy thật rộng gấp khoảng $\sqrt{11} \approx 3{,}3$ lần khoảng ngây thơ.

Hai cách xử lý đúng: (1) **bootstrap hai chiều** — lấy mẫu lại có hoàn lại *cả người nghe lẫn câu văn*, tính MOS trên ma trận con, lặp lại nhiều lần, lấy phân vị (Efron và Tibshirani, 1993); (2) **mô hình hiệu ứng hỗn hợp** (*linear mixed-effects model*) với hiệu ứng ngẫu nhiên cho người nghe và câu (ví dụ với gói lme4 của Bates và cộng sự, 2015). Mã dưới đây mô phỏng điểm MOS theo mô hình trên và *đo trực tiếp* độ phủ của hai loại khoảng tin cậy.

```python
import numpy as np
from scipy import stats

def simulate_mos(rng, n_listeners=20, n_sentences=30, mu=3.8,
                 sd_listener=0.5, sd_sentence=0.3, sd_noise=0.6):
    """Điểm MOS giả: r_ij = mu + độ lệch người nghe + độ khó câu + nhiễu,
    làm tròn và kẹp vào thang 1..5. Trả về ma trận (người nghe, câu)."""
    b = rng.normal(0, sd_listener, (n_listeners, 1))
    c = rng.normal(0, sd_sentence, (1, n_sentences))
    e = rng.normal(0, sd_noise, (n_listeners, n_sentences))
    return np.clip(np.round(mu + b + c + e), 1, 5)

def ci_t(r, level=0.95):
    """CI 'ngây thơ': coi mọi điểm là độc lập."""
    x = r.ravel(); n = len(x)
    h = stats.t.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return x.mean() - h, x.mean() + h

def ci_bootstrap_2way(r, rng, n_boot=500, level=0.95):
    """Bootstrap hai chiều: lấy mẫu lại CẢ người nghe lẫn câu (có hoàn lại)."""
    L, S = r.shape
    means = np.empty(n_boot)
    for k in range(n_boot):
        li = rng.integers(0, L, L); si = rng.integers(0, S, S)
        means[k] = r[np.ix_(li, si)].mean()
    a = (1 - level) / 2
    return tuple(np.quantile(means, [a, 1 - a]))

rng = np.random.default_rng(0)
r = simulate_mos(rng)
print(f"MOS = {r.mean():.2f}")
print("CI t (ngây thơ): [%.2f, %.2f]" % ci_t(r))
print("CI bootstrap 2 chiều: [%.2f, %.2f]" % ci_bootstrap_2way(r, rng))

# Kiểm tra độ phủ (coverage): CI 95% có chứa giá trị thật trong ~95% lần lặp?
true_mos = simulate_mos(np.random.default_rng(1), 3000, 3000).mean()   # xấp xỉ kỳ vọng thật
hit_t = hit_b = 0
n_rep = 100
for _ in range(n_rep):
    r = simulate_mos(rng)
    lo, hi = ci_t(r); hit_t += lo <= true_mos <= hi
    lo, hi = ci_bootstrap_2way(r, rng, n_boot=200); hit_b += lo <= true_mos <= hi
print(f"độ phủ thực tế: t ngây thơ {hit_t/n_rep:.0%}, bootstrap 2 chiều {hit_b/n_rep:.0%}")
assert hit_b > hit_t
```

Kết quả nói lên tất cả: khoảng tin cậy "ngây thơ" chỉ chứa giá trị thật trong khoảng 38% số lần lặp, thay vì 95% như nó tuyên bố. Bootstrap hai chiều cho độ phủ khoảng 97%: hơi thận trọng, vì với thiết kế chéo người nghe × câu, lấy mẫu lại cả hai chiều có xu hướng ước lượng dư phần phương sai nhiễu (Owen, 2007) — một sai lệch về phía an toàn, chấp nhận được trong thực hành.

**Cỡ mẫu.** Với độ lệch chuẩn của điểm xấp xỉ $\sigma$ và nửa độ rộng mong muốn $h$, cần $n \approx (1{,}96\,\sigma / h)^2$ điểm *độc lập*. Với $\sigma \approx 1$ và $h = 0{,}1$, đó là khoảng 384 điểm độc lập — rồi nhân với hệ số thiết kế. Hệ quả thực tế, phù hợp với phân tích của Wester và cộng sự (2015): **tăng số người nghe** hiệu quả hơn nhiều so với tăng số câu mỗi người nghe chấm, vì số hạng $\sigma_b^2/L$ chỉ giảm khi $L$ tăng.

### 38.2.3 CMOS, SMOS, AB và ABX

- **CMOS** (*Comparison MOS*, dựa trên thang CCR của P.800): người nghe nghe cặp (A, B) cùng câu văn và chấm từ −3 ("B kém hơn nhiều") đến +3 ("B tốt hơn nhiều"). So sánh trực tiếp nhạy hơn MOS tuyệt đối khi hai hệ thống gần nhau. Thứ tự A/B phải được ngẫu nhiên hóa và đổi dấu khi phân tích.
- **SMOS** (*Similarity MOS*): người nghe nghe một mẫu tham chiếu của người nói mục tiêu và một mẫu tổng hợp, chấm mức giống nhau về *giọng* (không phải nội dung hay chất lượng) trên thang 1–5. Hướng dẫn phải nhấn mạnh bỏ qua khác biệt nội dung.
- **AB preference**: chọn mẫu tốt hơn (có thể cho phép "không phân biệt được"). Với $n$ lượt chọn (bỏ qua lượt hòa), dưới giả thuyết không có khác biệt, số lượt chọn A tuân theo $\text{Binomial}(n, 0{,}5)$; dùng kiểm định nhị thức.
- **ABX**: nghe A, B và X, cho biết X giống A hay B hơn. Thường dùng để đo độ giống người nói hoặc kiểm tra khả năng phân biệt.

### 38.2.4 MUSHRA

**MUSHRA** (*MUltiple Stimuli with Hidden Reference and Anchor*, ITU-R BS.1534) trình bày *đồng thời* nhiều phiên bản của cùng một đoạn: một tham chiếu được công khai, một tham chiếu ẩn trong số các mẫu cần chấm, một hoặc vài mỏ neo chất lượng thấp (thường là bản lọc thông thấp), và các hệ thống cần so sánh. Người nghe chấm từng mẫu trên thang trượt 0–100 và có thể nghe lại tùy ý. Ưu điểm: phân biệt tốt các hệ thống chất lượng cao, vì người nghe được so sánh trực tiếp. Khuyến nghị bao gồm bước sàng lọc người nghe sau thí nghiệm: theo BS.1534, loại người nghe chấm tham chiếu ẩn dưới 90 điểm ở hơn 15% số mục (hãy kiểm tra phiên bản khuyến nghị bạn áp dụng). Trong TTS, nhiều nghiên cứu dùng biến thể "giống MUSHRA" không có tham chiếu công khai, vì không có "đáp án đúng" duy nhất cho ngôn điệu của một câu.

### 38.2.5 Người nghe và đánh giá qua nền tảng đám đông

- **Người nghe là người bản ngữ** của ngôn ngữ và, với tiếng Việt, cần cân nhắc **phương ngữ của người nghe**: người nghe miền Bắc có thể chấm thấp một giọng miền Nam chỉ vì khác phương ngữ chứ không vì kém tự nhiên. Ghi lại phương ngữ của người nghe và phân tích theo nhóm nếu cần.
- **Điều kiện nghe**: tai nghe, môi trường yên tĩnh; yêu cầu và (nếu có thể) kiểm tra bằng một bài nghe nhỏ đầu phiên.
- **Câu kiểm tra sự chú ý** (*gold/attention checks*): chèn mẫu bị làm hỏng rõ ràng (phải bị chấm thấp) và mẫu lặp lại (phải được chấm nhất quán). Loại người nghe không đạt theo quy tắc *đặt trước khi xem kết quả*.
- **Nền tảng đám đông**: ITU-T P.808 đưa ra quy trình cho đánh giá chất lượng tiếng nói bằng crowdsourcing; CrowdMOS (Ribeiro và cộng sự, 2011) là một trong những công trình sớm. Rủi ro gồm người làm vội, bot, thiết bị nghe kém; trả công hợp lý giúp cải thiện chất lượng.
- **Văn bản dài**: chấm câu đơn lẻ không phản ánh trải nghiệm nghe đoạn văn hay sách nói. Clark và cộng sự (2019) cho thấy điểm chấm theo câu và theo đoạn có thể khác nhau; với ứng dụng sách nói, hãy đánh giá cả trên đoạn văn.

## 38.3 Thống kê cho so sánh hệ thống

Khi so sánh hệ thống A và B, câu hỏi là: khác biệt quan sát được có thể chỉ do ngẫu nhiên không?

- **So sánh cặp** khi có thể: nếu cùng người nghe chấm cả A và B (hoặc cùng câu được tổng hợp bởi A và B), hãy phân tích *hiệu* theo người nghe hoặc theo câu. Ghép cặp loại bỏ độ lệch người nghe khỏi phép so sánh, tăng mạnh độ nhạy.
- **Kiểm định**: $t$-test cặp giả định hiệu có phân phối gần chuẩn; **Wilcoxon dấu-hạng** (Wilcoxon, 1945) chỉ dùng thứ hạng, phù hợp hơn với điểm MOS vốn là thang thứ bậc; với hai nhóm độc lập, dùng Mann–Whitney U. Với dữ liệu phức tạp (nhiều người nghe, nhiều câu, thiết kế không cân bằng), mô hình hiệu ứng hỗn hợp là lựa chọn nguyên tắc nhất.
- **So sánh bội**: so sánh hệ thống cơ sở với $m$ hệ thống khác bằng $m$ kiểm định ở mức 0,05 khiến xác suất có ít nhất một dương tính giả tăng lên tới xấp xỉ $1 - 0{,}95^m$ (khoảng 23% với $m = 5$ nếu các kiểm định độc lập). Hiệu chỉnh **Holm** (Holm, 1979) kiểm soát xác suất này (FWER) và luôn mạnh hơn Bonferroni; **Benjamini–Hochberg** kiểm soát tỉ lệ phát hiện sai (FDR) khi có rất nhiều so sánh.
- **Ý nghĩa thống kê khác ý nghĩa thực tế**: với đủ nhiều người nghe, chênh lệch MOS 0,03 cũng có thể "có ý nghĩa thống kê" nhưng không ai nghe ra. Báo cáo cả độ lớn hiệu ứng và khoảng tin cậy của nó.

Mã dưới đây so sánh một hệ thống cơ sở với ba hệ thống khác bằng Wilcoxon trên điểm trung bình theo người nghe (thiết kế cặp), rồi hiệu chỉnh Holm.

```python
import numpy as np
from scipy import stats

def holm(pvals):
    """Hiệu chỉnh Holm–Bonferroni: trả về p đã hiệu chỉnh (kiểm soát FWER)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])            # bảo toàn tính đơn điệu
        adj[i] = min(1.0, running)
    return adj

rng = np.random.default_rng(1)
n_listeners = 24
listener_bias = rng.normal(0, 0.5, n_listeners)
true_quality = {"A (cơ sở)": 3.6, "B": 3.65, "C": 3.95, "D": 3.6}
# Mỗi người nghe chấm 20 câu của mỗi hệ thống; ta dùng điểm TRUNG BÌNH theo người nghe
scores = {name: np.clip(q + listener_bias[:, None] + rng.normal(0, 0.7, (n_listeners, 20)),
                        1, 5).mean(axis=1)
          for name, q in true_quality.items()}                # mỗi giá trị: (n_listeners,)

base = scores["A (cơ sở)"]
names, pvals = [], []
for name in ["B", "C", "D"]:
    diff = scores[name] - base                                # so sánh CẶP theo người nghe
    p = stats.wilcoxon(diff).pvalue                           # Wilcoxon dấu-hạng
    names.append(name); pvals.append(p)
    print(f"{name} - A: chênh lệch trung bình {diff.mean():+.3f}, p thô = {p:.4f}")

for name, p_adj in zip(names, holm(pvals)):
    verdict = "khác biệt có ý nghĩa" if p_adj < 0.05 else "chưa đủ bằng chứng"
    print(f"{name}: p Holm = {p_adj:.4f} -> {verdict}")

assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
assert holm(pvals)[names.index("C")] < 0.05
```

Chỉ hệ thống C (có chất lượng thật cao hơn 0,35 điểm) được kết luận là khác biệt. Thử đổi hạt giống vài lần: đôi khi hệ thống D (chất lượng thật *bằng* A) cũng "có ý nghĩa" — đó chính là dương tính giả mà mức ý nghĩa 5% cho phép, và là lý do không nên tin một kết quả biên duy nhất.

## 38.4 Chỉ số khách quan

### 38.4.1 Mel-cepstral distortion (MCD) với DTW

**Trực giác.** MCD (Kubichek, 1993) đo khoảng cách giữa *đường bao phổ* của tiếng nói tổng hợp và tiếng nói tham chiếu, khung theo khung. Cepstrum bậc thấp mô tả đường bao phổ (âm sắc, formant) mà bỏ qua cấu trúc hài của F0, nên MCD chủ yếu phản ánh sai khác về âm sắc.

**Hình thức hóa.** Với vector mel-cepstrum $\mathbf{c}_t = (c_{t,1}, \ldots, c_{t,D})$ của khung tham chiếu và $\hat{\mathbf{c}}_t$ của khung tổng hợp (bỏ $c_0$ vì nó chỉ phản ánh năng lượng tổng), MCD của một khung là

$$
\mathrm{MCD}_t = \frac{10}{\ln 10} \sqrt{2 \sum_{d=1}^{D} \left(c_{t,d} - \hat{c}_{t,d}\right)^2} \quad \text{(dB)},
$$

và MCD của một câu là trung bình trên các khung đã được căn chỉnh.

**Các hằng số đến từ đâu?** Giả sử $c_n$ là hệ số cepstrum của **log phổ công suất** (logarit tự nhiên), tức là

$$
\ln P(\omega) = \sum_{n=-\infty}^{\infty} c_n e^{-j\omega n} = c_0 + 2\sum_{n=1}^{\infty} c_n \cos(n\omega), \qquad c_{-n} = c_n.
$$

Gọi $\Delta(\omega) = \ln P(\omega) - \ln \hat{P}(\omega)$ và $\Delta c_n = c_n - \hat{c}_n$. Theo định lý Parseval cho chuỗi Fourier,

$$
\frac{1}{2\pi}\int_{-\pi}^{\pi} \Delta(\omega)^2 \, d\omega = \sum_{n=-\infty}^{\infty} \Delta c_n^2 = \Delta c_0^2 + 2\sum_{n=1}^{\infty} \Delta c_n^2.
$$

Đổi sang decibel: hiệu log công suất tính bằng dB là $10\log_{10}(P/\hat{P}) = \frac{10}{\ln 10}\Delta(\omega)$. Vậy căn bậc hai của vế trái nhân $\frac{10}{\ln 10}$ là **sai khác trung bình bình phương (RMS) giữa hai log phổ, tính bằng dB**. Bỏ $\Delta c_0$ (loại bỏ khác biệt độ to tổng) và cắt chuỗi ở $D$ hệ số (chỉ so đường bao đã làm mượt), ta thu được đúng công thức MCD. Như vậy MCD xấp xỉ khoảng cách RMS (dB) giữa hai đường bao phổ đã trừ trung bình, trên thang tần số mel.

Lập luận này cũng giải thích vì sao **MCD giữa các bài báo thường không so sánh được**: các bộ công cụ khác nhau tính cepstrum từ log biên độ hay log công suất, theo phân tích mel-cepstral (như SPTK) hay theo kiểu MFCC từ bộ lọc mel, với $D$ khác nhau, có hay không loại khung im lặng — mỗi lựa chọn đổi giá trị tuyệt đối. Chỉ so MCD giữa các hệ thống được tính bằng *cùng một* đoạn mã.

**Căn chỉnh thời gian.** Tiếng nói tổng hợp và bản ghi tham chiếu có độ dài khác nhau, nên cần căn chỉnh động theo thời gian (*dynamic time warping*, DTW). Với ma trận khoảng cách $d(i, j)$ giữa khung $i$ của tham chiếu và khung $j$ của tổng hợp, chi phí tích lũy là

$$
D(i, j) = d(i, j) + \min\big\{D(i-1, j),\ D(i, j-1),\ D(i-1, j-1)\big\}, \qquad D(0,0) = 0,
$$

và đường đi tối ưu được truy vết ngược từ $D(T_x, T_y)$. Mã dưới đây tính cepstrum từ log-mel công suất với chuẩn hóa đúng như công thức trên (chia DCT-II không chuẩn hóa cho $2M$, $M$ là số kênh mel), rồi tính MCD trên đường DTW.

```python
import numpy as np
import librosa
from scipy.fft import dct

def mel_cepstrum(s, sr, n_mels=80, n_ceps=13, n_fft=1024, hop=256):
    """Cepstrum của log-mel CÔNG SUẤT, chuẩn hóa sao cho
    ln P(ω_m) ≈ c_0 + 2 Σ_n c_n cos(n ω_m); bỏ c0 (năng lượng). Trả về (T, n_ceps)."""
    mel = librosa.feature.melspectrogram(y=s, sr=sr, n_fft=n_fft, hop_length=hop,
                                         n_mels=n_mels, power=2.0)      # (n_mels, T)
    c = dct(np.log(np.maximum(mel, 1e-5)), type=2, axis=0) / (2 * n_mels)  # (n_mels, T)
    return c[1:n_ceps + 1].T

def dtw_path(X, Y):
    """DTW cổ điển với khoảng cách Euclid giữa các khung. Trả về đường đi (i, j)."""
    d = np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1))         # (Tx, Ty)
    Tx, Ty = d.shape
    D = np.full((Tx + 1, Ty + 1), np.inf); D[0, 0] = 0.0
    for i in range(1, Tx + 1):
        for j in range(1, Ty + 1):
            D[i, j] = d[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j, path = Tx, Ty, []
    while i > 0 and j > 0:                                               # truy vết ngược
        path.append((i - 1, j - 1))
        k = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        i, j = (i - 1, j - 1) if k == 0 else ((i - 1, j) if k == 1 else (i, j - 1))
    return path[::-1]

def mcd_dtw(ref, syn, sr):
    """MCD (dB) trung bình trên đường DTW: (10/ln10) * sqrt(2 * sum_d (c_d - ĉ_d)^2)."""
    X, Y = mel_cepstrum(ref, sr), mel_cepstrum(syn, sr)
    path = dtw_path(X, Y)
    diff = np.array([X[i] - Y[j] for i, j in path])                     # (len(path), D)
    return float(np.mean(10 / np.log(10) * np.sqrt(2 * (diff ** 2).sum(axis=1))))

# Tín hiệu thử: nguyên âm tổng hợp có F0 trượt 120 -> 180 Hz
sr = 16000
t = np.arange(int(0.6 * sr)) / sr
f0 = 120 + 100 * t
phase = 2 * np.pi * np.cumsum(f0) / sr
rng = np.random.default_rng(0)
floor = lambda n: 1e-3 * rng.standard_normal(n)                         # nền nhiễu ~ -60 dB
ref = sum(np.sin(k * phase) / k for k in range(1, 20)) * np.hanning(len(t)) + floor(len(t))
delayed = np.concatenate([floor(int(0.1 * sr)), ref])                   # trễ 100 ms
noisy = ref + 0.02 * rng.standard_normal(len(ref))                      # thêm nhiễu trắng

print(f"MCD(ref, ref)     = {mcd_dtw(ref, ref, sr):.2f} dB")
print(f"MCD(ref, delayed) = {mcd_dtw(ref, delayed, sr):.2f} dB  (DTW bù được độ trễ)")
print(f"MCD(ref, noisy)   = {mcd_dtw(ref, noisy, sr):.2f} dB")
assert mcd_dtw(ref, ref, sr) < 1e-6
assert mcd_dtw(ref, delayed, sr) < mcd_dtw(ref, noisy, sr)
```

Một vài lưu ý thực hành: (1) DTW "bù" được khác biệt thời gian, nên MCD gần như **mù với lỗi trường độ và nhịp**; (2) khung im lặng chiếm phần lớn có thể làm MCD đẹp hoặc xấu giả tạo — nhiều công cụ loại khung im lặng trước khi tính; (3) với vocoder nơ-ron hiện đại, MCD tương quan kém với cảm nhận của người nghe, nên hãy coi nó là chỉ số chẩn đoán âm sắc chứ không phải thước đo chất lượng tổng thể.

### 38.4.2 Các chỉ số F0

Với hai đường F0 đã căn chỉnh (cùng đường DTW ở trên, hoặc căn chỉnh theo trường độ), quy ước $f_0 = 0$ cho khung vô thanh:

- **Lỗi hữu thanh/vô thanh** (*V/UV error*): tỉ lệ khung có quyết định hữu thanh khác nhau.
- **RMSE F0** trên các khung hữu thanh ở cả hai, nên tính theo **cent** ($1200 \log_2(\hat{f}_0 / f_0)$) thay vì Hz để không phụ thuộc giọng cao hay thấp.
- **Tương quan F0** (Pearson): đo *hình dạng* đường nét, bỏ qua độ lệch và tỉ lệ.
- **GPE** (*gross pitch error*): tỉ lệ khung hữu thanh có sai lệch tương đối trên 20%.
- **FFE** (*F0 frame error*, Chu và Alwan, 2009): tỉ lệ khung có lỗi V/UV hoặc lỗi thô, gộp hai loại lỗi thành một số.

```python
import numpy as np

def f0_metrics(f0_ref, f0_syn):
    """Các chỉ số F0 trên hai chuỗi ĐÃ căn chỉnh (cùng độ dài); 0 = khung vô thanh."""
    v_r, v_s = f0_ref > 0, f0_syn > 0
    both = v_r & v_s
    cents = 1200 * np.log2(f0_syn[both] / f0_ref[both])          # sai lệch theo cent
    gross = np.abs(f0_syn[both] / f0_ref[both] - 1) > 0.2         # lệch > 20%
    return {
        "vuv_error": float(np.mean(v_r != v_s)),                   # sai quyết định hữu/vô thanh
        "f0_rmse_cent": float(np.sqrt(np.mean(cents ** 2))),
        "f0_corr": float(np.corrcoef(f0_ref[both], f0_syn[both])[0, 1]),
        "gpe": float(np.mean(gross)),                              # gross pitch error
        "ffe": float((np.sum(v_r != v_s) + np.sum(gross)) / len(f0_ref)),  # F0 frame error
    }

# Ví dụ: đường F0 đi lên (kiểu thanh sắc) và ba kiểu lỗi của bản tổng hợp
T = 50
rise = np.linspace(180, 260, T)                    # tham chiếu (Hz)
compressed = 220 + 0.3 * (rise - 220)              # đúng hướng nhưng bị "nén" biên độ
falling = rise[::-1].copy()                        # sai hướng: đi xuống
octave = rise.copy(); octave[20:25] /= 2           # lỗi quãng tám (thường do bộ trích F0)
for name, syn in [("nén", compressed), ("đi xuống", falling), ("quãng tám", octave)]:
    print(f"{name:10s}", {k: round(v, 3) for k, v in f0_metrics(rise, syn).items()})

m = f0_metrics(rise, compressed)
assert m["f0_corr"] > 0.999 and m["f0_rmse_cent"] > 50   # tương quan "hoàn hảo" vẫn có thể sai
assert f0_metrics(rise, falling)["f0_corr"] < -0.999
assert f0_metrics(rise, octave)["gpe"] == 5 / 50
```

Ví dụ minh họa vì sao cần nhiều chỉ số: đường nét bị "nén" (đúng hướng nhưng biên độ nhỏ — một kiểu lỗi hay gặp khi mô hình làm mượt quá mức) có tương quan **bằng 1** nhưng RMSE hơn 130 cent; còn vài khung lỗi quãng tám (thường do bộ trích F0, không phải do mô hình) làm RMSE vọt lên trong khi GPE chỉ 10%. Vì vậy hãy báo cáo GPE riêng và tính RMSE trên các khung không có lỗi thô.

!!! note "Góc tiếng Việt"
    Với ngôn ngữ thanh điệu, chỉ số F0 trung bình trên toàn câu che giấu điều quan trọng nhất: **hình dạng đường nét trong từng âm tiết**. Một cách đánh giá hữu ích hơn là căn chỉnh theo âm tiết (dùng forced alignment, [Chương 21](../phan-5-mo-hinh-am-hoc/ch21-alignment.md)), chuẩn hóa F0 theo người nói (z-score trên log-F0), lấy mẫu lại mỗi âm tiết thành một số điểm cố định (ví dụ 10 điểm), rồi so sánh đường nét theo từng loại thanh. Lưu ý rằng thanh ngã và nặng thường có đoạn thanh quản hóa, nơi bộ trích F0 dễ báo vô thanh hoặc lỗi quãng tám ([Chương 15](../phan-3-tin-hieu-ngu-am/ch15-ngu-am-tieng-viet.md)); đừng nhầm lỗi của bộ trích F0 với lỗi của mô hình.

### 38.4.3 Độ dễ hiểu: WER/CER qua ASR

Quy trình chuẩn: tổng hợp câu → nhận dạng bằng một hệ ASR mạnh → so với văn bản đầu vào bằng khoảng cách chỉnh sửa (Levenshtein). Với $S$ phép thay thế, $D$ phép xóa, $I$ phép chèn trên đường căn chỉnh tối ưu và $N$ token tham chiếu,

$$
\mathrm{WER} = \frac{S + D + I}{N}.
$$

Khoảng cách chỉnh sửa được tính bằng quy hoạch động: $E(i, j) = \min\{E(i-1, j) + 1,\ E(i, j-1) + 1,\ E(i-1, j-1) + \mathbb{1}[x_i \ne y_j]\}$, cùng cấu trúc với DTW ở trên. Lưu ý WER có thể vượt 100% khi có nhiều phép chèn (ví dụ mô hình ảo giác sinh thêm câu).

**Với tiếng Việt**, chữ viết tách *âm tiết* bằng khoảng trắng, nên "WER" tính trên các token cách nhau bởi khoảng trắng thực chất là **tỉ lệ lỗi âm tiết**. CER (trên ký tự, có dấu) nhạy hơn với lỗi nhỏ. Một chẩn đoán đặc biệt hữu ích cho tiếng Việt là tách **lỗi thanh điệu** khỏi lỗi phân đoạn:

- **CER bỏ thanh**: bỏ các dấu thanh (giữ dấu mũ, dấu móc, dấu trăng) rồi tính CER. Nếu CER cao nhưng CER bỏ thanh thấp, lỗi chủ yếu là thanh điệu.
- **Tỉ lệ lỗi thanh**: trong các cặp âm tiết được căn chỉnh có *cùng phần không thanh*, tỉ lệ cặp khác thanh.

**Chuẩn hóa trước khi chấm** quan trọng không kém bản thân ASR. Văn bản tham chiếu và đầu ra ASR phải được đưa về cùng dạng: NFC, chữ thường, bỏ dấu câu, thống nhất vị trí dấu (*hoà* → *hòa*), và **chuẩn hóa số/viết tắt** theo cùng một quy tắc (ASR có thể viết "2024" trong khi văn bản tham chiếu viết "hai nghìn không trăm hai mươi tư"). Bộ chuẩn hóa của Whisper được thiết kế cho tiếng Anh; với tiếng Việt, hãy dùng bộ chuẩn hóa văn bản của [Chương 17](../phan-4-frontend/ch17-chuan-hoa-van-ban.md).

```python
import re
import unicodedata

TONES = "".join(map(chr, (0x300, 0x301, 0x303, 0x309, 0x323)))  # huyền, sắc, ngã, hỏi, nặng (NFD)

def fix_tone_placement(syl):
    """Thống nhất kiểu bỏ dấu 'hòa, khỏe, thủy' (dấu trên nguyên âm đầu của oa/oe/uy
    khi không có âm cuối). 'quý' giữ nguyên vì 'qu' là âm đầu."""
    nfd = unicodedata.normalize("NFD", syl)
    tone = [c for c in nfd if c in TONES]
    base = "".join(c for c in nfd if c not in TONES)
    if tone and re.search(r"(?<!q)(oa|oe|uy)$", base):
        base = base[:-1] + tone[0] + base[-1]      # gắn dấu sau nguyên âm áp chót
        return unicodedata.normalize("NFC", base)
    return unicodedata.normalize("NFC", syl)

def normalize_vi(text):
    """Chuẩn hóa TRƯỚC khi chấm: NFC, chữ thường, bỏ dấu câu, thống nhất vị trí dấu.
    (Số, ngày tháng, viết tắt: hãy áp dụng CÙNG bộ chuẩn hóa văn bản của Chương 17.)"""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(fix_tone_placement(w) for w in text.split())

def strip_tones(text):
    nfd = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", "".join(c for c in nfd if c not in TONES))

def align(ref, hyp):
    """Khoảng cách Levenshtein + truy vết: danh sách (phép, ref_tok, hyp_tok)."""
    R, H = len(ref), len(hyp)
    D = [[i + j if i * j == 0 else 0 for j in range(H + 1)] for i in range(R + 1)]
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            D[i][j] = min(D[i-1][j] + 1, D[i][j-1] + 1, D[i-1][j-1] + (ref[i-1] != hyp[j-1]))
    ops, i, j = [], R, H
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i-1][j-1] + (ref[i-1] != hyp[j-1]):
            ops.append(("=" if ref[i-1] == hyp[j-1] else "S", ref[i-1], hyp[j-1])); i, j = i-1, j-1
        elif i > 0 and D[i][j] == D[i-1][j] + 1:
            ops.append(("D", ref[i-1], None)); i -= 1
        else:
            ops.append(("I", None, hyp[j-1])); j -= 1
    return ops[::-1]

def error_rate(ref_toks, hyp_toks):
    ops = align(ref_toks, hyp_toks)
    return sum(op != "=" for op, _, _ in ops) / max(1, len(ref_toks))

def evaluate(ref, hyp):
    r, h = normalize_vi(ref), normalize_vi(hyp)
    syl_ops = align(r.split(), h.split())
    same_base = [(a, b) for op, a, b in syl_ops if op in "=S" and strip_tones(a) == strip_tones(b)]
    return {"WER(âm tiết)": error_rate(r.split(), h.split()),
            "CER": error_rate(list(r.replace(" ", "")), list(h.replace(" ", ""))),
            "CER bỏ thanh": error_rate(list(strip_tones(r).replace(" ", "")),
                                       list(strip_tones(h).replace(" ", ""))),
            "lỗi thanh điệu": sum(a != b for a, b in same_base) / max(1, len(same_base))}

ref = "Hôm nay, trời hoà dịu; chúng tôi đi thăm Huế."
hyp = "hôm nay trời hòa dìu chúng tôi đi thăm huệ"    # 2 âm tiết sai THANH, đúng phần còn lại
res = evaluate(ref, hyp)
print({k: round(v, 3) for k, v in res.items()})
assert normalize_vi("Hoà THUỶ quý!") == "hòa thủy quý"
assert res["CER bỏ thanh"] == 0.0 and abs(res["WER(âm tiết)"] - 2 / 10) < 1e-9
```

**Cái bẫy của đánh giá bằng ASR:**

- **Sàn lỗi của ASR**: luôn báo cáo WER của chính ASR trên **bản ghi thật** của cùng tập câu. Nếu ASR có WER 5% trên tiếng nói thật, WER 6% của TTS không có nghĩa là TTS "sai 6%".
- **ASR "sửa giúp"**: mô hình ngôn ngữ bên trong ASR có thể đoán đúng một từ bị phát âm sai, nhất là với cặp tối thiểu thanh điệu trong ngữ cảnh gợi ý mạnh. Dùng câu khung trung tính khi kiểm tra thanh điệu (Mục 38.5).
- **Chọn ASR**: với tiếng Việt, các lựa chọn phổ biến tính đến thời điểm biên soạn gồm Whisper đa ngôn ngữ (Radford và cộng sự, 2023) và các bản tinh chỉnh cho tiếng Việt như PhoWhisper (Le và cộng sự, 2024). Hãy ghi rõ tên và phiên bản mô hình, vì WER giữa các ASR khác nhau không so sánh được.
- **ASR nhạy với phương ngữ**: một ASR huấn luyện chủ yếu trên giọng Bắc có thể cho WER cao hơn với giọng Nam dù TTS hoàn toàn đúng.

### 38.4.4 Độ giống người nói

Độ giống người nói (SIM, còn gọi SECS) được đo bằng cosine giữa embedding người nói của mẫu tổng hợp và của bản ghi tham chiếu, dùng một mô hình xác minh người nói như ECAPA-TDNN (Desplanques và cộng sự, 2020) hoặc mô hình dựa trên WavLM (Chen và cộng sự, 2022) được tinh chỉnh cho xác minh. Cách tính và cài đặt có ở [Chương 28](../phan-7-e2e-bieu-cam/ch28-da-nguoi-noi-voice-cloning.md). Ba lưu ý:

- **Giá trị SIM chỉ so sánh được khi dùng cùng mô hình embedding.** Một bài báo dùng mô hình A báo SIM 0,65 và bài khác dùng mô hình B báo 0,60 không cho biết hệ thống nào tốt hơn.
- **So với cái gì**: một số công trình (ví dụ Voicebox, Le và cộng sự, 2023) phân biệt độ giống so với *bản ghi gốc* và so với *bản ghi đã qua vocoder*; cách thứ hai loại bỏ ảnh hưởng của vocoder. Hãy ghi rõ cách bạn dùng.
- SIM cao không bảo đảm người nghe thấy giống; với kết luận quan trọng, bổ sung SMOS.

### 38.4.5 Bộ dự đoán MOS tự động

Các mô hình học dự đoán MOS từ âm thanh không cần tham chiếu: MOSNet (Lo và cộng sự, 2019), DNSMOS (Reddy và cộng sự, 2021), NISQA (Mittag và cộng sự, 2021 — dự đoán cả các chiều như nhiễu, méo màu, gián đoạn), UTMOS (Saeki và cộng sự, 2022 — hệ thống tham gia VoiceMOS Challenge 2022). Chúng rất hữu ích để sàng lọc và theo dõi, nhưng có giới hạn rõ ràng:

- **Lệch miền**: được huấn luyện trên một số bộ dữ liệu, ngôn ngữ và loại hệ thống nhất định; độ chính xác giảm khi gặp hệ thống mới, ngôn ngữ mới hoặc kiểu lỗi mới. Kết quả của VoiceMOS Challenge (Huang và cộng sự, 2022) cho thấy khả năng tổng quát hóa ra ngoài miền là thách thức chính.
- **Mù nội dung**: một câu đọc sai chữ, sai thanh nhưng nghe "mượt" vẫn có thể được chấm cao.
- **Khó xếp hạng các hệ thống tốt**: độ phân giải ở vùng chất lượng cao thường kém.
- **Có thể bị "khai thác"**: tối ưu trực tiếp theo một bộ dự đoán MOS (ví dụ làm tín hiệu thưởng) có thể làm tăng điểm mà không tăng chất lượng thật.

Với tiếng Việt, đừng mặc định một bộ dự đoán MOS đáng tin. Hãy **hiệu chuẩn**: chạy một bài nghe nhỏ, rồi đo tương quan hạng (Spearman hoặc Kendall) giữa điểm dự đoán và MOS thật, ở mức hệ thống và mức câu.

### 38.4.6 PESQ, STOI và FAD

- **PESQ** (ITU-T P.862; Rix và cộng sự, 2001) và **STOI** (Taal và cộng sự, 2011) là chỉ số *có tham chiếu*, được thiết kế cho viễn thông và tăng cường tiếng nói, giả định tín hiệu thử là phiên bản *bị suy giảm* của cùng một tín hiệu tham chiếu, căn chỉnh theo thời gian. TTS không thỏa giả định này (ngôn điệu, trường độ khác bản ghi thật), nên dùng PESQ/STOI cho TTS là sai mục đích. Chúng hợp lý hơn cho **copy synthesis của vocoder** hoặc **codec** ([Chương 31](../phan-8-mo-hinh-lon/ch31-audio-codec-speech-token.md)), với lưu ý PESQ chỉ hoạt động ở 8 kHz/16 kHz.
- **FAD** (*Fréchet Audio Distance*, Kilgour và cộng sự, 2019) so sánh *phân phối* embedding của tập âm thanh tổng hợp và tập âm thanh thật. Khớp mỗi tập bằng một Gaussian $(\boldsymbol{\mu}_r, \boldsymbol{\Sigma}_r)$ và $(\boldsymbol{\mu}_g, \boldsymbol{\Sigma}_g)$:

$$
\mathrm{FAD} = \lVert \boldsymbol{\mu}_r - \boldsymbol{\mu}_g \rVert_2^2 + \operatorname{tr}\!\Big(\boldsymbol{\Sigma}_r + \boldsymbol{\Sigma}_g - 2\big(\boldsymbol{\Sigma}_r \boldsymbol{\Sigma}_g\big)^{1/2}\Big).
$$

  FAD không cần cặp tham chiếu, phù hợp đánh giá ở mức tập, nhưng phụ thuộc mạnh vào mô hình embedding và **bị chệch theo cỡ mẫu** (tập nhỏ cho FAD lớn hơn), nên chỉ so sánh FAD tính với cùng embedding và cùng số mẫu.

## 38.5 Bộ dữ liệu đánh giá

### 38.5.1 Các bộ đánh giá phổ biến

- **Một người nói**: tập kiểm tra giữ riêng của chính bộ dữ liệu (ví dụ vài trăm câu LJSpeech không dùng khi huấn luyện).
- **Zero-shot tiếng Anh**: tập con của LibriSpeech test-clean (VALL-E dùng các câu dài 4–10 giây), tập kiểm tra của LibriTTS.
- **Seed-TTS-eval** (Anastassiou và cộng sự, 2024): theo mô tả của nhóm tác giả, gồm tập tiếng Anh lấy từ Common Voice, tập tiếng Trung, và một tập "khó" với câu lặp từ, câu líu lưỡi; được nhiều mô hình zero-shot dùng làm chuẩn so sánh, kèm quy trình tính WER và SIM.
- **Câu khó tự xây**: câu rất dài, câu lặp từ, tên riêng hiếm, số liệu, trộn mã, câu hỏi.

Tính đến thời điểm biên soạn (2026), theo hiểu biết của chúng tôi, tiếng Việt **chưa có một bộ đánh giá TTS chuẩn được cộng đồng chấp nhận rộng rãi**. Tập kiểm tra người nói đã thấy/chưa thấy của PhoAudiobook ([Chương 36](ch36-du-lieu-tts.md)) là một lựa chọn cho nghiên cứu, theo điều khoản của bộ dữ liệu. Với sản phẩm, bạn nên tự xây bộ kiểm thử theo mục tiếp theo.

### 38.5.2 Bộ kiểm thử tiếng Việt

Một bộ kiểm thử tiếng Việt tốt kiểm tra những điểm mà chỉ số trung bình bỏ sót:

| Nhóm | Mục đích | Ví dụ |
|---|---|---|
| Cặp tối thiểu thanh điệu | Phân biệt đủ 6 thanh | *ma, mà, má, mả, mã, mạ*; *bàn, bán, bạn*; *nghĩ, nghỉ*; *cổ, cỗ* |
| Thanh trong ngữ cảnh | Chuyển tiếp thanh liền kề | chuỗi nhiều thanh ngã/hỏi/nặng liên tiếp |
| Số và đơn vị | Chuẩn hóa văn bản | *21* (hai mươi mốt), *15* (mười lăm), *105* (một trăm linh/lẻ năm), *1.250.000 đ*, *25%* |
| Ngày giờ | Mơ hồ định dạng | *2/9*, *9h30*, *thế kỷ XXI* |
| Viết tắt | Đọc theo chữ hay theo từ | *TP.HCM*, *UBND*, *PGS.TS.*, *COVID-19* |
| Từ mượn, trộn mã | Chính sách phát âm | *email, Facebook, deadline* trong câu tiếng Việt |
| Ngữ điệu | Câu hỏi, tiểu từ | *Anh đi à?*, *Chị nghỉ đi nhé.* |
| Phương ngữ | Nhất quán phương ngữ mục tiêu | từ có *tr/ch*, *s/x*, *r/d/gi*, âm cuối *-n/-ng*, *-t/-c* |
| Văn bản dài | Độ ổn định | đoạn văn 5–10 câu |

**Kiểm tra thanh điệu.** Đặt từ cần thử vào **câu khung trung tính** (*carrier sentence*), ví dụ "Tôi nói chữ ___ một lần nữa", để vừa tránh đọc theo kiểu từ đơn lẻ, vừa không để ngữ cảnh gợi ý từ đúng cho ASR. Có ba cách chấm, theo thứ tự độ tin cậy giảm dần:

1. **Người nghe nhận diện cưỡng bức**: nghe câu và chọn từ nghe được trong 6 phương án khác thanh. Tỉ lệ đúng và *ma trận nhầm lẫn* thanh (ví dụ hỏi bị nghe thành ngã bao nhiêu lần) là kết quả rất giàu thông tin.
2. **Tỉ lệ lỗi thanh qua ASR**, như hàm `evaluate` ở trên — rẻ, nhưng chịu ảnh hưởng của mô hình ngôn ngữ trong ASR.
3. **Bộ phân loại thanh điệu** trên đường F0 của âm tiết — cần được kiểm định trước trên tiếng nói thật.

**Kiểm tra đọc số và viết tắt.** Với mỗi câu, liệt kê *các* cách đọc chấp nhận được (ví dụ *nghìn/ngàn*, *linh/lẻ*, *hai mươi lăm/hai mươi nhăm* tùy phương ngữ và quy ước). Chấm đúng/sai thủ công hoặc bằng ASR + so khớp với danh sách cách đọc. Bài kiểm tra này thực chất kiểm tra cả front-end; hãy chạy riêng front-end trên cùng bộ câu để biết lỗi nằm ở front-end hay ở mô hình.

**Kiểm tra phương ngữ.** Nhờ người nghe thuộc phương ngữ mục tiêu đánh giá "giọng có nhất quán là giọng [Bắc/Nam] không", và ghi lại các âm tiết bị đọc theo phương ngữ khác.

## 38.6 Đo tốc độ và độ trễ

Chi tiết đo hiệu năng triển khai nằm ở [Chương 39](ch39-trien-khai-production.md) và hàm đo RTF ở [Chương 26](../phan-6-vocoder/ch26-diffusion-vocoder-va-so-sanh.md). Khi báo cáo trong bối cảnh đánh giá:

- **Hệ số thời gian thực** $\mathrm{RTF} = t_{\text{tổng hợp}} / t_{\text{âm thanh}}$; RTF $< 1$ nghĩa là nhanh hơn thời gian thực.
- **Độ trễ đến âm thanh đầu tiên** (*time to first audio*) cho hệ thống streaming ([Chương 35](../phan-8-mo-hinh-lon/ch35-streaming-speech-llm.md)).
- **Quy trình đo**: chạy khởi động (*warm-up*) trước; với GPU, đồng bộ (`torch.cuda.synchronize()`) trước khi bấm giờ; báo cáo phần cứng, độ chính xác số (fp32/fp16/bf16), kích thước batch, số bước lấy mẫu (với diffusion/flow), phân phối độ dài câu; báo cáo trung vị và phân vị 95 thay vì chỉ trung bình.

## 38.7 Báo cáo kết quả

Một báo cáo đánh giá đáng tin nên có:

- **Hệ thống**: phiên bản, checkpoint (có dùng EMA không), dữ liệu huấn luyện, vocoder.
- **Tập đánh giá**: nguồn, số câu, độ dài, cách chọn; người nói đã thấy hay chưa thấy.
- **Đánh giá chủ quan**: phương pháp (MOS/CMOS/MUSHRA), câu hỏi và nhãn thang, mỏ neo, nền tảng, số người nghe, số điểm, tiêu chí loại người nghe, phương ngữ người nghe, điều kiện nghe, khoảng tin cậy (và cách tính), kiểm định và hiệu chỉnh so sánh bội.
- **Đánh giá khách quan**: tên và phiên bản ASR, quy tắc chuẩn hóa văn bản trước khi chấm, mô hình embedding cho SIM, bộ dự đoán MOS, tần số lấy mẫu khi tính (có lấy mẫu lại không), sàn lỗi ASR trên bản ghi thật.
- **Mẫu nghe**: chọn ngẫu nhiên theo quy tắc định trước, công bố kèm văn bản.
- **Hiệu năng**: phần cứng, RTF, độ trễ.

Nguyên tắc tổng quát: người đọc phải có đủ thông tin để **tái lập** đánh giá, và để **nghi ngờ đúng chỗ**.

## Lỗi thường gặp và mẹo thực hành

- **Khoảng tin cậy ngây thơ** coi mọi điểm MOS là độc lập — quá hẹp, có thể chỉ phủ khoảng 40% thay vì 95%. Dùng bootstrap hai chiều hoặc mô hình hiệu ứng hỗn hợp.
- **So sánh MOS giữa các bài nghe khác nhau.** MOS là thang tương đối với các mẫu trong cùng bài.
- **Không có bản ghi thật làm mỏ neo**: không biết MOS 4,1 là gần trần hay còn xa.
- **Quá ít người nghe, quá nhiều câu mỗi người.** Tăng số người nghe hiệu quả hơn.
- **Không chuẩn hóa văn bản trước khi tính WER**: dấu câu, chữ hoa, kiểu đặt dấu, chữ số làm WER tăng giả tạo; với tiếng Việt, nhớ NFC và thống nhất vị trí dấu.
- **Không báo cáo sàn lỗi ASR** trên bản ghi thật.
- **Dùng PESQ/STOI cho đầu ra TTS** không căn chỉnh với tham chiếu.
- **So sánh SIM, MCD hay FAD tính bằng công cụ khác nhau** giữa các bài báo.
- **Tin tuyệt đối vào bộ dự đoán MOS**, nhất là với tiếng Việt chưa được hiệu chuẩn.
- **Chọn mẫu nghe đẹp** (*cherry-picking*) để công bố. Chọn ngẫu nhiên theo quy tắc định trước.
- **Bỏ qua phương ngữ của người nghe** khi đánh giá giọng tiếng Việt.
- **Chỉ báo cáo trung bình**: thêm phân phối (biểu đồ hộp), tỉ lệ câu thất bại (WER trên một ngưỡng) — người dùng nhớ những câu hỏng, không nhớ trung bình.

## Tóm tắt

- Chất lượng TTS nhiều chiều: tự nhiên, dễ hiểu, giống người nói, ngôn điệu, độ bền, hiệu năng. Mỗi chiều cần phép đo riêng; chỉ số khách quan để sàng lọc, bài nghe để quyết định.
- MOS (ACR, P.800) cần câu hỏi rõ, mỏ neo, ngẫu nhiên hóa, thiết kế cân bằng. CMOS nhạy hơn cho hệ thống gần nhau; MUSHRA phân biệt tốt hệ thống chất lượng cao.
- Điểm MOS tương quan theo người nghe và câu: phương sai của trung bình có thành phần $\sigma_b^2/L$, nên khoảng tin cậy ngây thơ quá hẹp. Dùng bootstrap hai chiều hoặc mô hình hiệu ứng hỗn hợp; tăng số người nghe.
- So sánh hệ thống: thiết kế cặp, Wilcoxon dấu-hạng, hiệu chỉnh Holm cho so sánh bội; phân biệt ý nghĩa thống kê và ý nghĩa thực tế.
- MCD xấp xỉ khoảng cách RMS (dB) giữa hai đường bao log phổ (suy ra từ Parseval); chỉ so sánh trong cùng một công cụ; mù với lỗi nhịp do DTW.
- Chỉ số F0 cần dùng kết hợp: tương quan bỏ sót lỗi nén biên độ, RMSE bị lỗi quãng tám chi phối; với tiếng Việt, so đường nét theo từng âm tiết.
- WER/CER qua ASR cần chuẩn hóa văn bản giống nhau cho cả hai phía, báo cáo sàn lỗi ASR; với tiếng Việt, tách lỗi thanh bằng CER bỏ thanh và tỉ lệ lỗi thanh.
- Bộ dự đoán MOS, PESQ, STOI, FAD đều có giới hạn; hiệu chuẩn bộ dự đoán MOS với người nghe tiếng Việt trước khi tin.
- Xây bộ kiểm thử tiếng Việt: cặp tối thiểu thanh điệu trong câu khung, số và viết tắt, từ mượn, ngữ điệu, phương ngữ, văn bản dài.

## Bài tập

**Bài 1 [Cơ bản].** Một bài MOS có 15 người nghe, mỗi người chấm 40 câu của hệ thống A. Độ lệch chuẩn của 600 điểm là 0,9. (a) Tính nửa độ rộng khoảng tin cậy 95% theo công thức ngây thơ. (b) Nếu tương quan nội nhóm theo người nghe là $\rho = 0{,}3$, số mẫu hiệu dụng là bao nhiêu và nửa độ rộng xấp xỉ là bao nhiêu?

??? success "Lời giải"
    (a) $1{,}96 \times 0{,}9 / \sqrt{600} \approx 0{,}072$. (b) Hệ số thiết kế $1 + (40-1)\times 0{,}3 = 12{,}7$; $n_{\text{hiệu dụng}} = 600/12{,}7 \approx 47$; nửa độ rộng $\approx 1{,}96 \times 0{,}9/\sqrt{47} \approx 0{,}26$, rộng hơn khoảng 3,6 lần. (Đây là phép xấp xỉ chỉ tính đến tương quan theo người nghe; tương quan theo câu làm khoảng còn rộng hơn.)

**Bài 2 [Cơ bản].** Giải thích vì sao WER của TTS đo bằng ASR có thể *thấp hơn* tỉ lệ lỗi phát âm thực sự mà người nghe cảm nhận. Đề xuất hai cách giảm hiện tượng này khi kiểm tra thanh điệu tiếng Việt.

??? success "Lời giải"
    Mô hình ngôn ngữ trong ASR ưu tiên chuỗi từ có xác suất cao, nên có thể "sửa" một âm tiết phát âm sai thanh thành từ đúng khi ngữ cảnh gợi ý mạnh (ví dụ "cảm ơn" dù thanh hỏi bị đọc gần thanh ngã). Cách giảm: (1) dùng câu khung trung tính không gợi ý từ đúng; (2) dùng ASR với mô hình ngôn ngữ yếu hoặc giải mã không dùng LM ngoài; (3) bổ sung bài nhận diện cưỡng bức với người nghe.

**Bài 3 [Nâng cao].** Chứng minh rằng với mô hình $r_{ij} = \mu + b_i + c_j + \varepsilon_{ij}$ (các thành phần độc lập, kỳ vọng 0) và thiết kế đầy đủ $L$ người nghe × $S$ câu, $\operatorname{Var}(\bar{r}) = \sigma_b^2/L + \sigma_c^2/S + \sigma_\varepsilon^2/(LS)$. Từ đó, với ngân sách cố định $LS = 600$ điểm, $\sigma_b^2 = 0{,}25$, $\sigma_c^2 = 0{,}09$, $\sigma_\varepsilon^2 = 0{,}36$, hãy tìm cặp $(L, S)$ cho phương sai nhỏ nhất.

??? tip "Gợi ý"
    $\bar{r} = \mu + \bar{b} + \bar{c} + \bar{\varepsilon}$ với $\bar{b}$ là trung bình của $L$ biến độc lập, $\bar{c}$ của $S$ biến, $\bar{\varepsilon}$ của $LS$ biến. Với $LS$ cố định, cực tiểu hóa $\sigma_b^2/L + \sigma_c^2 L/600$ theo $L$.

??? success "Lời giải"
    Do độc lập, phương sai của tổng bằng tổng phương sai, và trung bình của $k$ biến độc lập cùng phương sai $\sigma^2$ có phương sai $\sigma^2/k$, ta có đẳng thức cần chứng minh. Với $LS = 600$: $f(L) = 0{,}25/L + 0{,}09 L/600 + 0{,}36/600$. Đạo hàm bằng 0: $L^2 = 0{,}25 \times 600 / 0{,}09 \approx 1667$, $L \approx 41$, $S \approx 15$. Kết luận: nhiều người nghe, mỗi người ít câu — phù hợp với khuyến nghị tăng số người nghe.

**Bài 4 [Nâng cao].** Giải thích vì sao MCD tính sau DTW gần như không phạt một hệ thống nói chậm gấp rưỡi bản ghi thật. Đề xuất một chỉ số bổ sung đo sai khác về nhịp/trường độ.

??? success "Lời giải"
    DTW cho phép một khung tham chiếu ghép với nhiều khung tổng hợp; khi nói chậm đều, đường DTW có độ dốc khác 1 nhưng mỗi cặp khung ghép vẫn có đường bao phổ giống nhau, nên MCD gần như không đổi. Chỉ số bổ sung: tỉ lệ độ dài tổng (thời lượng tổng hợp / thời lượng thật), sai số trường độ theo âm tiết sau forced alignment (RMSE của log trường độ), hoặc độ lệch của độ dốc đường DTW so với 1.

**Bài 5 [Lập trình].** Mở rộng `evaluate` ở Mục 38.4.3 để trả về **ma trận nhầm lẫn thanh điệu** $6 \times 6$ (hàng: thanh tham chiếu, cột: thanh nhận dạng) trên các cặp âm tiết cùng phần không thanh. Kiểm thử với một danh sách câu có cài sẵn lỗi hỏi→ngã và sắc→hỏi.

??? tip "Gợi ý"
    Tái sử dụng `align` trên chuỗi âm tiết, và hàm xác định thanh từ dấu tổ hợp sau khi phân rã NFD như `split_syllable` ở [Chương 36](ch36-du-lieu-tts.md).

**Bài 6 [Lập trình].** Viết hàm `paired_bootstrap(scores_a, scores_b, n_boot)` nhận hai ma trận điểm (người nghe × câu) của hai hệ thống trên *cùng* người nghe và câu, lấy mẫu lại người nghe và câu, và trả về khoảng tin cậy 95% cho hiệu MOS. So sánh độ rộng với khoảng tin cậy khi coi hai hệ thống độc lập.

**Bài 7 [Lập trình].** Chạy mô phỏng ở Mục 38.3 với 1 000 hạt giống khác nhau và đếm tỉ lệ lần hệ thống D (chất lượng thật bằng A) bị kết luận "khác biệt có ý nghĩa" trước và sau hiệu chỉnh Holm. Kết quả có khớp với lý thuyết không?

## Tài liệu tham khảo

1. Anastassiou, P. và cộng sự (2024). *Seed-TTS: A Family of High-Quality Versatile Speech Generation Models*. arXiv preprint.
2. Bates, D., Mächler, M., Bolker, B. và Walker, S. (2015). *Fitting Linear Mixed-Effects Models Using lme4*. Journal of Statistical Software.
3. Benjamini, Y. và Hochberg, Y. (1995). *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing*. Journal of the Royal Statistical Society, Series B.
4. Benoît, C., Grice, M. và Hazan, V. (1996). *The SUS Test: A Method for the Assessment of Text-to-Speech Synthesis Intelligibility Using Semantically Unpredictable Sentences*. Speech Communication.
5. Chen, S. và cộng sự (2022). *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing*. IEEE Journal of Selected Topics in Signal Processing.
6. Chu, W. và Alwan, A. (2009). *Reducing F0 Frame Error of F0 Tracking Algorithms under Noisy Conditions with an Unvoiced/Voiced Classification Frontend*. ICASSP.
7. Clark, R., Silen, H., Kenter, T. và Leith, R. (2019). *Evaluating Long-form Text-to-Speech: Comparing the Ratings of Sentences and Paragraphs*. ISCA Speech Synthesis Workshop (SSW10).
8. Desplanques, B., Thienpondt, J. và Demuynck, K. (2020). *ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification*. Interspeech.
9. Efron, B. và Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
10. Holm, S. (1979). *A Simple Sequentially Rejective Multiple Test Procedure*. Scandinavian Journal of Statistics.
11. Huang, W.-C. và cộng sự (2022). *The VoiceMOS Challenge 2022*. Interspeech.
12. ITU-R (2015). *Recommendation BS.1534-3: Method for the Subjective Assessment of Intermediate Quality Level of Audio Systems*.
13. ITU-T (1996). *Recommendation P.800: Methods for Subjective Determination of Transmission Quality*.
14. ITU-T (2018, sửa đổi về sau). *Recommendation P.808: Subjective Evaluation of Speech Quality with a Crowdsourcing Approach*.
15. Kilgour, K., Zuluaga, M., Roblek, D. và Sharifi, M. (2019). *Fréchet Audio Distance: A Reference-Free Metric for Evaluating Music Enhancement Algorithms*. Interspeech.
16. Kirkland, A. và cộng sự (2023). *Stuck in the MOS Pit: A Critical Analysis of MOS Test Methodology in TTS Evaluation*. ISCA Speech Synthesis Workshop (SSW12).
17. Kubichek, R. (1993). *Mel-Cepstral Distance Measure for Objective Speech Quality Assessment*. IEEE Pacific Rim Conference on Communications, Computers and Signal Processing.
18. Le, M. và cộng sự (2023). *Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale*. NeurIPS.
19. Le, T.-T., Nguyen, L. T. và Nguyen, D. Q. (2024). *PhoWhisper: Automatic Speech Recognition for Vietnamese*. ICLR Tiny Papers.
20. Lo, C.-C. và cộng sự (2019). *MOSNet: Deep Learning-Based Objective Assessment for Voice Conversion*. Interspeech.
21. Mittag, G., Naderi, B., Chehadi, A. và Möller, S. (2021). *NISQA: A Deep CNN-Self-Attention Model for Multidimensional Speech Quality Prediction with Crowdsourced Datasets*. Interspeech.
22. Owen, A. B. (2007). *The Pigeonhole Bootstrap*. The Annals of Applied Statistics.
23. Radford, A. và cộng sự (2023). *Robust Speech Recognition via Large-Scale Weak Supervision*. ICML.
24. Reddy, C. K. A., Gopal, V. và Cutler, R. (2021). *DNSMOS: A Non-Intrusive Perceptual Objective Speech Quality Metric to Evaluate Noise Suppressors*. ICASSP.
25. Ribeiro, F., Florêncio, D., Zhang, C. và Seltzer, M. (2011). *CrowdMOS: An Approach for Crowdsourcing Mean Opinion Score Studies*. ICASSP.
26. Rix, A. W., Beerends, J. G., Hollier, M. P. và Hekstra, A. P. (2001). *Perceptual Evaluation of Speech Quality (PESQ) — A New Method for Speech Quality Assessment of Telephone Networks and Codecs*. ICASSP.
27. Saeki, T. và cộng sự (2022). *UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022*. Interspeech.
28. Taal, C. H., Hendriks, R. C., Heusdens, R. và Jensen, J. (2011). *An Algorithm for Intelligibility Prediction of Time–Frequency Weighted Noisy Speech*. IEEE Transactions on Audio, Speech, and Language Processing.
29. Wester, M., Valentini-Botinhao, C. và Henter, G. E. (2015). *Are We Using Enough Listeners? No! — An Empirically-Supported Critique of Interspeech 2014 TTS Evaluations*. Interspeech.
30. Wilcoxon, F. (1945). *Individual Comparisons by Ranking Methods*. Biometrics Bulletin.
