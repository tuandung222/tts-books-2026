# Chương 18. G2P và biểu diễn đầu vào

!!! abstract "Mục tiêu học tập"
    Sau chương này, bạn có thể:

    - **So sánh** các đơn vị biểu diễn đầu vào cho TTS (ký tự, âm vị, byte, subword/BPE) theo độ dài chuỗi, kích thước từ vựng, khả năng khái quát hóa và yêu cầu dữ liệu.
    - **Giải thích** vì sao chính tả tiếng Việt gần như biểu âm và chỉ ra chính xác những chỗ ánh xạ chữ–âm không một-một.
    - **Cài đặt** một bộ G2P dựa trên luật cho tiếng Việt (giọng Bắc) gồm bộ phân tích âm tiết, ánh xạ sang IPA và token thanh điệu, kèm kiểm thử.
    - **Phân tích** ảnh hưởng của các cách biểu diễn thanh điệu (token riêng, gộp vào nguyên âm, đặc trưng song song) tới độ dài chuỗi, từ vựng và căn chỉnh.
    - **Lựa chọn** phương ngữ đích và chiến lược xử lý từ ngoại lai (từ điển, đánh vần, G2P nơ-ron, espeak-ng).
    - **Thiết kế** một bộ ký hiệu có phiên bản, gồm token đặc biệt, dấu câu và kỹ thuật chèn blank (*add_blank*) của VITS.
    - **Đánh giá** tác động của token hóa BPE/byte lên văn bản tiếng Việt trong các mô hình TTS dựa trên LLM.

!!! info "Kiến thức cần có"
    - [Chương 15](../phan-3-tin-hieu-ngu-am/ch15-ngu-am-tieng-viet.md): hệ thống âm vị, cấu trúc âm tiết và thanh điệu tiếng Việt.
    - [Chương 17](ch17-chuan-hoa-van-ban.md): chuẩn hóa văn bản — đầu ra của Chương 17 là đầu vào của chương này.
    - [Chương 5](../phan-1-nen-tang/ch05-transformer.md): embedding, attention và độ phức tạp theo độ dài chuỗi.

Sau khi chuẩn hóa ([Chương 17](ch17-chuan-hoa-van-ban.md)), văn bản chỉ còn chữ và dấu câu: "hà nội mùa thu, trời trong xanh!". Câu hỏi của chương này là: **mô hình âm học nên "nhìn" câu đó dưới dạng chuỗi ký hiệu nào?** Đó có thể là 32 ký tự, 25 âm vị kèm 7 token thanh điệu, vài chục byte, hay chừng mười token BPE. Lựa chọn này ảnh hưởng tới mọi thứ phía sau: mô hình phải học gì, chuỗi dài bao nhiêu, căn chỉnh dễ hay khó, và mô hình xử lý từ lạ ra sao.

## 18.1. Mô hình TTS "nhìn" văn bản như thế nào?

### 18.1.1. Từ ký hiệu đến embedding

Mọi mô hình TTS nơ-ron bắt đầu bằng một bảng tra **embedding**. Gọi $\mathcal{V}$ là bộ ký hiệu (từ vựng) và $\mathbf{x} = (x_1, \ldots, x_N)$, $x_i \in \{1, \ldots, |\mathcal{V}|\}$, là chuỗi đầu vào (đúng như [bảng ký hiệu](../gioi-thieu/ky-hieu.md) của sách). Lớp embedding là một ma trận $\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times C}$, và véc-tơ của token thứ $i$ là hàng $x_i$ của $\mathbf{E}$:

$$
\mathbf{e}_i = \mathbf{E}^{\top} \mathbf{o}_{x_i} \in \mathbb{R}^{C},
$$

với $\mathbf{o}_{x_i}$ là véc-tơ one-hot. Bộ mã hóa văn bản (CNN, BiLSTM, Transformer…) biến $(\mathbf{e}_1, \ldots, \mathbf{e}_N)$ thành biểu diễn ẩn $\mathbf{H} = (\mathbf{h}^{\text{text}}_1, \ldots, \mathbf{h}^{\text{text}}_N)$, rồi bộ giải mã căn chỉnh $\mathbf{H}$ với $T$ khung mel. Ba hệ quả trực tiếp của việc chọn $\mathcal{V}$:

1. **Mô hình phải học gì.** Nếu $\mathcal{V}$ là ký tự, mô hình phải tự học quy tắc chữ → âm. Nếu $\mathcal{V}$ là âm vị, việc đó đã được G2P làm sẵn.
2. **Độ dài $N$.** Chi phí self-attention tăng theo $O(N^2)$, chi phí căn chỉnh đơn điệu (MAS, [Chương 21](../phan-5-mo-hinh-am-hoc/ch21-alignment.md)) theo $O(NT)$. Mỗi token cũng là một "neo" để căn chỉnh: token quá ngắn về mặt âm học (một dấu thanh tách rời, một byte) làm căn chỉnh khó hơn.
3. **Độ thưa.** Mỗi hàng của $\mathbf{E}$ chỉ được cập nhật khi ký hiệu tương ứng xuất hiện. Từ vựng càng lớn, mỗi ký hiệu càng ít dữ liệu; ký hiệu hiếm hầu như không được học.

### 18.1.2. Bốn lựa chọn và sự đánh đổi

| Đơn vị | $\vert\mathcal{V}\vert$ điển hình | Độ dài $N$ | Cần G2P? | Từ lạ / ngoại lai | Thường dùng khi |
|---|---|---|---|---|---|
| Ký tự (grapheme) | vài chục – vài trăm | trung bình | không | mô hình tự đoán | dữ liệu vừa–lớn, chính tả nông |
| Âm vị (IPA) + thanh | vài chục | trung bình | có | phụ thuộc G2P | dữ liệu ít, cần kiểm soát phát âm |
| Byte UTF-8 | 256 | dài nhất | không | luôn biểu diễn được | mô hình đa ngôn ngữ rất lớn |
| Subword/BPE | hàng nghìn – hàng trăm nghìn | ngắn nhất | không | tách thành mảnh | TTS dựa trên LLM ([Phần VIII](../phan-8-mo-hinh-lon/ch32-codec-language-model.md)) |

Khái niệm then chốt để chọn giữa ký tự và âm vị là **độ sâu chính tả** (*orthographic depth*): mức độ cách viết quyết định cách đọc. Có thể hình thức hóa bằng entropy có điều kiện $H(\text{phát âm} \mid \text{chữ viết})$: với tiếng Anh, đại lượng này lớn ("read" có hai cách đọc; "ough" trong *though, through, tough* đọc khác nhau), nên mô hình dựa trên ký tự cần rất nhiều dữ liệu để học, và G2P đóng vai trò quan trọng. Với tiếng Việt, $H(\text{phát âm} \mid \text{chữ viết})$ gần bằng 0 đối với từ thuần Việt và Hán Việt trong một phương ngữ cố định: cách đọc được xác định gần như hoàn toàn từ chính tả.

Kinh nghiệm thực tế (cần kiểm chứng trên dữ liệu của bạn):

- **Dữ liệu ít** (vài giờ đến vài chục giờ một người nói): đầu vào âm vị thường giúp mô hình hội tụ nhanh hơn và ít lỗi phát âm hơn, vì mô hình chỉ cần học âm học, không cần học chính tả.
- **Dữ liệu lớn, mô hình lớn**: ký tự (hoặc BPE) hoạt động tốt, đơn giản hóa pipeline và tránh lan truyền lỗi của G2P. Nhiều mô hình zero-shot hiện đại như E2/F5-TTS dùng thẳng ký tự ([Chương 33](../phan-8-mo-hinh-lon/ch33-nar-quy-mo-lon.md)).
- **Tiếng Việt** nằm ở vùng giữa: vì chính tả nông, mô hình ký tự học khá dễ; nhưng các vấn đề Unicode (NFC/NFD, vị trí dấu), từ ngoại lai và lựa chọn phương ngữ vẫn khiến một front-end âm vị tường minh có giá trị, nhất là khi dữ liệu ít hoặc cần kiểm soát chặt.

## 18.2. Chính tả tiếng Việt: gần như biểu âm

### 18.2.1. Cấu trúc âm tiết (nhắc lại)

[Chương 15](../phan-3-tin-hieu-ngu-am/ch15-ngu-am-tieng-viet.md) trình bày đầy đủ âm vị học tiếng Việt; ở đây chỉ nhắc những gì cần cho G2P. Mỗi âm tiết gồm tối đa năm thành phần:

$$
\text{âm tiết} = (\text{âm đầu}) + (\text{âm đệm}) + \text{âm chính} + (\text{âm cuối}) + \text{thanh điệu},
$$

trong đó chỉ âm chính và thanh điệu là bắt buộc (thanh ngang không có dấu). Ví dụ "hoàng" = *h* + *o* (âm đệm /w/) + *a* + *ng* + thanh huyền; "ấy" = (âm đầu khuyết, thực tế là âm tắc thanh hầu [ʔ]) + *â* + *y* (bán âm cuối /j/) + thanh sắc.

### 18.2.2. Những chỗ ánh xạ chữ – âm không một-một

Chính tả tiếng Việt "gần như" biểu âm chứ không hoàn toàn. Các bất quy tắc đều **tất định theo ngữ cảnh chính tả**, nên luật xử lý được hết:

**Một âm, nhiều cách viết:**

- /k/: *c* (trước a, ă, â, o, ô, ơ, u, ư), *k* (trước i, y, e, ê, iê), *q* (chỉ trong *qu*).
- /ɣ/: *g* và *gh* (trước i, e, ê); /ŋ/: *ng* và *ngh* (trước i, e, ê).
- /i/: *i* và *y* (*kỹ* và *kĩ* cùng đọc); /iə/: *iê*, *yê* (đầu âm tiết hoặc sau âm đệm), *ia*, *ya* (vần mở).
- /uə/: *uô* và *ua*; /ɨə/: *ươ* và *ưa* (dạng *-a* dùng khi không có âm cuối: *mùa*, *cửa*).
- Âm đầu /z/ (giọng Hà Nội): *d*, *gi*, *r*.

**Một chữ, nhiều âm:**

- *a* là /aː/ (dài) trong *an, ai, ao* nhưng là /a/ (ngắn, như *ă*) trong *au, ay*: "cau" ≠ "cao", "tay" ≠ "tai".
- *a* trong *anh, ach* có chất lượng khác hẳn *a* thường (thường được mô tả là gần [ɛ] hoặc [a] kèm lướt [j], tùy tác giả và tùy giọng).
- *o* là âm đệm /w/ trong *oa, oă, oe* (*hoa, hoặc, khoe*) nhưng là âm chính /ɔ/ trong *oi, on, ong*.
- *u* là âm đệm /w/ trong *uâ, uê, uy, uơ* (*tuần, thuế, thủy, thuở*) và sau *q*, nhưng là âm chính trong *ui, un, ua*.
- *i, y* là âm chính trong *in, ý* nhưng là bán âm cuối /j/ trong *ai, ây, ôi*; *o, u* là bán âm cuối /w/ trong *ao, âu, êu*.
- *gi* đứng trước nguyên âm là âm đầu /z/ (*gia, giữa*); nhưng trong *gì, gìn* chữ *i* vừa thuộc *gi* vừa là âm chính; trong *giếng, giêng* phần *i* hòa vào nguyên âm đôi /iə/.

Đặc điểm quan trọng: tất cả các trường hợp trên được quyết định bởi **chữ cái lân cận trong cùng âm tiết**. Không có hiện tượng như *read* (hiện tại) / *read* (quá khứ) của tiếng Anh, nơi cần ngữ cảnh câu. Vì vậy, với từ thuần Việt và Hán Việt, một bộ G2P dựa trên luật có thể đạt độ chính xác gần như tuyệt đối — sai số còn lại chủ yếu đến từ từ ngoại lai, lỗi chính tả và **lựa chọn mô tả ngữ âm** (Mục 18.3.3).

## 18.3. Cài đặt G2P dựa trên luật

### 18.3.1. Bộ phân tích âm tiết

[Chương 15](../phan-3-tin-hieu-ngu-am/ch15-ngu-am-tieng-viet.md) xây dựng một bộ phân tích âm tiết đầy đủ; để chương này tự chứa, chúng ta viết lại một phiên bản gọn. Thuật toán gồm năm bước:

1. **Tách thanh điệu** qua NFD: dấu thanh là một trong năm ký tự kết hợp (huyền U+0300, sắc U+0301, hỏi U+0309, ngã U+0303, nặng U+0323). Bỏ chúng đi rồi NFC lại, ta được phần "xương" của âm tiết với các dấu phụ â, ê, ơ, ư… còn nguyên. Ưu điểm lớn: cách này **không phụ thuộc vị trí đặt dấu** — "hòa" và "hoà" cho cùng kết quả, nên G2P miễn nhiễm với vấn đề ở [Mục 17.2.2](ch17-chuan-hoa-van-ban.md).
2. **Tách âm đầu**: so khớp tiền tố dài nhất trong danh sách âm đầu chính tả (*ngh* trước *ng* trước *n*). Xử lý đặc biệt *gi* (khi theo sau không phải nguyên âm, hoặc là *ê*) và *qu* (âm đầu /k/ kèm âm đệm /w/).
3. **Tách âm đệm**: *o* đứng trước *a, ă, e*; *u* đứng trước *â, ê, y, ơ*.
4. **Tách âm cuối**: thử phụ âm cuối dài trước (*ng, nh, ch* rồi *c, m, n, p, t*); nếu phần còn lại không phải âm chính hợp lệ, thử bán âm cuối *i, y, o, u*.
5. **Kiểm tra chính tả**: quy tắc *c/k, g/gh, ng/ngh* và quy tắc **âm tiết tắc** (âm cuối *p, t, c, ch* chỉ đi với thanh sắc hoặc nặng). Bước này biến bộ phân tích thành một bộ **nhận diện tiếng Việt** rẻ tiền: "facebook", "zoom", "bàt" đều bị từ chối.

```python
import unicodedata
from dataclasses import dataclass

TONE_MARKS = {"̀": 2, "́": 3, "̉": 4, "̃": 5, "̣": 6}
TONE_NAMES = {1: "ngang", 2: "huyền", 3: "sắc", 4: "hỏi", 5: "ngã", 6: "nặng"}
VOWELS = set("aăâeêioôơuưy")
ONSETS = sorted(["ngh", "ng", "nh", "ch", "tr", "th", "ph", "kh", "gh", "gi", "qu", "b", "c",
                 "d", "đ", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "x"],
                key=len, reverse=True)
CODAS = ["ng", "nh", "ch", "c", "m", "n", "p", "t"]     # phụ âm cuối (dài trước)
NUCLEI = {"a", "ă", "â", "e", "ê", "i", "y", "o", "ô", "ơ", "u", "ư",
          "iê", "yê", "ia", "ya", "uô", "ua", "ươ", "ưa", "oo"}


@dataclass
class Syllable:
    onset: str     # âm đầu (chữ viết), "" nếu khuyết
    medial: str    # âm đệm: "o", "u" hoặc ""
    nucleus: str   # âm chính
    coda: str      # âm cuối: phụ âm, bán âm (i/y/o/u) hoặc ""
    tone: int      # 1..6: ngang, huyền, sắc, hỏi, ngã, nặng


def split_tone(word: str) -> tuple[str, int]:
    nfd = unicodedata.normalize("NFD", word.lower())
    marks = [TONE_MARKS[c] for c in nfd if c in TONE_MARKS]
    if len(marks) > 1:
        raise ValueError(f"nhiều dấu thanh: {word!r}")
    base = unicodedata.normalize("NFC", "".join(c for c in nfd if c not in TONE_MARKS))
    return base, (marks[0] if marks else 1)


def parse_syllable(word: str) -> Syllable:
    """Tách âm tiết chính tả tiếng Việt; ném ValueError nếu không hợp lệ."""
    base, tone = split_tone(word)
    onset = next((o for o in ONSETS if base.startswith(o)), "")
    rest = base[len(onset):]
    if onset == "gi" and (not rest or rest[0] not in VOWELS or rest[0] == "ê"):
        rest = "i" + rest                       # gì, gìn: "i" vừa thuộc "gi" vừa là âm chính; giếng → iê
    medial = ""
    if onset == "qu":
        medial = "u"                            # "qu" = /k/ + âm đệm /w/
    elif len(rest) >= 2 and ((rest[0] == "o" and rest[1] in "aăe")
                             or (rest[0] == "u" and rest[1] in "âêyơ")):
        medial, rest = rest[0], rest[1:]        # hoa, hoặc, khoe, tuần, thuế, thủy, thuở
    coda = next((c for c in CODAS if rest.endswith(c) and len(rest) > len(c)), "")
    nucleus = rest[:len(rest) - len(coda)]
    if not coda and nucleus not in NUCLEI and len(nucleus) >= 2 and nucleus[-1] in "iyou":
        nucleus, coda = nucleus[:-1], nucleus[-1]   # bán âm cuối: ai, ao, ưu, uôi, ươu...
    if nucleus not in NUCLEI:
        raise ValueError(f"không phải âm tiết tiếng Việt: {word!r}")
    _check_spelling(onset, medial + nucleus, coda, tone, word)
    return Syllable(onset, medial, nucleus, coda, tone)


def _check_spelling(onset: str, vowels: str, coda: str, tone: int, word: str) -> None:
    front = vowels[:1] in ("i", "e", "ê", "y")
    if (onset in ("k", "gh", "ngh") and not front) or (onset in ("c", "g", "ng") and front):
        raise ValueError(f"sai quy tắc c/k, g/gh, ng/ngh: {word!r}")
    if coda in ("p", "t", "c", "ch") and tone not in (3, 6):
        raise ValueError(f"âm tiết tắc chỉ mang thanh sắc/nặng: {word!r}")
```

### 18.3.2. Ánh xạ sang IPA (giọng Bắc)

Bảng dưới tóm tắt ánh xạ dùng trong mã nguồn (giọng Hà Nội, theo hướng âm vị học, có tham khảo mô tả của Kirby, 2011). Cột cuối là giá trị khi bật tùy chọn **giữ tương phản chính tả** (Mục 18.4).

| Âm đầu (chữ) | IPA | | Âm chính (chữ) | IPA | | Âm cuối (chữ) | IPA |
|---|---|---|---|---|---|---|---|
| b | ɓ | | a | aː (ngắn *a* trong *au, ay*; *ɛ* trong *anh, ach*) | | c | k |
| đ | ɗ | | ă | a | | ch | c |
| c, k, qu | k | | â | ə | | m, n | m, n |
| ch | c | | e | ɛ | | ng | ŋ |
| tr | c (giữ tương phản: ʈ) | | ê | e | | nh | ɲ |
| d, gi | z | | i, y | i | | p, t | p, t |
| r | z (giữ tương phản: r) | | o (oo) | ɔ | | i, y (bán âm) | j |
| s | s (giữ tương phản: ʂ) | | ô | o | | o, u (bán âm) | w |
| x | s | | ơ | əː | | | |
| g, gh | ɣ | | u | u | | | |
| kh | x | | ư | ɨ | | | |
| ng, ngh | ŋ | | iê, yê, ia, ya | iə | | | |
| nh | ɲ | | uô, ua | uə | | | |
| ph | f | | ươ, ưa | ɨə | | | |
| th | tʰ | | | | | | |
| (khuyết) | ʔ | | | | | | |

Còn lại *h, l, m, n, p, t, v* giữ nguyên ký hiệu. Âm đệm luôn là /w/.

```python
ONSET_IPA = {"": "ʔ", "b": "ɓ", "c": "k", "k": "k", "qu": "k", "ch": "c", "d": "z",
             "đ": "ɗ", "g": "ɣ", "gh": "ɣ", "gi": "z", "h": "h", "kh": "x", "l": "l",
             "m": "m", "n": "n", "ng": "ŋ", "ngh": "ŋ", "nh": "ɲ", "p": "p", "ph": "f",
             "r": "z", "s": "s", "t": "t", "th": "tʰ", "tr": "c", "v": "v", "x": "s"}
# Giữ tương phản chính tả mà giọng Hà Nội đã nhập làm một (hữu ích khi dữ liệu lẫn giọng)
SPELLING_CONTRASTS = {"r": "r", "s": "ʂ", "tr": "ʈ"}
NUCLEUS_IPA = {"a": "aː", "ă": "a", "â": "ə", "e": "ɛ", "ê": "e", "i": "i", "y": "i",
               "o": "ɔ", "ô": "o", "ơ": "əː", "u": "u", "ư": "ɨ", "oo": "ɔ",
               "iê": "iə", "yê": "iə", "ia": "iə", "ya": "iə",
               "uô": "uə", "ua": "uə", "ươ": "ɨə", "ưa": "ɨə"}
CODA_IPA = {"": None, "c": "k", "ch": "c", "m": "m", "n": "n", "ng": "ŋ", "nh": "ɲ",
            "p": "p", "t": "t", "i": "j", "y": "j", "o": "w", "u": "w"}


def syllable_to_ipa(s: Syllable, keep_contrasts: bool = False) -> list[str]:
    """Âm tiết → danh sách âm vị IPA (giọng Bắc/Hà Nội), CHƯA gồm thanh điệu."""
    onset = ONSET_IPA[s.onset]
    if keep_contrasts:
        onset = SPELLING_CONTRASTS.get(s.onset, onset)
    phones = [onset]
    if s.medial:
        phones.append("w")
    nucleus = NUCLEUS_IPA[s.nucleus]
    if s.nucleus == "a" and s.coda in ("u", "y"):
        nucleus = "a"                    # au, ay: a ngắn (= ă), khác ao, ai
    elif s.nucleus == "a" and s.coda in ("nh", "ch"):
        nucleus = "ɛ"                    # anh, ach: một quy ước phổ biến (xem Góc tiếng Việt)
    phones.append(nucleus)
    if CODA_IPA[s.coda]:
        phones.append(CODA_IPA[s.coda])
    return phones


def syllable_tokens(word: str, tone_mode: str = "separate", **kw):
    """Ba cách biểu diễn thanh điệu:
    'separate': [..., 'T3']           — thanh là một token riêng ở cuối âm tiết
    'fused'   : nguyên âm mang thanh  — 'iə3' (từ vựng lớn hơn, chuỗi không dài thêm)
    'feature' : (phones, tones)       — chuỗi thanh song song, cùng độ dài với phones
    """
    s = parse_syllable(word)
    phones = syllable_to_ipa(s, **kw)
    if tone_mode == "separate":
        return phones + [f"T{s.tone}"]
    if tone_mode == "fused":
        i = 2 if s.medial else 1          # vị trí âm chính: sau âm đầu (và âm đệm)
        return phones[:i] + [f"{phones[i]}{s.tone}"] + phones[i + 1:]
    if tone_mode == "feature":
        return phones, [s.tone] * len(phones)
    raise ValueError(tone_mode)
```

Hàm `syllable_tokens` cài sẵn ba cách biểu diễn thanh điệu mà Mục 18.5 sẽ phân tích.

### 18.3.3. Kiểm thử và các lựa chọn mô tả

```python
G2P_CASES = {  # từ → "âm vị ... thanh" (giọng Bắc, đã nhập r/d/gi, s/x, tr/ch)
    "Việt": "v iə t T6",   "Nam": "n aː m T1",      "quốc": "k w o k T3",
    "gia": "z aː T1",      "giếng": "z iə ŋ T3",    "gì": "z i T2",
    "nghiêng": "ŋ iə ŋ T1", "khuya": "x w iə T1",   "người": "ŋ ɨə j T2",
    "hoa": "h w aː T1",    "tuần": "t w ə n T2",    "thủy": "tʰ w i T4",
    "anh": "ʔ ɛ ɲ T1",     "sách": "s ɛ c T3",      "rượu": "z ɨə w T6",
    "khuỷu": "x w i w T4", "ăn": "ʔ a n T1",        "ấy": "ʔ ə j T3",
    "cau": "k a w T1",     "cao": "k aː w T1",      "tay": "t a j T1",
    "tai": "t aː j T1",    "mùa": "m uə T2",        "cửa": "k ɨə T4",
    "yêu": "ʔ iə w T1",    "uống": "ʔ uə ŋ T3",     "oan": "ʔ w aː n T1",
    "huế": "h w e T3",     "thuở": "tʰ w əː T4",    "giữa": "z ɨə T5",
    "kìa": "k iə T2",      "trường": "c ɨə ŋ T2",   "xoong": "s ɔ ŋ T1",
}
for word, expected in G2P_CASES.items():
    got = " ".join(syllable_tokens(word))
    assert got == expected, (word, got, expected)

# Giữ tương phản chính tả (r ≠ d, s ≠ x, tr ≠ ch)
assert syllable_tokens("trường", keep_contrasts=True)[0] == "ʈ"
assert syllable_tokens("rồi", keep_contrasts=True)[0] == "r"
assert syllable_tokens("sông", keep_contrasts=True)[0] == "ʂ"

# Ba cách biểu diễn thanh điệu
assert syllable_tokens("tiếng", tone_mode="fused") == ["t", "iə3", "ŋ"]
assert syllable_tokens("hoà", tone_mode="fused") == ["h", "w", "aː2"]
assert syllable_tokens("tiếng", tone_mode="feature") == (["t", "iə", "ŋ"], [3, 3, 3])

# Bộ kiểm tra chính tả bắt được từ ngoại lai / lỗi gõ
for bad in ["facebook", "ok", "zoom", "ge", "ka", "ngi", "bàt", "ci"]:
    try:
        parse_syllable(bad)
        raise AssertionError(f"lẽ ra phải từ chối {bad!r}")
    except ValueError:
        pass
print("Tất cả kiểm thử G2P đều qua.")
```

!!! note "Góc tiếng Việt"
    Một số điểm trong bảng IPA là **lựa chọn mô tả**, không phải sự thật duy nhất, và các tài liệu ngữ âm học mô tả khác nhau:

    - **Vần *anh, ach*.** Có tác giả phân tích *a* ở đây là biến thể của /ɛ/ và *nh, ch* là biến thể của /ŋ, k/ sau nguyên âm hàng trước; có mô tả ngữ âm giọng Hà Nội gần với [ajŋ], [ajk]. Mã nguồn chọn "ɛ ɲ" và "ɛ c": điều quan trọng với TTS là *anh* phải **khác** *eng* (xẻng, leng keng) và *ach* phải khác *éc* — cả hai lựa chọn đều thỏa mãn.
    - **Âm cuối sau nguyên âm tròn môi.** Trong giọng Bắc, *ông, ung, ốc, úc* thường được phát âm với động tác khép môi ở cuối (thường ghi [ŋ͡m], [k͡p]). Vì hiện tượng này **dự đoán được** từ nguyên âm, ta không cần ký hiệu riêng: mô hình âm học sẽ tự học từ ngữ cảnh.
    - ***ư*** được ghi là /ɨ/ hoặc /ɯ/ tùy tài liệu; *â, ă* được ghi là nguyên âm ngắn.

    Nguyên tắc: chọn một quy ước, ghi rõ trong tài liệu dự án, và đảm bảo mọi cặp âm tiết **khác âm** cho ra chuỗi ký hiệu khác nhau. Chính sự nhất quán, chứ không phải độ "chuẩn" của ký hiệu IPA, quyết định chất lượng TTS.

## 18.4. Chọn phương ngữ đích

Chữ viết tiếng Việt chung cho mọi miền, nhưng cách đọc khác nhau. Một bộ G2P bắt buộc phải nhắm tới **một** phương ngữ. Một số khác biệt thường được mô tả (chi tiết và mức độ biến thể xem [Chương 15](../phan-3-tin-hieu-ngu-am/ch15-ngu-am-tieng-viet.md)):

| Hiện tượng | Giọng Bắc (Hà Nội) | Giọng Nam (TP. Hồ Chí Minh) |
|---|---|---|
| *r / d / gi* | nhập làm một: [z] | *d, gi* (và thường cả *v*) thành [j]; *r* có nhiều biến thể |
| *s / x*, *tr / ch* | nhập làm một: [s], [c] | trong lối nói cẩn thận thường giữ phân biệt |
| Thanh hỏi / ngã | phân biệt | thường nhập làm một |
| Âm cuối *-n / -ng*, *-t / -c* | phân biệt | sau một số nguyên âm thường nhập làm một |
| Vần *-nh / -ch* | xem Góc tiếng Việt ở trên | thường gần [n], [t] |
| *qu-*, âm đệm | [kw] | *qu* thường thành [w] |

Hệ quả cho thiết kế:

1. **Phương ngữ của G2P phải trùng với phương ngữ của người nói trong dữ liệu.** Nếu dùng G2P giọng Bắc cho dữ liệu giọng Nam, cùng một chuỗi âm vị /z/ sẽ ứng với âm thanh [j] — mô hình vẫn học được (vì ánh xạ vẫn nhất quán), nhưng các ký hiệu không còn mang nghĩa IPA, và mọi luật hậu kỳ dựa trên IPA (ví dụ chuyển đổi giọng) sẽ sai.
2. **Nhập hay không nhập?** Hàm `syllable_to_ipa` có tùy chọn `keep_contrasts`. Nhập (*r, d, gi* → /z/) làm bộ ký hiệu nhỏ hơn và phản ánh đúng giọng Hà Nội. Giữ tương phản chính tả có lợi khi dữ liệu trộn nhiều người nói (có người phân biệt *tr/ch*, có người không), hoặc khi bạn muốn **một** front-end dùng cho nhiều giọng: mô hình đa người nói sẽ tự học cách mỗi giọng hiện thực hóa từng ký hiệu thông qua embedding người nói ([Chương 28](../phan-7-e2e-bieu-cam/ch28-da-nguoi-noi-voice-cloning.md)). Đây là một lý do khiến nhiều hệ thống đa giọng tiếng Việt chọn đầu vào gần với chính tả.
3. **Thanh điệu theo phương ngữ.** Nếu giọng đích nhập hỏi/ngã, bạn có thể giữ hai token riêng (để cùng front-end dùng được cho giọng Bắc) và để mô hình học rằng chúng trùng nhau ở giọng Nam — đơn giản và ít rủi ro hơn việc nhập sớm ở G2P.

## 18.5. Biểu diễn thanh điệu

Thanh điệu là thông tin âm học quan trọng bậc nhất của tiếng Việt: sai thanh là sai từ (*ma, mà, má, mả, mã, mạ*), và lỗi thanh là một trong những lỗi dễ nhận ra nhất ở TTS tiếng Việt. Về mặt âm học, thanh điệu là đường nét F0 (và chất giọng — ví dụ thanh hỏi, ngã, nặng thường đi kèm hiện tượng thanh quản hóa) trải trên **toàn bộ phần vần** (âm chính và âm cuối vang), không gắn với một phân đoạn duy nhất. Câu hỏi là làm sao đưa thông tin "trải dài" đó vào một chuỗi token rời rạc.

### 18.5.1. Ba phương án

**(a) Token thanh riêng.** Thêm một token sau mỗi âm tiết: "t iə ŋ T3". Đơn giản, từ vựng nhỏ (chỉ thêm 6 ký hiệu), dễ đọc. Nhược điểm: chuỗi dài thêm một token mỗi âm tiết; và với các mô hình dựa trên **trường độ** (FastSpeech 2, VITS, [Chương 22](../phan-5-mo-hinh-am-hoc/ch22-fastspeech.md), [Chương 27](../phan-7-e2e-bieu-cam/ch27-vits.md)), mỗi token phải được gán một số khung $d_i \ge 1$ — nhưng token thanh không có phân đoạn âm thanh riêng. Bộ căn chỉnh thường giải quyết bằng cách gán cho nó vài khung ở cuối vần, nên mô hình vẫn hoạt động, nhưng ranh giới trở nên khó diễn giải. Có thể đặt token thanh **trước** âm tiết thay vì sau; với bộ mã hóa có ngữ cảnh hai chiều, khác biệt có lẽ nhỏ, nhưng nên kiểm chứng bằng thí nghiệm.

**(b) Gộp thanh vào âm chính.** "t iə3 ŋ". Chuỗi không dài thêm và thông tin thanh nằm đúng ở hạt nhân âm tiết. Nhược điểm: số ký hiệu nguyên âm nhân lên sáu lần (14 âm chính × 6 thanh = 84 ký hiệu), mỗi ký hiệu ít dữ liệu hơn; mô hình không "biết" rằng *a3* và *e3* có chung đường nét thanh, trừ khi tự học được.

**(c) Thanh như một đặc trưng song song.** Giữ chuỗi âm vị nguyên vẹn và gắn cho **mỗi** âm vị một nhãn thanh của âm tiết chứa nó: phones = (t, iə, ŋ), tones = (3, 3, 3). Embedding đầu vào là tổng hai bảng tra:

$$
\mathbf{e}_i = \mathbf{E}_{\text{phone}}^{\top} \mathbf{o}_{x_i} + \mathbf{E}_{\text{tone}}^{\top} \mathbf{o}_{t_i}, \qquad \mathbf{E}_{\text{tone}} \in \mathbb{R}^{6 \times C}.
$$

Phép cộng (thay vì ghép) giữ nguyên số chiều $C$ và tương đương với việc học embedding cho từng cặp (âm vị, thanh) dưới ràng buộc **phân tách cộng tính** — chính ràng buộc này giúp chia sẻ thống kê: mọi âm vị mang thanh sắc cùng dùng một véc-tơ $\mathbf{E}_{\text{tone}}[3]$. Cách làm tương tự (embedding thanh điệu cộng vào embedding âm vị) được dùng trong nhiều hệ thống TTS cho các ngôn ngữ có thanh điệu. Nhược điểm duy nhất là pipeline dữ liệu phải mang hai chuỗi song song (và mọi phép biến đổi chuỗi, như chèn blank, phải áp dụng cho cả hai).

Ngoài ra còn hai phương án "ngầm" khi dùng **đầu vào ký tự**:

- **Ký tự NFC**: dấu thanh gộp vào chữ cái ("ế" là một ký hiệu) — giống phương án (b) nhưng từ vựng còn lớn hơn (chữ có dấu phụ × 5 dấu thanh, cả hoa lẫn thường nếu không đưa về chữ thường).
- **Ký tự NFD**: dấu thanh là ký hiệu riêng ngay sau nguyên âm mang dấu — giống phương án (a) nhưng vị trí token thanh phụ thuộc quy tắc đặt dấu, và các dấu phụ (mũ, móc, trăng) cũng tách riêng, làm chuỗi dài thêm đáng kể.

### 18.5.2. So sánh định lượng trên một câu

```python
sentence = "Hà Nội mùa thu, trời trong xanh!"
n_chars = len(sentence)
sep = g2p(sentence)                                   # thanh là token riêng
fused = g2p(sentence, tone_mode="fused")              # thanh gộp vào âm chính
n_syl = sum(t.startswith("T") for t in sep)
print(f"ký tự: {n_chars} | âm tiết: {n_syl} | token (thanh riêng): {len(sep)} "
      f"| token (thanh gộp): {len(fused)} | sau add_blank: {2 * len(sep) + 1}")
assert len(sep) == len(fused) + n_syl                 # mỗi âm tiết thêm đúng 1 token thanh
```

Với câu ví dụ 7 âm tiết, phương án (a) cho 32 token, phương án (b) cho 25 token (đúng bằng 32 trừ đi 7 token thanh), và khi chèn blank (Mục 18.8) chuỗi dài 65 token. Về tổng quát, gọi $P$ là số âm vị, $S$ là số âm tiết và $B$ là số token ranh giới/dấu câu:

$$
N_{(a)} = P + S + B, \qquad N_{(b)} = N_{(c)} = P + B, \qquad N_{\text{blank}} = 2N + 1.
$$

Không có kết luận chung rằng phương án nào luôn tốt nhất; nó phụ thuộc kiến trúc (attention hay trường độ), lượng dữ liệu và cách đánh giá. Gợi ý thực hành: với mô hình dựa trên trường độ và dữ liệu ít, (c) thường là lựa chọn "an toàn"; với mô hình dựa trên attention hoặc khi muốn pipeline đơn giản, (a) là điểm khởi đầu tốt. Dù chọn gì, hãy đánh giá riêng **độ chính xác thanh điệu** bằng các cặp tối thiểu ([Chương 38](../phan-9-thuc-tien/ch38-danh-gia-tts.md)).

!!! note "Góc tiếng Việt"
    Thanh sắc và thanh nặng ở **âm tiết tắc** (âm cuối *p, t, c, ch*: *các, cạc, học, hấp*) ngắn và có đường nét F0 khác với ở âm tiết mở hay âm tiết có âm cuối vang. Một số mô tả ngữ âm tách chúng thành hai thanh riêng (nâng tổng số lên tám "thanh" theo truyền thống mô tả thanh điệu vùng Đông Nam Á). Với TTS, việc tách là **không bắt buộc**, vì âm cuối tắc đã có trong chuỗi âm vị và bộ mã hóa thấy được ngữ cảnh đó. Nhưng nếu bạn dùng phương án (c) với một bộ mã hóa rất nông, việc tách hai token "sắc tắc", "nặng tắc" là một thí nghiệm rẻ đáng thử (Bài tập 7).

## 18.6. Tách từ, từ loại và thông tin ngôn ngữ bổ sung

Khoảng trắng trong tiếng Việt phân tách **âm tiết**; một **từ** có thể gồm nhiều âm tiết ("học sinh", "thành phố Hồ Chí Minh"). G2P tiếng Việt làm việc theo âm tiết nên không cần ranh giới từ. Vậy tách từ (*word segmentation*) có ích gì cho TTS?

- **Ngắt nghỉ và ngôn điệu.** Người nói gần như không bao giờ ngắt hơi *bên trong* một từ, và ranh giới cụm từ thường trùng ranh giới từ/cụm cú pháp ([Chương 16](../phan-3-tin-hieu-ngu-am/ch16-ngon-dieu.md)). Một mô hình dự đoán ngắt nghỉ hưởng lợi rõ rệt từ thông tin từ và từ loại (*part-of-speech*, POS).
- **Tra từ điển từ mượn.** Các từ như "Facebook" hay "Ê-đê" là một từ; tra từ điển theo từ dễ hơn theo âm tiết.
- **Đặc trưng từ mô hình ngôn ngữ.** Các mô hình như PhoBERT (Nguyen và Nguyen, 2020) được tiền huấn luyện trên văn bản **đã tách từ**, nên muốn dùng biểu diễn của chúng để dự đoán ngôn điệu ([Chương 29](../phan-7-e2e-bieu-cam/ch29-prosody-cam-xuc-phong-cach.md)) thì phải tách từ đúng như lúc tiền huấn luyện.

Các công cụ tách từ và gán nhãn từ loại tiếng Việt phổ biến gồm VnCoreNLP (Vu và cộng sự, 2018) với bộ tách từ RDRsegmenter (Nguyen và cộng sự, 2018), underthesea và pyvi. Các công cụ này thường biểu diễn từ nhiều âm tiết bằng dấu gạch dưới ("học_sinh"). Tính đến thời điểm biên soạn (2026), hãy kiểm tra phiên bản, độ chính xác trên miền văn bản của bạn và giấy phép trước khi đưa vào sản phẩm.

Trong chuỗi đầu vào, ranh giới có thể được biểu diễn bằng token: ở mã nguồn của chương, token `|` đánh dấu ranh giới âm tiết; bạn có thể thêm một token riêng cho ranh giới từ (ví dụ thay `|` bằng `#` giữa hai từ) để cung cấp tín hiệu ngắt nghỉ cho mô hình. Một điểm thuận lợi đáng nói: khác với tiếng Anh, tiếng Việt hầu như không có **từ đồng tự khác âm** (*heteronym*) trong từ vựng thuần Việt và Hán Việt, nên G2P không cần từ loại để chọn cách đọc.

## 18.7. Từ ngoại lai, từ mượn và G2P cho tiếng nước ngoài

Văn bản tiếng Việt hiện đại chứa nhiều từ không tuân theo chính tả tiếng Việt: tên riêng nước ngoài (Paris, Obama), thương hiệu (Facebook, iPhone), thuật ngữ (email, internet), từ mượn đã Việt hóa một phần (xăng, ô tô, ca-mê-ra). Bộ phân tích ở Mục 18.3 từ chối chúng, và ta cần một chiến lược dự phòng. Các lựa chọn, từ đơn giản đến phức tạp:

1. **Từ điển Việt hóa** (*respelling lexicon*): ghi cách đọc bằng **chính tả tiếng Việt** ("facebook" → "phây búc", "email" → "i meo"), rồi cho qua G2P tiếng Việt. Ưu điểm lớn: không cần thêm ký hiệu âm vị nào, và người không chuyên cũng sửa được từ điển. Đây là cách làm hiệu quả nhất cho vài nghìn từ mượn thông dụng.
2. **Đánh vần** theo tên chữ cái tiếng Việt: dự phòng cuối cùng, an toàn nhưng nghe không tự nhiên với từ dài.
3. **G2P tiếng Anh + ánh xạ âm vị**: tra từ điển phát âm tiếng Anh như CMUdict (hơn một trăm nghìn mục, ký hiệu ARPAbet) hoặc dùng G2P nơ-ron tiếng Anh (ví dụ thư viện `g2p_en`) cho từ ngoài từ điển, rồi ánh xạ âm vị tiếng Anh sang âm vị tiếng Việt gần nhất. Đây là chủ đề của [Chương 30](../phan-7-e2e-bieu-cam/ch30-da-ngon-ngu-code-switching.md) (trộn mã Việt–Anh).
4. **espeak-ng**: bộ tổng hợp formant mã nguồn mở có quy tắc G2P cho rất nhiều ngôn ngữ, thường được dùng như một **bộ phiên âm** (*phonemizer*) thông qua thư viện Python `phonemizer` (Bernard và Titeux, 2021). espeak-ng có hỗ trợ tiếng Việt (tùy phiên bản có các biến thể theo miền; kiểm tra bằng lệnh `espeak-ng --voices=vi`). Ưu điểm: một công cụ cho mọi ngôn ngữ, đầu ra IPA. Nhược điểm: quy ước ký hiệu và lựa chọn phương ngữ có thể không khớp với bộ G2P của bạn, chất lượng quy tắc khác nhau giữa các ngôn ngữ, và kết quả có thể thay đổi giữa các phiên bản — hãy cố định phiên bản và kiểm thử.
5. **G2P nơ-ron**: coi G2P là bài toán dịch chuỗi ký tự sang chuỗi âm vị. Các phương pháp kinh điển gồm mô hình chuỗi kết hợp (*joint-sequence models*, Bisani và Ney, 2008), mô hình n-gram trên WFST như Phonetisaurus (Novak và cộng sự, 2016), Transformer seq2seq (Yolchuyeva và cộng sự, 2019) và mô hình đa ngôn ngữ dựa trên byte như ByT5 (Zhu và cộng sự, 2022). Với từ thuần Việt, G2P nơ-ron không cần thiết; với từ ngoại lai, nó chỉ hữu ích nếu bạn có từ điển huấn luyện đủ lớn.

Một cạm bẫy: bộ nhận diện dựa trên chính tả (Mục 18.3.1) không phát hiện được từ tiếng Anh **trùng hình thức** với âm tiết tiếng Việt hợp lệ ("me", "in", "an", "to", "pin"). Khi văn bản có nhiều tiếng Anh, cần một bộ nhận diện ngôn ngữ cấp token dùng ngữ cảnh — xem [Chương 30](../phan-7-e2e-bieu-cam/ch30-da-ngon-ngu-code-switching.md).

Khối mã dưới đây ghép mọi thứ thành hàm `g2p` cấp câu: tách token chữ và dấu câu, tra từ điển từ mượn, G2P từng âm tiết, đánh vần khi cần, chèn ranh giới âm tiết và giữ dấu câu.

```python
import re

# Từ điển "Việt hóa" cách đọc từ mượn: ghi lại bằng chính tả tiếng Việt rồi cho qua G2P
LEXICON = {"facebook": "phây búc", "email": "i meo", "ok": "ô kê",
           "video": "vi đê ô", "internet": "in tơ nét"}
LETTER_NAMES = dict(zip("abcdđefghijklmnopqrstuvwxyz",
                        ["a", "bê", "xê", "dê", "đê", "e", "ép", "gờ", "hát", "i", "gi", "ca",
                         "lờ", "mờ", "nờ", "o", "pê", "quy", "rờ", "ét", "tê", "u", "vê",
                         "vê kép", "ích", "i", "dét"]))
PUNCT = {",": ",", ";": ",", ":": ",", ".": ".", "…": ".", "!": "!", "?": "?"}
WORD_SEP = "|"                                  # ranh giới âm tiết


def word_to_syllables(word: str) -> list[str]:
    """Từ (có thể là từ ngoại lai) → danh sách âm tiết tiếng Việt đọc được."""
    w = word.lower()
    if w in LEXICON:
        return LEXICON[w].split()
    try:
        parse_syllable(w)
        return [w]
    except ValueError:                          # không phải âm tiết Việt → đánh vần
        return [s for c in w for s in LETTER_NAMES.get(c, "").split()]


def g2p(text: str, tone_mode: str = "separate", **kw) -> list[str]:
    """Câu đã chuẩn hóa (Chương 17) → chuỗi token âm vị + thanh + dấu câu."""
    out: list[str] = []
    for m in re.finditer(r"[^\W\d_]+|[,;:.…!?]", text):
        tok = m.group(0)
        if tok in PUNCT:
            if out and out[-1] == WORD_SEP:
                out.pop()                       # bỏ ranh giới thừa trước dấu câu
            out.append(PUNCT[tok])
            continue
        for syl in word_to_syllables(tok):
            out += syllable_tokens(syl, tone_mode, **kw) + [WORD_SEP]
    if out and out[-1] == WORD_SEP:
        out.pop()
    return out


print(" ".join(g2p("Xin chào, tôi dùng Facebook.")))
# Kết quả in ra:
# s i n T1 | c aː w T2 , t o j T1 | z u ŋ T2 | f ə j T1 | ɓ u k T3 .
```

## 18.8. Thiết kế bộ ký hiệu và chuỗi đầu vào

### 18.8.1. Nguyên tắc

Bộ ký hiệu $\mathcal{V}$ là **hợp đồng** giữa front-end và mô hình: một checkpoint chỉ dùng được với đúng bộ ký hiệu nó đã được huấn luyện. Vài nguyên tắc rút ra từ kinh nghiệm:

- **Suy ra từ bảng ánh xạ, không từ dữ liệu.** Nếu bạn xây bộ ký hiệu bằng cách gom các ký hiệu xuất hiện trong tập huấn luyện, một ký hiệu hiếm vắng mặt trong dữ liệu sẽ trở thành OOV lúc suy luận. Suy ra từ bảng G2P bảo đảm mọi đầu ra hợp lệ của G2P đều có trong từ vựng.
- **Cố định thứ tự và đánh phiên bản.** Đổi thứ tự hai ký hiệu là đổi nghĩa của mọi hàng trong $\mathbf{E}$. Lưu bộ ký hiệu kèm checkpoint.
- **Id 0 cho padding.** Phần lớn mã nguồn dùng id 0 để đệm chuỗi trong batch và tạo mặt nạ độ dài.
- **Giữ dấu câu.** Dấu phẩy, dấu chấm, dấu hỏi, dấu chấm than mang thông tin ngắt nghỉ và ngữ điệu (câu hỏi lên giọng — [Chương 16](../phan-3-tin-hieu-ngu-am/ch16-ngon-dieu.md)). Gom các dấu tương đương (";", ":" → ","; "…" → ".") để mỗi ký hiệu có đủ dữ liệu.
- **Token ranh giới và token đặc biệt.** Ranh giới âm tiết/từ, token khoảng lặng (một số pipeline dùng ký hiệu kiểu `sp`, `sil` lấy từ bộ căn chỉnh cưỡng bức), token bắt đầu/kết thúc câu (BOS/EOS) cho mô hình tự hồi quy, token ngôn ngữ cho mô hình đa ngôn ngữ ([Chương 30](../phan-7-e2e-bieu-cam/ch30-da-ngon-ngu-code-switching.md)).
- **Báo lỗi to, đừng bỏ qua lặng lẽ.** Lúc huấn luyện, một ký hiệu lạ phải làm chương trình dừng (`KeyError`); lúc suy luận, có cơ chế dự phòng rõ ràng và ghi log.

### 18.8.2. Chèn blank (*add_blank*)

Các bản cài đặt chính thức của Glow-TTS (Kim và cộng sự, 2020) và VITS (Kim và cộng sự, 2021) có tùy chọn `add_blank`: chèn một token "trống" vào giữa mọi cặp token và ở hai đầu,

$$
(x_1, x_2, \ldots, x_N) \;\longmapsto\; (b, x_1, b, x_2, \ldots, b, x_N, b), \qquad N' = 2N + 1,
$$

và tùy chọn này được bật trong các cấu hình phổ biến của VITS (trong mã nguồn gốc, blank dùng chung id 0 với padding). Trực giác: hai mô hình này học căn chỉnh **cứng, đơn điệu** bằng thuật toán MAS, trong đó mỗi token phải nhận ít nhất một khung ([Chương 21](../phan-5-mo-hinh-am-hoc/ch21-alignment.md)). Nhưng âm thanh thật có những đoạn **chuyển tiếp** (đồng cấu âm giữa hai âm vị) không thuộc hẳn về token nào. Token blank cho bộ căn chỉnh một chỗ để "gửi" các khung chuyển tiếp đó, nhờ đó các token thật được căn khớp với phần ổn định của âm vị. Điều này tương tự vai trò của ký hiệu blank trong CTC. Cái giá là chuỗi dài gấp đôi, nên chi phí attention tăng khoảng bốn lần và chi phí MAS tăng gấp đôi. Mức cải thiện cụ thể phụ thuộc mô hình và dữ liệu; hãy kiểm chứng bằng thí nghiệm trên dữ liệu của bạn.

### 18.8.3. Cài đặt

```python
PAD = "_"   # id 0: vừa là padding, vừa là "blank" khi chèn (giống cách làm của VITS)


def build_symbols() -> list[str]:
    """Bộ ký hiệu cố định, suy ra từ các bảng ánh xạ — không phụ thuộc dữ liệu."""
    phones = (set(ONSET_IPA.values()) | set(SPELLING_CONTRASTS.values()) | {"w", "a", "ɛ"}
              | set(NUCLEUS_IPA.values()) | {v for v in CODA_IPA.values() if v})
    tones = [f"T{t}" for t in range(1, 7)]
    return [PAD, WORD_SEP] + sorted(set(PUNCT.values())) + tones + sorted(phones)


SYMBOLS = build_symbols()
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}


def intersperse(ids: list[int], blank: int = 0) -> list[int]:
    """[x1, x2, ..., xN] → [b, x1, b, x2, ..., b, xN, b]: độ dài 2N + 1."""
    out = [blank] * (2 * len(ids) + 1)
    out[1::2] = ids
    return out


def encode(text: str, add_blank: bool = True) -> list[int]:
    ids = [SYMBOL_TO_ID[t] for t in g2p(text)]          # KeyError = ký hiệu ngoài bộ
    return intersperse(ids) if add_blank else ids


tokens = g2p("Hà Nội mùa thu, trời trong xanh!")
ids = encode("Hà Nội mùa thu, trời trong xanh!")
assert len(ids) == 2 * len(tokens) + 1 and ids[0::2] == [0] * (len(tokens) + 1)
assert [SYMBOLS[i] for i in ids[1::2]] == tokens                  # giải mã ngược khớp
for word in G2P_CASES:                                            # không có ký hiệu OOV
    assert all(t in SYMBOL_TO_ID for t in g2p(word)), word
assert g2p("Tôi dùng email") == g2p("tôi dùng i meo")             # từ điển từ mượn
print(len(SYMBOLS), "ký hiệu:", " ".join(SYMBOLS))
print(" ".join(tokens))
```

Bộ ký hiệu kết quả có 51 phần tử: padding, ranh giới, 4 dấu câu, 6 token thanh và 39 âm vị IPA (kể cả ba âm *r, ʂ, ʈ* chỉ dùng khi giữ tương phản chính tả). So với hàng trăm ký tự có dấu của đầu vào ký tự NFC, đây là một từ vựng rất gọn.

## 18.9. Token hóa cho TTS dựa trên LLM

### 18.9.1. BPE và byte

Các hệ thống TTS dựa trên mô hình ngôn ngữ lớn ([Chương 32](../phan-8-mo-hinh-lon/ch32-codec-language-model.md)–[Chương 34](../phan-8-mo-hinh-lon/ch34-he-thong-lai-hien-dai.md)) thường đưa văn bản qua **bộ token hóa của LLM văn bản** nền tảng, không qua G2P. Các bộ token hóa này dùng **mã hóa cặp byte** (*byte-pair encoding*, BPE; Sennrich và cộng sự, 2016) hoặc các biến thể như SentencePiece (Kudo và Richardson, 2018) và BPE trên byte (Radford và cộng sự, 2019). Ý tưởng của BPE: bắt đầu từ các ký hiệu cơ sở (ký tự hoặc byte), lặp lại việc **gộp cặp ký hiệu liền kề xuất hiện nhiều nhất** thành một ký hiệu mới, cho đến khi đạt kích thước từ vựng mong muốn. Từ phổ biến trở thành một token; từ hiếm bị cắt thành nhiều mảnh.

### 18.9.2. Tiếng Việt dưới góc nhìn của bộ token hóa

Hai sự thật về mã hóa giải thích vì sao tiếng Việt thường "đắt" với các bộ token hóa được huấn luyện chủ yếu trên tiếng Anh:

```python
import unicodedata

s = "Người Việt nói tiếng Việt"
for form in ("NFC", "NFD"):
    t = unicodedata.normalize(form, s)
    print(f"{form}: {len(t):2d} ký tự (code point), {len(t.encode('utf-8')):2d} byte UTF-8")
print([(c, len(c.encode("utf-8"))) for c in unicodedata.normalize("NFC", "ườ")])
print([(f"U+{ord(c):04X}", len(c.encode("utf-8"))) for c in unicodedata.normalize("NFD", "ườ")])
# Kết quả:
# NFC: 25 ký tự (code point), 35 byte UTF-8
# NFD: 35 ký tự (code point), 45 byte UTF-8
# [('ư', 2), ('ờ', 3)]
# [('U+0075', 1), ('U+031B', 2), ('U+006F', 1), ('U+031B', 2), ('U+0300', 2)]
```

Một chữ có dấu như "ờ" tốn 3 byte ở NFC và tới 5 byte (ba code point) ở NFD. Một bộ BPE trên byte chưa thấy nhiều tiếng Việt sẽ phải dùng nhiều token cho một âm tiết; tỷ lệ **số token trên âm tiết** (*fertility*) cao làm chuỗi dài ra, tăng chi phí suy luận và làm mỗi token mang ít nghĩa hơn. Thí nghiệm BPE đồ chơi dưới đây cho thấy thêm một hiện tượng: với văn bản NFD, các phép gộp có thể tạo ra những token "vô nghĩa" về ngữ âm như *dấu sắc + ng* (dấu thanh dính vào âm cuối thay vì nguyên âm).

```python
from collections import Counter


def merge_word(word: tuple, pair: tuple) -> tuple:
    out, i = [], 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            out.append(word[i] + word[i + 1])
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def learn_bpe(words: list[str], num_merges: int):
    """BPE kinh điển (Sennrich và cộng sự, 2016) trên ký tự, trong phạm vi từng âm tiết."""
    vocab = Counter(tuple(w) for w in words)
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for word, freq in vocab.items():
            for a, b in zip(word, word[1:]):
                pairs[a, b] += freq
        if not pairs:
            break
        best = max(pairs, key=pairs.get)          # cặp xuất hiện nhiều nhất
        merges.append(best)
        vocab = Counter({merge_word(w, best): f for w, f in vocab.items()})
    return merges, vocab


corpus = ("người việt nói tiếng việt . tiếng việt có sáu thanh . "
          "người nói tiếng việt cần nói đúng thanh").split()
for form in ("NFC", "NFD"):
    words = [unicodedata.normalize(form, w) for w in corpus]
    merges, vocab = learn_bpe(words, num_merges=12)
    n_tokens = sum(len(w) * f for w, f in vocab.items())
    n_chars = sum(len(w) for w in words)
    print(f"{form}: {n_chars} ký hiệu ban đầu → {n_tokens} token sau 12 phép gộp")
    print("   các token dài nhất:",
          sorted({t for w in vocab for t in w}, key=len, reverse=True)[:5])
# Kết quả:
# NFC: 74 ký hiệu ban đầu → 35 token sau 12 phép gộp
#    các token dài nhất: ['tiếng', 'người', 'việt', 'nói', 'ng']
# NFD: 102 ký hiệu ban đầu → 53 token sau 12 phép gộp
#    (các token dài nhất gồm 'việt', 'nói' và các mảnh lẻ như dấu sắc dính với 'ng')
```

Với cùng số phép gộp, văn bản NFC được nén tốt hơn (35 so với 53 token) và các token học được trùng với âm tiết thật. Bài học cho kỹ sư:

- **Luôn NFC trước bộ token hóa**, và dùng đúng dạng chuẩn hóa mà mô hình gốc đã thấy lúc tiền huấn luyện.
- **Đo fertility** của bộ token hóa trên văn bản tiếng Việt của bạn (số token chia số âm tiết). Nếu quá cao, cân nhắc **mở rộng từ vựng** (thêm các âm tiết tiếng Việt phổ biến làm token mới, khởi tạo embedding mới từ trung bình embedding các mảnh cũ) — kỹ thuật này đòi hỏi tinh chỉnh lại mô hình.
- **Kiểm tra ký tự ngoài từ vựng.** Một số mô hình TTS dùng từ vựng ký tự được xây từ dữ liệu huấn luyện gốc (thường không có tiếng Việt); khi tinh chỉnh cho tiếng Việt, các ký tự như "ự", "ẫ" có thể rơi vào token "không xác định" và bị bỏ qua lặng lẽ. Đây là lỗi phổ biến khi tinh chỉnh mô hình zero-shot cho tiếng Việt ([Dự án 4](../phan-10-du-an/du-an-4-zero-shot-tts.md)).
- **Chuẩn hóa văn bản vẫn cần.** Mô hình LLM đọc được văn bản thô không có nghĩa là đọc đúng "1.005.021" ([Chương 17](ch17-chuan-hoa-van-ban.md)).

## Lỗi thường gặp và mẹo thực hành

- **Bộ ký hiệu suy ra từ dữ liệu.** Một âm vị hiếm (ví dụ vần *oong*, âm *p* đầu từ mượn) không có trong tập huấn luyện sẽ gây `KeyError` hoặc bị bỏ qua lúc suy luận. Suy ra bộ ký hiệu từ bảng ánh xạ và chạy G2P trên một danh sách âm tiết đầy đủ để kiểm tra.
- **G2P lúc huấn luyện khác lúc suy luận.** Sửa một luật (ví dụ đổi "anh" từ "ɛ ɲ" sang "a j ŋ") mà không huấn luyện lại sẽ làm mô hình nhận chuỗi chưa từng thấy. Đánh phiên bản G2P cùng bộ ký hiệu.
- **Tách dấu thanh bằng regex trên NFC.** Cách làm sai phổ biến là liệt kê thủ công các chữ có dấu; dễ sót "ỹ", "Ỵ" hoặc chữ hoa. Dùng NFD như mã nguồn ở Mục 18.3.1.
- **Không chuẩn hóa chữ hoa.** "Việt" và "việt" là hai chuỗi khác nhau với mô hình ký tự. Hãy đưa về chữ thường trước G2P (thông tin viết hoa, nếu cần cho ngôn điệu, nên truyền bằng kênh riêng).
- **Phương ngữ G2P không khớp người nói.** Triệu chứng: mô hình học chậm, một số âm (ví dụ *r*, *v* ở giọng Nam) nghe "lai". Kiểm tra bằng cách nghe dữ liệu và so với đầu ra G2P cho các cặp tối thiểu.
- **Quên áp dụng chèn blank cho chuỗi đặc trưng song song.** Nếu dùng phương án thanh (c), chuỗi tone phải được chèn blank đúng như chuỗi phone (thường dùng một id thanh "trung tính" cho blank), nếu không hai chuỗi lệch nhau.
- **Tin espeak-ng hoặc G2P bên ngoài mà không kiểm thử.** Chạy công cụ trên danh sách cặp tối thiểu (*ma/mà/má/mả/mã/mạ*, *tr/ch*, *s/x*) để chắc chắn quy ước phù hợp với giọng đích.
- **Bỏ dấu câu khỏi chuỗi đầu vào.** Mô hình mất tín hiệu ngắt nghỉ và ngữ điệu câu hỏi; giọng đọc trở nên đều đều, liền mạch.

## Tóm tắt

- Lựa chọn đơn vị đầu vào (ký tự, âm vị, byte, BPE) quyết định mô hình phải học gì, độ dài chuỗi $N$ và độ thưa của embedding; lựa chọn tốt phụ thuộc lượng dữ liệu, kích thước mô hình và độ sâu chính tả của ngôn ngữ.
- Chính tả tiếng Việt rất nông: các bất quy tắc (c/k/q, g/gh, ng/ngh, i/y, iê/ia, *a* trong *au/ay/anh*, âm đệm *o/u*, *gi*, *qu*) đều tất định theo ngữ cảnh trong âm tiết, nên G2P dựa trên luật đạt độ chính xác gần như tuyệt đối với từ thuần Việt và Hán Việt.
- Bộ G2P gồm: tách thanh qua NFD (miễn nhiễm với vị trí đặt dấu), phân tích âm đầu/âm đệm/âm chính/âm cuối, kiểm tra chính tả (đồng thời là bộ nhận diện từ ngoại lai), và ánh xạ sang IPA theo một phương ngữ đích.
- Phương ngữ của G2P phải trùng với người nói; tùy chọn giữ tương phản chính tả hữu ích cho dữ liệu nhiều giọng.
- Thanh điệu có thể biểu diễn bằng token riêng, gộp vào nguyên âm, hoặc đặc trưng song song cộng vào embedding; mỗi cách đánh đổi độ dài chuỗi, kích thước từ vựng và sự rõ ràng của căn chỉnh.
- Tách từ không cần cho G2P tiếng Việt nhưng hữu ích cho dự đoán ngắt nghỉ, tra từ điển từ mượn và dùng đặc trưng của các mô hình như PhoBERT.
- Từ ngoại lai: từ điển Việt hóa là giải pháp hiệu quả nhất; dự phòng bằng đánh vần, G2P tiếng Anh + ánh xạ, espeak-ng hoặc G2P nơ-ron.
- Bộ ký hiệu là hợp đồng giữa front-end và mô hình: suy ra từ bảng ánh xạ, đánh phiên bản, giữ dấu câu; *add_blank* nhân đôi độ dài chuỗi để giúp căn chỉnh cứng.
- Với TTS dựa trên LLM, tiếng Việt chịu chi phí token cao; luôn NFC, đo fertility và kiểm tra ký tự ngoài từ vựng khi tinh chỉnh.

## Bài tập

**Bài 1 [Cơ bản].** Phân tích bằng tay (âm đầu, âm đệm, âm chính, âm cuối, thanh) và viết chuỗi IPA giọng Bắc cho: *khuya, quyết, giếng, gìn, thuở, ngoằn, oai, chuyện*. Kiểm tra bằng `syllable_tokens`.

??? success "Lời giải"
    *khuya*: kh + u + ya + ∅ + ngang → x w iə T1. *quyết*: qu (/k/ + /w/) + yê + t + sắc → k w iə t T3. *giếng*: gi + iê + ng + sắc → z iə ŋ T3. *gìn*: gi + i + n + huyền → z i n T2. *thuở*: th + u + ơ + ∅ + hỏi → tʰ w əː T4. *ngoằn*: ng + o + ă + n + huyền → ŋ w a n T2. *oai*: ∅ + o + a + i + ngang → ʔ w aː j T1. *chuyện*: ch + u + yê + n + nặng → c w iə n T6.

**Bài 2 [Cơ bản].** Vì sao "quốc" và "cuốc" cho hai chuỗi âm vị khác nhau, dù đều bắt đầu bằng âm /k/? Viết hai chuỗi đó và chỉ ra thành phần nào khác nhau.

??? tip "Gợi ý"
    Trong "quốc", *u* thuộc âm đầu *qu* (âm đệm /w/) và âm chính là *ô*: k w o k. Trong "cuốc", *uô* là nguyên âm đôi: k uə k.

**Bài 3 [Nâng cao].** Với một câu gồm $S$ âm tiết, $P$ âm vị và $B$ token ranh giới/dấu câu, hãy tính độ dài chuỗi đầu vào cho: (i) phương án thanh (a), (ii) phương án (b), (iii) đầu vào ký tự NFC, (iv) ký tự NFD, mỗi trường hợp có và không có *add_blank*. Ước lượng tỷ lệ tăng chi phí self-attention của (a) + blank so với (b) không blank khi $P \approx 3S$.

??? success "Lời giải"
    (a) $P + S + B$; (b) $P + B$; với blank: $2N + 1$. Khi $P \approx 3S$ và bỏ qua $B$: (a) có $N \approx 4S$, sau blank $\approx 8S$; (b) có $N \approx 3S$. Chi phí self-attention tỉ lệ $N^2$, nên tỷ lệ là $(8S)^2 / (3S)^2 = 64/9 \approx 7.1$ lần. Với ký tự, $N$ là số ký tự (NFC) hoặc số code point (NFD, lớn hơn đáng kể do mỗi dấu phụ và dấu thanh là một code point riêng).

**Bài 4 [Nâng cao].** Thiết kế bảng ánh xạ cho giọng Nam dựa trên các khác biệt ở Mục 18.4: liệt kê những thay đổi cần làm trong `ONSET_IPA`, `CODA_IPA` và cách xử lý thanh hỏi/ngã. Thảo luận: nên nhập hỏi/ngã ở G2P hay để mô hình tự học? Nêu ưu và nhược điểm của mỗi cách.

??? tip "Gợi ý"
    Các thay đổi điển hình: *d, gi, v* → [j]; *qu* → [w] (bỏ /k/); âm cuối *nh, ch* → [n], [t]; *n/ng* và *t/c* sau một số nguyên âm nhập làm một (cần luật theo nguyên âm, không chỉ theo âm cuối). Nhập hỏi/ngã ở G2P làm từ vựng nhỏ hơn và phản ánh đúng giọng, nhưng front-end không còn dùng chung được cho giọng Bắc; để mô hình tự học thì linh hoạt hơn nhưng tốn dữ liệu cho hai ký hiệu thực chất là một. Hãy tham khảo [Chương 15](../phan-3-tin-hieu-ngu-am/ch15-ngu-am-tieng-viet.md) và mô tả ngữ âm học trước khi chốt.

**Bài 5 [Lập trình].** Cài đặt tùy chọn `dialect="south"` cho `syllable_to_ipa` theo bảng của Bài 4, kèm ít nhất mười ca kiểm thử (ví dụ *vui, quà, anh, sách, bàn, bàng, mắt, mắc, hỏi, ngã*). Bảo đảm các kiểm thử giọng Bắc cũ vẫn qua.

**Bài 6 [Lập trình].** Dùng hàm `learn_bpe` để học 200 phép gộp trên một văn bản tiếng Việt dài vài nghìn âm tiết (ví dụ một chương sách thuộc phạm vi công cộng). Đo fertility (số token / số âm tiết) trên một đoạn văn khác cho cả NFC và NFD. Sau bao nhiêu phép gộp thì phần lớn âm tiết phổ biến trở thành một token?

**Bài 7 [Lập trình].** Thêm tùy chọn `checked_tones=True` cho `syllable_tokens`: âm tiết tắc mang thanh sắc/nặng dùng token `T3c`/`T6c`. Cập nhật `build_symbols` và viết kiểm thử cho *học, hát, các, cạc, ấp*. Suy nghĩ: tùy chọn này làm thay đổi phân phối token thế nào trên một kho văn bản thực?

## Tài liệu tham khảo

1. Kirby, J. P. (2011). *Vietnamese (Hanoi Vietnamese)*. Journal of the International Phonetic Association, 41(3), 381–392.
2. Sennrich, R., Haddow, B., và Birch, A. (2016). *Neural machine translation of rare words with subword units*. ACL 2016.
3. Kudo, T., và Richardson, J. (2018). *SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing*. EMNLP 2018 (System Demonstrations).
4. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., và Sutskever, I. (2019). *Language models are unsupervised multitask learners*. Báo cáo kỹ thuật, OpenAI.
5. Bisani, M., và Ney, H. (2008). *Joint-sequence models for grapheme-to-phoneme conversion*. Speech Communication, 50(5).
6. Novak, J. R., Minematsu, N., và Hirose, K. (2016). *Phonetisaurus: Exploring grapheme-to-phoneme conversion with joint n-gram models in the WFST framework*. Natural Language Engineering, 22(6).
7. Yolchuyeva, S., Németh, G., và Gyires-Tóth, B. (2019). *Transformer based grapheme-to-phoneme conversion*. Interspeech 2019.
8. Zhu, J., Zhang, C., và Jurgens, D. (2022). *ByT5 model for massively multilingual grapheme-to-phoneme conversion*. Interspeech 2022.
9. Bernard, M., và Titeux, H. (2021). *Phonemizer: Text to phones transcription for multiple languages in Python*. Journal of Open Source Software, 6(68).
10. Vu, T., Nguyen, D. Q., Nguyen, D. Q., Dras, M., và Johnson, M. (2018). *VnCoreNLP: A Vietnamese natural language processing toolkit*. NAACL 2018 (Demonstrations).
11. Nguyen, D. Q., Nguyen, D. Q., Vu, T., Dras, M., và Johnson, M. (2018). *A fast and accurate Vietnamese word segmenter*. LREC 2018.
12. Nguyen, D. Q., và Nguyen, A. T. (2020). *PhoBERT: Pre-trained language models for Vietnamese*. Findings of EMNLP 2020.
13. Kim, J., Kim, S., Kong, J., và Yoon, S. (2020). *Glow-TTS: A generative flow for text-to-speech via monotonic alignment search*. NeurIPS 2020.
14. Kim, J., Kong, J., và Son, J. (2021). *Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech*. ICML 2021.
15. Taylor, P. (2009). *Text-to-Speech Synthesis*. Cambridge University Press.
