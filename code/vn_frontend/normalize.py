"""Chuẩn hóa văn bản tiếng Việt cho TTS: chuyển văn bản viết sang dạng đọc.

Thứ tự các bước quan trọng (quy tắc cụ thể chạy trước quy tắc tổng quát):

1. làm sạch Unicode (NFC, ký tự vô hình, dấu gạch/nháy, ký tự giống nhau);
2. email, URL;
3. viết tắt theo từ điển (TP.HCM, UBND, PGS.TS.);
4. số điện thoại;
5. ngày tháng (30/4/1975, 3/2024, "ngày 2/9");
6. giờ (14:30, 7h30);
7. khoảng (5-10), số âm; tiền tệ, phần trăm, đơn vị đo, "trên một đơn vị"
   (50.000đ/kg);
8. số thứ tự, tháng, số La Mã có ngữ cảnh (thế kỷ XXI, quý IV);
9. phân số;
10. tách chữ-số dính nhau (U23), đọc mọi số còn lại;
11. từ viết tắt viết hoa không có trong từ điển: đánh vần (VTV);
12. ký hiệu (&, +, =, ~), dọn khoảng trắng, chuẩn hóa vị trí dấu thanh.
"""
from __future__ import annotations

import re

from .lexicon import (
    ABBREVIATIONS, CURRENCY_PREFIX, CURRENCY_SUFFIX, LETTER_NAMES, LOANWORDS, UNITS,
)
from .numbers import (
    DEFAULT_STYLE, NumberStyle, decimal_to_words, digits_to_words, month_to_words,
    number_to_words, ordinal_to_words, roman_to_int,
)
from .orthography import clean_unicode, normalize_tone_marks
from .syllable import is_vietnamese_syllable

NUM = r"\d+(?:[.,]\d+)*"
_I = re.IGNORECASE


def _alt(keys) -> str:
    """Tạo nhóm lựa chọn regex, khóa dài thử trước."""
    return "|".join(re.escape(k) for k in sorted(keys, key=len, reverse=True))


def read_number(tok: str, st: NumberStyle = DEFAULT_STYLE) -> str:
    """Đọc một chuỗi số có thể có dấu phân cách: 1.000.000, 3,14, 3.5, 0912, 1.2.3."""
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", tok):  # dấu chấm phân cách hàng nghìn
        return number_to_words(int(tok.replace(".", "")), st)
    if re.fullmatch(r"\d{1,3}(?:,\d{3}){2,}", tok):  # kiểu Anh: 1,000,000
        return number_to_words(int(tok.replace(",", "")), st)
    m = re.fullmatch(r"(\d+),(\d+)", tok)
    if m:  # dấu phẩy thập phân kiểu Việt
        return decimal_to_words(m[1], m[2], "phẩy", st)
    m = re.fullmatch(r"(\d+)\.(\d+)", tok)
    if m:  # dấu chấm thập phân (thường gặp trong văn bản kỹ thuật: 3.5, phiên bản)
        return decimal_to_words(m[1], m[2], "chấm", st)
    if tok.isdigit():
        if (len(tok) > 1 and tok[0] == "0") or len(tok) > 15:
            return digits_to_words(tok)  # mã số, số có số 0 đứng đầu
        return number_to_words(int(tok), st)
    parts = re.split(r"([.,])", tok)  # ví dụ 1.2.3
    return " ".join("chấm" if p == "." else "phẩy" if p == "," else read_number(p, st)
                    for p in parts if p)


# ---------------------------------------------------------------------------
# Các biểu thức chính quy (biên dịch một lần)
# ---------------------------------------------------------------------------
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL_RE = re.compile(r"(?:https?://|www\.)[^\s,;]+[^\s,;.!?)]", _I)
_ABBR_RE = re.compile(
    r"(?<![\w.])(?:"
    + "|".join(re.escape(k) if k.endswith(".") else re.escape(k) + r"(?!\w)"
               for k in sorted(ABBREVIATIONS, key=len, reverse=True))
    + ")"
)
_PHONE_RE = re.compile(r"(?<![\w.,+])(\+84 ?|0)(\d{2,4}(?:[ .-]?\d{2,4}){1,3})(?![\w])")
_DATE_DMY_RE = re.compile(r"(?<![\d/.-])(\d{1,2})([/.-])(\d{1,2})\2(\d{4})(?![\d/.-]?\d)")
_DATE_ISO_RE = re.compile(r"(?<![\d/.-])(\d{4})-(\d{1,2})-(\d{1,2})(?![\d/.-]?\d)")
_DATE_MY_RE = re.compile(r"(?<![\d/.-])(\d{1,2})/(\d{4})(?![\d/])")
_DATE_DM_RE = re.compile(r"(?i:\b(ngày|mùng|mồng|hôm|sáng|trưa|chiều|tối|đêm))\s+(\d{1,2})/(\d{1,2})(?![\d/])")
_TIME_COLON_RE = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?::(\d{2}))?(?![\d:])")
_TIME_H_RE = re.compile(r"(?<![\w.,])(\d{1,2})(?:h|g(?=\d))(\d{2})?(?:ph|p|')?(?!\w)")
_CUR_SUFFIX_RE = re.compile(rf"({NUM})\s?({_alt(CURRENCY_SUFFIX)})(?!\w)")
_CUR_PREFIX_RE = re.compile(rf"({_alt(CURRENCY_PREFIX)})\s?({NUM})")
_KTR_RE = re.compile(rf"(?<![\w.,])({NUM})\s?(k|K|tr)(?!\w)")
_PERCENT_RE = re.compile(rf"({NUM})\s?%")
_UNIT_RE = re.compile(rf"({NUM})\s?({_alt(UNITS)})(?![\w²³])")
_CUR_PER_RE = re.compile(r"(đồng|đô la Mỹ|đô la|ơ rô|bảng Anh|yên)\s?/\s?(?=[^\W\d_])")
_PER_RE = re.compile(
    rf"\s?/\s?({_alt(UNITS)}|tháng|ngày|năm|giờ|tuần|người|lần|chiếc|cái|tấn|lít|vé|suất)(?![\w²³])")
_ORDINAL_RE = re.compile(r"(?i:\b(thứ|hạng|giải))\s+(\d{1,3})(?![\d.,]?\d)")
_MONTH_RE = re.compile(r"(?i:\b(tháng))\s+(\d{1,2})(?![\d.,]?\d)")
_ROMAN_CTX_RE = re.compile(
    r"(?i:\b(thế kỷ|thế kỉ|quý|khóa|khoá|kỳ|kì|đại hội|chương|phần|tập|hồi|thứ))"
    r"\s+([IVX]+|[IVXLCDM]{2,})(?:\s?/\s?(\d{4}))?(?![\w])")
_QUARTER_RE = re.compile(r"(?i:\b(quý))\s+(\d)\s?/\s?(\d{4})(?!\d)")
_FRACTION_RE = re.compile(r"(?<![\w/.,])(\d+)/(\d+)(?![\w/])")
_RANGE_RE = re.compile(rf"(?<![\w.,])({NUM})\s?-\s?({NUM})")
_NEG_RE = re.compile(r"(?<![\w.,])-(?=\d)")
_LETTER_DIGIT_RE = re.compile(r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])")
_NUM_RE = re.compile(NUM)
_ACRONYM_RE = re.compile(r"(?<![\w])([A-ZĐ]{2,}|[B-DF-HJ-NP-TV-XZĐ])(?![\w])")


def _spell(token: str) -> str:
    """Đánh vần một từ viết tắt theo tên chữ cái tiếng Việt."""
    return " ".join(LETTER_NAMES.get(c, c) for c in token.lower())


def _read_phone(m: re.Match[str]) -> str:
    prefix, rest = m.group(1).strip(), m.group(2)
    groups = re.split(r"[ .-]", rest)
    n_digits = len(prefix.lstrip("+")) + sum(len(g) for g in groups)
    if not 9 <= n_digits <= 13:
        return m.group(0)
    head = "cộng " + digits_to_words(prefix[1:]) if prefix.startswith("+") else "không"
    if len(groups) == 1:  # số viết liền: tách nhóm 3-3-3 cho dễ nghe (sau đầu số)
        g = groups[0]
        groups = [g[i:i + 3] for i in range(0, len(g), 3)]
        if len(groups[-1]) == 1:  # tránh nhóm cuối chỉ một chữ số
            groups[-2:] = [groups[-2] + groups[-1]]
    if prefix == "0":
        groups[0] = "0" + groups[0]
        return ", ".join(digits_to_words(g) for g in groups)
    return head + ", " + ", ".join(digits_to_words(g) for g in groups)


def normalize_text(text: str, *, tone_style: str = "old",
                   style: NumberStyle | None = None) -> str:
    """Chuyển văn bản viết sang dạng đọc (chữ), giữ nguyên dấu câu.

    Args:
        text: văn bản đầu vào (bất kỳ dạng Unicode nào).
        tone_style: ``"old"`` (hòa, khỏe, thủy — quy ước của giáo trình) hoặc
            ``"new"`` (hoà, khoẻ, thuỷ).
        style: cách đọc số (``NumberStyle``): nghìn/ngàn, linh/lẻ, tư/bốn, lăm/nhăm.

    Hàm có tính lũy đẳng: ``normalize_text(normalize_text(x)) == normalize_text(x)``.
    """
    st = style or DEFAULT_STYLE
    num = lambda s: read_number(s, st)  # noqa: E731

    text = clean_unicode(text)

    # Email và URL: đọc ký hiệu
    def email(m: re.Match[str]) -> str:
        local, domain = m.group(0).split("@", 1)
        say = lambda s: re.sub(r"\.", " chấm ", s).replace("-", " gạch ngang ")  # noqa: E731
        return f" {say(local)} a còng {say(domain)} "

    def url(m: re.Match[str]) -> str:
        u = re.sub(r"^https?://", "", m.group(0), flags=_I)
        u = u.replace(".", " chấm ").replace("/", " gạch chéo ")
        return " " + u.replace("-", " gạch ngang ").replace("_", " gạch dưới ") + " "

    text = _EMAIL_RE.sub(email, text)
    text = _URL_RE.sub(url, text)

    # Viết tắt theo từ điển (hai lượt: "TP.HCM" -> "thành phố HCM" -> "... Hồ Chí Minh")
    for _ in range(2):
        text = _ABBR_RE.sub(lambda m: f" {ABBREVIATIONS[m.group(0)]} ", text)

    # Số điện thoại
    text = _PHONE_RE.sub(_read_phone, text)

    # Ngày tháng
    def dmy(m: re.Match[str]) -> str:
        return dmy_values(m, int(m[1]), int(m[3]), int(m[4]))

    def dmy_values(m: re.Match[str], d: int, mo: int, y: int) -> str:
        if not (1 <= d <= 31 and 1 <= mo <= 12):
            return m.group(0)
        before = m.string[:m.start()].rstrip().lower()
        lead = "" if before.endswith(("ngày", "mùng", "mồng", "hôm")) else "ngày "
        return f"{lead}{number_to_words(d, st)} tháng {month_to_words(mo, st)} năm {number_to_words(y, st)}"

    def my(m: re.Match[str]) -> str:
        mo, y = int(m[1]), int(m[2])
        if not 1 <= mo <= 12:
            return m.group(0)
        before = m.string[:m.start()].rstrip().lower()
        lead = "" if before.endswith("tháng") else "tháng "
        return f"{lead}{month_to_words(mo, st)} năm {number_to_words(y, st)}"

    def dm(m: re.Match[str]) -> str:
        d, mo = int(m[2]), int(m[3])
        if not (1 <= d <= 31 and 1 <= mo <= 12):
            return m.group(0)
        return f"{m[1]} {number_to_words(d, st)} tháng {month_to_words(mo, st)}"

    text = _DATE_ISO_RE.sub(lambda m: dmy_values(m, int(m[3]), int(m[2]), int(m[1])), text)
    text = _DATE_DMY_RE.sub(dmy, text)
    text = _DATE_MY_RE.sub(my, text)
    text = _DATE_DM_RE.sub(dm, text)

    # Giờ
    def hms(h: int, mi: int, s: int | None) -> str | None:
        if h > 24 or mi > 59 or (s is not None and s > 59):
            return None
        out = f"{number_to_words(h, st)} giờ"
        if mi or s:
            out += f" {number_to_words(mi, st)} phút"
        if s:
            out += f" {number_to_words(s, st)} giây"
        return out

    def time_colon(m: re.Match[str]) -> str:
        r = hms(int(m[1]), int(m[2]), int(m[3]) if m[3] else None)
        return r if r is not None else m.group(0)

    def time_h(m: re.Match[str]) -> str:
        r = hms(int(m[1]), int(m[2] or 0), None)
        return r if r is not None else m.group(0)

    text = _TIME_COLON_RE.sub(time_colon, text)
    text = _TIME_H_RE.sub(time_h, text)

    # Khoảng (5-10) và số âm: trước tiền tệ/đơn vị để "20-25°C", "-5°C" đọc đúng
    text = _RANGE_RE.sub(lambda m: f"{m[1]} đến {m[2]}", text)
    text = _NEG_RE.sub("âm ", text)

    # Tiền tệ, phần trăm, đơn vị
    text = _CUR_PREFIX_RE.sub(lambda m: f"{num(m[2])} {CURRENCY_PREFIX[m[1]]}", text)
    text = _CUR_SUFFIX_RE.sub(lambda m: f"{num(m[1])} {CURRENCY_SUFFIX[m[2]]}", text)
    text = _KTR_RE.sub(lambda m: f"{num(m[1])} {'triệu' if m[2] == 'tr' else 'nghìn'}", text)
    text = _PERCENT_RE.sub(lambda m: f"{num(m[1])} phần trăm", text)
    text = _UNIT_RE.sub(lambda m: f"{num(m[1])} {UNITS[m[2]]}", text)
    text = _PER_RE.sub(lambda m: f" một {UNITS.get(m[1], m[1])}", text)
    text = _CUR_PER_RE.sub(lambda m: f"{m[1]} một ", text)

    # Số thứ tự, tháng, số La Mã theo ngữ cảnh
    text = _ORDINAL_RE.sub(lambda m: f"{m[1]} {ordinal_to_words(int(m[2]), st)}", text)
    text = _MONTH_RE.sub(
        lambda m: f"{m[1]} {month_to_words(int(m[2]), st)}" if 1 <= int(m[2]) <= 12 else m.group(0),
        text)

    def roman(m: re.Match[str]) -> str:
        try:
            v = roman_to_int(m[2])
        except ValueError:
            return m.group(0)
        word = ordinal_to_words(v, st) if m[1].lower() == "thứ" else number_to_words(v, st)
        tail = f" năm {number_to_words(int(m[3]), st)}" if m[3] else ""
        return f"{m[1]} {word}{tail}"

    text = _ROMAN_CTX_RE.sub(roman, text)
    text = _QUARTER_RE.sub(
        lambda m: f"{m[1]} {number_to_words(int(m[2]), st)} năm {number_to_words(int(m[3]), st)}",
        text)

    # Phân số
    text = _FRACTION_RE.sub(lambda m: f"{num(m[1])} phần {num(m[2])}", text)

    # Tách chữ-số dính nhau (U23, COVID-19) rồi đọc mọi số còn lại
    text = re.sub(r"(?<=[^\W\d_])-(?=\d)", " ", text)
    text = _LETTER_DIGIT_RE.sub(" ", text)
    text = _NUM_RE.sub(lambda m: num(m.group(0)), text)

    # Từ viết hoa: từ mượn, âm tiết tiếng Việt viết hoa, hay từ viết tắt cần đánh vần
    def acronym(m: re.Match[str]) -> str:
        tok = m.group(1)
        if tok.lower() in LOANWORDS:
            return LOANWORDS[tok.lower()]
        if is_vietnamese_syllable(tok):
            return tok
        return _spell(tok)

    text = _ACRONYM_RE.sub(acronym, text)

    # Ký hiệu còn sót
    for sym, word in (("&", " và "), ("+", " cộng "), ("=", " bằng "), ("×", " nhân "),
                      ("~", " khoảng "), ("≈", " xấp xỉ "), ("°", " độ "), ("@", " a còng "),
                      ("%", " phần trăm ")):
        text = text.replace(sym, word)
    text = re.sub(r"[#*^|\\<>{}]", " ", text)
    text = re.sub(r"(?<=\w)/(?=\w)", " ", text)  # và/hoặc -> và hoặc
    text = re.sub(r"(?<=\s)-+(?=\s)|(?<=\w)-(?=\s)|(?<=\s)-(?=\w)", " ", text)
    text = re.sub(r"(?m)^[ ]*-[ ]*", "", text)  # gạch đầu dòng hội thoại

    # Dọn khoảng trắng quanh dấu câu
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r" +([,.;:!?…)\]])", r"\1", text)
    text = re.sub(r"([(\[]) +", r"\1", text)
    text = re.sub(r" *\n *", "\n", text)
    return normalize_tone_marks(text.strip(), tone_style)
