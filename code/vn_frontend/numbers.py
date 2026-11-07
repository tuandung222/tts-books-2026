"""Đọc số tiếng Việt: số đếm, số thứ tự, số thập phân, đọc từng chữ số, số La Mã.

Các quy tắc chính (đã có kiểm thử trong ``tests/test_vn_frontend.py``):

* 10 = "mười", 20..90 = "... mươi" (không dấu);
* hàng đơn vị 1 sau "mươi" đọc "mốt" (21 = hai mươi mốt), nhưng 11 = mười một;
* hàng đơn vị 4 sau "mươi" đọc "tư" hoặc "bốn" (tùy chọn), nhưng 14 = mười bốn;
* hàng đơn vị 5 sau "mười/mươi" đọc "lăm" (hoặc "nhăm" sau "mươi"), nhưng
  sau "linh/lẻ" và khi đứng một mình vẫn là "năm";
* hàng chục bằng 0 giữa số: "linh" (miền Bắc) hoặc "lẻ" (miền Nam);
* nhóm ba chữ số không đứng đầu luôn đọc đủ, kể cả "không trăm";
* các bậc nghìn (ngàn), triệu, tỷ; trên tỷ ghép lại: nghìn tỷ, triệu tỷ, tỷ tỷ.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

DIGITS = ("không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín")


@dataclass(frozen=True)
class NumberStyle:
    """Các lựa chọn phương ngữ/phong cách khi đọc số."""

    thousand: str = "nghìn"  # hoặc "ngàn"
    zero_tens: str = "linh"  # hoặc "lẻ"
    four: str = "tư"  # đọc hàng đơn vị 4 sau "mươi": "tư" hoặc "bốn"
    five: str = "lăm"  # đọc hàng đơn vị 5 sau "mươi": "lăm" hoặc "nhăm"
    billion: str = "tỷ"  # hoặc "tỉ"


DEFAULT_STYLE = NumberStyle()


def _two_digits(tens: int, units: int, st: NumberStyle) -> list[str]:
    """Đọc phần chục-đơn vị khi ``tens >= 1``."""
    if tens == 1:
        if units == 0:
            return ["mười"]
        if units == 5:
            return ["mười", "lăm"]
        return ["mười", DIGITS[units]]  # 11 mười một, 14 mười bốn
    words = [DIGITS[tens], "mươi"]
    if units == 0:
        return words
    if units == 1:
        return words + ["mốt"]
    if units == 4:
        return words + [st.four]
    if units == 5:
        return words + [st.five]
    return words + [DIGITS[units]]


def _three_digits(n: int, full: bool, st: NumberStyle) -> list[str]:
    """Đọc một nhóm 0 < n < 1000. ``full``: đọc đủ "không trăm" (nhóm không đứng đầu)."""
    h, t, u = n // 100, (n // 10) % 10, n % 10
    words: list[str] = []
    if full or h > 0:
        words += [DIGITS[h], "trăm"]
    if t == 0:
        if u:
            if words:
                words.append(st.zero_tens)  # 105 = một trăm linh năm
            words.append(DIGITS[u])
        return words
    return words + _two_digits(t, u, st)


def _read(n: int, full: bool, st: NumberStyle) -> list[str]:
    """Đọc số nguyên dương ``n``; ``full=True`` khi ``n`` là phần tiếp sau một bậc lớn hơn."""
    if n >= 10**9:
        high, low = divmod(n, 10**9)
        words = _read(high, full, st) + [st.billion]
        if low:
            words += _read(low, True, st)
        return words
    words: list[str] = []
    groups = ((n // 10**6) % 1000, "triệu"), ((n // 1000) % 1000, st.thousand), (n % 1000, "")
    for value, scale in groups:
        if value == 0:
            continue  # bỏ qua nhóm toàn số 0
        words += _three_digits(value, full or bool(words), st)
        if scale:
            words.append(scale)
    return words


def number_to_words(n: int, style: NumberStyle = DEFAULT_STYLE) -> str:
    """Đọc số nguyên (có thể âm) thành chữ.

    >>> number_to_words(2026)
    'hai nghìn không trăm hai mươi sáu'
    """
    if n < 0:
        return "âm " + number_to_words(-n, style)
    if n == 0:
        return "không"
    return " ".join(_read(n, False, style))


def digits_to_words(digits: str) -> str:
    """Đọc từng chữ số: ``"0912"`` -> ``"không chín một hai"``."""
    return " ".join(DIGITS[int(c)] for c in digits if c.isdigit())


def decimal_to_words(int_part: str, frac_part: str, sep: str = "phẩy",
                     style: NumberStyle = DEFAULT_STYLE) -> str:
    """Đọc số thập phân. Phần lẻ ≤ 3 chữ số và không bắt đầu bằng 0 được đọc như
    số nguyên (3,14 = ba phẩy mười bốn); ngược lại đọc từng chữ số
    (0,05 = không phẩy không năm)."""
    left = number_to_words(int(int_part), style) if int_part else "không"
    if len(frac_part) <= 3 and not frac_part.startswith("0"):
        right = number_to_words(int(frac_part), style)
    else:
        right = digits_to_words(frac_part)
    return f"{left} {sep} {right}"


def ordinal_to_words(n: int, style: NumberStyle = DEFAULT_STYLE) -> str:
    """Số thứ tự đứng sau "thứ/hạng": 1 = nhất, 4 = tư, còn lại như số đếm."""
    if n == 1:
        return "nhất"
    if n == 4:
        return "tư"
    return number_to_words(n, style)


def month_to_words(m: int, style: NumberStyle = DEFAULT_STYLE) -> str:
    """Tên tháng sau chữ "tháng": tháng tư (4), các tháng khác đọc như số đếm."""
    return "tư" if m == 4 else number_to_words(m, style)


_ROMAN_RE = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(s: str) -> int:
    """Đổi số La Mã hợp lệ (I..MMMCMXCIX) sang số nguyên; ném ``ValueError`` nếu sai."""
    if not s or not _ROMAN_RE.match(s):
        raise ValueError(f"không phải số La Mã hợp lệ: {s!r}")
    total = 0
    for i, c in enumerate(s):
        v = _ROMAN_VALUES[c]
        nxt = _ROMAN_VALUES[s[i + 1]] if i + 1 < len(s) else 0
        total += -v if v < nxt else v
    return total
