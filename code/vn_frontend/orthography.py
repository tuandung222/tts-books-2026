"""Chính tả tiếng Việt: Unicode NFC, tách/gắn dấu thanh, chuẩn hóa vị trí dấu.

Quy ước đánh số thanh điệu dùng trong toàn bộ gói ``vn_frontend``::

    1 = ngang (không dấu)   2 = huyền (`)   3 = sắc (´)
    4 = hỏi (?)             5 = ngã (~)     6 = nặng (.)

Mọi hàm ở đây là Python thuần, không phụ thuộc thư viện ngoài.
"""
from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Thanh điệu
# ---------------------------------------------------------------------------
NGANG, HUYEN, SAC, HOI, NGA, NANG = 1, 2, 3, 4, 5, 6
TONE_NAMES = {1: "ngang", 2: "huyền", 3: "sắc", 4: "hỏi", 5: "ngã", 6: "nặng"}

# Dấu thanh ở dạng ký tự tổ hợp (combining) sau khi tách NFD.
_MARK_TO_TONE = {
    "̀": HUYEN,  # dấu huyền (grave)
    "́": SAC,  # dấu sắc (acute)
    "̉": HOI,  # dấu hỏi (hook above)
    "̃": NGA,  # dấu ngã (tilde)
    "̣": NANG,  # dấu nặng (dot below)
    "̀": HUYEN,  # biến thể cũ của dấu huyền
    "́": SAC,  # biến thể cũ của dấu sắc
}
_TONE_TO_MARK = {HUYEN: "̀", SAC: "́", HOI: "̉", NGA: "̃", NANG: "̣"}

VOWELS = frozenset("aăâeêioôơuưy")
# Nguyên âm mang dấu phụ (mũ, trăng, móc): dấu thanh ưu tiên đặt lên chúng.
_DIACRITIC_VOWELS = frozenset("ăâêôơư")

# Âm đầu viết (sắp xếp dài trước để so khớp tham lam).
ONSETS = (
    "ngh", "ch", "gh", "gi", "kh", "ng", "nh", "ph", "qu", "th", "tr",
    "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "x",
)
CONSONANT_CODAS = frozenset({"", "c", "ch", "m", "n", "ng", "nh", "p", "t"})


def split_tone(word: str) -> tuple[str, int]:
    """Tách dấu thanh khỏi một từ.

    Trả về ``(chuỗi không dấu thanh ở dạng NFC, số thanh 1..6)``. Các dấu phụ
    của nguyên âm (â, ă, ê, ô, ơ, ư) và chữ đ được giữ nguyên.

    >>> split_tone("Việt")
    ('Viêt', 6)
    """
    tone = NGANG
    kept: list[str] = []
    for ch in unicodedata.normalize("NFD", word):
        t = _MARK_TO_TONE.get(ch)
        if t is None:
            kept.append(ch)
            continue
        if tone != NGANG and t != tone:
            raise ValueError(f"từ có nhiều hơn một dấu thanh: {word!r}")
        tone = t
    return unicodedata.normalize("NFC", "".join(kept)), tone


def add_tone(char: str, tone: int) -> str:
    """Gắn dấu thanh ``tone`` lên một ký tự nguyên âm (kết quả ở dạng NFC)."""
    if tone == NGANG:
        return char
    return unicodedata.normalize("NFC", char + _TONE_TO_MARK[tone])


def split_orthographic(base: str) -> tuple[str, int, int]:
    """Tách một âm tiết **không dấu thanh, chữ thường** thành các phần chính tả.

    Trả về ``(âm_đầu_viết, vị_trí_bắt_đầu_cụm_nguyên_âm, vị_trí_kết_thúc)``.
    Cụm nguyên âm là ``base[start:end]``; phần còn lại ``base[end:]`` là âm cuối
    phụ âm. Hai trường hợp đặc biệt:

    * ``gi`` + nguyên âm (``gia``, ``giữ``): ``gi`` là âm đầu, cụm nguyên âm bắt
      đầu sau ``gi``. ``gi`` không có nguyên âm theo sau (``gì``, ``gìn``): chữ
      ``i`` được dùng chung, vừa thuộc âm đầu vừa là âm chính.
    * ``qu``: ``u`` là âm đệm, không thuộc cụm nguyên âm khi đặt dấu.

    Ném ``ValueError`` nếu không có nguyên âm hoặc âm cuối không hợp lệ.
    """
    onset = ""
    for cand in ONSETS:
        if base.startswith(cand):
            onset = cand
            break
    start = len(onset)
    if onset == "gi" and (len(base) == 2 or base[2] not in VOWELS):
        start = 1  # "gì", "gìn": chữ i là âm chính
    elif onset == "qu" and (len(base) == 2 or base[2] not in VOWELS):
        raise ValueError(f"không phải âm tiết tiếng Việt: {base!r}")
    end = start
    while end < len(base) and base[end] in VOWELS:
        end += 1
    if end == start:
        raise ValueError(f"không có nguyên âm: {base!r}")
    if base[end:] not in CONSONANT_CODAS:
        raise ValueError(f"âm cuối không hợp lệ: {base!r}")
    return onset, start, end


def tone_position(base: str, style: str = "old") -> int:
    """Vị trí (chỉ số ký tự) cần đặt dấu thanh trong âm tiết ``base``.

    ``style="old"``: kiểu đặt dấu truyền thống mà giáo trình dùng (hóa, khỏe,
    thủy). ``style="new"``: kiểu "mới" (hoá, khoẻ, thuỷ). Hai kiểu chỉ khác
    nhau ở các vần mở ``oa``, ``oe``, ``uy``.
    """
    if style not in ("old", "new"):
        raise ValueError("style phải là 'old' hoặc 'new'")
    low = base.lower()
    _, start, end = split_orthographic(low)
    cluster = low[start:end]
    has_coda = end < len(low)
    marked = [i for i, c in enumerate(cluster) if c in _DIACRITIC_VOWELS]
    if marked:
        idx = marked[-1]  # "ươ" -> đặt trên ơ; "uô" -> ô; "iê" -> ê
    elif len(cluster) == 1:
        idx = 0
    elif has_coda:
        idx = len(cluster) - 1  # "oan" -> a, "uynh" -> y
    elif len(cluster) == 3:
        idx = 1  # "oai" -> a, "uyu" -> y, "oeo" -> e
    elif style == "new" and cluster in ("oa", "oe", "uy"):
        idx = 1
    else:
        idx = 0  # "ai" -> á, "ua" -> ủ (của), "ia" -> í (mía), old: hóa
    return start + idx


def apply_tone(base: str, tone: int, style: str = "old") -> str:
    """Đặt dấu thanh lên âm tiết không dấu ``base``, giữ nguyên chữ hoa/thường."""
    if tone == NGANG:
        return base
    pos = tone_position(base, style)
    ch = base[pos]
    marked = add_tone(ch.lower(), tone)
    if ch.isupper():
        marked = marked.upper()
    return base[:pos] + marked + base[pos + 1:]


_WORD_RE = re.compile(r"[^\W\d_]+")


def normalize_tone_marks(text: str, style: str = "old") -> str:
    """Đưa vị trí dấu thanh về một kiểu thống nhất (mặc định kiểu "cũ": hòa).

    Từ nào không phân tích được như âm tiết tiếng Việt (từ nước ngoài, lỗi gõ
    có hai dấu thanh...) được giữ nguyên.
    """

    def fix(m: re.Match[str]) -> str:
        word = m.group(0)
        try:
            base, tone = split_tone(word)
            if tone == NGANG:
                return word
            return apply_tone(base, tone, style)
        except ValueError:
            return word

    return _WORD_RE.sub(fix, unicodedata.normalize("NFC", text))


# ---------------------------------------------------------------------------
# Làm sạch Unicode
# ---------------------------------------------------------------------------
_CHAR_MAP = {
    " ": " ",  # khoảng trắng không ngắt dòng
    " ": " ",
    " ": " ",
    "​": "",  # zero-width space
    "‌": "",
    "‍": "",
    "⁠": "",
    "﻿": "",  # BOM
    "­": "",  # soft hyphen
    "–": "-",  # en dash
    "−": "-",  # dấu trừ toán học
    "‐": "-",
    "‑": "-",
    "“": '"',
    "”": '"',
    "„": '"',
    "«": '"',
    "»": '"',
    "‘": "'",
    "’": "'",
    "Ð": "Đ",  # Ð (Eth, tiếng Iceland) trông giống hệt Đ
    "Ɖ": "Đ",  # Ɖ (African D)
    "\t": " ",
    "\r": "",
}
# Ký tự Kirin trông giống chữ Latinh; chỉ thay khi từ đó có lẫn chữ Latinh.
_CYRILLIC_LOOKALIKE = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "Х": "X",
})
_CYR_RE = re.compile(r"[Ѐ-ӿ]")
_LAT_RE = re.compile(r"[A-Za-zÀ-ỹĐđ]")
_TOKEN_RE = re.compile(r"\S+")


def clean_unicode(text: str) -> str:
    """NFC, bỏ ký tự vô hình, thống nhất dấu gạch/nháy, sửa ký tự "giống mà khác".

    Giữ nguyên xuống dòng (để tách câu/đoạn); gộp các khoảng trắng liên tiếp.
    """
    text = unicodedata.normalize("NFC", text)
    text = "".join(_CHAR_MAP.get(c, c) for c in text)

    def fix_mixed(m: re.Match[str]) -> str:
        tok = m.group(0)
        if _CYR_RE.search(tok) and _LAT_RE.search(tok):
            return tok.translate(_CYRILLIC_LOOKALIKE)
        return tok

    text = _TOKEN_RE.sub(fix_mixed, text)
    text = text.replace("...", "…")
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return unicodedata.normalize("NFC", text.strip())
