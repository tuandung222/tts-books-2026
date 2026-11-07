"""G2P tiếng Việt dựa trên luật: chữ viết -> âm vị (IPA) + token thanh điệu.

Đầu ra của ``g2p`` là danh sách token:

* âm vị IPA của từng âm tiết theo thứ tự âm đầu, âm đệm, âm chính, âm cuối;
* một token thanh ``T1``..``T6`` ở **cuối** mỗi âm tiết
  (1 ngang, 2 huyền, 3 sắc, 4 hỏi, 5 ngã, 6 nặng);
* ``" "`` (``WORD_BOUNDARY``) giữa hai từ viết cách nhau bởi khoảng trắng; các
  âm tiết nối bằng ``_`` hoặc ``-`` (``học_sinh``, ``ki-lô-mét``) không có ranh giới;
* dấu câu ``, . ! ? ; : …`` giữ làm token ngắt nghỉ.

Bảng âm vị là một **lựa chọn có chủ đích, đơn giản hóa** dựa trên mô tả giọng Hà
Nội; ``dialect="south"`` là xấp xỉ đơn giản giọng Sài Gòn. Xem trang Dự án 1 để
biết lý do của từng lựa chọn và cách mở rộng.
"""
from __future__ import annotations

import re
import unicodedata

from .lexicon import LETTER_NAMES, LOANWORDS
from .normalize import normalize_text, read_number
from .orthography import HOI, NGA
from .syllable import Syllable, parse_syllable

WORD_BOUNDARY = " "
TONE_TOKENS = ("T1", "T2", "T3", "T4", "T5", "T6")
PUNCTUATION = (",", ".", "!", "?", ";", ":", "…")
_PUNCT_MAP = {"—": ",", "(": ",", ")": ",", "[": ",", "]": ",", "\n": "."}

# --- Âm đầu (chữ viết -> IPA) -------------------------------------------------
ONSET_NORTH = {
    "": "ʔ", "b": "ɓ", "c": "k", "k": "k", "q": "k", "ch": "c", "tr": "c",
    "d": "z", "gi": "z", "r": "z", "đ": "ɗ", "g": "ɣ", "gh": "ɣ", "h": "h",
    "kh": "x", "l": "l", "m": "m", "n": "n", "ng": "ŋ", "ngh": "ŋ", "nh": "ɲ",
    "p": "p", "ph": "f", "s": "s", "x": "s", "t": "t", "th": "tʰ", "v": "v",
}
# Giọng Nam (đơn giản hóa): giữ tr/s quặt lưỡi, d/gi/v -> j, r -> ʐ (thực tế r
# biến thiên mạnh giữa các người nói: [ɹ], [ʐ], [r]...).
ONSET_SOUTH = dict(ONSET_NORTH, tr="ʈ", s="ʂ", r="ʐ", d="j", gi="j", v="j")

GLIDE = {"o": "w", "u": "w"}

NUCLEUS = {
    "a": "a", "ă": "ă", "â": "ə̆", "e": "ɛ", "ê": "e", "i": "i", "y": "i",
    "o": "ɔ", "ô": "o", "ơ": "ə", "u": "u", "ư": "ɨ",
    "iê": "iə", "yê": "iə", "ia": "iə", "ya": "iə",
    "uô": "uə", "ua": "uə", "ươ": "ɨə", "ưa": "ɨə",
    "oo": "ɔː", "ôô": "oː",
}

CODA = {
    "c": "k", "ch": "c", "m": "m", "n": "n", "ng": "ŋ", "nh": "ɲ", "p": "p", "t": "t",
    "i": "j", "y": "j", "o": "w", "u": "w",
}

# Toàn bộ âm vị mà các bảng trên có thể sinh ra (thứ tự cố định để ID ổn định).
PHONES = (
    "ʔ", "ɓ", "c", "ɗ", "f", "h", "k", "l", "m", "n", "ŋ", "ɲ", "p", "s", "t", "tʰ",
    "v", "x", "z", "ɣ", "ʈ", "ʂ", "ʐ", "j", "w",
    "a", "ă", "ə̆", "ɛ", "e", "i", "ɔ", "o", "ə", "u", "ɨ", "iə", "uə", "ɨə", "ɔː", "oː",
)

DIALECTS = ("north", "south")


def syllable_to_phones(syl: Syllable | str, dialect: str = "north",
                       with_tone: bool = True) -> list[str]:
    """Đổi một âm tiết sang danh sách âm vị (kèm token thanh nếu ``with_tone``)."""
    if dialect not in DIALECTS:
        raise ValueError(f"dialect phải là một trong {DIALECTS}, nhận {dialect!r}")
    if isinstance(syl, str):
        syl = parse_syllable(syl)
    south = dialect == "south"
    onsets = ONSET_SOUTH if south else ONSET_NORTH

    phones: list[str] = []
    if south and syl.onset == "q":
        phones.append("w")  # qu -> [w] ở giọng Nam: "quả" ~ [wa]
    else:
        phones.append(onsets[syl.onset])
        if syl.glide:
            phones.append(GLIDE[syl.glide])

    nucleus = NUCLEUS[syl.nucleus]
    coda = CODA.get(syl.coda, "") if syl.coda else ""
    if syl.nucleus == "a" and syl.coda in ("y", "u"):
        nucleus = "ă"  # "ay", "au": a ngắn /ăj/, /ăw/ (khác "ai", "ao")
    if not south and syl.nucleus == "a" and syl.coda in ("nh", "ch"):
        nucleus = "ɛ"  # "anh", "ach": phân tích phổ biến coi âm chính là e ngắn
    if south:
        front = nucleus in ("i", "e")
        if syl.coda == "n" and not front:
            coda = "ŋ"  # "an" ~ "ang"
        elif syl.coda == "t" and not front:
            coda = "k"  # "at" ~ "ac"
        elif syl.coda == "nh":
            coda = "n"  # "anh" ~ [an], "inh" ~ [ɨn] (đơn giản hóa)
        elif syl.coda == "ch":
            coda = "t"
    phones.append(nucleus)
    if coda:
        phones.append(coda)

    if with_tone:
        tone = HOI if (south and syl.tone == NGA) else syl.tone  # Nam: hỏi ≡ ngã
        phones.append(TONE_TOKENS[tone - 1])
    return phones


def _word_to_syllables(word: str) -> list[str]:
    """Từ không phân tích được: tra từ mượn, đọc số, hoặc đánh vần từng chữ cái."""
    low = word.lower()
    if low in LOANWORDS:
        return LOANWORDS[low].split()
    if low.isdigit():
        return read_number(low).split()
    out: list[str] = []
    for ch in low:
        if ch.isdigit():
            out.extend(read_number(ch).split())
        elif ch in LETTER_NAMES:
            out.extend(LETTER_NAMES[ch].split())
        else:
            base = unicodedata.normalize("NFD", ch)[0]  # chữ có dấu lạ: bỏ dấu
            if base in LETTER_NAMES:
                out.extend(LETTER_NAMES[base].split())
    return out


def _syllable_tokens(syl: str, dialect: str) -> list[str]:
    try:
        return syllable_to_phones(syl, dialect)
    except ValueError:
        out: list[str] = []
        for s in _word_to_syllables(syl):
            out.extend(syllable_to_phones(s, dialect))
        return out


_TOKEN_RE = re.compile(r"[^\W_]+(?:[_-][^\W_]+)*|[,.!?;:…—()\[\]\n]")


def g2p(text: str, dialect: str = "north", *, normalize: bool = True) -> list[str]:
    """Chuyển văn bản sang chuỗi token âm vị + thanh điệu + ranh giới + dấu câu.

    >>> g2p("Tiếng Việt")
    ['t', 'iə', 'ŋ', 'T3', ' ', 'v', 'iə', 't', 'T6']
    """
    if dialect not in DIALECTS:
        raise ValueError(f"dialect phải là một trong {DIALECTS}, nhận {dialect!r}")
    if normalize:
        text = normalize_text(text)
    tokens: list[str] = []
    for m in _TOKEN_RE.finditer(text.lower()):
        tok = m.group(0)
        if len(tok) == 1 and not tok.isalnum():  # dấu câu
            p = _PUNCT_MAP.get(tok, tok)
            if tokens and tokens[-1] == WORD_BOUNDARY:
                tokens.pop()
            if tokens and tokens[-1] in PUNCTUATION:
                continue  # gộp dấu câu liên tiếp
            if tokens:
                tokens.append(p)
            continue
        if tokens and tokens[-1] not in PUNCTUATION:
            tokens.append(WORD_BOUNDARY)
        for syl in re.split(r"[_-]", tok):
            tokens.extend(_syllable_tokens(syl, dialect))
    while tokens and tokens[-1] == WORD_BOUNDARY:
        tokens.pop()
    return tokens
