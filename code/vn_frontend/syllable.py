"""Phân tích âm tiết tiếng Việt theo chính tả: âm đầu, âm đệm, âm chính, âm cuối, thanh.

Mô hình cấu trúc âm tiết theo truyền thống ngữ âm học tiếng Việt (ví dụ Đoàn
Thiện Thuật)::

    âm tiết = [âm đầu] + vần{ [âm đệm] + âm chính + [âm cuối] } + thanh điệu

Các thành phần trả về là **chữ viết** (chữ thường, không dấu thanh); việc đổi
sang âm vị (IPA) do ``vn_frontend.g2p`` đảm nhiệm.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from .orthography import SAC, NANG, TONE_NAMES, VOWELS, split_orthographic, split_tone

# Âm chính hợp lệ (chữ viết). Nguyên âm đôi có hai cách viết tùy âm cuối:
# iê/yê/uô/ươ khi có âm cuối, ia/ya/ua/ưa khi không có âm cuối.
NUCLEI = frozenset({
    "a", "ă", "â", "e", "ê", "i", "y", "o", "ô", "ơ", "u", "ư",
    "iê", "yê", "ia", "ya", "uô", "ua", "ươ", "ưa",
    "oo", "ôô",  # chỉ gặp trong vài từ: xoong, boong, cồông (hiếm)
})
_CLOSED_DIPHTHONGS = frozenset({"iê", "yê", "uô", "ươ"})
_OPEN_DIPHTHONGS = frozenset({"ia", "ya", "ua", "ưa"})
_SEMIVOWEL_CODAS = frozenset("iyou")
_CHECKED_CODAS = frozenset({"p", "t", "c", "ch"})
_FRONT_START = ("e", "ê", "i", "y")


@dataclass(frozen=True)
class Syllable:
    """Một âm tiết đã phân tích.

    Thuộc tính:
        onset:   âm đầu viết (``""`` nếu không có; ``"q"`` trong ``qu``; ``"gi"``)
        glide:   âm đệm viết: ``"o"``, ``"u"`` hoặc ``""``
        nucleus: âm chính viết (``"a"``, ``"iê"``, ``"ươ"``...)
        coda:    âm cuối viết: phụ âm (``c ch m n ng nh p t``), bán nguyên âm
                 (``i y o u``) hoặc ``""``
        tone:    số thanh 1..6 = ngang, huyền, sắc, hỏi, ngã, nặng
    """

    onset: str
    glide: str
    nucleus: str
    coda: str
    tone: int

    @property
    def tone_name(self) -> str:
        return TONE_NAMES[self.tone]

    @property
    def rhyme(self) -> str:
        """Vần (không dấu thanh) theo chữ viết, ví dụ ``"oang"``."""
        return self.glide + self.nucleus + self.coda

    @property
    def is_checked(self) -> bool:
        """Âm tiết khép (tắc) kết thúc bằng p, t, c, ch."""
        return self.coda in _CHECKED_CODAS


def parse_syllable(syllable: str, strict: bool = False) -> Syllable:
    """Phân tích một âm tiết chữ quốc ngữ.

    Ví dụ: ``parse_syllable("hoàng") == Syllable("h", "o", "a", "ng", 2)``.

    Với ``strict=True``, kiểm tra thêm các ràng buộc chính tả và âm vị học
    (c/k/q, g/gh, ng/ngh; âm tiết khép chỉ mang thanh sắc hoặc nặng; nguyên âm
    đôi viết đúng dạng mở/đóng...). Ném ``ValueError`` nếu không hợp lệ.
    """
    word = unicodedata.normalize("NFC", syllable.strip())
    if not word or not all(c.isalpha() for c in word):
        raise ValueError(f"không phải âm tiết: {syllable!r}")
    base, tone = split_tone(word)
    low = base.lower()
    if any(c not in VOWELS and c not in "bcdđghklmnpqrstvx" for c in low):
        raise ValueError(f"chứa chữ cái ngoài bảng chữ cái tiếng Việt: {syllable!r}")
    onset_w, start, end = split_orthographic(low)
    vowels, cons_coda = low[start:end], low[end:]

    if onset_w == "qu":
        onset, glide = "q", "u"
    else:
        onset, glide = onset_w, ""
        if len(vowels) >= 2 and vowels[0] == "o" and vowels[1] in "aăe":
            glide, vowels = "o", vowels[1:]  # oa, oă, oe
        elif len(vowels) >= 2 and vowels[0] == "u" and vowels[1] in "yêâơ":
            glide, vowels = "u", vowels[1:]  # uy, uê, uâ, uơ (+ uya, uyê)

    if cons_coda:
        nucleus, coda = vowels, cons_coda
    elif vowels in _OPEN_DIPHTHONGS:
        nucleus, coda = vowels, ""
    elif len(vowels) >= 2 and vowels[-1] in _SEMIVOWEL_CODAS and vowels[:-1] in NUCLEI:
        nucleus, coda = vowels[:-1], vowels[-1]
    else:
        nucleus, coda = vowels, ""

    # "giếng", "giết", "giễu": chữ i của "gi" dùng chung với nguyên âm đôi iê.
    if onset == "gi" and nucleus == "ê" and coda:
        nucleus = "iê"
    if nucleus not in NUCLEI:
        raise ValueError(f"vần không hợp lệ: {syllable!r}")

    syl = Syllable(onset, glide, nucleus, coda, tone)
    if strict:
        _check_strict(syl, syllable)
    return syl


def _check_strict(s: Syllable, original: str) -> None:
    def bad(reason: str) -> None:
        raise ValueError(f"{original!r}: {reason}")

    first = (s.glide or s.nucleus)[:1]
    front = (s.glide == "" and s.nucleus.startswith(_FRONT_START))
    if s.onset == "c" and (front or s.glide):
        bad("trước e, ê, i, y phải viết k; âm /kw/ phải viết qu")
    if s.onset == "k" and not front:
        bad("k chỉ đứng trước e, ê, i, y")
    if s.onset in ("g", "ng") and front:
        bad("trước e, ê, i phải viết gh/ngh")
    if s.onset in ("gh", "ngh") and not (front and first != "y"):
        bad("gh/ngh chỉ đứng trước e, ê, i")
    if s.coda in _CHECKED_CODAS and s.tone not in (SAC, NANG):
        bad("âm tiết khép (p, t, c, ch) chỉ mang thanh sắc hoặc nặng")
    if s.nucleus in _CLOSED_DIPHTHONGS and not s.coda:
        bad("iê/yê/uô/ươ cần âm cuối (vần mở viết ia/ya/ua/ưa)")
    if s.nucleus in _OPEN_DIPHTHONGS and s.coda:
        bad("ia/ya/ua/ưa không đi với âm cuối")
    if s.nucleus.startswith("y") and len(s.nucleus) == 2 and not (s.glide or s.onset == ""):
        bad("yê/ya chỉ đứng đầu âm tiết hoặc sau âm đệm u")
    if s.nucleus == "ă" and s.coda not in {"c", "m", "n", "ng", "p", "t"}:
        bad("ă cần âm cuối phụ âm")
    if s.nucleus == "â" and not s.coda:
        bad("â cần âm cuối")
    if s.coda in ("nh", "ch") and s.nucleus not in ("a", "ê", "i", "y"):
        bad("nh/ch chỉ đứng sau a, ê, i/y")
    if s.coda == "y" and s.nucleus not in ("a", "â"):
        bad("bán nguyên âm cuối y chỉ đứng sau a, â")
    if s.coda == "o" and s.nucleus not in ("a", "e"):
        bad("bán nguyên âm cuối o chỉ đứng sau a, e")


def is_vietnamese_syllable(word: str) -> bool:
    """``True`` nếu ``word`` là một âm tiết tiếng Việt hợp lệ theo luật chính tả."""
    try:
        parse_syllable(word, strict=True)
        return True
    except ValueError:
        return False
