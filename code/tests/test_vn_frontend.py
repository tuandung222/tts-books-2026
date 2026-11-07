"""Kiểm thử cho gói vn_frontend (Dự án 1). Chạy: ``cd code && python -m pytest``."""
from __future__ import annotations

import unicodedata

import pytest

from vn_frontend import (
    BLANK_ID, PAD_ID, SPACE_ID, SYMBOL_TO_ID, SYMBOLS, NumberStyle, Syllable, decimal_to_words,
    digits_to_words, g2p, intersperse, is_vietnamese_syllable, normalize_text, normalize_tone_marks,
    number_to_words, ordinal_to_words, parse_syllable, roman_to_int, sequence_to_symbols,
    split_sentences, split_tone, syllable_to_phones, text_to_sequence,
)
from vn_frontend.__main__ import main as cli_main
from vn_frontend.orthography import apply_tone, clean_unicode

# ---------------------------------------------------------------------------
# Đọc số
# ---------------------------------------------------------------------------
NUMBERS = [
    (0, "không"),
    (1, "một"),
    (4, "bốn"),
    (5, "năm"),
    (10, "mười"),
    (11, "mười một"),  # không phải "mười mốt"
    (14, "mười bốn"),  # không phải "mười tư"
    (15, "mười lăm"),  # không phải "mười năm"
    (20, "hai mươi"),  # "mươi" không dấu
    (21, "hai mươi mốt"),
    (24, "hai mươi tư"),
    (25, "hai mươi lăm"),
    (30, "ba mươi"),
    (41, "bốn mươi mốt"),
    (55, "năm mươi lăm"),
    (99, "chín mươi chín"),
    (100, "một trăm"),
    (101, "một trăm linh một"),  # không phải "mốt"
    (104, "một trăm linh bốn"),
    (105, "một trăm linh năm"),  # sau "linh" là "năm", không phải "lăm"
    (110, "một trăm mười"),
    (111, "một trăm mười một"),
    (115, "một trăm mười lăm"),
    (121, "một trăm hai mươi mốt"),
    (500, "năm trăm"),
    (1000, "một nghìn"),
    (1001, "một nghìn không trăm linh một"),
    (1010, "một nghìn không trăm mười"),
    (1015, "một nghìn không trăm mười lăm"),
    (1021, "một nghìn không trăm hai mươi mốt"),
    (1050, "một nghìn không trăm năm mươi"),
    (1100, "một nghìn một trăm"),
    (1975, "một nghìn chín trăm bảy mươi lăm"),
    (2000, "hai nghìn"),
    (2005, "hai nghìn không trăm linh năm"),
    (2026, "hai nghìn không trăm hai mươi sáu"),
    (10_000, "mười nghìn"),
    (15_000, "mười lăm nghìn"),
    (21_000, "hai mươi mốt nghìn"),
    (100_000, "một trăm nghìn"),
    (1_000_000, "một triệu"),
    (1_000_001, "một triệu không trăm linh một"),
    (1_001_000, "một triệu không trăm linh một nghìn"),
    (1_234_567, "một triệu hai trăm ba mươi tư nghìn năm trăm sáu mươi bảy"),
    (1_000_000_000, "một tỷ"),
    (1_000_000_005, "một tỷ không trăm linh năm"),
    (1_005_000_000, "một tỷ không trăm linh năm triệu"),
    (2_500_000_000, "hai tỷ năm trăm triệu"),
    (1_000_000_000_000, "một nghìn tỷ"),
    (3_000_000_000_000_000, "ba triệu tỷ"),
]


@pytest.mark.parametrize("n,expected", NUMBERS)
def test_number_to_words(n, expected):
    assert number_to_words(n) == expected


def test_number_styles():
    south = NumberStyle(thousand="ngàn", zero_tens="lẻ", four="bốn", five="nhăm")
    assert number_to_words(1005, south) == "một ngàn không trăm lẻ năm"
    assert number_to_words(24, south) == "hai mươi bốn"
    assert number_to_words(25, south) == "hai mươi nhăm"
    assert number_to_words(15, south) == "mười lăm"  # "nhăm" chỉ dùng sau "mươi"
    assert number_to_words(14, south) == "mười bốn"
    assert number_to_words(-7) == "âm bảy"


def test_decimal_digits_ordinal_roman():
    assert decimal_to_words("3", "14") == "ba phẩy mười bốn"
    assert decimal_to_words("0", "05") == "không phẩy không năm"
    assert decimal_to_words("2", "5") == "hai phẩy năm"
    assert decimal_to_words("3", "14159") == "ba phẩy một bốn một năm chín"
    assert digits_to_words("0912") == "không chín một hai"
    assert ordinal_to_words(1) == "nhất"
    assert ordinal_to_words(4) == "tư"
    assert ordinal_to_words(2) == "hai"
    assert roman_to_int("XXI") == 21
    assert roman_to_int("MCMXLV") == 1945
    assert roman_to_int("IV") == 4
    with pytest.raises(ValueError):
        roman_to_int("IIII")


# ---------------------------------------------------------------------------
# Chuẩn hóa văn bản
# ---------------------------------------------------------------------------
NORMALIZE_CASES = [
    ("Ngày 2/9/1945.", "Ngày hai tháng chín năm một nghìn chín trăm bốn mươi lăm."),
    ("30/4/1975", "ngày ba mươi tháng tư năm một nghìn chín trăm bảy mươi lăm"),
    ("2024-05-12", "ngày mười hai tháng năm năm hai nghìn không trăm hai mươi tư"),
    ("tháng 3/2024", "tháng ba năm hai nghìn không trăm hai mươi tư"),
    ("sáng 2/9", "sáng hai tháng chín"),
    ("tháng 4", "tháng tư"),
    ("tỷ lệ 1/2", "tỷ lệ một phần hai"),
    ("14:30", "mười bốn giờ ba mươi phút"),
    ("8:05", "tám giờ năm phút"),
    ("9:00", "chín giờ"),
    ("7h30", "bảy giờ ba mươi phút"),
    ("7h sáng", "bảy giờ sáng"),
    ("50.000đ", "năm mươi nghìn đồng"),
    ("50.000 VNĐ", "năm mươi nghìn đồng"),
    ("100 USD", "một trăm đô la Mỹ"),
    ("$100", "một trăm đô la"),
    ("50k", "năm mươi nghìn"),
    ("15tr/tháng", "mười lăm triệu một tháng"),
    ("15%", "mười lăm phần trăm"),
    ("3,5%", "ba phẩy năm phần trăm"),
    ("1.000.000", "một triệu"),
    ("3,14", "ba phẩy mười bốn"),
    ("120km", "một trăm hai mươi ki lô mét"),
    ("60 km/h", "sáu mươi ki lô mét trên giờ"),
    ("-5°C", "âm năm độ xê"),
    ("5-10%", "năm đến mười phần trăm"),
    ("TP.HCM", "thành phố Hồ Chí Minh"),
    ("UBND", "ủy ban nhân dân"),
    ("PGS.TS. Lê", "phó giáo sư tiến sĩ Lê"),
    ("VTV", "vê tê vê"),
    ("U23", "U hai mươi ba"),
    ("thứ 4", "thứ tư"),
    ("Thứ 7", "Thứ bảy"),
    ("hạng 1", "hạng nhất"),
    ("thế kỷ XXI", "thế kỷ hai mươi mốt"),
    ("Chương IV", "Chương bốn"),
    ("0912345678", "không chín một hai, ba bốn năm, sáu bảy tám"),
    ("0912 345 678", "không chín một hai, ba bốn năm, sáu bảy tám"),
    ("mã 007", "mã không không bảy"),
    ("Hoà bình", "Hòa bình"),
    ("COVID-19", "cô vít mười chín"),
    ("A & B", "A và bê"),
]


@pytest.mark.parametrize("raw,expected", NORMALIZE_CASES)
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


def test_normalize_number_style_and_tone_style():
    south = NumberStyle(thousand="ngàn", zero_tens="lẻ")
    assert normalize_text("1.005 người", style=south) == "một ngàn không trăm lẻ năm người"
    assert normalize_text("hòa thủy khỏe", tone_style="new") == "hoà thuỷ khoẻ"


def test_normalize_is_idempotent():
    text = ("UBND TP.HCM họp lúc 14h30 ngày 15/3/2025, giá 50.000đ/kg tăng 3,5%; "
            "PGS.TS. Nguyễn Văn A, đội U23 VN thắng 2-1. Hoà bình!")
    once = normalize_text(text)
    assert normalize_text(once) == once
    assert not any(c.isdigit() for c in once)


def test_normalize_unicode_cleanup():
    nfd = unicodedata.normalize("NFD", "Tiếng Việt")
    assert normalize_text(nfd) == "Tiếng Việt"
    assert unicodedata.is_normalized("NFC", normalize_text(nfd))
    assert clean_unicode("Ðà Nẵng​") == "Đà Nẵng"  # Ð (Eth) -> Đ
    assert clean_unicode("“Xin chào”...") == '"Xin chào"…'


# ---------------------------------------------------------------------------
# Dấu thanh
# ---------------------------------------------------------------------------
TONE_PLACEMENT = [  # (kiểu mới hoặc lẫn lộn, kiểu cũ chuẩn của giáo trình)
    ("hoà", "hòa"), ("hoá", "hóa"), ("khoẻ", "khỏe"), ("thuỷ", "thủy"), ("tuỳ", "tùy"),
    ("xoã", "xõa"), ("HOÀ", "HÒA"), ("Thuỷ", "Thủy"),
]
UNCHANGED = [
    "hoàng", "quý", "quả", "quyển", "người", "tiếng", "của", "mùa", "giữa", "khuỷu",
    "ngoài", "khuyến", "thuở", "gì", "giường", "rượu", "muối", "khuấy", "hoặc", "xoáy",
    "ngoẹo", "huỳnh", "mía", "thuế", "xuân", "Việt", "giếng", "gìn", "yến", "ở",
]


@pytest.mark.parametrize("raw,expected", TONE_PLACEMENT)
def test_tone_mark_normalization(raw, expected):
    assert normalize_tone_marks(raw) == expected


@pytest.mark.parametrize("word", UNCHANGED)
def test_tone_mark_roundtrip(word):
    assert normalize_tone_marks(word) == word
    base, tone = split_tone(word)
    assert apply_tone(base, tone) == word


# ---------------------------------------------------------------------------
# Tách câu
# ---------------------------------------------------------------------------
def test_split_sentences_basic():
    text = "Xin chào. Tôi là An! Bạn khỏe không? Tôi ổn…"
    assert split_sentences(text) == ["Xin chào.", "Tôi là An!", "Bạn khỏe không?", "Tôi ổn…"]


def test_split_sentences_does_not_break_abbreviations_numbers():
    text = "Ông đến TP. Hồ Chí Minh lúc 3.5 giờ. PGS. Lan nói: “Được.” Rồi đi."
    assert split_sentences(text) == [
        "Ông đến TP. Hồ Chí Minh lúc 3.5 giờ.", "PGS. Lan nói: “Được.”", "Rồi đi."]
    assert split_sentences("Chờ chút… rồi đi tiếp. Xong") == ["Chờ chút… rồi đi tiếp.", "Xong"]
    assert split_sentences("Dòng một\n\nDòng hai") == ["Dòng một", "Dòng hai"]


def test_split_sentences_max_chars():
    long = "Hôm nay trời đẹp, chúng ta đi chơi công viên, sau đó ăn trưa ở quán quen."
    parts = split_sentences(long, max_chars=30)
    assert all(len(p) <= 30 for p in parts)
    assert " ".join(parts) == long


# ---------------------------------------------------------------------------
# Phân tích âm tiết
# ---------------------------------------------------------------------------
SYLLABLES = [
    ("hoàng", ("h", "o", "a", "ng", 2)),
    ("quốc", ("q", "u", "ô", "c", 3)),
    ("giá", ("gi", "", "a", "", 3)),
    ("gì", ("gi", "", "i", "", 2)),
    ("giếng", ("gi", "", "iê", "ng", 3)),
    ("người", ("ng", "", "ươ", "i", 2)),
    ("khuya", ("kh", "u", "ya", "", 1)),
    ("khuỷu", ("kh", "u", "y", "u", 4)),
    ("thủy", ("th", "u", "y", "", 4)),
    ("ăn", ("", "", "ă", "n", 1)),
    ("yêu", ("", "", "yê", "u", 1)),
    ("uống", ("", "", "uô", "ng", 3)),
    ("oanh", ("", "o", "a", "nh", 1)),
    ("hoặc", ("h", "o", "ă", "c", 6)),
    ("xoong", ("x", "", "oo", "ng", 1)),
    ("nghiêng", ("ngh", "", "iê", "ng", 1)),
    ("Việt", ("v", "", "iê", "t", 6)),
    ("ngã", ("ng", "", "a", "", 5)),
    ("quyết", ("q", "u", "yê", "t", 3)),
    ("thuở", ("th", "u", "ơ", "", 4)),
    ("rượu", ("r", "", "ươ", "u", 6)),
    ("của", ("c", "", "ua", "", 4)),
    ("mưa", ("m", "", "ưa", "", 1)),
    ("máy", ("m", "", "a", "y", 3)),
    ("ĐƯỜNG", ("đ", "", "ươ", "ng", 2)),
]


@pytest.mark.parametrize("word,parts", SYLLABLES)
def test_parse_syllable(word, parts):
    s = parse_syllable(word)
    assert (s.onset, s.glide, s.nucleus, s.coda, s.tone) == parts
    assert s == Syllable(*parts)


def test_parse_syllable_extras():
    s = parse_syllable("học")
    assert s.tone_name == "nặng" and s.is_checked and s.rhyme == "oc"
    assert parse_syllable(unicodedata.normalize("NFD", "người")) == parse_syllable("người")


@pytest.mark.parametrize("bad", ["", "xyz", "bcd", "hello", "anhh", "qu", "123", "a-b"])
def test_parse_syllable_invalid(bad):
    with pytest.raises(ValueError):
        parse_syllable(bad)


@pytest.mark.parametrize("word,ok", [
    ("kiến", True), ("cá", True), ("ghế", True), ("nghĩ", True), ("quê", True),
    ("ceo", False),  # trước e phải viết k
    ("kan", False),  # k chỉ đứng trước e, ê, i, y
    ("nge", False), ("ghy", False), ("gha", False),
    ("tàp", False),  # âm tiết khép chỉ mang sắc/nặng
    ("tiê", False), ("mian", False), ("ăy", False),
])
def test_is_vietnamese_syllable(word, ok):
    assert is_vietnamese_syllable(word) is ok


# ---------------------------------------------------------------------------
# G2P và bảng ký hiệu
# ---------------------------------------------------------------------------
def test_g2p_examples_north():
    assert g2p("Tiếng Việt") == ["t", "iə", "ŋ", "T3", " ", "v", "iə", "t", "T6"]
    assert g2p("Xin chào, Việt Nam!") == [
        "s", "i", "n", "T1", " ", "c", "a", "w", "T2", ",",
        "v", "iə", "t", "T6", " ", "n", "a", "m", "T1", "!"]
    assert syllable_to_phones("ăn") == ["ʔ", "ă", "n", "T1"]  # âm đầu rỗng -> tắc thanh hầu
    assert syllable_to_phones("quả") == ["k", "w", "a", "T4"]
    assert syllable_to_phones("máy") == ["m", "ă", "j", "T3"]  # ay = a ngắn
    assert syllable_to_phones("mai") == ["m", "a", "j", "T1"]
    assert syllable_to_phones("anh") == ["ʔ", "ɛ", "ɲ", "T1"]
    assert syllable_to_phones("giữ") == ["z", "ɨ", "T5"]
    assert syllable_to_phones("rừng") == syllable_to_phones("dừng")  # r/d/gi gộp ở Hà Nội


def test_g2p_south_differences():
    assert syllable_to_phones("quả", "south") == ["w", "a", "T4"]
    assert syllable_to_phones("ngã", "south")[-1] == "T4"  # hỏi/ngã gộp
    assert syllable_to_phones("Việt", "south") == ["j", "iə", "k", "T6"]
    assert syllable_to_phones("an", "south") == syllable_to_phones("ang", "south")
    assert syllable_to_phones("tin", "south")[-2] == "n"
    assert syllable_to_phones("anh", "south") == ["ʔ", "a", "n", "T1"]
    with pytest.raises(ValueError):
        g2p("xin chào", dialect="central")


def test_g2p_boundaries_and_punctuation():
    toks = g2p("học_sinh đi học... (ở trường) — vui!")
    assert toks.count(" ") == 3  # học_sinh | đi | học ... ở | trường
    assert "…" in toks and "," in toks and toks[-1] == "!"
    assert toks[0] != " " and not any(a in ",.!?" and b == " " for a, b in zip(toks, toks[1:]))
    assert g2p("a,,, b") == ["ʔ", "a", "T1", ","] + g2p("b")  # gộp dấu câu lặp
    # từ không phải tiếng Việt: tra từ mượn hoặc đánh vần
    assert g2p("Facebook") == g2p("phây_búc")  # một từ: không chèn ranh giới
    assert g2p("xyz") == g2p("ích_i_dài_dét")


def test_symbols_and_sequences():
    assert SYMBOLS[PAD_ID] == "<pad>" and PAD_ID == 0
    assert SYMBOLS[SPACE_ID] == " " and SYMBOLS[BLANK_ID] == "<blank>"
    assert len(set(SYMBOLS)) == len(SYMBOLS)
    for p in [",", ".", "!", "?", "T1", "T6", "ʔ", "iə"]:
        assert p in SYMBOL_TO_ID
    text = ("Ngày 2/9/1945, UBND TP.HCM thông báo: giá 50.000đ/kg tăng 3,5%! "
            "Facebook, iPhone và 7h30 sáng thứ 4 — khoẻ không? Huỳnh Khuỷu Xoong Quyết")
    for dialect in ("north", "south"):
        ids = text_to_sequence(text, dialect)
        assert ids and all(0 < i < len(SYMBOLS) for i in ids)
        assert SYMBOL_TO_ID["<unk>"] not in ids
        assert sequence_to_symbols(ids) == g2p(text, dialect)
    assert intersperse([7, 8]) == [BLANK_ID, 7, BLANK_ID, 8, BLANK_ID]


def test_cli(capsys):
    assert cli_main(["normalize", "Giá 50.000đ"]) == 0
    assert capsys.readouterr().out.strip() == "Giá năm mươi nghìn đồng"
    assert cli_main(["g2p", "xin chào"]) == 0
    assert capsys.readouterr().out.strip() == "s i n T1 | c a w T2"
