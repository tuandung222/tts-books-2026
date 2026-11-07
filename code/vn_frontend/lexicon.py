"""Từ điển nhỏ dùng cho chuẩn hóa: viết tắt, tên chữ cái, đơn vị, tiền tệ, từ mượn.

Đây là các bảng **minh họa**, đủ cho kiểm thử và ví dụ trong giáo trình. Hệ
thống thực tế cần bảng lớn hơn nhiều và nên được kiểm duyệt theo miền dữ liệu
(tin tức, y tế, pháp luật...). Cách đọc ở đây là một lựa chọn; hãy giữ nhất quán
với cách người đọc (voice talent) trong dữ liệu huấn luyện của bạn.
"""
from __future__ import annotations

# Viết tắt -> dạng đọc. Khóa phân biệt hoa/thường. Khóa kết thúc bằng "." được
# so khớp kể cả khi dính liền chữ phía sau ("TP.HCM").
ABBREVIATIONS: dict[str, str] = {
    "TP.": "thành phố",
    "Tp.": "thành phố",
    "TP": "thành phố",
    "TPHCM": "thành phố Hồ Chí Minh",
    "HCM": "Hồ Chí Minh",
    "UBND": "ủy ban nhân dân",
    "HĐND": "hội đồng nhân dân",
    "PGS.TS.": "phó giáo sư tiến sĩ",
    "PGS.TS": "phó giáo sư tiến sĩ",
    "GS.TS.": "giáo sư tiến sĩ",
    "GS.TS": "giáo sư tiến sĩ",
    "PGS.": "phó giáo sư",
    "GS.": "giáo sư",
    "TS.": "tiến sĩ",
    "ThS.": "thạc sĩ",
    "ThS": "thạc sĩ",
    "BS.": "bác sĩ",
    "KS.": "kỹ sư",
    "THPT": "trung học phổ thông",
    "THCS": "trung học cơ sở",
    "ĐHQG": "đại học quốc gia",
    "ĐH": "đại học",
    "VN": "Việt Nam",
    "CSGT": "cảnh sát giao thông",
    "BHXH": "bảo hiểm xã hội",
    "BHYT": "bảo hiểm y tế",
    "CLB": "câu lạc bộ",
    "HLV": "huấn luyện viên",
    "NXB": "nhà xuất bản",
    "SĐT": "số điện thoại",
    "TTXVN": "Thông tấn xã Việt Nam",
    "TW": "trung ương",
    "v.v.": "vân vân",
    "v.v": "vân vân",
    "USD": "đô la Mỹ",
    "VNĐ": "đồng",
    "VND": "đồng",
}

# Tên chữ cái khi đánh vần từ viết tắt (VTV -> vê tê vê). Cách gọi tên chữ cái
# khác nhau giữa các vùng và ngữ cảnh ("lờ"/"e lờ", "mờ"/"em"); đây là một lựa chọn.
LETTER_NAMES: dict[str, str] = {
    "a": "a", "ă": "á", "â": "ớ", "b": "bê", "c": "xê", "d": "dê", "đ": "đê",
    "e": "e", "ê": "ê", "f": "ép", "g": "giê", "h": "hát", "i": "i", "j": "gi",
    "k": "ca", "l": "e lờ", "m": "em", "n": "en", "o": "o", "ô": "ô", "ơ": "ơ",
    "p": "pê", "q": "quy", "r": "e rờ", "s": "ét", "t": "tê", "u": "u", "ư": "ư",
    "v": "vê", "w": "vê kép", "x": "ích", "y": "i dài", "z": "dét",
}

# Đơn vị đo (đứng ngay sau số). Khóa dài phải được thử trước khóa ngắn.
UNITS: dict[str, str] = {
    "km/h": "ki lô mét trên giờ",
    "m/s": "mét trên giây",
    "km²": "ki lô mét vuông", "km2": "ki lô mét vuông",
    "cm²": "xen ti mét vuông", "cm2": "xen ti mét vuông",
    "cm³": "xen ti mét khối", "cm3": "xen ti mét khối",
    "m²": "mét vuông", "m2": "mét vuông",
    "m³": "mét khối", "m3": "mét khối",
    "km": "ki lô mét", "cm": "xen ti mét", "mm": "mi li mét", "m": "mét",
    "kg": "ki lô gam", "mg": "mi li gam", "g": "gam",
    "ml": "mi li lít",
    "ha": "héc ta",
    "°C": "độ xê", "ºC": "độ xê", "°F": "độ ép",
    "kWh": "ki lô oát giờ", "kW": "ki lô oát", "MW": "mê ga oát", "W": "oát",
    "GB": "gi ga bai", "MB": "mê ga bai", "KB": "ki lô bai", "TB": "tê ra bai",
    "GHz": "gi ga héc", "MHz": "mê ga héc", "kHz": "ki lô héc", "Hz": "héc",
    "dB": "đề xi ben",
}

# Ký hiệu/mã tiền tệ đứng SAU số.
CURRENCY_SUFFIX: dict[str, str] = {
    "VNĐ": "đồng", "VND": "đồng", "vnđ": "đồng", "vnd": "đồng", "đồng": "đồng",
    "đ": "đồng", "₫": "đồng",
    "USD": "đô la Mỹ", "usd": "đô la Mỹ", "$": "đô la",
    "EUR": "ơ rô", "€": "ơ rô", "£": "bảng Anh", "¥": "yên", "JPY": "yên",
}
# Ký hiệu tiền tệ đứng TRƯỚC số ($100, €5).
CURRENCY_PREFIX: dict[str, str] = {"$": "đô la", "€": "ơ rô", "£": "bảng Anh", "¥": "yên"}

# Từ mượn/ tên riêng nước ngoài đọc theo kiểu Việt hóa (khóa viết thường).
# Chính sách đọc từ nước ngoài là một quyết định thiết kế (xem Chương 30).
LOANWORDS: dict[str, str] = {
    "covid": "cô vít",
    "email": "i meo",
    "gmail": "gờ meo",
    "facebook": "phây búc",
    "youtube": "diu túp",
    "internet": "in tơ nét",
    "online": "on lai",
    "video": "vi đê ô",
    "wifi": "oai phai",
    "ok": "ô kê",
    "okay": "ô kê",
    "website": "oép sai",
    "app": "áp",
}
