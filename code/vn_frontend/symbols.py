"""Bảng ký hiệu (symbol set) và ánh xạ văn bản -> chuỗi ID cho mô hình.

Thứ tự ``SYMBOLS`` là **hợp đồng** giữa front-end và mô hình đã huấn luyện: đổi
thứ tự hoặc chèn ký hiệu vào giữa sẽ làm hỏng checkpoint cũ. Muốn thêm ký hiệu,
hãy nối vào CUỐI danh sách.
"""
from __future__ import annotations

from .g2p import PHONES, PUNCTUATION, TONE_TOKENS, WORD_BOUNDARY, g2p

PAD = "<pad>"  # ID 0: đệm batch
UNK = "<unk>"  # ký hiệu lạ (không nên xuất hiện nếu front-end đúng)
BLANK = "<blank>"  # chèn xen kẽ kiểu VITS (add_blank)
BOS = "<bos>"
EOS = "<eos>"

SPECIALS = (PAD, UNK, BLANK, BOS, EOS, WORD_BOUNDARY)
SYMBOLS: list[str] = [*SPECIALS, *PUNCTUATION, *PHONES, *TONE_TOKENS]
SYMBOL_TO_ID: dict[str, int] = {s: i for i, s in enumerate(SYMBOLS)}
assert len(SYMBOL_TO_ID) == len(SYMBOLS), "SYMBOLS có phần tử trùng lặp"

PAD_ID = SYMBOL_TO_ID[PAD]
UNK_ID = SYMBOL_TO_ID[UNK]
BLANK_ID = SYMBOL_TO_ID[BLANK]
BOS_ID = SYMBOL_TO_ID[BOS]
EOS_ID = SYMBOL_TO_ID[EOS]
SPACE_ID = SYMBOL_TO_ID[WORD_BOUNDARY]


def text_to_sequence(text: str, dialect: str = "north") -> list[int]:
    """Văn bản thô -> chuẩn hóa -> G2P -> danh sách ID trong ``SYMBOLS``."""
    return [SYMBOL_TO_ID.get(t, UNK_ID) for t in g2p(text, dialect)]


def sequence_to_symbols(ids: list[int]) -> list[str]:
    """Chiều ngược lại (để gỡ lỗi): danh sách ID -> danh sách ký hiệu."""
    return [SYMBOLS[i] for i in ids]


def intersperse(ids: list[int], item: int = BLANK_ID) -> list[int]:
    """Chèn ``item`` xen kẽ và ở hai đầu: [a, b] -> [blank, a, blank, b, blank]."""
    out = [item] * (2 * len(ids) + 1)
    out[1::2] = ids
    return out
