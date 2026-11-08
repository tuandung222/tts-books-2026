"""Tách câu cho văn bản tiếng Việt (dựa trên luật, không cần mô hình)."""
from __future__ import annotations

import re

from .lexicon import ABBREVIATIONS

# Các viết tắt có dấu chấm: dấu chấm của chúng KHÔNG kết thúc câu.
_ABBR_WITH_DOT = tuple(sorted({k for k in ABBREVIATIONS if k.endswith(".")}
                              | {"TP.", "Tp.", "Q.", "P.", "St.", "Mr.", "Mrs.", "Dr.", "TS.", "Th.S."},
                              key=len, reverse=True))
_TERMINATOR_RE = re.compile(r"[.!?…]+[\"')\]»”’]*")
_CLAUSE_BREAK_RE = re.compile(r"[,;:]\s")


def _starts_sentence(ch: str) -> bool:
    return ch.isupper() or ch.isdigit() or ch in "\"'“«([-—"


def _split_line(line: str) -> list[str]:
    out, start = [], 0
    for m in _TERMINATOR_RE.finditer(line):
        end = m.end()
        rest = line[end:]
        stripped = rest.lstrip()
        if stripped and (len(stripped) == len(rest) or not _starts_sentence(stripped[0])):
            continue  # không có khoảng trắng phía sau, hoặc câu sau không viết hoa
        before = line[start:end]
        if m.group(0) == "." and before.endswith(_ABBR_WITH_DOT):
            continue  # "TP. Hồ Chí Minh", "PGS. Nguyễn..."
        if m.group(0) == "." and re.search(r"(?:^|\s)[A-ZĐ]\.$", before):
            continue  # chữ cái viết tắt tên người: "Nguyễn V. A"
        out.append(line[start:end].strip())
        start = end
    tail = line[start:].strip()
    if tail:
        out.append(tail)
    return [s for s in out if s]


def _split_long(sentence: str, max_chars: int) -> list[str]:
    """Chia câu quá dài tại dấu phẩy/chấm phẩy/hai chấm gần giới hạn nhất, sau đó tại khoảng trắng."""
    parts: list[str] = []
    while len(sentence) > max_chars:
        cut = -1
        for m in _CLAUSE_BREAK_RE.finditer(sentence, 0, max_chars + 1):
            cut = m.start() + 1
        if cut <= 0:
            cut = sentence.rfind(" ", 0, max_chars + 1)
        if cut <= 0:
            cut = max_chars
        parts.append(sentence[:cut].strip())
        sentence = sentence[cut:].strip()
    if sentence:
        parts.append(sentence)
    return parts


def split_sentences(text: str, max_chars: int | None = None) -> list[str]:
    """Tách văn bản thành danh sách câu.

    Quy tắc: xuống dòng luôn ngắt câu; ``. ! ? …`` (có thể kèm dấu đóng ngoặc/
    nháy) ngắt câu khi theo sau là khoảng trắng và một ký tự mở đầu câu (chữ hoa,
    chữ số, dấu mở ngoặc/nháy, gạch đầu dòng) hoặc hết văn bản. Không ngắt sau
    các viết tắt có dấu chấm (``TP.``, ``PGS.``) và chữ cái viết tắt trong tên.

    ``max_chars``: nếu đặt, câu dài hơn sẽ được chia tiếp tại ranh giới mệnh đề
    (hữu ích cho TTS streaming và cho mô hình kém ổn định với câu dài).
    """
    sentences: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            sentences.extend(_split_line(line))
    if max_chars:
        sentences = [p for s in sentences for p in _split_long(s, max_chars)]
    return sentences
