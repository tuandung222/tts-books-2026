"""Tiền xử lý và chia đoạn (chunking) văn bản cho TTS streaming.

Nếu gói ``vn_frontend`` của Dự án 1 import được, dịch vụ dùng ``normalize_text``
và ``split_sentences`` của nó. Nếu không, dịch vụ rơi về một bộ xử lý tối giản
(chỉ chuẩn hóa NFC, khoảng trắng và tách câu bằng biểu thức chính quy) để vẫn chạy
được và để kiểm thử không phụ thuộc vào Dự án 1.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bộ xử lý văn bản (front-end) tối giản dùng làm phương án dự phòng
# ---------------------------------------------------------------------------
_SPACE_RE = re.compile(r"\s+")
_CONTROL_RE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f​﻿]")
# Dấu kết câu (có thể kèm dấu đóng ngoặc/nháy) và theo sau là khoảng trắng hoặc hết chuỗi
_SENT_END_RE = re.compile(r"[.!?…]+[\"'”’»)\]]*(?=\s|$)")
# Dấu kết câu đã chắc chắn kết thúc: phải có khoảng trắng phía sau (dùng cho chế độ tăng dần)
_SENT_END_WS_RE = re.compile(r"[.!?…]+[\"'”’»)\]]*\s+")
# Ranh giới cụm: sau dấu phẩy, chấm phẩy, hai chấm, gạch ngang dài
_PHRASE_SPLIT_RE = re.compile(r"(?<=[,;:–—])\s+")
# Một số chữ viết tắt hay đứng trước dấu chấm mà không kết thúc câu
_ABBREVIATIONS = {"tp", "ts", "ths", "pgs", "gs", "bs", "ks", "ths.bs", "q", "p", "mr", "mrs", "ms", "dr", "st", "v.v"}


def basic_normalize(text: str) -> str:
    """Chuẩn hóa tối thiểu: NFC, bỏ ký tự điều khiển, gộp khoảng trắng."""
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def _is_abbreviation(text: str, dot_pos: int) -> bool:
    """Từ đứng ngay trước dấu chấm ở ``dot_pos`` có phải chữ viết tắt/chữ cái đầu tên không."""
    if text[dot_pos] != ".":
        return False
    words = text[:dot_pos].split()
    if not words:
        return False
    last = words[-1].lower()
    return last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha())


def basic_split_sentences(text: str) -> list[str]:
    """Tách câu bằng biểu thức chính quy, tránh cắt sau các chữ viết tắt phổ biến."""
    text = basic_normalize(text)
    sentences, start = [], 0
    for m in _SENT_END_RE.finditer(text):
        if _is_abbreviation(text, m.start()):
            continue
        sentence = text[start:m.end()].strip()
        if sentence:
            sentences.append(sentence)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


@dataclass(frozen=True)
class TextFrontend:
    """Hai thao tác văn bản mà dịch vụ cần: chuẩn hóa và tách câu."""

    name: str
    normalize: Callable[[str], str]
    split_sentences: Callable[[str], list[str]]


BASIC_FRONTEND = TextFrontend("basic", basic_normalize, basic_split_sentences)


def _with_fallback(fn: Callable, fallback: Callable, what: str) -> Callable:
    """Bọc một hàm front-end: nếu nó lỗi với một đầu vào lạ, ghi log và dùng phương án dự phòng.

    Trong dịch vụ thật, một lỗi chuẩn hóa văn bản không nên làm hỏng cả yêu cầu;
    nhưng phải được ghi log (và đếm) để sửa ở front-end.
    """
    def wrapped(text: str):
        try:
            return fn(text)
        except Exception:  # noqa: BLE001 - cố ý bắt rộng ở ranh giới dịch vụ
            logger.exception("vn_frontend.%s lỗi, dùng bộ dự phòng", what)
            return fallback(text)
    return wrapped


def load_vn_frontend() -> TextFrontend | None:
    """Thử nạp front-end tiếng Việt của Dự án 1; trả về ``None`` nếu không có."""
    try:
        import vn_frontend  # type: ignore[import-not-found]
        normalize = vn_frontend.normalize_text
        split = vn_frontend.split_sentences
    except Exception:  # noqa: BLE001 - thiếu gói hoặc gói lỗi: đều dùng dự phòng
        return None
    return TextFrontend(
        "vn_frontend",
        _with_fallback(normalize, basic_normalize, "normalize_text"),
        _with_fallback(split, basic_split_sentences, "split_sentences"),
    )


def default_frontend() -> TextFrontend:
    """Front-end mặc định: ``vn_frontend`` nếu có, ngược lại là bộ tối giản."""
    return load_vn_frontend() or BASIC_FRONTEND


# ---------------------------------------------------------------------------
# Chia đoạn cho streaming
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChunkConfig:
    """Trần độ dài (số ký tự) của đoạn thứ k: min(max_chars, first_max_chars * growth**k).

    Đoạn đầu ngắn để giảm thời gian tới byte âm thanh đầu tiên (TTFB); các đoạn sau
    được phép dài dần vì khi đó đã có âm thanh trong bộ đệm phía client.
    """

    first_max_chars: int = 60
    max_chars: int = 220
    growth: float = 2.0

    def limit(self, k: int) -> int:
        return int(min(self.max_chars, self.first_max_chars * self.growth ** k))


def _split_words(text: str, limit: int) -> list[str]:
    """Cắt tham lam theo ranh giới từ; một từ dài hơn ``limit`` được giữ nguyên."""
    pieces, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > limit:
            pieces.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    if cur:
        pieces.append(cur)
    return pieces


def _pack(pieces: list[str], limit: int) -> list[str]:
    """Gộp tham lam các mảnh liên tiếp miễn là không vượt ``limit``."""
    out: list[str] = []
    for p in pieces:
        if out and len(out[-1]) + 1 + len(p) <= limit:
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return out


def split_long(text: str, limit: int) -> list[str]:
    """Chia một câu dài thành các mảnh <= ``limit`` ký tự.

    Ưu tiên ranh giới cụm (dấu phẩy, chấm phẩy...) vì ngắt ở đó nghe tự nhiên;
    chỉ khi một cụm vẫn quá dài mới cắt theo ranh giới từ (không bao giờ cắt giữa âm tiết).
    """
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    for phrase in _PHRASE_SPLIT_RE.split(text):
        pieces.extend([phrase] if len(phrase) <= limit else _split_words(phrase, limit))
    return _pack([p for p in pieces if p], limit)


def _has_content(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


def chunk_text(text: str, config: ChunkConfig | None = None,
               frontend: TextFrontend | None = None) -> list[str]:
    """Tách câu, chuẩn hóa rồi gom thành các đoạn có trần độ dài tăng dần."""
    config = config or ChunkConfig()
    fe = frontend or default_frontend()
    sentences = [fe.normalize(s) for s in fe.split_sentences(text)]
    queue = deque(s for s in sentences if s and _has_content(s))
    chunks: list[str] = []
    while queue:
        limit = config.limit(len(chunks))
        s = queue.popleft()
        if len(s) > limit:
            head, *rest = split_long(s, limit)
            queue.extendleft(reversed(rest))  # phần còn lại xử lý ở vòng sau (trần lớn hơn)
            chunks.append(head)
            continue
        while queue and len(s) + 1 + len(queue[0]) <= limit:  # gộp câu ngắn
            s = f"{s} {queue.popleft()}"
        chunks.append(s)
    return chunks


class IncrementalChunker:
    """Nhận văn bản theo từng mảnh (ví dụ token do LLM sinh ra) và nhả ra các đoạn đã đủ để đọc.

    - Nhả một đoạn khi gặp ranh giới câu *đã chắc chắn* (dấu kết câu + khoảng trắng):
      chưa thấy khoảng trắng thì "3." vẫn có thể là "3.5", nên phải chờ.
    - Nếu bộ đệm dài hơn trần hiện tại mà chưa có ranh giới câu, cắt ở ranh giới
      cụm hoặc từ gần nhất để không bắt người nghe chờ quá lâu.
    """

    def __init__(self, config: ChunkConfig | None = None) -> None:
        self.config = config or ChunkConfig()
        self._buf = ""
        self._emitted = 0

    def _find_cut(self) -> int | None:
        limit = self.config.limit(self._emitted)
        for m in _SENT_END_WS_RE.finditer(self._buf):
            if _is_abbreviation(self._buf, m.start()):
                continue
            if m.start() <= limit:
                return m.end()
            break  # câu đầu tiên quá dài: xử lý như bộ đệm quá dài bên dưới
        if len(self._buf) <= limit:
            return None
        window = self._buf[:limit + 1]
        for sep in (", ", "; ", ": ", " – ", " — "):
            idx = window.rfind(sep)
            if idx > 0:
                return idx + len(sep)
        idx = window.rfind(" ")
        if idx > 0:
            return idx + 1
        idx = self._buf.find(" ", limit)  # một "từ" rất dài (URL...): cắt ngay sau nó
        return idx + 1 if idx > 0 else None

    def push(self, delta: str) -> list[str]:
        """Thêm một mảnh văn bản; trả về các đoạn hoàn chỉnh (có thể rỗng)."""
        self._buf += delta
        out: list[str] = []
        while (cut := self._find_cut()) is not None:
            piece, self._buf = self._buf[:cut].strip(), self._buf[cut:]
            if piece:
                out.append(piece)
                self._emitted += 1
        return out

    def flush(self) -> list[str]:
        """Kết thúc luồng văn bản: nhả phần còn lại (chia nhỏ nếu quá dài)."""
        rest, self._buf = self._buf.strip(), ""
        if not rest:
            return []
        pieces = split_long(rest, self.config.limit(self._emitted))
        self._emitted += len(pieces)
        return pieces
