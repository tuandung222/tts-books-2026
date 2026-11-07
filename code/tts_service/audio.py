"""Tiện ích âm thanh cho dịch vụ TTS: chuyển đổi PCM 16-bit, header WAV, chia khung, fade.

Mọi hàm ở đây chỉ dùng numpy và thư viện chuẩn để dịch vụ không phụ thuộc
thư viện âm thanh nặng.
"""
from __future__ import annotations

import io
import struct
import wave
from collections.abc import Iterator

import numpy as np

PCM_SAMPLE_WIDTH = 2  # 16 bit = 2 byte mỗi mẫu
_INT16_MAX = 32767


def float_to_pcm16(audio: np.ndarray) -> bytes:
    """Chuyển dạng sóng float trong [-1, 1] sang PCM 16-bit little-endian.

    Giá trị vượt ngoài [-1, 1] bị cắt (clip) thay vì bị tràn số: tràn số int16
    tạo ra tiếng "nổ" rất khó chịu, còn clip chỉ làm méo nhẹ.
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)  # (L,)
    if not np.all(np.isfinite(x)):
        # NaN/Inf từ mô hình (ví dụ fp16 tràn số) -> thay bằng im lặng
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, -1.0, 1.0)
    return np.round(x * _INT16_MAX).astype("<i2").tobytes()


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    """Phép ngược của :func:`float_to_pcm16` (sai số lượng tử <= 1/32767)."""
    if len(pcm) % PCM_SAMPLE_WIDTH:
        raise ValueError("Độ dài dữ liệu PCM 16-bit phải là số chẵn byte")
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / _INT16_MAX  # (L,)


def wav_header(sample_rate: int, num_samples: int | None = None, channels: int = 1) -> bytes:
    """Tạo header WAV (RIFF) 44 byte cho PCM 16-bit.

    Khi ``num_samples`` là ``None`` (streaming, chưa biết độ dài), các trường kích
    thước được đặt bằng 0xFFFFFFFF — đa số trình phát (ffmpeg, trình duyệt) chấp
    nhận quy ước này và đọc đến khi luồng kết thúc.
    """
    block_align = channels * PCM_SAMPLE_WIDTH
    byte_rate = sample_rate * block_align
    if num_samples is None:
        data_size = riff_size = 0xFFFFFFFF
    else:
        data_size = num_samples * block_align
        riff_size = 36 + data_size
    return (
        b"RIFF" + struct.pack("<I", riff_size) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate,
                                byte_rate, block_align, 8 * PCM_SAMPLE_WIDTH)
        + b"data" + struct.pack("<I", data_size)
    )


def to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Đóng gói toàn bộ dạng sóng float thành một tệp WAV hoàn chỉnh (trong bộ nhớ)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(PCM_SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(float_to_pcm16(audio))
    return buf.getvalue()


def iter_frames(pcm: bytes, frame_samples: int) -> Iterator[bytes]:
    """Chia một khối PCM thành các khung cố định ``frame_samples`` mẫu (khung cuối có thể ngắn hơn).

    ``frame_samples <= 0`` nghĩa là gửi nguyên khối.
    """
    if frame_samples <= 0:
        if pcm:
            yield pcm
        return
    step = frame_samples * PCM_SAMPLE_WIDTH
    for start in range(0, len(pcm), step):
        yield pcm[start:start + step]


def apply_fade(audio: np.ndarray, sample_rate: int, fade_ms: float = 5.0) -> np.ndarray:
    """Fade-in/fade-out dạng nửa chu kỳ cosin ở hai đầu một đoạn âm thanh.

    Khi ghép các đoạn sinh độc lập, biên đoạn hiếm khi bằng 0 đúng lúc; bước nhảy
    biên độ tạo ra tiếng "tách" (click). Vài mili-giây fade là đủ để loại bỏ nó
    mà không nghe thấy khác biệt.
    """
    x = np.asarray(audio, dtype=np.float32).copy()  # (L,)
    n = min(int(sample_rate * fade_ms / 1000.0), len(x) // 2)
    if n <= 0:
        return x
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n, dtype=np.float32))  # 0 -> 1
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x
