"""Dịch vụ TTS streaming (Dự án 5 của giáo trình).

Các thành phần chính:

- :mod:`tts_service.synthesizer` — giao diện ``Synthesizer`` và các cài đặt (giả lập, ONNX).
- :mod:`tts_service.text` — chuẩn hóa, tách câu, chia đoạn (dùng ``vn_frontend`` nếu có).
- :mod:`tts_service.engine` — bộ máy streaming (thread pool, pipeline, đo TTFB/RTF).
- :mod:`tts_service.server` — ứng dụng FastAPI (HTTP + WebSocket).
- :mod:`tts_service.client`, :mod:`tts_service.loadtest` — client và thử tải.
"""
from .audio import float_to_pcm16, pcm16_to_float, to_wav_bytes, wav_header
from .engine import AudioChunk, TTSEngine
from .metrics import Metrics, StreamStats
from .synthesizer import (CallableSynthesizer, OnnxVitsSynthesizer, Synthesizer,
                          ToneSynthesizer, syllable_tone)
from .text import ChunkConfig, IncrementalChunker, TextFrontend, chunk_text, default_frontend

__all__ = [
    "AudioChunk", "CallableSynthesizer", "ChunkConfig", "IncrementalChunker", "Metrics",
    "OnnxVitsSynthesizer", "StreamStats", "Synthesizer", "TTSEngine", "TextFrontend",
    "ToneSynthesizer", "chunk_text", "default_frontend", "float_to_pcm16", "pcm16_to_float",
    "syllable_tone", "to_wav_bytes", "wav_header",
]
