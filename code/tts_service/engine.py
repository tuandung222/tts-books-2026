"""Bộ máy streaming: chạy Synthesizer trong thread pool, sinh đoạn k+1 trong khi gửi đoạn k.

Đây là phần "lõi" độc lập với giao thức mạng: HTTP và WebSocket đều dùng chung.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .audio import apply_fade, float_to_pcm16
from .metrics import StreamStats
from .synthesizer import Synthesizer
from .text import ChunkConfig, TextFrontend, chunk_text, default_frontend


@dataclass
class AudioChunk:
    """Một đoạn âm thanh đã sinh, sẵn sàng gửi đi."""

    index: int
    text: str
    pcm: bytes          # PCM 16-bit little-endian, mono
    n_samples: int
    compute_s: float    # thời gian Synthesizer chạy cho đoạn này
    queue_wait_s: float  # thời gian chờ tới lượt dùng mô hình


async def _as_async(source: Iterable[str] | AsyncIterable[str]) -> AsyncIterator[str]:
    if hasattr(source, "__aiter__"):
        async for item in source:  # type: ignore[union-attr]
            yield item
    else:
        for item in source:  # type: ignore[union-attr]
            yield item


class TTSEngine:
    """Điều phối việc tổng hợp từng đoạn văn bản.

    - ``max_concurrency``: số lời gọi ``synthesize`` chạy song song tối đa (số luồng
      của thread pool). Với một GPU và mô hình không batch, giá trị 1–2 thường hợp lý.
    - ``prefetch``: số đoạn được phép sinh trước khi đoạn trước được gửi xong. Giới hạn
      này tạo *áp lực ngược* (backpressure): client đọc chậm thì máy chủ ngừng sinh,
      nhường mô hình cho người khác.
    """

    def __init__(self, synthesizer: Synthesizer, *, max_concurrency: int = 1,
                 frontend: TextFrontend | None = None,
                 chunk_config: ChunkConfig | None = None,
                 prefetch: int = 1, fade_ms: float = 5.0) -> None:
        self.synthesizer = synthesizer
        self.sample_rate = synthesizer.sample_rate
        self.frontend = frontend or default_frontend()
        self.chunk_config = chunk_config or ChunkConfig()
        self.prefetch = max(1, prefetch)
        self.fade_ms = fade_ms
        # Thread pool riêng: số luồng = giới hạn đồng thời. Khi một yêu cầu bị hủy, lời gọi
        # đang chạy trong luồng vẫn chạy nốt, nhưng không có lời gọi MỚI nào vượt giới hạn.
        self._executor = ThreadPoolExecutor(max_workers=max_concurrency,
                                            thread_name_prefix="tts-synth")

    # -- văn bản ---------------------------------------------------------------
    def plan(self, text: str) -> list[str]:
        """Văn bản thô -> danh sách đoạn đã chuẩn hóa, sẵn sàng tổng hợp."""
        return chunk_text(text, self.chunk_config, self.frontend)

    def normalize(self, text: str) -> str:
        return self.frontend.normalize(text)

    # -- mô hình ---------------------------------------------------------------
    async def synthesize_chunk(self, text: str, speaker: str | None,
                               speed: float) -> tuple[np.ndarray, float, float]:
        """Chạy Synthesizer trong thread pool; trả về (audio, thời_gian_chờ, thời_gian_tính)."""
        loop = asyncio.get_running_loop()
        t_submit = time.perf_counter()

        def job() -> tuple[np.ndarray, float, float]:
            t0 = time.perf_counter()
            audio = self.synthesizer.synthesize(text, speaker=speaker, speed=speed)
            return audio, t0 - t_submit, time.perf_counter() - t0

        return await loop.run_in_executor(self._executor, job)

    async def warmup(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self._executor, self.synthesizer.warmup)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.synthesizer.close()

    # -- streaming -------------------------------------------------------------
    async def stream(self, chunks: Iterable[str] | AsyncIterable[str], *,
                     speaker: str | None = None, speed: float = 1.0,
                     stats: StreamStats | None = None) -> AsyncIterator[AudioChunk]:
        """Sinh lần lượt các đoạn; đoạn k+1 được tổng hợp trong khi đoạn k đang được gửi.

        Người gọi nên dùng ``contextlib.aclosing`` để khi dừng giữa chừng (client
        ngắt kết nối, lệnh hủy), tác vụ sinh nền cũng bị hủy ngay.
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=self.prefetch)
        end = object()

        async def producer() -> None:
            try:
                index = 0
                async for text in _as_async(chunks):
                    audio, wait_s, compute_s = await self.synthesize_chunk(text, speaker, speed)
                    audio = apply_fade(audio, self.sample_rate, self.fade_ms)
                    await queue.put(AudioChunk(index, text, float_to_pcm16(audio),
                                               len(audio), compute_s, wait_s))
                    index += 1
                await queue.put(end)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - chuyển lỗi sang phía tiêu thụ
                await queue.put(exc)

        task = asyncio.create_task(producer())
        try:
            while True:
                item = await queue.get()
                if item is end:
                    break
                if isinstance(item, Exception):
                    raise item
                if stats is not None:
                    stats.n_chunks += 1
                    stats.audio_samples += item.n_samples
                    stats.compute_s += item.compute_s
                    stats.queue_wait_s += item.queue_wait_s
                    if stats.t_first_audio is None and item.n_samples > 0:
                        stats.t_first_audio = time.perf_counter()  # TTFB phía máy chủ
                yield item
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
