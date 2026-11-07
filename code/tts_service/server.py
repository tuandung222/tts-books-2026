"""Ứng dụng FastAPI: HTTP (trả trọn tệp hoặc chunked) và WebSocket streaming cho TTS.

Chạy thử::

    cd code
    uvicorn tts_service.server:create_app --factory --port 8000

Giao thức WebSocket (``/v1/tts/ws``), mọi thông điệp điều khiển là JSON dạng văn bản:

- máy chủ -> client: ``ready`` (khi kết nối), ``start``, ``chunk`` (siêu dữ liệu của đoạn,
  theo sau là các khung nhị phân PCM 16-bit LE), ``end`` (kèm số liệu), ``cancelled``, ``error``.
- client -> máy chủ: ``synthesize`` (cả văn bản), ``text`` + ``flush`` (văn bản đến dần,
  ví dụ từ LLM), ``cancel`` (ngắt lời).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import aclosing, asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .audio import iter_frames, wav_header
from .engine import TTSEngine
from .metrics import Metrics, StreamStats
from .synthesizer import Synthesizer, synthesizer_from_env
from .text import ChunkConfig, IncrementalChunker, TextFrontend

logger = logging.getLogger("tts_service")

MIN_SPEED, MAX_SPEED = 0.5, 2.0


@dataclass
class ServiceConfig:
    max_text_chars: int = 5000      # giới hạn độ dài đầu vào (chống lạm dụng, chống OOM)
    max_concurrency: int = 1        # số lời gọi mô hình song song
    prefetch: int = 1               # số đoạn được sinh trước
    frame_ms: float = 200.0         # kích thước khung nhị phân gửi qua WebSocket
    warmup: bool = True
    chunk: ChunkConfig = field(default_factory=ChunkConfig)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ServiceConfig:
        env = os.environ if env is None else env
        return cls(
            max_text_chars=int(env.get("TTS_MAX_TEXT_CHARS", "5000")),
            max_concurrency=int(env.get("TTS_MAX_CONCURRENCY", "1")),
            prefetch=int(env.get("TTS_PREFETCH", "1")),
            frame_ms=float(env.get("TTS_FRAME_MS", "200")),
            warmup=env.get("TTS_WARMUP", "1") == "1",
            chunk=ChunkConfig(
                first_max_chars=int(env.get("TTS_FIRST_CHUNK_CHARS", "60")),
                max_chars=int(env.get("TTS_MAX_CHUNK_CHARS", "220")),
            ),
        )


class TTSRequest(BaseModel):
    text: str = Field(min_length=1)
    speaker: str | None = None
    speed: float = Field(1.0, ge=MIN_SPEED, le=MAX_SPEED)
    format: Literal["wav", "pcm"] = "wav"


def create_app(synthesizer: Synthesizer | None = None, config: ServiceConfig | None = None,
               frontend: TextFrontend | None = None) -> FastAPI:
    """Nhà máy (factory) tạo ứng dụng; không đối số thì đọc cấu hình từ biến môi trường."""
    config = config or ServiceConfig.from_env()
    synthesizer = synthesizer or synthesizer_from_env()
    engine = TTSEngine(synthesizer, max_concurrency=config.max_concurrency, frontend=frontend,
                       chunk_config=config.chunk, prefetch=config.prefetch)
    metrics = Metrics()
    sr = synthesizer.sample_rate
    frame_samples = int(sr * config.frame_ms / 1000)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if config.warmup:
            await engine.warmup()  # nạp mô hình trước khi báo "sẵn sàng"
        yield
        engine.close()

    app = FastAPI(title="Dịch vụ TTS tiếng Việt (streaming)", lifespan=lifespan)
    app.state.engine, app.state.metrics, app.state.config = engine, metrics, config

    def plan_or_raise(text: str) -> list[str]:
        if len(text) > config.max_text_chars:
            raise HTTPException(413, f"Văn bản dài hơn {config.max_text_chars} ký tự")
        chunks = engine.plan(text)
        if not chunks:
            raise HTTPException(422, "Văn bản không có nội dung đọc được")
        return chunks

    def log_done(kind: str, request_id: str, status: str, stats: StreamStats) -> None:
        logger.info(json.dumps({"event": "tts_done", "kind": kind, "request_id": request_id,
                                "status": status, **stats.as_dict()}, ensure_ascii=False))

    # ------------------------------------------------------------------ HTTP
    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "synthesizer": synthesizer.name, "sample_rate": sr,
                "frontend": engine.frontend.name}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus_metrics() -> PlainTextResponse:
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.post("/v1/tts")
    async def tts_full(req: TTSRequest) -> Response:
        """Trả về trọn tệp âm thanh: đơn giản, nhưng TTFB = toàn bộ thời gian tổng hợp."""
        chunks = plan_or_raise(req.text)
        request_id = uuid.uuid4().hex[:12]
        stats, status, pcm = StreamStats(sr, n_chars=len(req.text)), "error", bytearray()
        metrics.stream_started()
        try:
            async with aclosing(engine.stream(chunks, speaker=req.speaker, speed=req.speed,
                                              stats=stats)) as agen:
                async for chunk in agen:
                    pcm += chunk.pcm
            status = "ok"
        finally:
            # Với kiểu trả trọn tệp, byte âm thanh đầu tiên chỉ rời máy chủ khi xong hết
            stats.t_end = stats.t_first_audio = time.perf_counter()
            metrics.stream_finished("full", status, stats)
            log_done("full", request_id, status, stats)
        body = (wav_header(sr, len(pcm) // 2) + pcm) if req.format == "wav" else bytes(pcm)
        return Response(bytes(body), media_type="audio/wav" if req.format == "wav"
                        else "application/octet-stream",
                        headers={"X-Request-ID": request_id, "X-Sample-Rate": str(sr),
                                 "X-TTS-RTF": f"{stats.rtf or 0:.4f}",
                                 "X-TTS-Audio-Seconds": f"{stats.audio_s:.3f}"})

    @app.post("/v1/tts/stream")
    async def tts_stream(req: TTSRequest) -> StreamingResponse:
        """HTTP chunked: gửi PCM ngay khi từng đoạn sinh xong.

        Lưu ý: mã trạng thái 200 đã gửi đi trước khi tổng hợp; nếu lỗi giữa chừng, client
        chỉ thấy luồng bị cắt cụt. Đó là lý do giao thức WebSocket có thông điệp ``end``/``error``.
        """
        chunks = plan_or_raise(req.text)
        request_id = uuid.uuid4().hex[:12]
        stats = StreamStats(sr, n_chars=len(req.text))

        async def body() -> AsyncIterator[bytes]:
            status = "cancelled"
            metrics.stream_started()
            try:
                if req.format == "wav":
                    yield wav_header(sr, None)  # header "không rõ độ dài" cho streaming
                async with aclosing(engine.stream(chunks, speaker=req.speaker, speed=req.speed,
                                                  stats=stats)) as agen:
                    async for chunk in agen:
                        yield chunk.pcm
                status = "ok"
            except Exception:
                status = "error"
                raise
            finally:
                stats.t_end = time.perf_counter()
                metrics.stream_finished("stream", status, stats)
                log_done("stream", request_id, status, stats)

        media_type = "audio/wav" if req.format == "wav" else "application/octet-stream"
        return StreamingResponse(body(), media_type=media_type, headers={
            "X-Request-ID": request_id, "X-Sample-Rate": str(sr), "X-Audio-Encoding": "pcm_s16le",
            "Cache-Control": "no-store", "X-Accel-Buffering": "no"})  # tắt đệm của nginx

    # ------------------------------------------------------------- WebSocket
    @app.websocket("/v1/tts/ws")
    async def tts_ws(ws: WebSocket) -> None:
        await ws.accept()
        send_lock = asyncio.Lock()  # chỉ một coroutine được ghi vào socket tại một thời điểm

        async def send_json(obj: dict) -> None:
            async with send_lock:
                await ws.send_text(json.dumps(obj, ensure_ascii=False))

        async def send_bytes(data: bytes) -> None:
            async with send_lock:
                await ws.send_bytes(data)

        async def run(request_id: str, source, stats: StreamStats,
                      speaker: str | None, speed: float) -> None:
            status = "error"
            metrics.stream_started()
            try:
                await send_json({"type": "start", "request_id": request_id})
                async with aclosing(engine.stream(source, speaker=speaker, speed=speed,
                                                  stats=stats)) as agen:
                    async for chunk in agen:
                        await send_json({"type": "chunk", "request_id": request_id,
                                         "index": chunk.index, "text": chunk.text,
                                         "samples": chunk.n_samples})
                        for frame in iter_frames(chunk.pcm, frame_samples):
                            await send_bytes(frame)
                stats.t_end = time.perf_counter()
                status = "ok"
                await send_json({"type": "end", "request_id": request_id,
                                 "stats": stats.as_dict()})
            except asyncio.CancelledError:
                status = "cancelled"
                raise
            except Exception as exc:  # noqa: BLE001 - báo lỗi cho client thay vì đóng im lặng
                logger.exception("Lỗi khi tổng hợp %s", request_id)
                with suppress(Exception):
                    await send_json({"type": "error", "request_id": request_id, "message": str(exc)})
            finally:
                stats.t_end = stats.t_end or time.perf_counter()
                metrics.stream_finished("ws", status, stats)
                log_done("ws", request_id, status, stats)

        async def from_queue(q: asyncio.Queue) -> AsyncIterator[str]:
            while (item := await q.get()) is not None:
                yield item

        async def error(message: str, request_id: str | None = None) -> None:
            await send_json({"type": "error", "request_id": request_id, "message": message})

        await send_json({"type": "ready", "sample_rate": sr, "encoding": "pcm_s16le",
                         "channels": 1, "synthesizer": synthesizer.name})
        active: asyncio.Task | None = None
        active_id: str | None = None
        text_queue: asyncio.Queue | None = None   # chỉ dùng ở chế độ văn bản đến dần
        chunker: IncrementalChunker | None = None
        inc_stats: StreamStats | None = None
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                    kind = msg.get("type")
                except (json.JSONDecodeError, AttributeError):
                    await error("Thông điệp phải là đối tượng JSON")
                    continue
                busy = active is not None and not active.done()
                if kind in ("synthesize", "text"):
                    try:
                        speed = float(msg.get("speed", 1.0))
                    except (TypeError, ValueError):
                        speed = -1.0
                    if not MIN_SPEED <= speed <= MAX_SPEED:
                        await error(f"speed phải trong [{MIN_SPEED}, {MAX_SPEED}]")
                        continue
                if kind == "synthesize":
                    if busy:
                        await error("Đang xử lý một yêu cầu khác; gửi 'cancel' trước", active_id)
                        continue
                    text = str(msg.get("text", ""))
                    if len(text) > config.max_text_chars:
                        await error(f"Văn bản dài hơn {config.max_text_chars} ký tự")
                        continue
                    chunks = engine.plan(text)
                    if not chunks:
                        await error("Văn bản không có nội dung đọc được")
                        continue
                    active_id = str(msg.get("request_id") or uuid.uuid4().hex[:12])
                    active = asyncio.create_task(run(active_id, chunks, StreamStats(sr, n_chars=len(text)),
                                                     msg.get("speaker"), speed))
                elif kind == "text":
                    if text_queue is None:
                        if busy:
                            await error("Đang xử lý một yêu cầu khác; gửi 'cancel' trước", active_id)
                            continue
                        text_queue, chunker = asyncio.Queue(), IncrementalChunker(config.chunk)
                        inc_stats = StreamStats(sr)
                        active_id = str(msg.get("request_id") or uuid.uuid4().hex[:12])
                        active = asyncio.create_task(run(active_id, from_queue(text_queue), inc_stats,
                                                         msg.get("speaker"), speed))
                    delta = str(msg.get("text", ""))
                    inc_stats.n_chars += len(delta)
                    for piece in chunker.push(delta):
                        if (piece := engine.normalize(piece)) and any(c.isalnum() for c in piece):
                            text_queue.put_nowait(piece)
                elif kind == "flush":
                    if text_queue is not None:
                        for piece in chunker.flush():
                            if (piece := engine.normalize(piece)) and any(c.isalnum() for c in piece):
                                text_queue.put_nowait(piece)
                        text_queue.put_nowait(None)  # báo hết văn bản
                        text_queue = chunker = None
                elif kind == "cancel":
                    text_queue = chunker = None
                    if busy:
                        active.cancel()
                        with suppress(asyncio.CancelledError):
                            await active
                        await send_json({"type": "cancelled", "request_id": active_id})
                else:
                    await error(f"Loại thông điệp không hỗ trợ: {kind!r}")
        except WebSocketDisconnect:
            pass
        finally:
            if active is not None and not active.done():
                active.cancel()
                with suppress(asyncio.CancelledError):
                    await active

    return app
