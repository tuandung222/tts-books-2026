"""Client bất đồng bộ cho dịch vụ TTS (WebSocket và HTTP chunked), đo TTFB phía client.

Dùng từ dòng lệnh::

    python -m tts_service.client --url ws://localhost:8000/v1/tts/ws \
        --text "Xin chào các bạn." --out out.wav
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import wave
from dataclasses import dataclass, field

import httpx
from websockets.asyncio.client import connect

from .audio import PCM_SAMPLE_WIDTH

WAV_HEADER_BYTES = 44


@dataclass
class ClientResult:
    pcm: bytes
    sample_rate: int
    ttfb_s: float          # từ lúc gửi yêu cầu tới khi nhận byte ÂM THANH đầu tiên
    total_s: float         # từ lúc gửi yêu cầu tới khi nhận xong
    n_chunks: int = 0
    server_stats: dict = field(default_factory=dict)

    @property
    def audio_s(self) -> float:
        return len(self.pcm) / PCM_SAMPLE_WIDTH / self.sample_rate

    @property
    def rtf_client(self) -> float:
        return self.total_s / self.audio_s if self.audio_s else float("inf")


async def synthesize_ws(url: str, text: str, *, speaker: str | None = None,
                        speed: float = 1.0, timeout: float = 60.0) -> ClientResult:
    """Gửi một yêu cầu ``synthesize`` qua WebSocket và gom toàn bộ PCM trả về."""
    async with connect(url, max_size=None, open_timeout=timeout) as ws:
        ready = json.loads(await asyncio.wait_for(ws.recv(), timeout))
        assert ready["type"] == "ready", ready
        sample_rate = int(ready["sample_rate"])
        buf, ttfb, n_chunks = bytearray(), None, 0
        t0 = time.perf_counter()
        await ws.send(json.dumps({"type": "synthesize", "text": text,
                                  "speaker": speaker, "speed": speed}, ensure_ascii=False))
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout)
            if isinstance(msg, bytes):
                if ttfb is None:
                    ttfb = time.perf_counter() - t0
                buf += msg
                continue
            event = json.loads(msg)
            if event["type"] == "chunk":
                n_chunks += 1
            elif event["type"] == "end":
                total = time.perf_counter() - t0
                return ClientResult(bytes(buf), sample_rate, ttfb if ttfb is not None else total,
                                    total, n_chunks, event.get("stats", {}))
            elif event["type"] == "error":
                raise RuntimeError(event["message"])


async def synthesize_http_stream(base_url: str, text: str, *, speaker: str | None = None,
                                 speed: float = 1.0, timeout: float = 60.0) -> ClientResult:
    """Gọi ``POST /v1/tts/stream`` (định dạng WAV) và đọc dần thân phản hồi."""
    payload = {"text": text, "speaker": speaker, "speed": speed, "format": "wav"}
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        t0 = time.perf_counter()
        async with client.stream("POST", "/v1/tts/stream", json=payload) as resp:
            resp.raise_for_status()
            sample_rate = int(resp.headers["x-sample-rate"])
            buf, ttfb = bytearray(), None
            async for part in resp.aiter_bytes():
                buf += part
                # 44 byte đầu là header WAV, chưa phải âm thanh
                if ttfb is None and len(buf) > WAV_HEADER_BYTES:
                    ttfb = time.perf_counter() - t0
        total = time.perf_counter() - t0
    return ClientResult(bytes(buf[WAV_HEADER_BYTES:]), sample_rate,
                        ttfb if ttfb is not None else total, total)


def save_wav(path: str, pcm: bytes, sample_rate: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(PCM_SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def main() -> None:
    ap = argparse.ArgumentParser(description="Client thử dịch vụ TTS")
    ap.add_argument("--url", default="ws://localhost:8000/v1/tts/ws",
                    help="ws://... cho WebSocket hoặc http://host:port cho HTTP chunked")
    ap.add_argument("--text", required=True)
    ap.add_argument("--speaker", default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--out", default="out.wav")
    args = ap.parse_args()
    if args.url.startswith("ws"):
        coro = synthesize_ws(args.url, args.text, speaker=args.speaker, speed=args.speed)
    else:
        coro = synthesize_http_stream(args.url, args.text, speaker=args.speaker, speed=args.speed)
    res = asyncio.run(coro)
    save_wav(args.out, res.pcm, res.sample_rate)
    print(f"TTFB={1000 * res.ttfb_s:.1f} ms  tổng={res.total_s:.2f} s  "
          f"âm thanh={res.audio_s:.2f} s  RTF(client)={res.rtf_client:.3f}  -> {args.out}")


if __name__ == "__main__":
    main()
