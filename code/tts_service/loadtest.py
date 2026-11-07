"""Thử tải đơn giản cho dịch vụ TTS: vòng kín (số client cố định) hoặc vòng mở (Poisson).

Ví dụ::

    python -m tts_service.loadtest --url ws://localhost:8000/v1/tts/ws \
        --concurrency 8 --requests 64
    python -m tts_service.loadtest --url ws://localhost:8000/v1/tts/ws \
        --rate 4 --requests 100      # 4 yêu cầu/giây, đến theo quá trình Poisson
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass, field

import numpy as np

from .client import ClientResult, synthesize_http_stream, synthesize_ws

DEFAULT_TEXTS = [
    "Xin chào, tôi là trợ lý ảo của bạn.",
    "Hôm nay trời Hà Nội có mưa rào và dông vài nơi, nhiệt độ cao nhất khoảng ba mươi độ.",
    "Bạn có muốn nghe tiếp chương sau của cuốn sách không?",
    "Đơn hàng của quý khách đã được giao thành công. Cảm ơn quý khách đã tin dùng dịch vụ.",
]


@dataclass
class LoadTestReport:
    wall_s: float
    results: list[ClientResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        ok = self.results
        if not ok:
            return {"ok": 0, "errors": len(self.errors)}
        ttfb = np.array([r.ttfb_s for r in ok]) * 1000.0
        rtf = np.array([r.rtf_client for r in ok])
        audio = sum(r.audio_s for r in ok)
        p = lambda xs, q: round(float(np.percentile(xs, q)), 1)  # noqa: E731
        return {
            "ok": len(ok), "errors": len(self.errors),
            "ttfb_ms_p50": p(ttfb, 50), "ttfb_ms_p90": p(ttfb, 90),
            "ttfb_ms_p95": p(ttfb, 95), "ttfb_ms_p99": p(ttfb, 99),
            "rtf_client_mean": round(float(rtf.mean()), 4),
            "rtf_client_p95": round(float(np.percentile(rtf, 95)), 4),
            "requests_per_s": round(len(ok) / self.wall_s, 3),
            # số giây âm thanh phục vụ được trong mỗi giây đồng hồ: > 1 nghĩa là nhanh hơn thời gian thực
            "audio_s_per_wall_s": round(audio / self.wall_s, 3),
        }


async def _one(url: str, text: str, report: LoadTestReport) -> None:
    try:
        if url.startswith("ws"):
            res = await synthesize_ws(url, text)
        else:
            res = await synthesize_http_stream(url, text)
        report.results.append(res)
    except Exception as exc:  # noqa: BLE001 - lỗi là một kết quả cần đếm
        report.errors.append(repr(exc))


async def run_load_test(url: str, texts: list[str] | None = None, *, concurrency: int = 4,
                        requests: int = 32, rate: float | None = None,
                        max_in_flight: int = 256, seed: int = 0) -> LoadTestReport:
    """``rate=None``: vòng kín với ``concurrency`` client, mỗi client gửi yêu cầu mới ngay khi
    yêu cầu cũ xong. ``rate`` (yêu cầu/giây): vòng mở, thời điểm đến tuân theo quá trình Poisson
    bất kể máy chủ đã trả lời hay chưa — gần với lưu lượng thật hơn.
    """
    texts = texts or DEFAULT_TEXTS
    rng = random.Random(seed)
    report = LoadTestReport(wall_s=0.0)
    t0 = time.perf_counter()
    if rate is None:
        counter = iter(range(requests))

        async def worker() -> None:
            for i in counter:  # các worker dùng chung một bộ đếm
                await _one(url, texts[i % len(texts)], report)

        await asyncio.gather(*(worker() for _ in range(concurrency)))
    else:
        limit = asyncio.Semaphore(max_in_flight)

        async def guarded(text: str) -> None:
            async with limit:
                await _one(url, text, report)

        tasks = []
        for i in range(requests):
            await asyncio.sleep(rng.expovariate(rate))  # khoảng cách giữa hai lần đến ~ Exp(rate)
            tasks.append(asyncio.create_task(guarded(texts[i % len(texts)])))
        await asyncio.gather(*tasks)
    report.wall_s = time.perf_counter() - t0
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Thử tải dịch vụ TTS")
    ap.add_argument("--url", default="ws://localhost:8000/v1/tts/ws")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--rate", type=float, default=None, help="yêu cầu/giây (vòng mở)")
    args = ap.parse_args()
    report = asyncio.run(run_load_test(args.url, concurrency=args.concurrency,
                                       requests=args.requests, rate=args.rate))
    print(json.dumps(report.summary(), ensure_ascii=False, indent=2))
    for e in report.errors[:5]:
        print("Lỗi:", e)


if __name__ == "__main__":
    main()
