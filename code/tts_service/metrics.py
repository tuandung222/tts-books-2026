"""Đo lường cho dịch vụ TTS: TTFB, RTF và các bộ đếm theo định dạng văn bản của Prometheus.

Ta tự cài một registry rất nhỏ để không thêm phụ thuộc; trong sản phẩm thật nên
dùng thư viện ``prometheus_client`` (cùng định dạng xuất ra).
"""
from __future__ import annotations

import bisect
import threading
import time
from dataclasses import dataclass, field

# Mốc histogram (giây) cho TTFB và thời gian tính toán; mốc không thứ nguyên cho RTF
LATENCY_BUCKETS = (0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
RTF_BUCKETS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)


@dataclass
class StreamStats:
    """Số liệu của một yêu cầu tổng hợp (đo phía máy chủ)."""

    sample_rate: int
    n_chars: int = 0
    t_start: float = field(default_factory=time.perf_counter)
    t_first_audio: float | None = None
    t_end: float | None = None
    n_chunks: int = 0
    audio_samples: int = 0
    compute_s: float = 0.0     # tổng thời gian Synthesizer thực sự chạy
    queue_wait_s: float = 0.0  # tổng thời gian chờ đến lượt dùng mô hình

    @property
    def audio_s(self) -> float:
        return self.audio_samples / self.sample_rate

    @property
    def ttfb_s(self) -> float | None:
        return None if self.t_first_audio is None else self.t_first_audio - self.t_start

    @property
    def wall_s(self) -> float:
        return (self.t_end or time.perf_counter()) - self.t_start

    @property
    def rtf(self) -> float | None:
        """RTF tính toán = thời gian mô hình chạy / thời lượng âm thanh sinh ra."""
        return self.compute_s / self.audio_s if self.audio_samples else None

    @property
    def rtf_wall(self) -> float | None:
        """RTF "tường" = tổng thời gian phục vụ (gồm chờ hàng đợi, mạng) / thời lượng âm thanh."""
        return self.wall_s / self.audio_s if self.audio_samples else None

    def as_dict(self) -> dict:
        def r(x: float | None, k: int = 4) -> float | None:
            return None if x is None else round(x, k)
        return {
            "ttfb_ms": r(None if self.ttfb_s is None else 1000 * self.ttfb_s, 1),
            "audio_seconds": r(self.audio_s, 3),
            "wall_seconds": r(self.wall_s, 3),
            "compute_seconds": r(self.compute_s, 3),
            "queue_wait_seconds": r(self.queue_wait_s, 3),
            "rtf": r(self.rtf),
            "rtf_wall": r(self.rtf_wall),
            "chunks": self.n_chunks,
            "chars": self.n_chars,
        }


class _Histogram:
    def __init__(self, buckets: tuple[float, ...]) -> None:
        self.buckets = buckets
        self.counts = [0] * (len(buckets) + 1)  # ô cuối là +Inf
        self.sum = 0.0
        self.count = 0

    def observe(self, value: float) -> None:
        self.counts[bisect.bisect_left(self.buckets, value)] += 1
        self.sum += value
        self.count += 1


class Metrics:
    """Registry tối giản: counter, gauge, histogram; xuất ra định dạng Prometheus."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests: dict[tuple[str, str], int] = {}  # (endpoint, status) -> số yêu cầu
        self.audio_seconds_total = 0.0
        self.compute_seconds_total = 0.0
        self.active_streams = 0
        self.ttfb = _Histogram(LATENCY_BUCKETS)
        self.rtf = _Histogram(RTF_BUCKETS)
        self.queue_wait = _Histogram(LATENCY_BUCKETS)

    def stream_started(self) -> None:
        with self._lock:
            self.active_streams += 1

    def stream_finished(self, endpoint: str, status: str, stats: StreamStats | None) -> None:
        with self._lock:
            self.active_streams -= 1
            key = (endpoint, status)
            self.requests[key] = self.requests.get(key, 0) + 1
            if stats is None:
                return
            self.audio_seconds_total += stats.audio_s
            self.compute_seconds_total += stats.compute_s
            if stats.ttfb_s is not None:
                self.ttfb.observe(stats.ttfb_s)
            if stats.rtf is not None:
                self.rtf.observe(stats.rtf)
            self.queue_wait.observe(stats.queue_wait_s)

    @staticmethod
    def _render_hist(name: str, help_: str, h: _Histogram) -> list[str]:
        lines = [f"# HELP {name} {help_}", f"# TYPE {name} histogram"]
        cum = 0
        for le, c in zip(h.buckets, h.counts):
            cum += c
            lines.append(f'{name}_bucket{{le="{le}"}} {cum}')
        lines.append(f'{name}_bucket{{le="+Inf"}} {h.count}')
        lines += [f"{name}_sum {h.sum:.6f}", f"{name}_count {h.count}"]
        return lines

    def render(self) -> str:
        with self._lock:
            lines = ["# HELP tts_requests_total Số yêu cầu đã xử lý xong.",
                     "# TYPE tts_requests_total counter"]
            for (endpoint, status), n in sorted(self.requests.items()):
                lines.append(f'tts_requests_total{{endpoint="{endpoint}",status="{status}"}} {n}')
            lines += [
                "# HELP tts_audio_seconds_total Tổng số giây âm thanh đã sinh.",
                "# TYPE tts_audio_seconds_total counter",
                f"tts_audio_seconds_total {self.audio_seconds_total:.6f}",
                "# HELP tts_compute_seconds_total Tổng thời gian mô hình chạy.",
                "# TYPE tts_compute_seconds_total counter",
                f"tts_compute_seconds_total {self.compute_seconds_total:.6f}",
                "# HELP tts_active_streams Số luồng đang phục vụ.",
                "# TYPE tts_active_streams gauge",
                f"tts_active_streams {self.active_streams}",
            ]
            lines += self._render_hist("tts_ttfb_seconds", "Thời gian tới byte âm thanh đầu tiên.", self.ttfb)
            lines += self._render_hist("tts_rtf", "Hệ số thời gian thực (tính toán).", self.rtf)
            lines += self._render_hist("tts_queue_wait_seconds", "Thời gian chờ mô hình.", self.queue_wait)
        return "\n".join(lines) + "\n"
