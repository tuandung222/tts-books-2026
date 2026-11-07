"""Giao diện ``Synthesizer`` có thể thay thế và các cài đặt cụ thể.

- :class:`ToneSynthesizer`: bộ tổng hợp giả, tất định, sinh một âm có đường F0
  mô phỏng thô thanh điệu của từng âm tiết. Không cần mô hình, dùng cho kiểm thử
  và đo tải của chính hạ tầng dịch vụ.
- :class:`CallableSynthesizer`: bọc một hàm Python bất kỳ ``text -> waveform``.
- :class:`OnnxVitsSynthesizer`: chạy mô hình VITS đã xuất ONNX theo quy ước đầu
  vào kiểu Piper (``input``, ``input_lengths``, ``scales``, ``sid``) — xem Dự án 3.
"""
from __future__ import annotations

import os
import re
import time
import unicodedata
import zlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence

import numpy as np


class Synthesizer(ABC):
    """Hợp đồng tối thiểu giữa dịch vụ và một mô hình TTS.

    ``synthesize`` là hàm *đồng bộ* và có thể chậm (chạy mạng nơ-ron); dịch vụ sẽ
    gọi nó trong một luồng (thread) riêng để không chặn vòng lặp sự kiện.
    """

    name: str = "synthesizer"
    sample_rate: int = 22050

    @abstractmethod
    def synthesize(self, text: str, *, speaker: str | None = None,
                   speed: float = 1.0) -> np.ndarray:
        """Trả về dạng sóng float32 mono trong [-1, 1], hình dạng (L,)."""

    def warmup(self) -> None:
        """Chạy thử một lần để nạp trọng số, khởi tạo kernel... trước khi nhận tải thật."""
        self.synthesize("xin chào")

    def close(self) -> None:
        """Giải phóng tài nguyên (bộ nhớ GPU, phiên ONNX...)."""


# ---------------------------------------------------------------------------
# Bộ tổng hợp giả theo thanh điệu
# ---------------------------------------------------------------------------
_TONE_MARKS = {
    "̀": "huyen",  # dấu huyền
    "́": "sac",    # dấu sắc
    "̉": "hoi",    # dấu hỏi
    "̃": "nga",    # dấu ngã
    "̣": "nang",   # dấu nặng
}

# Đường F0 (đơn vị bán cung so với F0 nền) tại các mốc thời gian chuẩn hóa tau in [0, 1].
# Đây là phác họa rất thô theo mô tả giọng Bắc, CHỈ để kiểm thử — xem Chương 15.
TONE_CONTOURS: dict[str, tuple[list[float], list[float]]] = {
    "ngang": ([0.0, 1.0], [0.0, -0.5]),
    "huyen": ([0.0, 1.0], [-2.0, -6.0]),
    "sac": ([0.0, 0.3, 1.0], [-0.5, -1.0, 5.0]),
    "hoi": ([0.0, 0.6, 1.0], [-2.0, -7.0, -4.0]),
    "nga": ([0.0, 0.45, 0.55, 1.0], [-1.0, -2.0, 1.0, 6.0]),
    "nang": ([0.0, 1.0], [-3.0, -9.0]),
}
_DURATION_FACTOR = {"nang": 0.7}


def syllable_tone(syllable: str) -> str:
    """Nhận diện thanh điệu từ dấu thanh (sau khi tách tổ hợp NFD)."""
    for ch in unicodedata.normalize("NFD", syllable):
        if ch in _TONE_MARKS:
            return _TONE_MARKS[ch]
    return "ngang"


class ToneSynthesizer(Synthesizer):
    """Mỗi âm tiết -> một âm hài (3 họa âm) có đường F0 theo thanh điệu; dấu câu -> khoảng lặng.

    ``simulated_rtf > 0`` làm hàm "giả vờ" tốn thời gian tính toán bằng
    ``simulated_rtf * thời_lượng_âm_thanh`` giây, để thử tải và thử hủy yêu cầu.
    """

    name = "tone"

    def __init__(self, sample_rate: int = 22050, base_f0: float = 180.0,
                 syllable_s: float = 0.18, gap_s: float = 0.04,
                 comma_pause_s: float = 0.15, sentence_pause_s: float = 0.30,
                 amplitude: float = 0.3, simulated_rtf: float = 0.0) -> None:
        self.sample_rate = sample_rate
        self.base_f0 = base_f0
        self.syllable_s = syllable_s
        self.gap_s = gap_s
        self.comma_pause_s = comma_pause_s
        self.sentence_pause_s = sentence_pause_s
        self.amplitude = amplitude
        self.simulated_rtf = simulated_rtf

    def _speaker_f0(self, speaker: str | None) -> float:
        if not speaker:
            return self.base_f0
        shift = zlib.crc32(speaker.encode("utf-8")) % 13 - 6  # tất định, trong [-6, 6] bán cung
        return self.base_f0 * 2.0 ** (shift / 12.0)

    def _silence(self, seconds: float, speed: float) -> np.ndarray:
        return np.zeros(int(self.sample_rate * seconds / speed), dtype=np.float32)

    def _syllable(self, syllable: str, f0_base: float, speed: float) -> np.ndarray:
        tone = syllable_tone(syllable)
        n = max(int(self.sample_rate * self.syllable_s * _DURATION_FACTOR.get(tone, 1.0) / speed), 2)
        tau = np.linspace(0.0, 1.0, n, dtype=np.float64)                  # (n,)
        semitones = np.interp(tau, *TONE_CONTOURS[tone])
        f0 = f0_base * 2.0 ** (semitones / 12.0)                          # (n,) Hz
        phase = 2.0 * np.pi * np.cumsum(f0) / self.sample_rate           # pha liên tục
        wave = (np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)) / 1.75
        ramp = min(int(0.01 * self.sample_rate), n // 2)                 # 10 ms lên/xuống
        env = np.ones(n)
        if ramp > 0:
            r = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, ramp))
            env[:ramp] = r
            env[-ramp:] = r[::-1]
        if tone == "nga":  # chỗ "gãy" giữa âm tiết (thanh quản hóa) -> hạ biên độ
            env *= 1.0 - 0.8 * np.exp(-0.5 * ((tau - 0.5) / 0.06) ** 2)
        return (self.amplitude * wave * env).astype(np.float32)

    def synthesize(self, text: str, *, speaker: str | None = None,
                   speed: float = 1.0) -> np.ndarray:
        if speed <= 0:
            raise ValueError("speed phải dương")
        t_start = time.perf_counter()
        f0_base = self._speaker_f0(speaker)
        parts: list[np.ndarray] = []
        for token in re.findall(r"\w+|[^\w\s]", unicodedata.normalize("NFC", text)):
            if token[0].isalnum() or token[0] == "_":
                parts += [self._syllable(token, f0_base, speed), self._silence(self.gap_s, speed)]
            elif token in ",;:–—":
                parts.append(self._silence(self.comma_pause_s, speed))
            elif token in ".!?…":
                parts.append(self._silence(self.sentence_pause_s, speed))
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)  # (L,)
        if self.simulated_rtf > 0:
            target = self.simulated_rtf * len(audio) / self.sample_rate
            time.sleep(max(0.0, target - (time.perf_counter() - t_start)))
        return audio


# ---------------------------------------------------------------------------
# Bộ chuyển đổi (adapter) cho mô hình thật
# ---------------------------------------------------------------------------
class CallableSynthesizer(Synthesizer):
    """Bọc một hàm ``fn(text, speaker, speed) -> np.ndarray`` thành Synthesizer."""

    def __init__(self, fn: Callable[..., np.ndarray], sample_rate: int,
                 name: str = "callable") -> None:
        self._fn = fn
        self.sample_rate = sample_rate
        self.name = name

    def synthesize(self, text: str, *, speaker: str | None = None,
                   speed: float = 1.0) -> np.ndarray:
        return np.asarray(self._fn(text, speaker=speaker, speed=speed), dtype=np.float32).reshape(-1)


def intersperse(ids: Sequence[int], blank: int = 0) -> list[int]:
    """Chèn token trống giữa các token (``add_blank`` của VITS): [a, b] -> [0, a, 0, b, 0]."""
    out = [blank] * (2 * len(ids) + 1)
    out[1::2] = list(ids)
    return out


def _default_text_to_ids() -> Callable[[str], list[int]]:
    try:
        from vn_frontend import text_to_sequence  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Cần truyền text_to_ids hoặc cài gói vn_frontend (Dự án 1) để chuyển văn bản sang ID"
        ) from exc
    return text_to_sequence


class OnnxVitsSynthesizer(Synthesizer):
    """Chạy VITS đã xuất ONNX với quy ước đầu vào kiểu Piper.

    Đầu vào của đồ thị: ``input`` (int64, (1, N)), ``input_lengths`` (int64, (1,)),
    ``scales`` (float32, (3,) = [noise_scale, length_scale, noise_w]) và tùy chọn
    ``sid`` (int64, (1,)). Đầu ra đầu tiên là dạng sóng, mọi hình dạng đều được
    làm phẳng thành (L,).

    ``text_to_ids`` PHẢI giống hệt hàm dùng lúc huấn luyện (cùng bảng ký hiệu,
    cùng việc có/không chèn blank) — đây là nguồn lỗi số một khi triển khai.
    """

    name = "onnx-vits"

    def __init__(self, model_path: str | None = None, *, sample_rate: int = 22050,
                 text_to_ids: Callable[[str], Sequence[int]] | None = None,
                 add_blank: bool = False, noise_scale: float = 0.667,
                 length_scale: float = 1.0, noise_w: float = 0.8,
                 speaker_ids: Mapping[str, int] | None = None,
                 session=None, providers: Sequence[str] | None = None) -> None:
        if session is None:
            import onnxruntime as ort  # nạp muộn: chỉ cần khi dùng adapter này
            if model_path is None:
                raise ValueError("Cần model_path hoặc session")
            session = ort.InferenceSession(model_path, providers=list(providers or ["CPUExecutionProvider"]))
        self.session = session
        self.input_names = {i.name for i in session.get_inputs()}
        self.sample_rate = sample_rate
        self.text_to_ids = text_to_ids or _default_text_to_ids()
        self.add_blank = add_blank
        self.noise_scale, self.length_scale, self.noise_w = noise_scale, length_scale, noise_w
        self.speaker_ids = dict(speaker_ids or {})

    def synthesize(self, text: str, *, speaker: str | None = None,
                   speed: float = 1.0) -> np.ndarray:
        ids = list(self.text_to_ids(text))
        if self.add_blank:
            ids = intersperse(ids)
        if not ids:
            return np.zeros(0, dtype=np.float32)
        feeds = {
            "input": np.asarray([ids], dtype=np.int64),                       # (1, N)
            "input_lengths": np.asarray([len(ids)], dtype=np.int64),          # (1,)
            # length_scale lớn hơn -> nói chậm hơn, nên chia cho speed
            "scales": np.asarray([self.noise_scale, self.length_scale / speed, self.noise_w],
                                 dtype=np.float32),                            # (3,)
        }
        if "sid" in self.input_names:
            feeds["sid"] = np.asarray([self.speaker_ids.get(speaker or "", 0)], dtype=np.int64)
        audio = self.session.run(None, feeds)[0]
        return np.asarray(audio, dtype=np.float32).reshape(-1)              # (L,)


def synthesizer_from_env(env: Mapping[str, str] | None = None) -> Synthesizer:
    """Tạo Synthesizer từ biến môi trường (tiện cho Docker và ``uvicorn --factory``).

    - ``TTS_SYNTHESIZER``: ``tone`` (mặc định) hoặc ``onnx``
    - ``TTS_SAMPLE_RATE``: tần số lấy mẫu (mặc định 22050)
    - ``TTS_SIMULATED_RTF``: RTF giả lập cho ``tone``
    - ``TTS_ONNX_MODEL``, ``TTS_ADD_BLANK`` (0/1): cho ``onnx``
    """
    env = os.environ if env is None else env
    kind = env.get("TTS_SYNTHESIZER", "tone").lower()
    sample_rate = int(env.get("TTS_SAMPLE_RATE", "22050"))
    if kind == "tone":
        return ToneSynthesizer(sample_rate=sample_rate,
                               simulated_rtf=float(env.get("TTS_SIMULATED_RTF", "0")))
    if kind == "onnx":
        return OnnxVitsSynthesizer(env["TTS_ONNX_MODEL"], sample_rate=sample_rate,
                                   add_blank=env.get("TTS_ADD_BLANK", "0") == "1")
    raise ValueError(f"TTS_SYNTHESIZER không hợp lệ: {kind!r}")
