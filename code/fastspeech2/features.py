"""Trích đặc trưng cho FastSpeech 2: log-mel (tương thích HiFi-GAN), F0, năng lượng,
và chuyển kết quả căn chỉnh (TextGrid của MFA) thành trường độ theo khung."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from .config import AudioConfig


# ---------------------------------------------------------------------------
# Âm thanh và mel
# ---------------------------------------------------------------------------
def load_wav(path: str, sample_rate: int) -> np.ndarray:
    """Đọc wav bằng soundfile, trộn về mono, đổi tần số lấy mẫu nếu cần. -> float32 (L,)."""
    import librosa
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != sample_rate:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
    return wav.astype(np.float32)


@lru_cache(maxsize=4)
def _mel_basis(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> torch.Tensor:
    import librosa

    # librosa mặc định: thang mel Slaney + chuẩn hóa "slaney" — giống HiFi-GAN gốc
    return torch.from_numpy(librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax))


def stft_magnitude(wav: torch.Tensor, cfg: AudioConfig) -> torch.Tensor:
    """wav (B, L) -> biên độ STFT (B, n_fft//2+1, T) với T = L // hop.

    Đệm phản xạ (n_fft - hop)/2 mỗi bên và center=False, đúng như mã HiFi-GAN gốc,
    để mel của mô hình âm học khớp với mel mà vocoder được huấn luyện.
    """
    pad = (cfg.n_fft - cfg.hop_length) // 2
    y = F.pad(wav.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    window = torch.hann_window(cfg.win_length, device=wav.device)
    spec = torch.stft(y, cfg.n_fft, cfg.hop_length, cfg.win_length, window=window,
                      center=False, return_complex=True)
    return torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)


def mel_spectrogram(wav: torch.Tensor, cfg: AudioConfig) -> torch.Tensor:
    """wav (B, L) hoặc (L,) -> log-mel (B, F, T): log tự nhiên, kẹp dưới 1e-5."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    mag = stft_magnitude(wav, cfg)
    basis = _mel_basis(cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.f_min, cfg.f_max).to(wav.device)
    return torch.log(torch.clamp(basis @ mag, min=1e-5))


def energy_from_wav(wav: torch.Tensor, cfg: AudioConfig) -> torch.Tensor:
    """Năng lượng mỗi khung = chuẩn L2 của biên độ STFT (như FastSpeech 2). -> (B, T)."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    return torch.linalg.vector_norm(stft_magnitude(wav, cfg), dim=1)


# ---------------------------------------------------------------------------
# Cao độ
# ---------------------------------------------------------------------------
def extract_f0(wav: np.ndarray, cfg: AudioConfig) -> tuple[np.ndarray, np.ndarray]:
    """F0 (Hz) bằng pYIN của librosa, căn theo lưới khung của mel. -> (f0, voiced), dài T.

    pYIN chậm nhưng không cần cài thêm gì; với dữ liệu lớn có thể thay bằng WORLD
    (pyworld), CREPE hoặc RMVPE (Chương 14). Khung pYIN (center=True) lệch nửa bước
    nhảy so với khung mel — không đáng kể cho mục đích này.
    """
    import librosa

    T = len(wav) // cfg.hop_length
    f0, voiced, _ = librosa.pyin(wav, fmin=cfg.f0_min, fmax=cfg.f0_max, sr=cfg.sample_rate,
                                 frame_length=4 * cfg.hop_length * 2, hop_length=cfg.hop_length)
    f0 = np.nan_to_num(f0[:T], nan=0.0).astype(np.float32)
    voiced = voiced[:T].astype(bool) & (f0 > 0)
    if len(f0) < T:  # phòng hờ: đệm cho đủ T khung
        f0 = np.pad(f0, (0, T - len(f0)))
        voiced = np.pad(voiced, (0, T - len(voiced)))
    return f0, voiced


def interpolate_f0(f0: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    """Nội suy tuyến tính F0 qua các khung vô thanh (biên: lấy giá trị hữu thanh gần nhất)."""
    if not voiced.any():
        return np.zeros_like(f0)
    idx = np.arange(len(f0))
    return np.interp(idx, idx[voiced], f0[voiced]).astype(np.float32)


# ---------------------------------------------------------------------------
# Căn chỉnh: TextGrid (MFA) -> trường độ theo token
# ---------------------------------------------------------------------------
@dataclass
class Interval:
    start: float
    end: float
    label: str


SILENCE_LABELS = frozenset({"", "sil", "sp", "spn", "<eps>"})


def read_textgrid(path_or_text: str, tier: str = "phones") -> list[Interval]:
    """Đọc một tầng (tier) từ TextGrid dạng "long" (định dạng mặc định MFA xuất ra)."""
    text = path_or_text
    if "\n" not in path_or_text:
        with open(path_or_text, encoding="utf-8") as f:
            text = f.read()
    for block in re.split(r"\n\s*item \[\d+\]:", text)[1:]:
        name = re.search(r'name = "(.*)"', block)
        if name and name.group(1) == tier:
            pat = r'xmin = ([\d.eE+-]+)\s*\n\s*xmax = ([\d.eE+-]+)\s*\n\s*text = "(.*)"'
            return [Interval(float(a), float(b), lab.strip())
                    for a, b, lab in re.findall(pat, block)]
    raise ValueError(f"không tìm thấy tầng {tier!r} trong TextGrid")


def durations_from_intervals(tokens: list[str], intervals: list[Interval], cfg: AudioConfig,
                             phone_set: set[str]) -> tuple[list[int], int, int]:
    """Gán số khung cho từng token của G2P dựa trên các đoạn âm vị của MFA.

    * token âm vị nhận đúng đoạn MFA tương ứng (nhãn phải khớp, nếu không báo lỗi);
    * token thanh điệu nhận 0 khung (thanh là đặc tính của cả vần, không phải một đoạn);
    * ranh giới từ / dấu câu nhận đoạn lặng (sil/sp) nằm đúng vị trí đó, nếu có;
    * đoạn lặng ở chỗ không có token tương ứng được cộng vào token âm vị liền trước;
    * lặng đầu và cuối câu bị cắt bỏ: trả về (durations, start_sample, end_sample).
    """
    speech = [i for i, iv in enumerate(intervals) if iv.label not in SILENCE_LABELS]
    if not speech:
        raise ValueError("TextGrid không có đoạn âm vị nào")
    first, last = speech[0], speech[-1]
    ivs = intervals[first:last + 1]
    t0 = ivs[0].start
    frame = lambda t: int(round((t - t0) * cfg.sample_rate / cfg.hop_length))  # noqa: E731
    lengths = [frame(iv.end) - frame(iv.start) for iv in ivs]  # làm tròn biên -> tổng chính xác

    durations = [0] * len(tokens)
    k, last_phone = 0, None
    for j, tok in enumerate(tokens):
        if tok in phone_set:
            while k < len(ivs) and ivs[k].label in SILENCE_LABELS:  # lặng "lạc chỗ"
                if last_phone is None:
                    raise ValueError("đoạn lặng trước âm vị đầu tiên")
                durations[last_phone] += lengths[k]
                k += 1
            if k >= len(ivs) or ivs[k].label != tok:
                got = ivs[k].label if k < len(ivs) else "<hết>"
                raise ValueError(f"lệch nhãn ở token {j}: G2P={tok!r}, MFA={got!r}")
            durations[j] = lengths[k]
            last_phone, k = j, k + 1
        elif k < len(ivs) and ivs[k].label in SILENCE_LABELS and last_phone is not None:
            durations[j] = lengths[k]  # ranh giới/dấu câu ứng với một đoạn lặng
            k += 1
    if k != len(ivs):
        raise ValueError("còn đoạn MFA chưa được gán cho token nào")
    start = int(round(t0 * cfg.sample_rate))
    end = start + sum(durations) * cfg.hop_length
    return durations, start, end


def fix_total(durations: np.ndarray, n_frames: int) -> np.ndarray:
    """Sửa lệch ±vài khung do làm tròn để tổng trường độ đúng bằng số khung mel."""
    d = durations.astype(np.int64).copy()
    diff = n_frames - int(d.sum())
    i = len(d) - 1
    while diff != 0 and i >= 0:
        if d[i] > 0 or diff > 0:
            step = diff if diff > 0 else max(diff, -int(d[i]))
            d[i] += step
            diff -= step
        i -= 1
    if diff != 0:
        raise ValueError("không thể sửa tổng trường độ")
    return d
