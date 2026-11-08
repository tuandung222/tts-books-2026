"""Dataset và collate cho dữ liệu đã tiền xử lý bởi ``preprocess.py``.

Cấu trúc thư mục::

    feats/
      train.jsonl, val.jsonl   # mỗi dòng: {"id", "text", "tokens", "n_frames", "speaker"}
      stats.json               # trung bình/độ lệch chuẩn log-F0 và năng lượng, min/max sau chuẩn hóa
      mel/<id>.npy             # (F, T) log-mel
      pitch/<id>.npy           # (T,)  log-F0 đã nội suy qua khung vô thanh
      energy/<id>.npy          # (T,)
      duration/<id>.npy        # (N,)  chỉ có khi dùng MFA
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

MEL_PAD_VALUE = float(np.log(1e-5))  # log-mel của "im lặng" theo quy ước kẹp 1e-5


class TTSDataset(Dataset):
    def __init__(self, feat_dir: str, split: str = "train", use_durations: bool = True):
        self.dir = feat_dir
        with open(os.path.join(feat_dir, f"{split}.jsonl"), encoding="utf-8") as f:
            self.items = [json.loads(line) for line in f if line.strip()]
        with open(os.path.join(feat_dir, "stats.json"), encoding="utf-8") as f:
            self.stats = json.load(f)
        self.use_durations = use_durations

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, kind: str, uid: str) -> np.ndarray:
        return np.load(os.path.join(self.dir, kind, f"{uid}.npy"))

    def __getitem__(self, i: int) -> dict:
        it, st = self.items[i], self.stats
        uid = it["id"]
        sample = {
            "id": uid,
            "tokens": torch.tensor(it["tokens"], dtype=torch.long),  # (N,)
            "mel": torch.from_numpy(self._load("mel", uid)).T.float(),  # (T, F)
            "pitch": torch.from_numpy((self._load("pitch", uid) - st["pitch_mean"]) / st["pitch_std"]).float(),
            "energy": torch.from_numpy((self._load("energy", uid) - st["energy_mean"]) / st["energy_std"]).float(),
            "speaker": int(it.get("speaker", 0)),
        }
        if self.use_durations:
            sample["duration"] = torch.from_numpy(self._load("duration", uid)).long()  # (N,)
            assert int(sample["duration"].sum()) == sample["mel"].size(0), f"{uid}: tổng trường độ != số khung"
        return sample


def collate(batch: list[dict]) -> dict:
    """Đệm về độ dài lớn nhất trong batch; trả về dict khớp tham số của FastSpeech2.forward."""
    B = len(batch)
    N = max(b["tokens"].numel() for b in batch)
    T = max(b["mel"].size(0) for b in batch)
    F = batch[0]["mel"].size(1)
    out = {
        "ids": [b["id"] for b in batch],
        "tokens": torch.zeros(B, N, dtype=torch.long),  # 0 = <pad>
        "token_lens": torch.tensor([b["tokens"].numel() for b in batch]),
        "mels": torch.full((B, T, F), MEL_PAD_VALUE),
        "mel_lens": torch.tensor([b["mel"].size(0) for b in batch]),
        "pitch": torch.zeros(B, T),
        "energy": torch.zeros(B, T),
        "speakers": torch.tensor([b["speaker"] for b in batch]),
    }
    has_dur = "duration" in batch[0]
    if has_dur:
        out["durations"] = torch.zeros(B, N, dtype=torch.long)
    for i, b in enumerate(batch):
        n, t = b["tokens"].numel(), b["mel"].size(0)
        out["tokens"][i, :n] = b["tokens"]
        out["mels"][i, :t] = b["mel"]
        out["pitch"][i, :t] = b["pitch"]
        out["energy"][i, :t] = b["energy"]
        if has_dur:
            out["durations"][i, :n] = b["duration"]
    return out


def random_batch(n_symbols: int, n_mels: int = 80, batch_size: int = 2, max_tokens: int = 12,
                 max_dur: int = 4, seed: int = 0) -> dict:
    """Batch giả ngẫu nhiên đúng định dạng ``collate`` (cho kiểm thử khói và --synthetic)."""
    g = torch.Generator().manual_seed(seed)
    items = []
    for i in range(batch_size):
        n = int(torch.randint(max_tokens // 2, max_tokens + 1, (1,), generator=g))
        dur = torch.randint(1, max_dur + 1, (n,), generator=g)
        t = int(dur.sum())
        items.append({
            "id": f"rand{i}",
            "tokens": torch.randint(1, n_symbols, (n,), generator=g),
            "duration": dur,
            "mel": torch.randn(t, n_mels, generator=g),
            "pitch": torch.randn(t, generator=g),
            "energy": torch.randn(t, generator=g),
            "speaker": 0,
        })
    return collate(items)
