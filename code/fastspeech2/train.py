"""Huấn luyện FastSpeech 2.

Ví dụ::

    # chạy thử vài bước trên dữ liệu giả (CPU, vài giây)
    python -m fastspeech2.train --synthetic --steps 3 --out runs/smoke

    # huấn luyện thật trên đặc trưng đã tiền xử lý
    python -m fastspeech2.train --data feats --out runs/fs2_vi [--config cfg.json] [--aligner]
"""
from __future__ import annotations

import argparse
import itertools
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import TTSDataset, collate, random_batch
from .loss import FastSpeech2Loss
from .model import FastSpeech2

MODEL_KEYS = ("tokens", "token_lens", "mels", "mel_lens", "durations", "pitch", "energy", "speakers")


def noam_lambda(warmup: int):
    """Hệ số tốc độ học kiểu Noam (đỉnh = 1 tại bước warmup, sau đó giảm theo 1/sqrt(bước))."""
    return lambda step: min((step + 1) / warmup, (warmup / (step + 1)) ** 0.5)


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def save_checkpoint(path: str, model, optimizer, scheduler, step: int, cfg: Config) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "step": step, "config": cfg.to_json()}, path)


def load_model(path: str, device: str = "cpu") -> tuple[FastSpeech2, Config]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    import json

    cfg = Config.from_dict(json.loads(ckpt["config"]))
    model = FastSpeech2(cfg.model).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval(), cfg


@torch.no_grad()
def validate(model, criterion, loader, device) -> dict:
    model.eval()
    sums, n = {}, 0
    for batch in loader:
        batch = to_device(batch, device)
        out = model(**{k: batch[k] for k in MODEL_KEYS if k in batch})
        _, logs = criterion(out, batch["mels"], batch["token_lens"], batch["mel_lens"])
        for k, v in logs.items():
            sums[k] = sums.get(k, 0.0) + v
        n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in sums.items()}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Huấn luyện FastSpeech 2")
    ap.add_argument("--data", help="thư mục đặc trưng (đầu ra của preprocess.py)")
    ap.add_argument("--out", required=True, help="thư mục lưu checkpoint")
    ap.add_argument("--config", help="tệp JSON cấu hình (mặc định: Config())")
    ap.add_argument("--aligner", action="store_true", help="học căn chỉnh cùng mô hình (không cần MFA)")
    ap.add_argument("--synthetic", action="store_true", help="dùng batch giả ngẫu nhiên để chạy thử")
    ap.add_argument("--steps", type=int, help="ghi đè train.max_steps")
    ap.add_argument("--resume", help="checkpoint để huấn luyện tiếp")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config) if args.config else Config()
    from vn_frontend import SYMBOLS  # bảng ký hiệu của Dự án 1

    cfg.model.n_symbols = len(SYMBOLS)
    cfg.model.use_aligner = cfg.model.use_aligner or args.aligner
    if args.steps:
        cfg.train.max_steps = args.steps
    if args.synthetic:  # mô hình nhỏ cho chạy thử
        cfg.model.d_model, cfg.model.ffn_dim, cfg.model.vp_channels = 32, 64, 32
        cfg.model.encoder_layers = cfg.model.decoder_layers = 1
        cfg.model.postnet_channels, cfg.train.warmup_steps = 32, 10
    elif args.data:
        import json

        with open(os.path.join(args.data, "stats.json"), encoding="utf-8") as f:
            stats = json.load(f)
        cfg.model.pitch_range = (stats["pitch_min"], stats["pitch_max"])
        cfg.model.energy_range = (stats["energy_min"], stats["energy_max"])
    else:
        ap.error("cần --data hoặc --synthetic")

    tc = cfg.train
    torch.manual_seed(tc.seed), random.seed(tc.seed), np.random.seed(tc.seed)
    device = torch.device(args.device)
    model = FastSpeech2(cfg.model).to(device)
    criterion = FastSpeech2Loss(tc.binarization_start_step, tc.binarization_warmup_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.lr, betas=tc.betas, eps=tc.eps,
                                  weight_decay=tc.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, noam_lambda(tc.warmup_steps))
    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        step = ckpt["step"]

    if args.synthetic:
        batches = itertools.repeat(random_batch(cfg.model.n_symbols, cfg.model.n_mels, batch_size=2))
        val_loader = None
    else:
        use_dur = not cfg.model.use_aligner
        train_set = TTSDataset(args.data, "train", use_durations=use_dur)
        loader = DataLoader(train_set, tc.batch_size, shuffle=True, collate_fn=collate,
                            num_workers=2, drop_last=True)
        batches = (b for _ in itertools.count() for b in loader)
        val_loader = DataLoader(TTSDataset(args.data, "val", use_durations=use_dur), tc.batch_size,
                                collate_fn=collate)

    print(f"Tham số: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} triệu | thiết bị: {device}")
    model.train()
    t0 = time.time()
    for batch in batches:
        if step >= tc.max_steps:
            break
        batch = to_device(batch, device)
        if cfg.model.use_aligner:
            batch.pop("durations", None)  # trường độ lấy từ bộ căn chỉnh
        out = model(**{k: batch[k] for k in MODEL_KEYS if k in batch})
        loss, logs = criterion(out, batch["mels"], batch["token_lens"], batch["mel_lens"], step)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1
        if step % tc.log_every == 0 or step == 1 or args.synthetic:
            msg = " ".join(f"{k}={v:.3f}" for k, v in logs.items())
            print(f"bước {step} | {msg} | grad={float(grad_norm):.2f} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e} | {time.time() - t0:.1f}s")
        if val_loader is not None and step % tc.val_every == 0:
            print("  kiểm định:", {k: round(v, 3) for k, v in validate(model, criterion, val_loader, device).items()})
        if step % tc.save_every == 0 or step == tc.max_steps:
            save_checkpoint(os.path.join(args.out, f"ckpt_{step:07d}.pt"), model, optimizer, scheduler, step, cfg)


if __name__ == "__main__":
    main()
