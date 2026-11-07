"""Suy luận FastSpeech 2 (+ HiFi-GAN) và xuất mel GTA để fine-tune vocoder.

Ví dụ::

    python -m fastspeech2.synthesize --checkpoint runs/fs2_vi/ckpt_0200000.pt \\
        --vocoder hifigan/g_02500000 --vocoder-config hifigan/config.json \\
        --text "Xin chào, đây là giọng nói tổng hợp." --out out.wav

    # mel "ground-truth aligned" (teacher forcing) cho từng câu trong tập huấn luyện
    python -m fastspeech2.synthesize --checkpoint ... --gta --data feats --gta-out ft_dataset
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import TTSDataset, collate
from .hifigan import Generator, load_generator
from .train import MODEL_KEYS, load_model, to_device


@torch.no_grad()
def synthesize(text: str, model, vocoder: Generator | None = None, dialect: str = "north",
               **controls) -> tuple[np.ndarray, np.ndarray | None]:
    """Văn bản -> (mel (F, T), sóng âm (L,) hoặc None). Văn bản dài: tách câu trước."""
    from vn_frontend import split_sentences, normalize_text, text_to_sequence

    device = next(model.parameters()).device
    mels, wavs = [], []
    for sent in split_sentences(normalize_text(text)):
        ids = torch.tensor(text_to_sequence(sent, dialect), device=device)
        mel, _ = model.infer(ids, **controls)  # (1, T, F)
        mels.append(mel[0].T.cpu())  # (F, T)
        if vocoder is not None:
            wavs.append(vocoder(mel.transpose(1, 2).to(next(vocoder.parameters()).device))[0, 0].cpu())
            wavs.append(torch.zeros(int(0.15 * 22050)))  # nghỉ ngắn giữa các câu
    mel_all = torch.cat(mels, dim=1).numpy()
    return mel_all, (torch.cat(wavs).numpy() if wavs else None)


@torch.no_grad()
def export_gta(model, data_dir: str, out_dir: str, split: str = "train", device: str = "cpu") -> int:
    """Lưu mel dự đoán khi ép (teacher-force) trường độ/cao độ/năng lượng thật.

    Mỗi tệp ``<id>.npy`` có hình dạng (F, T), cùng số khung với mel thật, nên vẫn
    khớp mẫu-với-khung với tệp wav gốc — đúng định dạng thư mục fine-tune của HiFi-GAN.
    """
    os.makedirs(out_dir, exist_ok=True)
    use_dur = not model.cfg.use_aligner
    loader = DataLoader(TTSDataset(data_dir, split, use_durations=use_dur), 8, collate_fn=collate)
    n = 0
    for batch in loader:
        batch = to_device(batch, device)
        out = model(**{k: batch[k] for k in MODEL_KEYS if k in batch})
        for i, uid in enumerate(batch["ids"]):
            t = int(batch["mel_lens"][i])
            np.save(os.path.join(out_dir, f"{uid}.npy"), out["mel_postnet"][i, :t].T.cpu().numpy())
            n += 1
    return n


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Suy luận FastSpeech 2")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--text")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--vocoder", help="checkpoint bộ sinh HiFi-GAN (định dạng repo gốc)")
    ap.add_argument("--vocoder-config", help="config_v1.json đi kèm checkpoint vocoder")
    ap.add_argument("--dialect", default="north")
    ap.add_argument("--duration-scale", type=float, default=1.0, help=">1: nói chậm hơn")
    ap.add_argument("--pitch-shift", type=float, default=0.0, help="dịch log-F0 (đơn vị độ lệch chuẩn)")
    ap.add_argument("--pitch-scale", type=float, default=1.0, help="giãn/nén biên độ cao độ")
    ap.add_argument("--energy-scale", type=float, default=1.0)
    ap.add_argument("--gta", action="store_true", help="xuất mel GTA thay vì tổng hợp văn bản")
    ap.add_argument("--data", help="thư mục đặc trưng (cho --gta)")
    ap.add_argument("--gta-out", default="ft_dataset")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    model, cfg = load_model(args.checkpoint, args.device)
    if args.gta:
        for split in ("train", "val"):
            print(split, export_gta(model, args.data, args.gta_out, split, args.device), "tệp")
        return
    vocoder = None
    if args.vocoder:
        conf = json.load(open(args.vocoder_config, encoding="utf-8")) if args.vocoder_config else None
        vocoder = load_generator(args.vocoder, conf, cfg.audio.n_mels).to(args.device)
    mel, wav = synthesize(args.text, model, vocoder, args.dialect, duration_scale=args.duration_scale,
                          pitch_shift=args.pitch_shift, pitch_scale=args.pitch_scale,
                          energy_scale=args.energy_scale)
    if wav is None:
        np.save(os.path.splitext(args.out)[0] + ".npy", mel)
        print("Không có vocoder: đã lưu mel", mel.shape)
    else:
        import soundfile as sf

        sf.write(args.out, wav, cfg.audio.sample_rate)
        print(f"Đã ghi {args.out} ({len(wav) / cfg.audio.sample_rate:.2f} s)")


if __name__ == "__main__":
    main()
