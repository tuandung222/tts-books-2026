"""Tiền xử lý dữ liệu cho FastSpeech 2.

Đầu vào: thư mục kiểu LJSpeech — ``metadata.csv`` (mỗi dòng ``id|văn bản`` hoặc
``id|văn bản gốc|văn bản chuẩn hóa``) và ``wavs/<id>.wav``.

Hai lệnh con::

    # 1) (tùy chọn) tạo corpus + từ điển phát âm cho Montreal Forced Aligner
    python -m fastspeech2.preprocess mfa --data DATA --out mfa_corpus
    #    ... chạy MFA (xem trang Dự án 2), được thư mục TextGrid ...

    # 2) trích đặc trưng: mel, log-F0, năng lượng (+ trường độ nếu có TextGrid)
    python -m fastspeech2.preprocess features --data DATA --out feats [--textgrids TG_DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from .config import AudioConfig
from .features import (
    durations_from_intervals, energy_from_wav, extract_f0, interpolate_f0, load_wav,
    mel_spectrogram, read_textgrid,
)


def read_metadata(data_dir: str) -> list[tuple[str, str]]:
    """Đọc metadata.csv -> danh sách (id, văn bản). Ưu tiên cột văn bản đã chuẩn hóa nếu có."""
    items = []
    with open(os.path.join(data_dir, "metadata.csv"), encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE):
            if row:
                items.append((row[0], row[-1]))
    return items


def toneless_phones(word: str, dialect: str) -> list[str]:
    """Âm vị của một từ, bỏ token thanh điệu: MFA chỉ cần ranh giới đoạn âm."""
    from vn_frontend import TONE_TOKENS, g2p

    return [p for p in g2p(word, dialect, normalize=False) if p not in TONE_TOKENS and p != " "]


def mfa_words(text: str) -> list[str]:
    """Các "từ" (âm tiết) cho tệp .lab của MFA, lấy từ văn bản đã chuẩn hóa."""
    import re

    return re.findall(r"[^\W\d_]+", text.lower())


def cmd_mfa(args) -> None:
    from vn_frontend import normalize_text

    os.makedirs(args.out, exist_ok=True)
    lexicon: dict[str, list[str]] = {}
    for uid, text in read_metadata(args.data):
        words = mfa_words(normalize_text(text))
        shutil.copy(os.path.join(args.data, "wavs", f"{uid}.wav"), os.path.join(args.out, f"{uid}.wav"))
        with open(os.path.join(args.out, f"{uid}.lab"), "w", encoding="utf-8") as f:
            f.write(" ".join(words) + "\n")
        for w in words:
            if w not in lexicon:
                lexicon[w] = toneless_phones(w, args.dialect)
    with open(args.dictionary, "w", encoding="utf-8") as f:
        for w in sorted(lexicon):
            f.write(f"{w}\t{' '.join(lexicon[w])}\n")
    print(f"Đã ghi {len(lexicon)} mục từ vào {args.dictionary} và corpus vào {args.out}")


def process_one(job: tuple) -> dict | None:
    """Trích đặc trưng cho một câu; trả về dòng manifest (hoặc None nếu bỏ qua)."""
    uid, text, args_d = job
    from vn_frontend import PHONES, g2p, normalize_text, text_to_sequence

    cfg = AudioConfig(**args_d["audio"])
    norm = normalize_text(text)
    tokens = g2p(norm, args_d["dialect"], normalize=False)
    ids = text_to_sequence(norm, args_d["dialect"])
    wav = load_wav(os.path.join(args_d["data"], "wavs", f"{uid}.wav"), cfg.sample_rate)

    durations = None
    if args_d["textgrids"]:
        tg = os.path.join(args_d["textgrids"], f"{uid}.TextGrid")
        if not os.path.exists(tg):
            return None
        try:
            durations, start, end = durations_from_intervals(tokens, read_textgrid(tg), cfg, set(PHONES))
        except ValueError as e:
            print(f"[bỏ qua] {uid}: {e}")
            return None
        wav = np.pad(wav, (0, max(0, end - len(wav))))[start:end]
    else:  # dùng bộ căn chỉnh học cùng mô hình: chỉ cắt lặng hai đầu
        import librosa

        wav, _ = librosa.effects.trim(wav, top_db=args_d["trim_db"])

    wav_t = torch.from_numpy(wav)
    mel = mel_spectrogram(wav_t, cfg)[0].numpy()  # (F, T)
    energy = energy_from_wav(wav_t, cfg)[0].numpy()  # (T,)
    f0, voiced = extract_f0(wav, cfg)
    log_f0 = np.log(np.maximum(interpolate_f0(f0, voiced), 1.0))  # log-F0, nội suy khung vô thanh
    T = mel.shape[1]
    if durations is not None and sum(durations) != T:
        print(f"[bỏ qua] {uid}: tổng trường độ {sum(durations)} != {T} khung")
        return None
    if not voiced.any():
        return None
    out = args_d["out"]
    np.save(os.path.join(out, "mel", f"{uid}.npy"), mel.astype(np.float32))
    np.save(os.path.join(out, "pitch", f"{uid}.npy"), log_f0.astype(np.float32))
    np.save(os.path.join(out, "energy", f"{uid}.npy"), energy.astype(np.float32))
    np.save(os.path.join(out, "voiced", f"{uid}.npy"), voiced)
    if durations is not None:
        np.save(os.path.join(out, "duration", f"{uid}.npy"), np.asarray(durations, dtype=np.int64))
    return {"id": uid, "text": norm, "tokens": ids, "n_frames": int(T), "speaker": 0}


def compute_stats(out: str, items: list[dict]) -> dict:
    """Trung bình/độ lệch chuẩn (log-F0 chỉ trên khung hữu thanh) và min/max sau chuẩn hóa."""
    pitch_vals, energy_vals = [], []
    for it in items:
        p = np.load(os.path.join(out, "pitch", f"{it['id']}.npy"))
        v = np.load(os.path.join(out, "voiced", f"{it['id']}.npy"))
        pitch_vals.append(p[v])
        energy_vals.append(np.load(os.path.join(out, "energy", f"{it['id']}.npy")))
    p_all, e_all = np.concatenate(pitch_vals), np.concatenate(energy_vals)
    st = {"pitch_mean": float(p_all.mean()), "pitch_std": float(p_all.std() + 1e-8),
          "energy_mean": float(e_all.mean()), "energy_std": float(e_all.std() + 1e-8)}
    p_norm = [(np.load(os.path.join(out, "pitch", f"{it['id']}.npy")) - st["pitch_mean"]) / st["pitch_std"]
              for it in items]
    e_norm = (e_all - st["energy_mean"]) / st["energy_std"]
    st.update(pitch_min=float(min(x.min() for x in p_norm)), pitch_max=float(max(x.max() for x in p_norm)),
              energy_min=float(e_norm.min()), energy_max=float(e_norm.max()))
    return st


def cmd_features(args) -> None:
    for sub in ("mel", "pitch", "energy", "voiced", "duration"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    args_d = {"audio": vars(AudioConfig()), "dialect": args.dialect, "data": args.data,
              "textgrids": args.textgrids, "out": args.out, "trim_db": args.trim_db}
    jobs = [(uid, text, args_d) for uid, text in read_metadata(args.data)]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            results = list(ex.map(process_one, jobs, chunksize=8))
    else:
        results = [process_one(j) for j in jobs]
    items = [r for r in results if r is not None]
    print(f"Giữ {len(items)}/{len(jobs)} câu")
    random.Random(args.seed).shuffle(items)
    splits = {"val": items[: args.val_size], "train": items[args.val_size:]}
    for name, rows in splits.items():
        with open(os.path.join(args.out, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(compute_stats(args.out, items), f, indent=2)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Tiền xử lý dữ liệu FastSpeech 2")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mfa", help="tạo corpus .wav/.lab và từ điển phát âm cho MFA")
    m.add_argument("--data", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--dictionary", default="vi_dict.txt")
    m.add_argument("--dialect", default="north")
    f = sub.add_parser("features", help="trích mel, log-F0, năng lượng, trường độ")
    f.add_argument("--data", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--textgrids", help="thư mục TextGrid của MFA (bỏ trống nếu dùng --aligner)")
    f.add_argument("--dialect", default="north")
    f.add_argument("--val-size", type=int, default=100)
    f.add_argument("--trim-db", type=float, default=40.0)
    f.add_argument("--workers", type=int, default=1)
    f.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)
    cmd_mfa(args) if args.cmd == "mfa" else cmd_features(args)


if __name__ == "__main__":
    main()
