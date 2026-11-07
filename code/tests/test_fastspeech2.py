"""Kiểm thử khói (smoke test) cho gói fastspeech2 (Dự án 2), chạy trên CPU với tensor nhỏ.

Chạy: ``cd code && OMP_NUM_THREADS=2 python -m pytest tests/test_fastspeech2.py``.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fastspeech2 import (  # noqa: E402
    FastSpeech2, FastSpeech2Loss, Generator, ModelConfig, average_by_duration, beta_binomial_prior,
    forward_sum_loss, length_regulate, maximum_path, merge_weight_norm,
)
from fastspeech2.config import AudioConfig  # noqa: E402
from fastspeech2.data import random_batch  # noqa: E402
from fastspeech2.features import (  # noqa: E402
    durations_from_intervals, energy_from_wav, extract_f0, fix_total, interpolate_f0, mel_spectrogram,
    read_textgrid,
)

torch.set_num_threads(2)
TINY = dict(n_symbols=60, n_mels=16, d_model=32, n_heads=2, encoder_layers=1, decoder_layers=1,
            ffn_dim=64, ffn_kernel=3, vp_channels=32, n_bins=16, postnet_channels=16, postnet_layers=3)
MODEL_KEYS = ("tokens", "token_lens", "mels", "mel_lens", "durations", "pitch", "energy")


def tiny_model(**overrides) -> FastSpeech2:
    torch.manual_seed(0)
    return FastSpeech2(ModelConfig(**{**TINY, **overrides}))


def test_length_regulate_matches_repeat_interleave():
    x = torch.randn(2, 4, 3)  # (B, N, C)
    d = torch.tensor([[2, 0, 3, 1], [1, 1, 1, 0]])
    out, lens = length_regulate(x, d)
    assert out.shape == (2, 6, 3) and lens.tolist() == [6, 3]
    for b in range(2):
        ref = torch.repeat_interleave(x[b], d[b], dim=0)
        assert torch.equal(out[b, : lens[b]], ref)
        assert torch.all(out[b, lens[b]:] == 0)  # vùng đệm bằng 0
    out2, _ = length_regulate(x, d, max_len=9)
    assert out2.shape == (2, 9, 3)


def test_average_by_duration():
    v = torch.tensor([[1.0, 3.0, 5.0, 7.0, 0.0]])
    d = torch.tensor([[2, 0, 2, 1]])
    assert torch.allclose(average_by_duration(v, d), torch.tensor([[2.0, 0.0, 6.0, 0.0]]))


@pytest.mark.parametrize("level", ["phoneme", "frame"])
def test_forward_loss_backward_step(level):
    model = tiny_model(pitch_level=level, energy_level=level)
    batch = random_batch(60, 16, batch_size=3, seed=1)
    out = model(**{k: batch[k] for k in MODEL_KEYS})
    B, T, F = batch["mels"].shape
    assert out["mel"].shape == (B, T, F) and out["mel_postnet"].shape == (B, T, F)
    assert torch.equal(out["mel_lens"], batch["mel_lens"])
    n_var = batch["tokens"].size(1) if level == "phoneme" else T
    assert out["pitch_pred"].shape == (B, n_var) and out["pitch_target"].shape == (B, n_var)
    loss, logs = FastSpeech2Loss()(out, batch["mels"], batch["token_lens"], batch["mel_lens"])
    assert torch.isfinite(loss) and set(logs) >= {"mel", "duration", "pitch", "energy", "total"}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = copy.deepcopy(model.mel_linear.weight.detach())
    opt.zero_grad()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    opt.step()
    assert not torch.equal(before, model.mel_linear.weight)


def test_inference_and_duration_control():
    model = tiny_model().eval()
    proj = model.variance_adaptor.duration_predictor.proj
    with torch.no_grad():  # ép bộ dự đoán ra log(d+1) = log 4  =>  d = 3 khung mỗi token
        proj.weight.zero_()
        proj.bias.fill_(math.log(4.0))
    tokens = torch.randint(1, 60, (7,))
    mel, lens = model.infer(tokens)
    assert mel.shape == (1, 21, 16) and lens.item() == 21
    mel2, lens2 = model.infer(tokens, duration_scale=2.0, pitch_shift=0.5, energy_scale=1.2)
    assert lens2.item() == 42 and torch.isfinite(mel2).all()


def brute_force_best_path(log_p: np.ndarray) -> float:
    """Liệt kê mọi căn chỉnh đơn điệu (mỗi token >= 1 khung) để kiểm tra MAS."""
    T, N = log_p.shape
    best = -np.inf
    for cuts in itertools.combinations(range(1, T), N - 1):
        bounds = (0, *cuts, T)
        score = sum(log_p[bounds[n]:bounds[n + 1], n].sum() for n in range(N))
        best = max(best, score)
    return best


def test_maximum_path_is_optimal_and_monotonic():
    rng = np.random.default_rng(0)
    for T, N in [(5, 3), (7, 4), (6, 1), (4, 4)]:
        log_p = np.log(rng.dirichlet(np.ones(N), size=T))
        path = maximum_path(log_p)
        assert np.all(path.sum(1) == 1)  # mỗi khung đúng một token
        assert np.all(path.sum(0) >= 1)  # mỗi token ít nhất một khung
        idx = path.argmax(1)
        assert idx[0] == 0 and idx[-1] == N - 1 and np.all(np.diff(idx) >= 0) and np.all(np.diff(idx) <= 1)
        assert np.isclose((path * log_p).sum(), brute_force_best_path(log_p))


def test_beta_binomial_prior_and_forward_sum():
    prior = beta_binomial_prior(5, 20)  # (T, N)
    assert prior.shape == (20, 5)
    assert torch.allclose(prior.sum(1), torch.ones(20), atol=1e-5)
    assert torch.all(torch.diff(prior.argmax(1)) >= 0)  # đỉnh trượt dần theo đường chéo
    logp = torch.log_softmax(torch.randn(2, 12, 5, requires_grad=True), dim=-1)
    loss = forward_sum_loss(logp, torch.tensor([5, 3]), torch.tensor([12, 8]))
    assert torch.isfinite(loss)
    loss.backward()


def test_aligner_mode_training_step():
    model = tiny_model(use_aligner=True)
    batch = random_batch(60, 16, batch_size=2, seed=2)
    keys = [k for k in MODEL_KEYS if k != "durations"]
    out = model(**{k: batch[k] for k in keys})
    assert torch.equal(out["duration_target"].sum(1), batch["mel_lens"])  # MAS phủ đủ mọi khung
    loss, logs = FastSpeech2Loss(binarization_start_step=0)(
        out, batch["mels"], batch["token_lens"], batch["mel_lens"], step=10)
    assert "forward_sum" in logs and torch.isfinite(loss)
    loss.backward()
    assert model.aligner.key_proj[0].weight.grad is not None


@pytest.mark.filterwarnings("ignore::FutureWarning")  # weight_norm kiểu cũ đã bị đánh dấu lỗi thời
@pytest.mark.parametrize("new_api", [False, True])
def test_hifigan_weight_norm_merge(new_api):
    torch.manual_seed(0)
    cfg = dict(n_mels=16, upsample_rates=(4, 2), upsample_kernel_sizes=(8, 4), upsample_initial_channel=16,
               resblock_kernel_sizes=(3,), resblock_dilation_sizes=((1, 3, 5),))
    wn_model = Generator(**cfg)
    wn = torch.nn.utils.parametrizations.weight_norm if new_api else torch.nn.utils.weight_norm
    for m in wn_model.modules():
        if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            wn(m)
    with torch.no_grad():  # đặt g khác chuẩn của v để phép gộp thực sự được kiểm tra
        for name, p in wn_model.named_parameters():
            if name.endswith("weight_g") or name.endswith("original0"):
                p.mul_(1.5)
    plain = Generator(**cfg)
    plain.load_state_dict(merge_weight_norm(wn_model.state_dict()))
    mel = torch.randn(1, 16, 10)
    with torch.no_grad():
        y_ref, y = wn_model(mel), plain(mel)
    assert y.shape == (1, 1, 80) and torch.allclose(y, y_ref, atol=1e-5)


def test_mel_energy_f0_features():
    cfg = AudioConfig()
    sr = cfg.sample_rate
    t = np.arange(int(0.5 * sr)) / sr
    wav = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    mel = mel_spectrogram(torch.from_numpy(wav), cfg)
    assert mel.shape == (1, cfg.n_mels, len(wav) // cfg.hop_length)
    assert energy_from_wav(torch.from_numpy(wav), cfg).shape == (1, len(wav) // cfg.hop_length)
    f0, voiced = extract_f0(wav, cfg)
    assert len(f0) == len(wav) // cfg.hop_length and voiced.mean() > 0.8
    assert abs(np.median(f0[voiced]) - 200) < 10
    f = np.array([0, 100, 0, 0, 130, 0], dtype=np.float32)
    assert np.allclose(interpolate_f0(f, f > 0), [100, 100, 110, 120, 130, 130])


TEXTGRID = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.0
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1.0
        intervals: size = {n}
{body}'''


def make_textgrid(intervals: list[tuple[float, float, str]]) -> str:
    body = "".join(
        f"        intervals [{i + 1}]:\n            xmin = {a}\n            xmax = {b}\n            text = \"{lab}\"\n"
        for i, (a, b, lab) in enumerate(intervals))
    return TEXTGRID.format(n=len(intervals), body=body)


def test_textgrid_to_durations():
    from vn_frontend import PHONES, g2p

    cfg = AudioConfig(sample_rate=1000, hop_length=10)  # 1 khung = 10 ms cho dễ tính
    tokens = g2p("xin chào, ba")  # s i n T1 | c a w T2 , ɓ a T1
    phones = [t for t in tokens if t in PHONES]
    ivs, t = [(0.0, 0.2, "")], 0.2  # lặng đầu 200 ms
    for k, p in enumerate(phones):
        ivs.append((round(t, 3), round(t + 0.05, 3), p))
        t += 0.05
        if k == 5:  # sau "chào": khoảng lặng ứng với dấu phẩy
            ivs.append((round(t, 3), round(t + 0.1, 3), "sp"))
            t += 0.1
    ivs.append((round(t, 3), round(t + 0.3, 3), "sil"))
    parsed = read_textgrid(make_textgrid(ivs))
    assert [iv.label for iv in parsed] == [lab for _, _, lab in ivs]
    dur, start, end = durations_from_intervals(tokens, parsed, cfg, set(PHONES))
    assert start == 200 and (end - start) == sum(dur) * cfg.hop_length
    for tok, d in zip(tokens, dur):
        assert d == (5 if tok in PHONES else 10 if tok == "," else 0)
    assert fix_total(np.array([3, 0, 2]), 7).tolist() == [3, 0, 4]
    assert fix_total(np.array([3, 0, 2]), 2).tolist() == [2, 0, 0]


def test_cli_train_then_synthesize(tmp_path, capsys):
    from fastspeech2.synthesize import synthesize
    from fastspeech2.train import load_model, main

    main(["--synthetic", "--steps", "2", "--out", str(tmp_path)])
    ckpt = tmp_path / "ckpt_0000002.pt"
    assert ckpt.exists()
    model, cfg = load_model(str(ckpt))
    mel, wav = synthesize("Xin chào. Hôm nay là 2/9.", model)
    assert mel.shape[0] == cfg.model.n_mels and wav is None
    tiny_voc = Generator(n_mels=cfg.model.n_mels, upsample_initial_channel=16)
    _, wav = synthesize("Xin chào.", model, tiny_voc)
    assert wav is not None and np.isfinite(wav).all()


def test_preprocess_pipeline(tmp_path):
    import soundfile as sf

    from fastspeech2.data import TTSDataset, collate
    from fastspeech2.preprocess import main as prep

    data = tmp_path / "data"
    (data / "wavs").mkdir(parents=True)
    sr = 22050
    texts = ["Xin chào.", "Hôm nay trời đẹp.", "Giá 50.000đ."]
    with open(data / "metadata.csv", "w", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            t = np.arange(int(0.6 * sr)) / sr
            f0 = 150 + 60 * t  # cao độ tăng dần
            sf.write(data / "wavs" / f"u{i}.wav", 0.3 * np.sin(2 * np.pi * np.cumsum(f0) / sr), sr)
            f.write(f"u{i}|{text}\n")
    prep(["mfa", "--data", str(data), "--out", str(tmp_path / "mfa"),
          "--dictionary", str(tmp_path / "dict.txt")])
    lines = (tmp_path / "dict.txt").read_text(encoding="utf-8").splitlines()
    assert "chào\tc a w" in lines and (tmp_path / "mfa" / "u0.lab").read_text(encoding="utf-8").strip() == "xin chào"

    feats = tmp_path / "feats"
    prep(["features", "--data", str(data), "--out", str(feats), "--val-size", "1"])
    stats = json.loads((feats / "stats.json").read_text())
    assert stats["pitch_std"] > 0 and stats["pitch_min"] < 0 < stats["pitch_max"]
    ds = TTSDataset(str(feats), "train", use_durations=False)
    assert len(ds) == 2
    batch = collate([ds[0], ds[1]])
    model = tiny_model(n_mels=80, use_aligner=True)
    out = model(**{k: batch[k] for k in MODEL_KEYS if k in batch})
    assert out["mel"].shape == batch["mels"].shape
    assert os.path.exists(feats / "mel" / "u0.npy")
