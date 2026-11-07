"""FastSpeech 2 rút gọn: encoder FFT -> variance adaptor (trường độ, cao độ, năng
lượng + length regulator) -> decoder FFT -> mel (+ PostNet).

Bố cục tensor: Transformer dùng (B, L, C); Conv1d dùng (B, C, L).
Mặt nạ đệm (padding mask) quy ước True = vị trí đệm.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aligner import AlignmentEncoder, beta_binomial_prior, binarize_alignment
from .config import ModelConfig


def make_pad_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """lengths (B,) -> mặt nạ (B, L), True tại vị trí đệm."""
    max_len = int(lengths.max()) if max_len is None else max_len
    return torch.arange(max_len, device=lengths.device)[None, :] >= lengths[:, None]


def sinusoid_table(n_pos: int, d: int) -> torch.Tensor:
    """Mã hóa vị trí hình sin (Vaswani và cộng sự, 2017): (n_pos, d)."""
    pos = torch.arange(n_pos, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    table = torch.zeros(n_pos, d)
    table[:, 0::2] = torch.sin(pos * div)
    table[:, 1::2] = torch.cos(pos * div)
    return table


class FFTBlock(nn.Module):
    """Khối Feed-Forward Transformer: self-attention + FFN bằng Conv1d, Post-LN."""

    def __init__(self, d: int, n_heads: int, ffn_dim: int, kernel: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.conv1 = nn.Conv1d(d, ffn_dim, kernel, padding=kernel // 2)
        self.conv2 = nn.Conv1d(ffn_dim, d, 1)
        self.norm2 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:  # x (B, L, C)
        a, _ = self.attn(x, x, x, key_padding_mask=pad, need_weights=False)
        x = self.norm1(x + self.dropout(a)).masked_fill(pad.unsqueeze(-1), 0.0)
        h = self.conv2(F.relu(self.conv1(x.transpose(1, 2)))).transpose(1, 2)  # (B, L, C)
        return self.norm2(x + self.dropout(h)).masked_fill(pad.unsqueeze(-1), 0.0)


class FFTStack(nn.Module):
    """Chồng các khối FFT + mã hóa vị trí; dùng cho cả encoder và decoder."""

    def __init__(self, n_layers: int, cfg: ModelConfig):
        super().__init__()
        self.register_buffer("pe", sinusoid_table(cfg.max_positions, cfg.d_model), persistent=False)
        self.layers = nn.ModuleList(
            FFTBlock(cfg.d_model, cfg.n_heads, cfg.ffn_dim, cfg.ffn_kernel, cfg.dropout)
            for _ in range(n_layers))

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        pe = self.pe[:L] if L <= self.pe.size(0) else sinusoid_table(L, x.size(2)).to(x.device)
        x = x + pe.unsqueeze(0)
        for layer in self.layers:
            x = layer(x, pad)
        return x


class VariancePredictor(nn.Module):
    """Conv1d -> ReLU -> LayerNorm -> Dropout (x2) -> Linear: dự đoán một đại lượng vô hướng."""

    def __init__(self, d_in: int, channels: int, kernel: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(d_in, channels, kernel, padding=kernel // 2),
                                    nn.Conv1d(channels, channels, kernel, padding=kernel // 2)])
        self.norms = nn.ModuleList([nn.LayerNorm(channels), nn.LayerNorm(channels)])
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:  # (B, L, C) -> (B, L)
        for conv, norm in zip(self.convs, self.norms):
            x = self.dropout(norm(F.relu(conv(x.transpose(1, 2)).transpose(1, 2))))
        return self.proj(x).squeeze(-1).masked_fill(pad, 0.0)


def length_regulate(x: torch.Tensor, durations: torch.Tensor, max_len: int | None = None):
    """Lặp mỗi véc-tơ token d_i lần. x (B, N, C), durations (B, N) số nguyên >= 0.

    Trả về (B, T, C), độ dài mel (B,). Khung t thuộc token i khi
    cum[i-1] <= t < cum[i], tìm bằng ``searchsorted`` thay vì vòng lặp Python.
    """
    B, N, C = x.shape
    mel_lens = durations.sum(1)  # (B,)
    T = int(mel_lens.max()) if max_len is None else max_len
    T = max(T, 1)
    cum = durations.cumsum(1)  # (B, N)
    frames = torch.arange(T, device=x.device).expand(B, T).contiguous()  # (B, T)
    idx = torch.searchsorted(cum, frames, right=True).clamp(max=N - 1)  # (B, T)
    out = torch.gather(x, 1, idx.unsqueeze(-1).expand(B, T, C))  # (B, T, C)
    out = out.masked_fill(make_pad_mask(mel_lens, T).unsqueeze(-1), 0.0)
    return out, mel_lens


def average_by_duration(values: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
    """Trung bình đặc trưng theo khung (B, T) trên đoạn khung của mỗi token -> (B, N)."""
    T = values.size(1)
    csum = F.pad(values.cumsum(1), (1, 0))  # (B, T+1)
    bounds = F.pad(durations.cumsum(1), (1, 0)).clamp(max=T)  # (B, N+1)
    total = csum.gather(1, bounds[:, 1:]) - csum.gather(1, bounds[:, :-1])
    return total / durations.clamp(min=1)


class VarianceAdaptor(nn.Module):
    """Dự đoán trường độ, cao độ, năng lượng; cộng embedding lượng tử hóa; length regulator."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        vp = lambda: VariancePredictor(cfg.d_model, cfg.vp_channels, cfg.vp_kernel, cfg.vp_dropout)  # noqa: E731
        self.duration_predictor, self.pitch_predictor, self.energy_predictor = vp(), vp(), vp()
        # n_bins-1 ngưỡng chia trục giá trị thành n_bins khoảng
        self.register_buffer("pitch_bins", torch.linspace(*cfg.pitch_range, cfg.n_bins - 1))
        self.register_buffer("energy_bins", torch.linspace(*cfg.energy_range, cfg.n_bins - 1))
        self.pitch_embedding = nn.Embedding(cfg.n_bins, cfg.d_model)
        self.energy_embedding = nn.Embedding(cfg.n_bins, cfg.d_model)

    def _add_pitch(self, x, pad, target, shift, scale):
        pred = self.pitch_predictor(x, pad)
        value = target if target is not None else pred * scale + shift
        return x + self.pitch_embedding(torch.bucketize(value, self.pitch_bins)), pred

    def _add_energy(self, x, pad, target, scale):
        pred = self.energy_predictor(x, pad)
        value = target if target is not None else pred * scale
        return x + self.energy_embedding(torch.bucketize(value, self.energy_bins)), pred

    def forward(self, x, src_pad, durations=None, pitch=None, energy=None, max_len=None,
                duration_scale=1.0, pitch_shift=0.0, pitch_scale=1.0, energy_scale=1.0):
        """x (B, N, C). durations (B, N); pitch/energy đúng cấp (token hoặc khung) hoặc None."""
        cfg = self.cfg
        log_d_pred = self.duration_predictor(x, src_pad)  # (B, N): dự đoán log(d + 1)
        p_pred = e_pred = None
        if cfg.pitch_level == "phoneme":
            x, p_pred = self._add_pitch(x, src_pad, pitch, pitch_shift, pitch_scale)
        if cfg.energy_level == "phoneme":
            x, e_pred = self._add_energy(x, src_pad, energy, energy_scale)

        if durations is None:  # suy luận: làm tròn trường độ dự đoán
            durations = torch.round((torch.exp(log_d_pred) - 1) * duration_scale)
            durations = durations.clamp(min=0).long().masked_fill(src_pad, 0)
            durations[durations.sum(1) == 0, 0] = 1  # tránh câu rỗng (attention toàn đệm -> NaN)
        x, mel_lens = length_regulate(x, durations, max_len)  # (B, T, C)
        mel_pad = make_pad_mask(mel_lens, x.size(1))

        if cfg.pitch_level == "frame":
            x, p_pred = self._add_pitch(x, mel_pad, pitch, pitch_shift, pitch_scale)
        if cfg.energy_level == "frame":
            x, e_pred = self._add_energy(x, mel_pad, energy, energy_scale)
        return x, dict(log_duration_pred=log_d_pred, pitch_pred=p_pred, energy_pred=e_pred,
                       durations=durations, mel_lens=mel_lens, mel_pad=mel_pad)


class PostNet(nn.Module):
    """5 lớp Conv1d (như Tacotron 2) dự đoán phần dư để làm sắc nét mel."""

    def __init__(self, n_mels: int, channels: int, n_layers: int, kernel: int = 5, dropout: float = 0.5):
        super().__init__()
        dims = [n_mels] + [channels] * (n_layers - 1) + [n_mels]
        self.convs = nn.ModuleList(nn.Sequential(
            nn.Conv1d(dims[i], dims[i + 1], kernel, padding=kernel // 2), nn.BatchNorm1d(dims[i + 1]))
            for i in range(n_layers))
        self.dropout = nn.Dropout(dropout)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:  # (B, T, F) -> (B, T, F)
        x = mel.transpose(1, 2)
        for i, conv in enumerate(self.convs):
            x = conv(x)
            x = self.dropout(torch.tanh(x) if i < len(self.convs) - 1 else x)
        return x.transpose(1, 2)


class FastSpeech2(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.n_symbols, cfg.d_model, padding_idx=0)
        self.encoder = FFTStack(cfg.encoder_layers, cfg)
        self.speaker_embedding = nn.Embedding(cfg.n_speakers, cfg.d_model) if cfg.n_speakers > 1 else None
        self.variance_adaptor = VarianceAdaptor(cfg)
        self.decoder = FFTStack(cfg.decoder_layers, cfg)
        self.mel_linear = nn.Linear(cfg.d_model, cfg.n_mels)
        self.postnet = PostNet(cfg.n_mels, cfg.postnet_channels, cfg.postnet_layers) if cfg.use_postnet else None
        self.aligner = (AlignmentEncoder(cfg.n_mels, cfg.d_model, temperature=cfg.aligner_temperature)
                        if cfg.use_aligner else None)

    def forward(self, tokens, token_lens, mels=None, mel_lens=None, durations=None, pitch=None,
                energy=None, speakers=None, duration_scale=1.0, pitch_shift=0.0, pitch_scale=1.0,
                energy_scale=1.0) -> dict:
        """tokens (B, N); mels (B, T, F); durations (B, N); pitch, energy: theo khung (B, T).

        Khi huấn luyện: truyền mels + (durations hoặc dùng aligner) + pitch + energy.
        Khi suy luận: chỉ truyền tokens, token_lens (và các hệ số điều khiển).
        """
        cfg = self.cfg
        src_pad = make_pad_mask(token_lens, tokens.size(1))  # (B, N)
        emb = self.embedding(tokens)  # (B, N, C)
        h = self.encoder(emb, src_pad)  # (B, N, C)
        if self.speaker_embedding is not None and speakers is not None:
            h = h + self.speaker_embedding(speakers).unsqueeze(1)

        out: dict = {"src_pad": src_pad}
        if self.aligner is not None and mels is not None:
            prior = torch.zeros(mels.size(0), mels.size(1), tokens.size(1), device=mels.device)
            for b in range(mels.size(0)):  # tiền nghiệm beta-binomial cho từng mẫu
                n, t = int(token_lens[b]), int(mel_lens[b])
                prior[b, :t, :n] = beta_binomial_prior(n, t).to(mels.device)
            attn_soft, attn_logprob = self.aligner(mels, emb, src_pad, prior)
            attn_hard = binarize_alignment(attn_soft, token_lens, mel_lens)  # (B, T, N)
            out.update(attn_soft=attn_soft, attn_logprob=attn_logprob, attn_hard=attn_hard)
            if durations is None:
                durations = attn_hard.sum(1).long()  # (B, N)

        # Đưa mục tiêu cao độ/năng lượng về đúng cấp (token hay khung)
        if pitch is not None and cfg.pitch_level == "phoneme" and durations is not None:
            pitch = average_by_duration(pitch, durations)
        if energy is not None and cfg.energy_level == "phoneme" and durations is not None:
            energy = average_by_duration(energy, durations)
        out.update(duration_target=durations, pitch_target=pitch, energy_target=energy)

        max_len = mels.size(1) if mels is not None else None
        x, va = self.variance_adaptor(h, src_pad, durations, pitch, energy, max_len,
                                      duration_scale, pitch_shift, pitch_scale, energy_scale)
        out.update(va)
        x = self.decoder(x, va["mel_pad"])  # (B, T, C)
        mel = self.mel_linear(x).masked_fill(va["mel_pad"].unsqueeze(-1), 0.0)  # (B, T, F)
        out["mel"] = mel
        out["mel_postnet"] = mel if self.postnet is None else (
            mel + self.postnet(mel)).masked_fill(va["mel_pad"].unsqueeze(-1), 0.0)
        return out

    @torch.no_grad()
    def infer(self, tokens: torch.Tensor, **controls) -> tuple[torch.Tensor, torch.Tensor]:
        """tokens (N,) hoặc (B, N) không đệm -> (mel (B, T, F), mel_lens (B,))."""
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        lens = torch.full((tokens.size(0),), tokens.size(1), dtype=torch.long, device=tokens.device)
        out = self.forward(tokens, lens, **controls)
        return out["mel_postnet"], out["mel_lens"]
