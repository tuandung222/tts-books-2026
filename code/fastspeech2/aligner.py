"""Bộ căn chỉnh học cùng mô hình (không cần MFA), theo hướng "One TTS Alignment To
Rule Them All" (Badlani và cộng sự, 2022) như cách FastPitch/RAD-TTS dùng:

* ma trận căn chỉnh mềm từ khoảng cách giữa embedding văn bản và khung mel;
* hàm mất mát forward-sum (giống CTC) để học căn chỉnh đơn điệu;
* tiền nghiệm beta-binomial giúp căn chỉnh "gần đường chéo" ngay từ đầu;
* nhị phân hóa bằng Monotonic Alignment Search (MAS) để lấy trường độ cứng.

Lý thuyết chi tiết: Chương 21.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentEncoder(nn.Module):
    """Tính log-xác suất căn chỉnh khung mel t với token n."""

    def __init__(self, n_mels: int, d_text: int, d_att: int = 80, temperature: float = 0.0005):
        super().__init__()
        self.temperature = temperature
        self.key_proj = nn.Sequential(  # phía văn bản
            nn.Conv1d(d_text, 2 * d_text, 3, padding=1), nn.ReLU(),
            nn.Conv1d(2 * d_text, d_att, 1))
        self.query_proj = nn.Sequential(  # phía mel
            nn.Conv1d(n_mels, 2 * n_mels, 3, padding=1), nn.ReLU(),
            nn.Conv1d(2 * n_mels, n_mels, 1), nn.ReLU(),
            nn.Conv1d(n_mels, d_att, 1))

    def forward(self, mels: torch.Tensor, text_emb: torch.Tensor, src_pad: torch.Tensor,
                prior: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """mels (B, T, F), text_emb (B, N, C), src_pad (B, N) True = đệm, prior (B, T, N).

        Trả về ``(attn_soft, attn_logprob)``, cùng hình dạng (B, T, N).
        """
        q = self.query_proj(mels.transpose(1, 2))  # (B, A, T)
        k = self.key_proj(text_emb.transpose(1, 2))  # (B, A, N)
        dist = (q.unsqueeze(-1) - k.unsqueeze(2)).pow(2).sum(1)  # (B, T, N)
        logits = (-self.temperature * dist).masked_fill(src_pad.unsqueeze(1), float("-inf"))
        logprob = F.log_softmax(logits, dim=-1)
        if prior is not None:
            logprob = logprob + torch.log(prior + 1e-8)
        attn_soft = torch.softmax(logprob, dim=-1)  # chuẩn hóa lại sau khi nhân tiền nghiệm
        return attn_soft, logprob


def forward_sum_loss(attn_logprob: torch.Tensor, token_lens: torch.Tensor,
                     mel_lens: torch.Tensor, blank_logprob: float = -1.0) -> torch.Tensor:
    """Mất mát forward-sum: -log tổng xác suất mọi căn chỉnh đơn điệu, tính bằng CTC.

    Mỗi token là một "nhãn" xuất hiện đúng một lần theo thứ tự 1..N; thêm một lớp
    blank (chỉ số 0) với log-xác suất cố định để dùng được ``F.ctc_loss``.
    """
    logp = F.pad(attn_logprob, (1, 0), value=blank_logprob)  # (B, T, N+1)
    losses = []
    for b in range(logp.size(0)):
        n, t = int(token_lens[b]), int(mel_lens[b])
        lp = F.log_softmax(logp[b, :t, : n + 1], dim=-1).unsqueeze(1)  # (T, 1, N+1)
        target = torch.arange(1, n + 1, device=logp.device).unsqueeze(0)  # (1, N)
        losses.append(F.ctc_loss(lp, target, torch.tensor([t]), torch.tensor([n]),
                                 blank=0, zero_infinity=True))
    return torch.stack(losses).mean()


def beta_binomial_prior(n_tokens: int, n_frames: int, scaling: float = 1.0) -> torch.Tensor:
    """Tiền nghiệm (n_frames, n_tokens): khung t ưu tiên token gần t·N/T.

    Hàng t là phân phối Beta-Binomial(N-1, a=s·t, b=s·(T+1-t)) trên chỉ số token.
    """
    n = n_tokens - 1
    k = torch.arange(n_tokens, dtype=torch.float64)  # (N,)
    t = torch.arange(1, n_frames + 1, dtype=torch.float64).unsqueeze(1)  # (T, 1)
    a, b = scaling * t, scaling * (n_frames + 1 - t)
    lbeta = lambda x, y: torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)  # noqa: E731
    log_comb = torch.lgamma(torch.tensor(n + 1.0)) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
    log_pmf = log_comb + lbeta(k + a, n - k + b) - lbeta(a, b)  # (T, N)
    return log_pmf.exp().float()


def maximum_path(log_p: np.ndarray) -> np.ndarray:
    """Monotonic Alignment Search: đường đơn điệu có tổng log-xác suất lớn nhất.

    log_p: (T, N) với T >= N. Mỗi token nhận ít nhất một khung, đi từ token 0 ở
    khung đầu tới token N-1 ở khung cuối. Trả về ma trận 0/1 hình dạng (T, N).
    """
    T, N = log_p.shape
    if T < N:
        raise ValueError(f"MAS cần số khung >= số token, nhận T={T}, N={N}")
    Q = np.full((T, N), -np.inf)
    Q[0, 0] = log_p[0, 0]
    for t in range(1, T):  # quy hoạch động: ở lại token cũ hoặc tiến thêm một token
        move = np.concatenate(([-np.inf], Q[t - 1, :-1]))
        Q[t] = np.maximum(Q[t - 1], move) + log_p[t]
    path = np.zeros((T, N), dtype=np.float32)
    n = N - 1
    for t in range(T - 1, -1, -1):  # truy vết ngược
        path[t, n] = 1.0
        if t > 0 and n > 0 and Q[t - 1, n - 1] >= Q[t - 1, n]:
            n -= 1
    return path


@torch.no_grad()
def binarize_alignment(attn_soft: torch.Tensor, token_lens: torch.Tensor,
                       mel_lens: torch.Tensor) -> torch.Tensor:
    """Áp MAS lên từng mẫu trong batch: (B, T, N) mềm -> (B, T, N) cứng."""
    hard = torch.zeros_like(attn_soft)
    log_attn = torch.log(attn_soft.clamp_min(1e-12)).cpu().numpy()
    for b in range(attn_soft.size(0)):
        n, t = int(token_lens[b]), int(mel_lens[b])
        hard[b, :t, :n] = torch.from_numpy(maximum_path(log_attn[b, :t, :n]))
    return hard


def bin_loss(attn_soft: torch.Tensor, attn_hard: torch.Tensor) -> torch.Tensor:
    """Kéo căn chỉnh mềm về phía căn chỉnh cứng: -mean log attn_soft tại các ô được chọn."""
    selected = torch.log(attn_soft.clamp_min(1e-12))[attn_hard == 1]
    return -selected.mean()
