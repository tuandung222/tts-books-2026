"""Hàm mất mát FastSpeech 2: L1 cho mel, MSE cho log-trường độ, cao độ, năng lượng;
cộng thêm forward-sum và bin loss khi dùng bộ căn chỉnh học cùng mô hình."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aligner import bin_loss, forward_sum_loss


class FastSpeech2Loss(nn.Module):
    def __init__(self, binarization_start_step: int = 6000, binarization_warmup_steps: int = 2000):
        super().__init__()
        self.bin_start = binarization_start_step
        self.bin_warmup = max(binarization_warmup_steps, 1)

    def bin_weight(self, step: int | None) -> float:
        """Trọng số bin loss: 0 trước bin_start, tăng tuyến tính lên 1 trong bin_warmup bước."""
        if step is None:
            return 1.0
        return min(max(step - self.bin_start, 0) / self.bin_warmup, 1.0)

    def forward(self, out: dict, mels: torch.Tensor, token_lens: torch.Tensor,
                mel_lens: torch.Tensor, step: int | None = None) -> tuple[torch.Tensor, dict]:
        mel_valid = ~out["mel_pad"]  # (B, T)
        src_valid = ~out["src_pad"]  # (B, N)
        mel_loss = F.l1_loss(out["mel"][mel_valid], mels[mel_valid])
        post_loss = F.l1_loss(out["mel_postnet"][mel_valid], mels[mel_valid])

        log_d_target = torch.log(out["duration_target"].float() + 1.0)
        dur_loss = F.mse_loss(out["log_duration_pred"][src_valid], log_d_target[src_valid])

        def variance_loss(pred, target):
            valid = src_valid if pred.shape == src_valid.shape else mel_valid
            return F.mse_loss(pred[valid], target[valid])

        pitch_loss = variance_loss(out["pitch_pred"], out["pitch_target"])
        energy_loss = variance_loss(out["energy_pred"], out["energy_target"])
        total = mel_loss + post_loss + dur_loss + pitch_loss + energy_loss
        logs = dict(mel=mel_loss, postnet=post_loss, duration=dur_loss, pitch=pitch_loss,
                    energy=energy_loss)

        if "attn_logprob" in out:  # bộ căn chỉnh học cùng mô hình
            ctc = forward_sum_loss(out["attn_logprob"], token_lens, mel_lens)
            w = self.bin_weight(step)
            b = bin_loss(out["attn_soft"], out["attn_hard"])
            total = total + ctc + w * b
            logs.update(forward_sum=ctc, bin=b)
        logs["total"] = total
        return total, {k: float(v) for k, v in logs.items()}
