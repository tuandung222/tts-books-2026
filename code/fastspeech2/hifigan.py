"""Bộ sinh HiFi-GAN (Kong và cộng sự, 2020) tối giản cho SUY LUẬN.

Kiến trúc và tên tham số theo mã nguồn gốc (biến thể ResBlock1, cấu hình V1/V2),
để nạp được checkpoint bộ sinh do tác giả công bố. Checkpoint gốc lưu trọng số
dưới dạng weight normalization (``weight_g``, ``weight_v``); ``load_generator``
gộp chúng thành ``weight`` nên mô hình ở đây không cần weight_norm.
Huấn luyện/fine-tune HiFi-GAN: dùng repo gốc hoặc mã ở Chương 25.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LRELU_SLOPE = 0.1


def get_padding(kernel: int, dilation: int = 1) -> int:
    return (kernel * dilation - dilation) // 2


class ResBlock1(nn.Module):
    def __init__(self, ch: int, kernel: int, dilations=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(nn.Conv1d(ch, ch, kernel, 1, dilation=d, padding=get_padding(kernel, d))
                                    for d in dilations)
        self.convs2 = nn.ModuleList(nn.Conv1d(ch, ch, kernel, 1, dilation=1, padding=get_padding(kernel))
                                    for _ in dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            x = x + c2(F.leaky_relu(c1(F.leaky_relu(x, LRELU_SLOPE)), LRELU_SLOPE))
        return x


class Generator(nn.Module):
    """mel (B, F, T) -> sóng âm (B, 1, T * prod(upsample_rates)), giá trị trong [-1, 1]."""

    def __init__(self, n_mels: int = 80, upsample_rates=(8, 8, 2, 2), upsample_kernel_sizes=(16, 16, 4, 4),
                 upsample_initial_channel: int = 512, resblock_kernel_sizes=(3, 7, 11),
                 resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5))):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.hop_length = 1
        for u in upsample_rates:
            self.hop_length *= u
        ch = upsample_initial_channel
        self.conv_pre = nn.Conv1d(n_mels, ch, 7, 1, padding=3)
        self.ups = nn.ModuleList(
            nn.ConvTranspose1d(ch // 2**i, ch // 2 ** (i + 1), k, u, padding=(k - u) // 2)
            for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)))
        self.resblocks = nn.ModuleList(
            ResBlock1(ch // 2 ** (i + 1), k, d)
            for i in range(len(self.ups))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes))
        self.conv_post = nn.Conv1d(ch // 2 ** len(self.ups), 1, 7, 1, padding=3)

    @classmethod
    def from_config(cls, h: dict, n_mels: int = 80) -> "Generator":
        """Tạo từ dict cấu hình kiểu ``config_v1.json`` của repo gốc."""
        if str(h.get("resblock", "1")) != "1":
            raise NotImplementedError("bản rút gọn chỉ hỗ trợ resblock '1' (cấu hình V1/V2)")
        return cls(n_mels, h["upsample_rates"], h["upsample_kernel_sizes"], h["upsample_initial_channel"],
                   h["resblock_kernel_sizes"], h["resblock_dilation_sizes"])

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(mel)
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, LRELU_SLOPE))
            x = sum(self.resblocks[i * self.num_kernels + j](x) for j in range(self.num_kernels)) / self.num_kernels
        x = self.conv_post(F.leaky_relu(x))  # độ dốc mặc định 0.01, như mã gốc
        return torch.tanh(x)


def merge_weight_norm(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Gộp (g, v) của weight normalization thành weight = g · v / ||v||.

    Hỗ trợ cả khóa kiểu cũ (``*.weight_g``/``*.weight_v``) lẫn kiểu parametrizations
    mới (``*.parametrizations.weight.original0/1``). Chuẩn lấy trên mọi chiều trừ chiều 0.
    """
    out: dict[str, torch.Tensor] = {}
    for key, val in state.items():
        if key.endswith(".weight_g") or key.endswith(".parametrizations.weight.original0"):
            prefix = key.rsplit(".weight_g", 1)[0] if key.endswith(".weight_g") else key.rsplit(
                ".parametrizations.weight.original0", 1)[0]
            v = state.get(prefix + ".weight_v", state.get(prefix + ".parametrizations.weight.original1"))
            norm = v.flatten(1).norm(dim=1).view(-1, *([1] * (v.dim() - 1)))
            out[prefix + ".weight"] = val * v / norm
        elif key.endswith(".weight_v") or key.endswith(".parametrizations.weight.original1"):
            continue
        else:
            out[key] = val
    return out


def load_generator(checkpoint_path: str, config: dict | None = None, n_mels: int = 80) -> Generator:
    """Nạp checkpoint bộ sinh (định dạng repo gốc: {"generator": state_dict})."""
    gen = Generator.from_config(config, n_mels) if config else Generator(n_mels)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = ckpt.get("generator", ckpt)
    gen.load_state_dict(merge_weight_norm(state))
    return gen.eval()
