"""Cấu hình cho FastSpeech 2 (Dự án 2). Mọi giá trị mặc định là điểm khởi đầu hợp lý,
không phải "chân lý": hãy điều chỉnh theo dữ liệu và phần cứng của bạn."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields


@dataclass
class AudioConfig:
    """Thông số trích đặc trưng. PHẢI khớp với vocoder (xem features.mel_spectrogram)."""

    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0
    f0_min: float = 60.0  # Hz; nới rộng cho giọng nữ cao hoặc thanh sắc/ngã vút cao
    f0_max: float = 600.0


@dataclass
class ModelConfig:
    """Siêu tham số mô hình. Mặc định gần với cấu hình phổ biến của FastSpeech 2
    (4 khối FFT cho encoder/decoder, d_model 256, 2 đầu attention, FFN Conv1d
    nhân 9, bộ dự đoán phương sai 2 lớp Conv1d nhân 3)."""

    n_symbols: int = 60  # = len(vn_frontend.SYMBOLS); train.py tự gán lại
    n_mels: int = 80
    d_model: int = 256
    n_heads: int = 2
    encoder_layers: int = 4
    decoder_layers: int = 4
    ffn_dim: int = 1024
    ffn_kernel: int = 9
    dropout: float = 0.2
    vp_channels: int = 256  # bộ dự đoán phương sai (variance predictor)
    vp_kernel: int = 3
    vp_dropout: float = 0.5
    n_bins: int = 256  # số khoảng lượng tử hóa cao độ/năng lượng
    pitch_level: str = "phoneme"  # "phoneme" (trung bình theo token) hoặc "frame"
    energy_level: str = "phoneme"
    pitch_range: tuple[float, float] = (-4.0, 4.0)  # min/max SAU chuẩn hóa z-score
    energy_range: tuple[float, float] = (-4.0, 4.0)
    use_postnet: bool = True
    postnet_channels: int = 256
    postnet_layers: int = 5
    n_speakers: int = 1
    use_aligner: bool = False  # True: học căn chỉnh cùng mô hình (không cần MFA)
    aligner_temperature: float = 0.0005
    max_positions: int = 4000  # độ dài tối đa của bảng mã hóa vị trí dựng sẵn

    def __post_init__(self) -> None:
        if self.pitch_level not in ("phoneme", "frame") or self.energy_level not in ("phoneme", "frame"):
            raise ValueError("pitch_level/energy_level phải là 'phoneme' hoặc 'frame'")
        if self.d_model % self.n_heads:
            raise ValueError("d_model phải chia hết cho n_heads")
        self.pitch_range = tuple(self.pitch_range)
        self.energy_range = tuple(self.energy_range)


@dataclass
class TrainConfig:
    batch_size: int = 16
    lr: float = 1e-3  # tốc độ học đỉnh sau warmup
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-9
    weight_decay: float = 0.0
    warmup_steps: int = 4000
    max_steps: int = 200_000
    grad_clip: float = 1.0
    binarization_start_step: int = 6000  # chỉ dùng khi use_aligner=True
    binarization_warmup_steps: int = 2000
    log_every: int = 100
    val_every: int = 2000
    save_every: int = 10_000
    seed: int = 1234


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        def build(kind, values):
            names = {f.name for f in fields(kind)}
            unknown = set(values) - names
            if unknown:
                raise ValueError(f"khóa cấu hình lạ cho {kind.__name__}: {sorted(unknown)}")
            return kind(**values)

        return cls(build(AudioConfig, d.get("audio", {})), build(ModelConfig, d.get("model", {})),
                   build(TrainConfig, d.get("train", {})))

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
