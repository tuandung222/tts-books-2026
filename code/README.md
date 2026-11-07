# Mã nguồn đi kèm giáo trình TTS 2026

Thư mục này chứa mã cho các dự án ở Phần X của giáo trình. Mọi gói đều có thể
import trực tiếp khi đứng trong thư mục `code/` (không cần `pip install`).

| Gói | Dự án | Phụ thuộc | Nội dung |
|---|---|---|---|
| `vn_frontend/` | [Dự án 1](../docs/phan-10-du-an/du-an-1-frontend-tieng-viet.md) | chỉ thư viện chuẩn Python | chuẩn hóa văn bản, tách câu, phân tích âm tiết, G2P (IPA + thanh điệu), bảng ký hiệu, CLI |
| `fastspeech2/` | [Dự án 2](../docs/phan-10-du-an/du-an-2-fastspeech2-hifigan.md) | PyTorch, NumPy, librosa, soundfile | FastSpeech 2 rút gọn, bộ căn chỉnh học cùng mô hình, trích đặc trưng, tiền xử lý, huấn luyện, suy luận, bộ sinh HiFi-GAN để suy luận |
| `tts_service/` | [Dự án 5](../docs/phan-10-du-an/du-an-5-tts-service-streaming.md) | FastAPI (tùy chọn) | dịch vụ TTS streaming |

## Cài đặt và kiểm thử

```bash
cd code
python -m venv .venv && source .venv/bin/activate
pip install torch numpy librosa soundfile pytest      # vn_frontend không cần gì thêm
OMP_NUM_THREADS=2 python -m pytest                    # toàn bộ kiểm thử, chạy trên CPU
python -m pytest tests/test_vn_frontend.py           # chỉ front-end (Python thuần)
```

## Dùng nhanh `vn_frontend`

```python
from vn_frontend import normalize_text, split_sentences, parse_syllable, g2p, text_to_sequence, SYMBOLS

normalize_text("UBND TP.HCM họp lúc 14h30 ngày 15/3, giá 50.000đ/kg.")
# 'ủy ban nhân dân thành phố Hồ Chí Minh họp lúc mười bốn giờ ba mươi phút ngày mười lăm tháng ba, giá năm mươi nghìn đồng một ki lô gam.'
parse_syllable("hoàng")      # Syllable(onset='h', glide='o', nucleus='a', coda='ng', tone=2)
g2p("Tiếng Việt")            # ['t', 'iə', 'ŋ', 'T3', ' ', 'v', 'iə', 't', 'T6']
text_to_sequence("Xin chào") # danh sách ID trong SYMBOLS
```

Thanh điệu được đánh số `1..6` = ngang, huyền, sắc, hỏi, ngã, nặng (token `T1`..`T6`).

Dòng lệnh: `python -m vn_frontend {normalize,sentences,parse,g2p,ids} "văn bản" [--dialect south]`.

## Dùng nhanh `fastspeech2`

```bash
python -m fastspeech2.train --synthetic --steps 3 --out runs/smoke        # chạy thử trên dữ liệu giả
python -m fastspeech2.preprocess features --data DATA --out feats         # trích đặc trưng
python -m fastspeech2.train --data feats --out runs/fs2_vi --aligner      # huấn luyện thật
python -m fastspeech2.synthesize --checkpoint runs/fs2_vi/ckpt_0200000.pt --text "Xin chào"
```

Quy trình đầy đủ (dữ liệu, căn chỉnh bằng MFA hoặc bộ căn chỉnh học cùng mô
hình, vocoder, đánh giá) được trình bày ở trang Dự án 2.

Giấy phép mã nguồn: MIT (xem chân trang của giáo trình). Dữ liệu và checkpoint
của bên thứ ba có giấy phép riêng — hãy kiểm tra trước khi dùng.
