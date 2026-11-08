# Giáo trình Tổng hợp tiếng nói (Text-to-Speech) 2026

[![Build & Deploy](https://github.com/tuandung222/tts-books-2026/actions/workflows/deploy.yml/badge.svg)](https://github.com/tuandung222/tts-books-2026/actions/workflows/deploy.yml)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](LICENSE)
[![Code: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE-CODE)

**Giáo trình mở bằng tiếng Việt về Text-to-Speech — từ nền tảng học sâu đến làm chủ các hệ thống TTS hiện đại.**

📖 **Đọc trực tuyến:** <https://tuandung222.github.io/tts-books-2026/>

## Nội dung

40 chương lý thuyết, 5 dự án thực hành có mã nguồn, và các phụ lục:

| Phần | Chương | Chủ đề |
|---|---|---|
| I. Nền tảng học sâu | 1–6 | Toán, mạng nơ-ron, tối ưu hóa, CNN/RNN, Transformer, PyTorch cho tiếng nói |
| II. Mô hình sinh | 7–12 | Tổng quan, VAE, GAN, normalizing flow, diffusion, flow matching |
| III. Tín hiệu và ngữ âm | 13–16 | Âm học, xử lý tín hiệu số, ngữ âm học tiếng Việt, ngôn điệu |
| IV. Xử lý văn bản | 17–18 | Chuẩn hóa văn bản tiếng Việt, G2P |
| V. Mô hình âm học | 19–23 | Lịch sử TTS, Tacotron, alignment, FastSpeech, Glow/Grad/Matcha-TTS |
| VI. Vocoder | 24–26 | WaveNet, WaveGlow, HiFi-GAN, BigVGAN, Vocos, DiffWave |
| VII. End-to-end và biểu cảm | 27–30 | VITS, đa người nói, cảm xúc/phong cách, trộn mã Việt–Anh |
| VIII. TTS thời mô hình lớn | 31–35 | Audio codec, VALL-E, F5-TTS, CosyVoice, speech LLM, streaming |
| IX. Thực tiễn | 36–40 | Dữ liệu, huấn luyện và gỡ lỗi, đánh giá, triển khai, đạo đức |
| X. Dự án | 1–5 | Frontend tiếng Việt, FastSpeech 2 + HiFi-GAN, VITS, zero-shot, dịch vụ streaming |

## Cấu trúc kho mã

```text
.
├── docs/                    # Nội dung giáo trình (Markdown, MkDocs Material)
│   ├── gioi-thieu/          # Lời nói đầu, lộ trình, ký hiệu, quy ước biên soạn
│   ├── phan-1-nen-tang/ … phan-10-du-an/
│   └── phu-luc/             # Thuật ngữ, danh mục bài báo, tài nguyên
├── code/                    # Mã nguồn các dự án thực hành + kiểm thử
├── scripts/check_book.py    # Kiểm tra cấu trúc chương, Unicode NFC, công thức
├── mkdocs.yml               # Cấu hình site và mục lục
└── .github/workflows/       # CI: kiểm tra, build, deploy GitHub Pages
```

## Build trên máy

```bash
pip install -r requirements.txt
python scripts/check_book.py --stats   # kiểm tra cấu trúc
mkdocs serve                           # xem tại http://127.0.0.1:8000
mkdocs build --strict                  # build đầy đủ như CI
```

Chạy kiểm thử mã nguồn dự án:

```bash
cd code
pip install -r requirements.txt
python -m pytest -q
```

## Triển khai

Mỗi lần có commit vào nhánh `main`, GitHub Actions ([`.github/workflows/deploy.yml`](.github/workflows/deploy.yml)) sẽ:

1. chạy `scripts/check_book.py` và `mkdocs build --strict`;
2. chạy kiểm thử mã nguồn trong `code/`;
3. build site và đẩy lên nhánh `gh-pages`, được GitHub Pages phục vụ tại địa chỉ ở trên.

Pull request chỉ chạy bước kiểm tra (1–2), không deploy.

> Lần đầu: nếu trang chưa hiển thị, vào **Settings → Pages**, chọn *Source* = **Deploy from a branch**, nhánh **`gh-pages`**, thư mục **`/ (root)`**.

## Đóng góp

Rất hoan nghênh mọi đóng góp: sửa lỗi, cải thiện diễn đạt, bổ sung ví dụ tiếng Việt, bài tập, hình minh họa. Vui lòng đọc [CONTRIBUTING.md](CONTRIBUTING.md) và [Quy ước biên soạn](docs/gioi-thieu/quy-uoc-bien-soan.md) trước khi gửi pull request.

## Giấy phép

- Nội dung giáo trình: [CC BY-SA 4.0](LICENSE).
- Mã nguồn: [MIT](LICENSE-CODE).
