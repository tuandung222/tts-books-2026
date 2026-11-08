"""Chạy máy chủ: ``python -m tts_service --port 8000 [--simulated-rtf 0.05]``.

Cấu hình chi tiết hơn đọc từ biến môi trường (xem ``ServiceConfig.from_env`` và
``synthesizer_from_env``).
"""
from __future__ import annotations

import argparse
import logging
import os

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser(description="Dịch vụ TTS streaming")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--synthesizer", default=None, help="tone | onnx (ghi đè TTS_SYNTHESIZER)")
    ap.add_argument("--onnx-model", default=None, help="đường dẫn mô hình ONNX (ghi đè TTS_ONNX_MODEL)")
    ap.add_argument("--simulated-rtf", type=float, default=None, help="RTF giả lập cho bộ 'tone'")
    ap.add_argument("--max-concurrency", type=int, default=None)
    args = ap.parse_args()
    for key, value in [("TTS_SYNTHESIZER", args.synthesizer), ("TTS_ONNX_MODEL", args.onnx_model),
                       ("TTS_SIMULATED_RTF", args.simulated_rtf),
                       ("TTS_MAX_CONCURRENCY", args.max_concurrency)]:
        if value is not None:
            os.environ[key] = str(value)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # Chỉ MỘT tiến trình: mỗi tiến trình nạp một bản mô hình riêng (xem Dự án 5, mục triển khai)
    uvicorn.run("tts_service.server:create_app", factory=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
