# Ảnh Docker cho dịch vụ TTS (Dự án 5). Build từ thư mục code/ của kho:
#   cd code && docker build -f tts_service/Dockerfile -t tts-service .
#   docker run --rm -p 8000:8000 -e TTS_SIMULATED_RTF=0.05 tts-service
# Lưu ý: tệp này chưa được build thử trong môi trường biên soạn; hãy kiểm tra lại.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2

WORKDIR /app
COPY tts_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Mã nguồn: dịch vụ + front-end tiếng Việt (Dự án 1). Mô hình nên được gắn vào qua volume
# hoặc tải lúc khởi động, không nên "nướng" vào ảnh.
COPY tts_service/ /app/tts_service/
COPY vn_frontend/ /app/vn_frontend/

RUN useradd --create-home --uid 10001 tts
USER tts

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

# Một worker mỗi container; mở rộng bằng cách chạy thêm container (mỗi container một bản mô hình).
CMD ["uvicorn", "tts_service.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--timeout-graceful-shutdown", "30"]
