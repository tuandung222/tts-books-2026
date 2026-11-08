"""Cấu hình pytest dùng chung cho thư mục ``code/``.

Tệp này nằm ở gốc ``code/`` để pytest đưa thư mục này vào ``sys.path``; nhờ
vậy ``import vn_frontend``, ``import fastspeech2``... hoạt động khi chạy
``cd code && python -m pytest`` (hoặc ``pytest``) mà không cần cài đặt gói.
"""
