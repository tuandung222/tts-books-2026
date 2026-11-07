# Hướng dẫn đóng góp

Cảm ơn bạn đã quan tâm tới giáo trình! Đây là dự án cộng đồng; mọi đóng góp, dù nhỏ, đều giúp ích cho người học sau.

## Bạn có thể đóng góp gì?

- **Báo lỗi**: sai công thức, sai kiến thức, lỗi chính tả, liên kết hỏng, mã không chạy → mở [issue](https://github.com/tuandung222/tts-books-2026/issues), ghi rõ chương, mục và nội dung đề xuất sửa.
- **Sửa nhanh**: bấm biểu tượng ✏️ ở đầu mỗi trang trên site để sửa trực tiếp trên GitHub và gửi pull request.
- **Bổ sung nội dung**: ví dụ tiếng Việt, bài tập, lời giải, hình minh họa, cập nhật mô hình mới. Với thay đổi lớn (thêm mục, viết lại chương), hãy mở issue thảo luận trước.
- **Mã nguồn dự án** trong `code/`: sửa lỗi, thêm kiểm thử, hỗ trợ phương ngữ, v.v.

## Quy trình

1. Fork kho mã và tạo nhánh mới từ `main`.
2. Đọc [Quy ước biên soạn](docs/gioi-thieu/quy-uoc-bien-soan.md) và [Ký hiệu toán học](docs/gioi-thieu/ky-hieu.md).
3. Sửa nội dung trong `docs/` (hoặc mã trong `code/`).
4. Kiểm tra trên máy:

    ```bash
    pip install -r requirements.txt
    python scripts/check_book.py
    mkdocs build --strict
    cd code && python -m pytest -q   # nếu sửa mã nguồn
    ```

5. Gửi pull request, mô tả ngắn gọn thay đổi và lý do. CI sẽ tự động kiểm tra.

## Nguyên tắc nội dung

- **Chính xác trên hết.** Không đưa số liệu, kết quả hay trích dẫn mà bạn không kiểm chứng được. Ghi nguồn cho thông tin quan trọng.
- **Tiếng Việt chuẩn, Unicode NFC.** Hầu hết trình soạn thảo hiện đại đã dùng NFC; `check_book.py` sẽ báo nếu không.
- **Giữ cấu trúc chương** như quy định (mục tiêu, kiến thức cần có, nội dung, lỗi thường gặp, tóm tắt, bài tập, tài liệu tham khảo).
- **Tôn trọng bản quyền.** Chỉ đưa hình ảnh, đoạn trích có giấy phép phù hợp và ghi nguồn.

## Giấy phép đóng góp

Khi gửi đóng góp, bạn đồng ý để nội dung được phát hành theo [CC BY-SA 4.0](LICENSE) và mã nguồn theo [MIT](LICENSE-CODE).
