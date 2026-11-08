# Quy ước biên soạn

Trang này dành cho người viết và người đóng góp. Giáo trình được nhiều người (và nhiều lượt biên soạn) cùng viết, nên mọi chương phải tuân thủ cùng một bộ quy ước về **cấu trúc**, **văn phong**, **thuật ngữ**, **ký hiệu** và **mã nguồn**. Người đọc cũng nên lướt qua trang này để biết sách được tổ chức ra sao.

## 1. Cấu trúc bắt buộc của một chương

Mỗi chương là **một tệp Markdown** và phải có các phần sau, theo đúng thứ tự:

1. **Tiêu đề cấp 1** dạng `# Chương N. Tên chương`. Mỗi tệp chỉ có đúng một tiêu đề cấp 1.
2. Hộp **Mục tiêu học tập** (dùng `!!! abstract "Mục tiêu học tập"`): 4–8 gạch đầu dòng, mỗi dòng bắt đầu bằng một động từ đo được (*giải thích*, *suy ra*, *cài đặt*, *so sánh*, *chẩn đoán*…).
3. Hộp **Kiến thức cần có** (`!!! info "Kiến thức cần có"`): liệt kê các chương trước cần đọc, có liên kết tương đối.
4. **Phần thân** chia thành các mục `##` và `###`. Mỗi khái niệm quan trọng nên đi theo nhịp:
   **trực giác → hình thức hóa toán học → ví dụ hoặc mã minh họa → liên hệ với TTS**.
5. Mục `## Lỗi thường gặp và mẹo thực hành`: kinh nghiệm thực tế, các "bẫy" khi cài đặt/huấn luyện.
6. Mục `## Tóm tắt`: 5–10 gạch đầu dòng chốt lại ý chính.
7. Mục `## Bài tập`: tối thiểu 5 bài, phân ba mức **[Cơ bản]**, **[Nâng cao]**, **[Lập trình]**. Bài nào có gợi ý hoặc lời giải thì đặt trong khối thu gọn `??? tip "Gợi ý"` hoặc `??? success "Lời giải"`.
8. Mục `## Tài liệu tham khảo`: danh sách đánh số, mỗi mục gồm *tác giả, năm, tên công trình, nơi công bố*.

Các mục tùy chọn nhưng được khuyến khích: `## Đọc thêm`, các hộp `!!! example`, `!!! tip`, `!!! warning`, `!!! note "Góc tiếng Việt"` (lưu ý riêng cho tiếng Việt).

## 2. Văn phong

- Viết bằng **tiếng Việt chuẩn, rõ ràng, mạch lạc**, giọng văn của một giáo trình đại học tốt: chính xác nhưng dễ tiếp cận. Xưng hô với người đọc là "bạn", tác giả xưng "chúng ta" khi cùng suy luận.
- Ưu tiên **giải thích vì sao** chứ không chỉ liệt kê *cái gì*. Mỗi thiết kế kiến trúc phải trả lời được: *nó giải quyết vấn đề gì của phương pháp trước?*
- Câu ngắn vừa phải; tránh văn dịch máy (ví dụ tránh "được thực hiện bởi", "nó là một cái mà…").
- **Không bịa đặt.** Không bịa số liệu, kết quả thực nghiệm, mã arXiv, tên tác giả hay tên bài báo. Nếu không chắc một con số, hãy mô tả định tính hoặc ghi rõ "theo báo cáo gốc", "xấp xỉ". Thông tin dễ lỗi thời (mô hình mới, giấy phép, kích thước dữ liệu) cần ghi "tính đến thời điểm biên soạn (2026)".
- Không quảng cáo sản phẩm; khi so sánh các hệ thống, nêu cả ưu và nhược điểm.

### Chính tả

- Văn bản phải ở dạng **Unicode NFC** (dựng sẵn). Không dùng tổ hợp dấu rời (NFD). Kịch bản `scripts/check_book.py` sẽ báo lỗi nếu vi phạm.
- Thống nhất cách đặt dấu kiểu phổ biến: **hóa, hòa, khỏe, thủy, quý** (dấu đặt trên nguyên âm thứ nhất của "oa, oe, uy" khi không có âm cuối).
- Dùng "kỹ thuật", "lý thuyết", "hệ thống", "mô hình", "dữ liệu".
- Số thập phân trong văn xuôi dùng dấu phẩy kiểu Việt (0,5) **hoặc** dấu chấm khi ở trong công thức, mã, bảng số liệu kỹ thuật — trong công thức và mã **luôn dùng dấu chấm**.

## 3. Thuật ngữ

- Lần đầu xuất hiện trong chương: viết **thuật ngữ tiếng Việt** kèm *tiếng Anh trong ngoặc*, ví dụ: "bộ mã hóa (*encoder*)", "ngôn điệu (*prosody*)".
- Sau đó dùng cách gọi tự nhiên nhất trong cộng đồng. Nhiều thuật ngữ được giữ nguyên tiếng Anh vì cộng đồng kỹ sư Việt Nam dùng phổ biến: *attention, encoder, decoder, embedding, token, vocoder, mel-spectrogram, batch, epoch, checkpoint, fine-tune, zero-shot, codec, flow matching*.
- Tên mô hình, bài báo, thư viện **giữ nguyên** (Tacotron 2, HiFi-GAN, VITS, PyTorch).
- Tra cứu và bổ sung vào [Phụ lục A – Thuật ngữ](../phu-luc/a-thuat-ngu.md). Một số quy ước cố định:

| Tiếng Anh | Dùng trong sách |
|---|---|
| deep learning / machine learning | học sâu / học máy |
| neural network | mạng nơ-ron |
| loss function | hàm mất mát |
| backpropagation | lan truyền ngược |
| optimizer / learning rate | bộ tối ưu / tốc độ học |
| overfitting / regularization | quá khớp / điều chuẩn |
| inference | suy luận |
| sampling rate | tần số lấy mẫu |
| fundamental frequency (F0) / pitch | tần số cơ bản (F0) / cao độ |
| prosody / tone | ngôn điệu / thanh điệu |
| phoneme / syllable | âm vị / âm tiết |
| acoustic model | mô hình âm học |
| alignment / duration | căn chỉnh / trường độ |
| text normalization | chuẩn hóa văn bản |
| autoregressive / non-autoregressive | tự hồi quy / không tự hồi quy |
| generative model | mô hình sinh |
| diffusion model | mô hình khuếch tán |
| normalizing flow | normalizing flow (luồng chuẩn hóa) |
| latent variable | biến ẩn |
| generator / discriminator | bộ sinh / bộ phân biệt |
| quantization / codebook | lượng tử hóa / sổ mã (codebook) |
| speaker / speaker embedding | người nói / embedding người nói |
| voice cloning | nhân bản giọng nói |
| receptive field | trường tiếp nhận |
| real-time factor (RTF) | hệ số thời gian thực (RTF) |

## 4. Toán học

- Dùng LaTeX: công thức trong dòng `$...$`, công thức khối `$$...$$` **đứng riêng một đoạn** (có dòng trống trước và sau).
- Tuân thủ [bảng ký hiệu](ky-hieu.md). Khi buộc phải dùng ký hiệu khác (để khớp bài báo gốc), hãy nói rõ.
- **Không dùng ký tự `$` cho tiền tệ** trong văn bản (sẽ bị hiểu là công thức) — viết "USD" hoặc "đô-la".
- Mỗi công thức quan trọng phải được giải thích bằng lời: các ký hiệu là gì, công thức nói lên điều gì.
- Khi suy ra (*derive*), trình bày từng bước; bước nào bỏ qua thì nói rõ là bỏ qua và vì sao.

## 5. Mã nguồn

- Ngôn ngữ: **Python ≥ 3.10**, **PyTorch 2.x**, `torchaudio`, `numpy`. Tránh phụ thuộc hiếm.
- Mã minh họa phải **chạy được** (đã kiểm thử với dữ liệu giả ngẫu nhiên, trên CPU) trừ khi ghi rõ là mã giả (*pseudo-code*) hoặc phác thảo.
- Định danh (biến, hàm, lớp) bằng tiếng Anh; chú thích (*comment*) bằng tiếng Việt.
- Ghi chú hình dạng tensor ở các dòng quan trọng, ví dụ `# (B, T, F)`. Quy ước: `B` = batch, `N` = số token văn bản, `T` = số khung, `F` = số kênh mel, `C` = số kênh ẩn, `L` = số mẫu âm thanh.
- Khối mã ngắn gọn (thường ≤ 80 dòng); mã dài đưa vào dự án ở Phần X.
- Ghi rõ ngôn ngữ cho khối mã: ` ```python `, ` ```bash `, ` ```yaml `.

## 6. Hình vẽ và sơ đồ

- Dùng **Mermaid** (` ```mermaid `) cho sơ đồ khối, luồng xử lý. Nhãn có dấu tiếng Việt hoặc ngoặc phải đặt trong dấu nháy kép: `A["Bộ mã hóa (Encoder)"]`.
- Bảng Markdown cho so sánh. Hình ảnh (nếu có) đặt trong `docs/assets/` và có chú thích nguồn, giấy phép.

## 7. Liên kết chéo

- Dùng **liên kết tương đối** tới tệp `.md`, ví dụ `[Chương 11](../phan-2-mo-hinh-sinh/ch11-diffusion.md)`.
- Tên tệp và đường dẫn các chương được cố định trong `mkdocs.yml`; không đổi tên tệp khi chưa cập nhật mục lục.

## 8. Tài liệu tham khảo

- Trích dẫn trong thân bài theo dạng *(Tác giả và cộng sự, năm)*, ví dụ: (Shen và cộng sự, 2018).
- Danh sách cuối chương ghi: `Họ tác giả đầu và cộng sự (năm). *Tên bài báo*. Nơi công bố.`
- Chỉ ghi mã arXiv khi **chắc chắn**; nếu không, bỏ qua — người đọc có thể tìm theo tên bài.

## 9. Kiểm tra trước khi gửi

```bash
pip install -r requirements.txt
python scripts/check_book.py      # kiểm tra cấu trúc, NFC, công thức
mkdocs build --strict             # build không được có cảnh báo
mkdocs serve                      # xem trước tại http://127.0.0.1:8000
```

Danh sách tự kiểm:

- [ ] Đủ các mục bắt buộc ở Mục 1.
- [ ] Không có số liệu, trích dẫn bịa đặt; thông tin không chắc chắn đã được rào đón.
- [ ] Mã minh họa đã chạy thử.
- [ ] Ký hiệu khớp với [bảng ký hiệu](ky-hieu.md).
- [ ] `check_book.py` và `mkdocs build --strict` đều qua.
