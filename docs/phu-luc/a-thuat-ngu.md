# Phụ lục A. Thuật ngữ Anh–Việt

Bảng dưới đây liệt kê các thuật ngữ dùng trong sách, theo thứ tự chữ cái tiếng Anh. Cột "Dùng trong sách" cho biết cách gọi mà giáo trình thống nhất sử dụng. Với nhiều thuật ngữ, cộng đồng kỹ sư Việt Nam quen dùng nguyên tiếng Anh; khi đó sách cũng giữ nguyên và chỉ chú thích tiếng Việt ở lần xuất hiện đầu tiên.

!!! tip "Tra cứu nhanh"
    Dùng ô tìm kiếm ở đầu trang (phím tắt ++s++ hoặc ++slash++) để tìm thuật ngữ trong toàn bộ giáo trình.

## A – C

| Tiếng Anh | Dùng trong sách | Giải thích ngắn |
|---|---|---|
| acoustic model | mô hình âm học | Mô hình biến chuỗi văn bản/âm vị thành đặc trưng âm học (thường là mel-spectrogram) |
| acoustic feature | đặc trưng âm học | Biểu diễn trung gian của tiếng nói: mel-spectrogram, F0, năng lượng… |
| acoustic token | token âm học | Token rời rạc từ audio codec, giữ chi tiết âm sắc và chất lượng âm thanh |
| activation function | hàm kích hoạt | Hàm phi tuyến áp dụng sau phép biến đổi tuyến tính |
| adversarial training | huấn luyện đối kháng | Huấn luyện bộ sinh và bộ phân biệt cạnh tranh nhau (GAN) |
| aliasing | chồng phổ (*aliasing*) | Thành phần tần số cao bị "gập" thành tần số thấp khi lấy mẫu không đủ |
| alignment | căn chỉnh (*alignment*) | Ánh xạ giữa token văn bản và các khung âm thanh |
| allophone | biến thể âm vị | Các cách phát âm khác nhau của cùng một âm vị |
| attention | cơ chế chú ý (*attention*) | Cơ chế tính tổ hợp có trọng số của các véc-tơ dựa trên độ tương đồng |
| autoregressive (AR) | tự hồi quy | Sinh từng phần tử dựa trên các phần tử đã sinh trước đó |
| backpropagation | lan truyền ngược | Thuật toán tính gradient qua đồ thị tính toán bằng quy tắc dây chuyền |
| batch | lô (*batch*) | Nhóm mẫu dữ liệu xử lý cùng lúc |
| bitrate | tốc độ bit | Số bit mỗi giây cần để biểu diễn âm thanh |
| checkpoint | checkpoint | Bản lưu trạng thái mô hình và bộ tối ưu tại một bước huấn luyện |
| classifier-free guidance (CFG) | classifier-free guidance | Kỹ thuật tăng mức tuân thủ điều kiện bằng cách ngoại suy giữa dự đoán có và không có điều kiện |
| code-switching | trộn mã (*code-switching*) | Dùng xen kẽ hai ngôn ngữ trong một câu, ví dụ Việt–Anh |
| codebook | sổ mã (*codebook*) | Tập véc-tơ đại diện dùng trong lượng tử hóa véc-tơ |
| codec (neural audio codec) | codec âm thanh nơ-ron | Mô hình nén âm thanh thành token rời rạc và giải nén lại |
| coda | âm cuối | Phụ âm hoặc bán nguyên âm kết thúc âm tiết |
| concatenative synthesis | tổng hợp ghép nối | Ghép các đoạn âm thanh ghi sẵn để tạo câu nói |
| conditional generation | sinh có điều kiện | Sinh dữ liệu theo một điều kiện cho trước (ví dụ văn bản) |
| convolution | tích chập | Phép toán trượt bộ lọc trên tín hiệu |
| causal convolution | tích chập nhân quả | Tích chập chỉ nhìn vào quá khứ, không nhìn tương lai |
| cross-entropy | entropy chéo | Hàm mất mát cho bài toán phân loại / token rời rạc |

## D – F

| Tiếng Anh | Dùng trong sách | Giải thích ngắn |
|---|---|---|
| decoder | bộ giải mã (*decoder*) | Phần mô hình sinh đầu ra từ biểu diễn ẩn |
| deepfake | deepfake (giọng giả mạo) | Nội dung giả mạo giọng hoặc hình ảnh người thật bằng AI |
| dequantization | khử lượng tử hóa | Thêm nhiễu để biến dữ liệu rời rạc thành liên tục khi huấn luyện flow |
| dialect | phương ngữ | Biến thể địa phương của ngôn ngữ (Bắc, Trung, Nam) |
| diffusion model | mô hình khuếch tán | Mô hình sinh học cách đảo ngược quá trình thêm nhiễu dần |
| dilated convolution | tích chập nhân giãn | Tích chập có khoảng cách giữa các phần tử bộ lọc, mở rộng trường tiếp nhận |
| diphthong | nguyên âm đôi | Nguyên âm có sự chuyển động cấu âm, ví dụ "iê", "uô", "ươ" |
| discriminator | bộ phân biệt | Mạng phân biệt dữ liệu thật và giả trong GAN |
| duration | trường độ | Độ dài thời gian (số khung) của mỗi âm vị/token |
| duration predictor | bộ dự đoán trường độ | Mô-đun dự đoán số khung cho mỗi token |
| embedding | embedding (véc-tơ nhúng) | Véc-tơ biểu diễn liên tục của một đối tượng rời rạc |
| encoder | bộ mã hóa (*encoder*) | Phần mô hình biến đầu vào thành biểu diễn ẩn |
| end-to-end | đầu-cuối (*end-to-end*) | Huấn luyện toàn bộ hệ thống như một mô hình duy nhất |
| energy | năng lượng | Độ lớn (biên độ bình phương) của tín hiệu trong một khung |
| epoch | epoch | Một lượt duyệt qua toàn bộ tập huấn luyện |
| exposure bias | thiên lệch phơi nhiễm | Sai lệch do huấn luyện bằng dữ liệu thật nhưng suy luận bằng đầu ra của chính mô hình |
| feature matching loss | hàm mất mát khớp đặc trưng | Khoảng cách giữa đặc trưng trung gian của bộ phân biệt với dữ liệu thật và giả |
| fine-tune | tinh chỉnh (*fine-tune*) | Huấn luyện tiếp mô hình đã huấn luyện sẵn trên dữ liệu mới |
| flow matching | flow matching (khớp dòng) | Huấn luyện trường véc-tơ vận tốc đưa phân phối nhiễu thành phân phối dữ liệu |
| forced alignment | căn chỉnh cưỡng bức | Căn chỉnh văn bản đã biết với âm thanh để lấy mốc thời gian |
| formant | formant | Tần số cộng hưởng của tuyến âm, quyết định màu sắc nguyên âm |
| frame | khung | Đoạn ngắn tín hiệu (thường 10–25 ms) dùng trong phân tích STFT |
| frontend | frontend (xử lý văn bản) | Phần xử lý văn bản trước mô hình âm học: chuẩn hóa, G2P… |
| fundamental frequency (F0) | tần số cơ bản (F0) | Tần số dao động của dây thanh, cảm nhận như cao độ |

## G – L

| Tiếng Anh | Dùng trong sách | Giải thích ngắn |
|---|---|---|
| generative model | mô hình sinh | Mô hình học phân phối dữ liệu để sinh mẫu mới |
| generator | bộ sinh | Mạng sinh dữ liệu trong GAN |
| glide (medial) | âm đệm | Bán nguyên âm /w/ giữa âm đầu và âm chính, như trong "hoa", "quê" |
| glottal stop | âm tắc thanh hầu | Phụ âm /ʔ/, xuất hiện ở đầu âm tiết bắt đầu bằng nguyên âm |
| gradient clipping | cắt gradient | Giới hạn độ lớn gradient để ổn định huấn luyện |
| grapheme | tự vị (chữ cái) | Đơn vị chữ viết |
| grapheme-to-phoneme (G2P) | chuyển tự vị sang âm vị (G2P) | Chuyển chữ viết thành chuỗi âm vị |
| ground-truth aligned (GTA) | mel căn theo dữ liệu thật (GTA) | Mel do mô hình âm học sinh ra khi được ép theo căn chỉnh thật, dùng để tinh chỉnh vocoder |
| hop size | bước nhảy (*hop*) | Khoảng cách (số mẫu) giữa hai khung STFT liên tiếp |
| in-context learning | học trong ngữ cảnh | Mô hình bắt chước giọng/phong cách từ đoạn mẫu trong đầu vào mà không cần tinh chỉnh |
| inference | suy luận | Dùng mô hình đã huấn luyện để sinh đầu ra |
| initial (onset) | âm đầu | Phụ âm mở đầu âm tiết |
| intonation | ngữ điệu | Đường nét cao độ ở cấp câu/ngữ đoạn |
| inverse STFT (iSTFT) | STFT ngược | Phép biến đổi khôi phục tín hiệu từ phổ STFT |
| KL divergence | phân kỳ KL | Độ đo khác biệt giữa hai phân phối xác suất |
| latency | độ trễ | Thời gian từ lúc nhận yêu cầu tới lúc có đầu ra |
| latent variable | biến ẩn | Biến không quan sát được, mô hình suy ra từ dữ liệu |
| learning rate | tốc độ học | Hệ số nhân với gradient khi cập nhật tham số |
| length regulator | bộ điều chỉnh độ dài | Mô-đun nhân bản biểu diễn của token theo trường độ (FastSpeech) |
| likelihood | hàm hợp lý | Xác suất (mật độ) của dữ liệu theo mô hình |
| loss function | hàm mất mát | Hàm đo sai số cần cực tiểu hóa khi huấn luyện |

## M – P

| Tiếng Anh | Dùng trong sách | Giải thích ngắn |
|---|---|---|
| mask | mặt nạ | Tensor đánh dấu vị trí hợp lệ (bỏ đệm) hoặc vị trí được phép nhìn |
| maximum likelihood estimation (MLE) | ước lượng hợp lý cực đại | Chọn tham số làm cực đại xác suất của dữ liệu |
| mean opinion score (MOS) | điểm ý kiến trung bình (MOS) | Điểm đánh giá chủ quan trung bình (thường thang 1–5) |
| mel scale | thang mel | Thang tần số mô phỏng cảm nhận cao độ của tai người |
| mel-spectrogram | mel-spectrogram | Phổ đồ được chiếu lên các băng tần thang mel, thường lấy log |
| mixed precision | độ chính xác hỗn hợp | Huấn luyện kết hợp số thực 16 bit và 32 bit |
| monotonic alignment search (MAS) | tìm kiếm căn chỉnh đơn điệu (MAS) | Thuật toán quy hoạch động tìm căn chỉnh đơn điệu tối ưu |
| multi-speaker | đa người nói | Một mô hình tổng hợp nhiều giọng |
| neural network | mạng nơ-ron | Mô hình gồm các lớp biến đổi tuyến tính và phi tuyến |
| non-autoregressive (NAR) | không tự hồi quy | Sinh toàn bộ chuỗi song song |
| normalizing flow | normalizing flow (luồng chuẩn hóa) | Chuỗi phép biến đổi khả nghịch giữa phân phối đơn giản và phân phối dữ liệu |
| nucleus (main vowel) | âm chính | Nguyên âm làm hạt nhân âm tiết |
| one-to-many | một-đối-nhiều | Một đầu vào ứng với nhiều đầu ra hợp lệ (một câu – nhiều cách đọc) |
| optimizer | bộ tối ưu | Thuật toán cập nhật tham số (SGD, Adam…) |
| overfitting | quá khớp | Mô hình học thuộc dữ liệu huấn luyện, kém tổng quát hóa |
| over-smoothing | hiện tượng nhòe (*over-smoothing*) | Đầu ra bị làm trơn, mất chi tiết do hồi quy về trung bình |
| phase | pha | Thành phần góc của hệ số phổ phức |
| phoneme | âm vị | Đơn vị âm thanh nhỏ nhất phân biệt nghĩa |
| phonation | phát thanh (chất giọng) | Cách dây thanh rung: bình thường, thì thào, kẹt (*creaky*)… |
| pitch | cao độ | Cảm nhận về độ cao của âm, tương quan chủ yếu với F0 |
| posterior collapse | sụp đổ hậu nghiệm | Bộ giải mã VAE phớt lờ biến ẩn, KL tiến về 0 |
| prompt | đoạn mẫu (*prompt*) | Âm thanh hoặc văn bản mẫu cung cấp cho mô hình để điều kiện hóa |
| prosody | ngôn điệu | Đặc tính siêu đoạn tính: ngữ điệu, nhịp, trọng âm, ngắt nghỉ |

## Q – S

| Tiếng Anh | Dùng trong sách | Giải thích ngắn |
|---|---|---|
| quantization | lượng tử hóa | Ánh xạ giá trị liên tục thành tập giá trị rời rạc |
| real-time factor (RTF) | hệ số thời gian thực (RTF) | Thời gian xử lý chia cho thời lượng âm thanh sinh ra; RTF < 1 là nhanh hơn thời gian thực |
| receptive field | trường tiếp nhận | Khoảng đầu vào ảnh hưởng tới một đầu ra |
| reference encoder | bộ mã hóa tham chiếu | Mạng nén một đoạn âm thanh mẫu thành véc-tơ phong cách |
| regularization | điều chuẩn | Kỹ thuật hạn chế quá khớp |
| reparameterization trick | thủ thuật tái tham số hóa | Viết mẫu ngẫu nhiên thành hàm khả vi của tham số và nhiễu |
| residual connection | kết nối tắt (*residual*) | Cộng đầu vào vào đầu ra của một khối |
| residual vector quantization (RVQ) | lượng tử hóa véc-tơ phần dư (RVQ) | Lượng tử hóa nhiều tầng, mỗi tầng mã hóa phần dư của tầng trước |
| rhyme | vần | Phần âm tiết gồm âm đệm, âm chính và âm cuối |
| sampling rate | tần số lấy mẫu | Số mẫu tín hiệu mỗi giây |
| score function | hàm điểm (*score*) | Gradient của log mật độ theo dữ liệu |
| semantic token | token ngữ nghĩa | Token rời rạc mang nội dung ngôn ngữ, ít thông tin âm sắc |
| sentence-final particle | tiểu từ tình thái cuối câu | Các từ như "à", "ạ", "nhé", "nhỉ" mang sắc thái ngữ điệu |
| short-time Fourier transform (STFT) | biến đổi Fourier thời gian ngắn | Phân tích tần số theo từng khung thời gian |
| speaker embedding | embedding người nói | Véc-tơ đại diện cho đặc điểm giọng của một người |
| speaker verification | xác minh người nói | Bài toán xác định hai đoạn âm thanh có cùng người nói không |
| spectrogram | phổ đồ (*spectrogram*) | Biểu diễn năng lượng tín hiệu theo thời gian và tần số |
| streaming | streaming (sinh theo luồng) | Sinh và trả âm thanh từng phần khi chưa sinh xong toàn bộ |
| syllable | âm tiết | Đơn vị phát âm; trong tiếng Việt thường trùng với một "tiếng" |

## T – Z

| Tiếng Anh | Dùng trong sách | Giải thích ngắn |
|---|---|---|
| teacher forcing | teacher forcing (ép theo dữ liệu thật) | Dùng đầu ra thật của bước trước làm đầu vào khi huấn luyện mô hình tự hồi quy |
| text normalization (TN) | chuẩn hóa văn bản | Chuyển số, ký hiệu, viết tắt thành dạng chữ đọc được |
| timbre | âm sắc | Đặc tính phân biệt giọng người này với người khác |
| tone | thanh điệu | Đường nét cao độ và chất giọng mang nghĩa từ vựng của âm tiết |
| tone sandhi | biến điệu | Thanh điệu thay đổi do ngữ cảnh (ít gặp trong tiếng Việt) |
| token | token | Đơn vị rời rạc của chuỗi đầu vào/đầu ra |
| training | huấn luyện | Quá trình tối ưu tham số mô hình trên dữ liệu |
| transposed convolution | tích chập chuyển vị | Tích chập dùng để tăng độ phân giải (upsampling) |
| unit selection | chọn đơn vị | Tổng hợp ghép nối bằng cách chọn đơn vị tối ưu từ kho lớn |
| vector field | trường véc-tơ | Hàm gán véc-tơ vận tốc cho mỗi điểm và thời điểm |
| vocoder | vocoder (bộ tạo sóng âm) | Mô hình biến đặc trưng âm học thành dạng sóng |
| voice cloning | nhân bản giọng nói | Tổng hợp giọng của một người cụ thể từ mẫu giọng của họ |
| voice conversion | chuyển đổi giọng nói | Biến giọng người A thành giọng người B, giữ nội dung |
| voiced / unvoiced | hữu thanh / vô thanh | Có / không có rung dây thanh |
| watermarking | thủy vân (*watermarking*) | Nhúng tín hiệu ẩn vào âm thanh để nhận diện nguồn gốc |
| waveform | dạng sóng | Tín hiệu âm thanh theo thời gian |
| weight decay | suy giảm trọng số | Điều chuẩn kéo trọng số về 0 |
| window function | hàm cửa sổ | Hàm nhân vào mỗi khung trước khi biến đổi Fourier |
| word segmentation | tách từ | Xác định ranh giới từ (trong tiếng Việt, từ có thể gồm nhiều âm tiết) |
| zero-shot | zero-shot | Tổng hợp giọng/ngôn ngữ chưa thấy khi huấn luyện, không cần tinh chỉnh |
