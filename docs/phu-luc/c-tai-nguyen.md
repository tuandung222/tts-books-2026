# Phụ lục C. Công cụ, thư viện và dữ liệu

Phụ lục này liệt kê các công cụ mã nguồn mở và tài nguyên thường dùng khi làm TTS, đặc biệt là TTS tiếng Việt.

!!! warning "Thông tin dễ lỗi thời"
    Danh sách được tổng hợp **tính đến thời điểm biên soạn (2026)**. Kho mã có thể đổi chủ, đổi tên hoặc ngừng bảo trì; giấy phép có thể thay đổi. Trước khi dùng, hãy kiểm tra trang gốc, đặc biệt là **giấy phép của mã nguồn, trọng số mô hình và dữ liệu** — ba thứ này thường có giấy phép khác nhau.

## 1. Bộ công cụ TTS tổng hợp

| Công cụ | Mô tả | Ghi chú |
|---|---|---|
| [ESPnet](https://github.com/espnet/espnet) | Bộ công cụ xử lý tiếng nói đầu-cuối (ASR, TTS, ST…), có công thức (*recipe*) cho nhiều mô hình TTS | Rất phù hợp cho nghiên cứu, tái lập kết quả |
| [Coqui TTS](https://github.com/coqui-ai/TTS) | Thư viện TTS phổ biến (Tacotron, Glow-TTS, VITS, YourTTS, XTTS…) | Công ty Coqui đã ngừng hoạt động; cộng đồng duy trì bản fork, ví dụ [idiap/coqui-ai-TTS](https://github.com/idiap/coqui-ai-TTS) |
| [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) | Khung huấn luyện mô hình hội thoại, có FastPitch, HiFi-GAN và nhiều mô hình TTS | Tối ưu cho GPU NVIDIA, huấn luyện phân tán |
| [SpeechBrain](https://github.com/speechbrain/speechbrain) | Bộ công cụ PyTorch cho xử lý tiếng nói | Có mô hình xác minh người nói ECAPA-TDNN tiện cho đánh giá |
| [Amphion](https://github.com/open-mmlab/Amphion) | Bộ công cụ sinh âm thanh (TTS, VC, vocoder, codec), gắn với bộ dữ liệu Emilia và một số mô hình như MaskGCT | Hữu ích để tìm hiểu các mô hình quy mô lớn |

## 2. Kho mã mô hình tiêu biểu

| Nhóm | Kho mã | Chương liên quan |
|---|---|---|
| Mô hình âm học | [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) | [Chương 23](../phan-5-mo-hinh-am-hoc/ch23-glow-grad-matcha.md) |
| End-to-end | [VITS](https://github.com/jaywalnut310/vits) (bản chính thức) | [Chương 27](../phan-7-e2e-bieu-cam/ch27-vits.md) |
| Biểu cảm | [StyleTTS 2](https://github.com/yl4579/StyleTTS2) | [Chương 29](../phan-7-e2e-bieu-cam/ch29-prosody-cam-xuc-phong-cach.md) |
| Vocoder | [HiFi-GAN](https://github.com/jik876/hifi-gan), [BigVGAN](https://github.com/NVIDIA/BigVGAN), [Vocos](https://github.com/gemelo-ai/vocos) | [Chương 25](../phan-6-vocoder/ch25-gan-vocoder.md) |
| Codec | [EnCodec](https://github.com/facebookresearch/encodec), [DAC](https://github.com/descriptinc/descript-audio-codec) | [Chương 31](../phan-8-mo-hinh-lon/ch31-audio-codec-speech-token.md) |
| Zero-shot NAR | [F5-TTS](https://github.com/SWivid/F5-TTS) | [Chương 33](../phan-8-mo-hinh-lon/ch33-nar-quy-mo-lon.md) |
| Hệ thống lai | [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) | [Chương 34](../phan-8-mo-hinh-lon/ch34-he-thong-lai-hien-dai.md) |
| Hội thoại | [Moshi](https://github.com/kyutai-labs/moshi) | [Chương 35](../phan-8-mo-hinh-lon/ch35-streaming-speech-llm.md) |

## 3. Triển khai và suy luận

| Công cụ | Mô tả |
|---|---|
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | Chạy mô hình ONNX trên CPU/GPU/thiết bị di động |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | Suy luận ASR/TTS ngoại tuyến trên nhiều nền tảng (Android, iOS, Raspberry Pi…), hỗ trợ nhiều mô hình TTS dạng ONNX |
| Piper | Hệ thống TTS nhẹ dựa trên VITS, xuất ONNX, chạy nhanh trên CPU; kho mã gốc đã được chuyển giao — hãy tìm phiên bản đang được bảo trì |
| [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) | Máy chủ phục vụ mô hình với batching động |

Xem [Chương 39](../phan-9-thuc-tien/ch39-trien-khai-production.md) và [Dự án 5](../phan-10-du-an/du-an-5-tts-service-streaming.md).

## 4. Xử lý tín hiệu và tiếng nói

| Thư viện | Dùng để |
|---|---|
| [librosa](https://github.com/librosa/librosa) | Phân tích âm thanh: STFT, mel, pYIN, hiển thị phổ đồ |
| [torchaudio](https://github.com/pytorch/audio) | Biến đổi âm thanh trên GPU trong PyTorch (lưu ý: API đọc/ghi tệp đã thay đổi ở các phiên bản gần đây) |
| soundfile | Đọc/ghi tệp WAV, FLAC ổn định |
| Praat / parselmouth | Phân tích ngữ âm học (F0, formant, cường độ), rất hữu ích để nghiên cứu thanh điệu tiếng Việt |
| pyworld | Giao diện Python cho vocoder WORLD: trích F0 (DIO/Harvest), đường bao phổ, độ không tuần hoàn |
| [Montreal Forced Aligner](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) | Căn chỉnh cưỡng bức văn bản – âm thanh |
| [espeak-ng](https://github.com/espeak-ng/espeak-ng) | Bộ tổng hợp formant đa ngôn ngữ, thường dùng như công cụ G2P (*phonemizer*) |

## 5. Xử lý ngôn ngữ tiếng Việt

| Công cụ | Dùng để |
|---|---|
| [underthesea](https://github.com/undertheseanlp/underthesea) | Tách từ, gán nhãn từ loại, chuẩn hóa văn bản tiếng Việt |
| [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP) | Tách từ, gán nhãn từ loại, nhận dạng thực thể, phân tích cú pháp |
| pyvi | Tách từ tiếng Việt nhẹ |
| [PhoWhisper](https://github.com/VinAIResearch/PhoWhisper) | Nhận dạng tiếng nói tiếng Việt (tinh chỉnh từ Whisper) — dùng để phiên âm dữ liệu và tính WER/CER |
| `vn_frontend` (trong kho mã này) | Chuẩn hóa văn bản và G2P tiếng Việt — xem [Dự án 1](../phan-10-du-an/du-an-1-frontend-tieng-viet.md) |

## 6. Đánh giá

| Công cụ | Dùng để |
|---|---|
| [jiwer](https://github.com/jitsi/jiwer) | Tính WER/CER |
| UTMOS, DNSMOS | Dự đoán MOS tự động (chỉ nên dùng để so sánh tương đối) |
| Mô hình xác minh người nói (WavLM, ECAPA-TDNN) | Tính độ giống người nói (SECS/SIM) |
| pesq, pystoi | PESQ, STOI — chủ yếu cho bài toán có tín hiệu tham chiếu |

Xem [Chương 38](../phan-9-thuc-tien/ch38-danh-gia-tts.md).

## 7. An toàn

| Công cụ | Dùng để |
|---|---|
| [AudioSeal](https://github.com/facebookresearch/audioseal) | Thủy vân âm thanh và phát hiện thủy vân |
| Các mô hình chống giả mạo (ví dụ AASIST) | Phát hiện tiếng nói tổng hợp |

Xem [Chương 40](../phan-9-thuc-tien/ch40-dao-duc-an-toan.md).

## 8. Bộ dữ liệu

Mô tả chi tiết, cách chọn và lưu ý giấy phép của từng bộ dữ liệu nằm ở [Chương 36](../phan-9-thuc-tien/ch36-du-lieu-tts.md). Tóm tắt nhanh:

| Bộ dữ liệu | Ngôn ngữ | Đặc điểm |
|---|---|---|
| [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) | Tiếng Anh | Một người nói (nữ), đọc sách; chuẩn "hello world" của TTS |
| VCTK | Tiếng Anh | Khoảng 110 người nói với nhiều giọng địa phương |
| [LibriTTS](https://www.openslr.org/60/) / [LibriTTS-R](https://www.openslr.org/141/) | Tiếng Anh | Đa người nói, từ sách nói LibriVox; bản R được phục hồi chất lượng |
| [Multilingual LibriSpeech](https://www.openslr.org/94/) | Nhiều ngôn ngữ châu Âu | Quy mô lớn, từ sách nói |
| Emilia | Nhiều ngôn ngữ | Dữ liệu "in-the-wild" quy mô rất lớn, dùng cho mô hình zero-shot |
| [Common Voice](https://commonvoice.mozilla.org/) | Nhiều ngôn ngữ, có tiếng Việt | Đóng góp cộng đồng, chất lượng không đồng đều, phù hợp hơn cho ASR |
| VIVOS | Tiếng Việt | Bộ dữ liệu ASR nhỏ, đọc trong môi trường yên tĩnh |
| Các bộ dữ liệu TTS tiếng Việt khác | Tiếng Việt | Dữ liệu của các kỳ thi VLSP, dữ liệu sách nói, dữ liệu thu thập từ Internet do cộng đồng công bố — xem Chương 36 |

!!! note "Góc tiếng Việt"
    Dữ liệu TTS tiếng Việt chất lượng cao, giấy phép rõ ràng còn khan hiếm. Nhiều bộ dữ liệu cộng đồng được thu thập từ Internet và chỉ cho phép dùng cho nghiên cứu. Nếu xây dựng sản phẩm thương mại, phương án an toàn nhất vẫn là **thu âm dữ liệu riêng có hợp đồng và sự đồng ý rõ ràng** của người đọc.
