# Chương 39. Tối ưu suy luận và triển khai

!!! abstract "Mục tiêu học tập"
    - Phân tích yêu cầu triển khai (độ trễ, thông lượng, chi phí, bộ nhớ, quyền riêng tư) và chọn mục tiêu phù hợp: máy chủ GPU, máy chủ CPU, thiết bị di động/biên, trình duyệt.
    - Xuất một mô hình PyTorch sang ONNX với trục thời gian động, kiểm chứng số học bằng ONNX Runtime và nhận diện các thành phần khó xuất trong mô hình TTS.
    - Suy ra sai số lượng tử hóa (quy tắc khoảng 6 dB mỗi bit), so sánh lượng tử hóa theo tensor và theo kênh, động và tĩnh, chỉ-trọng-số; đánh giá tác động lên chất lượng âm thanh.
    - Đo RTF và độ trễ đúng phương pháp (khởi động, phân vị, đồng bộ GPU) và dùng định luật Amdahl để chọn chỗ cần tối ưu.
    - Thiết kế một dịch vụ TTS: batching động, continuous batching cho phần LM, giao thức truyền (HTTP chunked, WebSocket, gRPC, WebRTC), mã hóa âm thanh, bộ nhớ đệm, SSML.
    - Cài đặt một endpoint streaming bằng FastAPI và đo độ trễ đến byte đầu tiên từ phía người dùng.
    - Lập kế hoạch giám sát, ước lượng chi phí và năng lực (định luật Little), và chọn bộ công cụ cho thiết bị biên.

!!! info "Kiến thức cần có"
    - PyTorch, `torch.compile` và vòng huấn luyện: [Chương 6](../phan-1-nen-tang/ch06-pytorch-cho-tieng-noi.md).
    - Các họ vocoder và hàm đo RTF: [Chương 25](../phan-6-vocoder/ch25-gan-vocoder.md), [Chương 26](../phan-6-vocoder/ch26-diffusion-vocoder-va-so-sanh.md).
    - TTS dựa trên LM và flow matching: [Chương 34](../phan-8-mo-hinh-lon/ch34-he-thong-lai-hien-dai.md); streaming và độ trễ: [Chương 35](../phan-8-mo-hinh-lon/ch35-streaming-speech-llm.md).
    - Chuẩn hóa văn bản: [Chương 17](../phan-4-frontend/ch17-chuan-hoa-van-ban.md); đánh giá chất lượng: [Chương 38](../phan-9-thuc-tien/ch38-danh-gia-tts.md).

Một mô hình TTS cho điểm MOS cao trong bài báo mới chỉ là một nửa câu chuyện. Nửa còn lại là đưa nó đến người dùng: trả lời trong vài trăm mili-giây, phục vụ hàng trăm luồng đồng thời trên một số GPU hữu hạn, chạy trên điện thoại không có mạng, không đọc sai khi gặp một chuỗi SSML lạ, và báo động khi chất lượng âm thanh âm thầm giảm sút sau một lần cập nhật. Chương này trình bày bộ công cụ kỹ thuật cho nửa thứ hai: xuất mô hình, lượng tử hóa, đo hiệu năng, kiến trúc phục vụ, giám sát và triển khai trên thiết bị biên.

Nguyên tắc xuyên suốt chương là **đo trước, tối ưu sau, và kiểm tra lại chất lượng sau mỗi lần tối ưu**. Nhiều kỹ thuật "tăng tốc" không tăng tốc trong hoàn cảnh cụ thể của bạn, và một số làm giảm chất lượng theo cách mà các thước đo số học không phát hiện được. Các đoạn mã trong chương đã được chạy trên CPU; các con số thời gian in ra chỉ minh họa cho một lần chạy trên máy của người biên soạn và sẽ khác trên máy của bạn.

## 39.1 Bức tranh triển khai

### Mục tiêu triển khai

| Mục tiêu | Điểm mạnh | Ràng buộc | Công cụ thường dùng |
|---|---|---|---|
| Máy chủ GPU | Thông lượng cao, chạy được mô hình lớn (LM vài trăm triệu đến vài tỉ tham số) | Chi phí, độ trễ mạng, cần quản lý batching | PyTorch, TensorRT, ONNX Runtime (CUDA), máy chủ phục vụ LLM, Triton |
| Máy chủ CPU | Rẻ, dễ mở rộng, sẵn có | Chỉ phù hợp mô hình nhỏ/vừa (VITS, Matcha + vocoder nhẹ) | ONNX Runtime, OpenVINO, PyTorch |
| Di động / thiết bị biên | Không cần mạng, quyền riêng tư, độ trễ mạng bằng 0 | Bộ nhớ, pin, nhiệt, kích thước ứng dụng | ONNX Runtime Mobile, sherpa-onnx, ExecuTorch, Core ML |
| Trình duyệt | Không cần cài đặt, chạy phía người dùng | Hiệu năng WebAssembly/WebGPU, kích thước tải về | ONNX Runtime Web, các thư viện JavaScript chạy mô hình ONNX |

Lựa chọn mục tiêu quyết định gần như mọi thứ phía sau. Một hệ thống lai LM + flow matching cỡ 0,5 tỉ tham số ([Chương 34](../phan-8-mo-hinh-lon/ch34-he-thong-lai-hien-dai.md)) hầu như chỉ hợp lý trên GPU; một mô hình VITS ([Chương 27](../phan-7-e2e-bieu-cam/ch27-vits.md)) có thể chạy thời gian thực trên CPU của điện thoại.

### Đo trước, tối ưu sau: định luật Amdahl

Giả sử một đường ống TTS gồm các khâu với tỉ lệ thời gian $p_1, p_2, \ldots$ ($\sum p_i = 1$). Nếu ta tăng tốc khâu chiếm tỉ lệ $p$ lên $s$ lần, tốc độ tổng thể tăng

$$
S = \frac{1}{(1 - p) + p/s}.
$$

Đây là định luật Amdahl (Amdahl, 1967). Ví dụ minh họa: frontend chiếm 5% thời gian, mô hình âm học 25%, vocoder 70%. Tăng tốc vocoder 4 lần cho $S = 1/(0{,}30 + 0{,}175) \approx 2{,}1$ lần; tăng tốc mô hình âm học 10 lần chỉ cho $S = 1/(0{,}75 + 0{,}025) \approx 1{,}29$ lần. Kể cả khi tăng tốc vô hạn một khâu, $S$ không vượt quá $1/(1 - p)$. Bài học: hãy **đo phân bố thời gian** (bằng `torch.profiler`, một trình profile lấy mẫu như py-spy, hoặc Nsight Systems trên GPU) trước khi quyết định tối ưu ở đâu. Với TTS, kết quả thường bất ngờ: chuẩn hóa văn bản viết bằng Python thuần, sao chép dữ liệu giữa CPU và GPU, hay mã hóa âm thanh đầu ra có thể chiếm tỉ lệ đáng kể.

## 39.2 Xuất mô hình: torch.export, ONNX, TensorRT

### Vì sao cần xuất mô hình?

Chạy trực tiếp mã PyTorch ở chế độ eager rất tiện khi phát triển, nhưng khi triển khai có ba bất lợi: chi phí của trình thông dịch Python trên mỗi phép toán (đáng kể với mô hình nhỏ hoặc với bước giải mã tự hồi quy), không có cơ hội tối ưu toàn đồ thị (gộp phép toán, gấp hằng số), và phụ thuộc vào toàn bộ PyTorch, thứ không thể mang lên điện thoại hay trình duyệt. **Xuất mô hình** (*export*) là chụp lại đồ thị tính toán thành một dạng biểu diễn độc lập, để một môi trường chạy (*runtime*) chuyên dụng thực thi.

### Các con đường chính

- **TorchScript**: cơ chế cũ của PyTorch. Tính đến thời điểm biên soạn (2026), PyTorch khuyến nghị dùng `torch.export` cho các dự án mới; bạn vẫn sẽ gặp TorchScript trong mã cũ.
- **`torch.export`**: chụp toàn bộ đồ thị thành một `ExportedProgram`, không cho phép "ngắt đồ thị" (*graph break*). Trục động được khai báo tường minh bằng `torch.export.Dim`. Đây là nền tảng cho bộ xuất ONNX mới, ExecuTorch và biên dịch trước (AOT).
- **ONNX + ONNX Runtime**: ONNX là định dạng trao đổi mô hình mở; ONNX Runtime là môi trường chạy với nhiều *execution provider* (CPU, CUDA, TensorRT, DirectML, Core ML, NNAPI, WebAssembly, WebGPU…). Đây là con đường phổ biến nhất để chạy TTS trên CPU, di động và trình duyệt.
- **TensorRT**: trình biên dịch và môi trường chạy của NVIDIA, thường nhận đầu vào ONNX, tối ưu mạnh cho một dòng GPU cụ thể (fp16, int8, gộp nhân). Hai ràng buộc cần nhớ: bản biên dịch (*engine*) gắn với phiên bản TensorRT và kiến trúc GPU; trục động phải khai báo qua "hồ sơ tối ưu" (*optimization profile*) gồm kích thước nhỏ nhất, thường gặp nhất và lớn nhất.

### Thực hành: xuất một vocoder đồ chơi sang ONNX

Đoạn mã sau định nghĩa một vocoder đồ chơi theo tinh thần HiFi-GAN (tích chập, tích chập chuyển vị để tăng mẫu, một khối dư), xuất sang ONNX với trục thời gian động, rồi so sánh đầu ra của ONNX Runtime với PyTorch ở ba độ dài khác nhau, kể cả độ dài khác với lúc xuất.

```python
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

torch.manual_seed(0)
torch.set_num_threads(2)

class TinyVocoder(nn.Module):
    """Vocoder đồ chơi kiểu HiFi-GAN thu nhỏ: mel (B, F, T) -> dạng sóng (B, 1, T*hop)."""
    def __init__(self, n_mels=80, ch=128, rates=(8, 8)):
        super().__init__()
        self.hop = int(np.prod(rates))
        self.pre = nn.Conv1d(n_mels, ch, kernel_size=7, padding=3)
        layers, c = [], ch
        for r in rates:                        # mỗi tầng tăng mẫu r lần
            layers += [nn.LeakyReLU(0.1),
                       nn.ConvTranspose1d(c, c // 2, 2 * r, stride=r, padding=r // 2)]
            c //= 2
        self.ups = nn.Sequential(*layers)
        self.res = nn.Conv1d(c, c, kernel_size=3, padding=3, dilation=3)
        self.post = nn.Conv1d(c, 1, kernel_size=7, padding=3)

    def forward(self, mel):                    # mel: (B, F, T)
        h = self.ups(self.pre(mel))            # (B, C/4, T*hop)
        h = h + self.res(nn.functional.leaky_relu(h, 0.1))
        return torch.tanh(self.post(h))        # (B, 1, T*hop)

model = TinyVocoder().eval()
mel = torch.randn(1, 80, 50)                   # (B, F, T)
assert model(mel).shape == (1, 1, 50 * model.hop)

# Xuất ONNX với trục thời gian động (exporter dựa trên torch.export)
T = torch.export.Dim("T", min=4, max=8192)
onnx_prog = torch.onnx.export(model, (mel,), dynamo=True,
                              input_names=["mel"], output_names=["wav"],
                              dynamic_shapes={"mel": {2: T}})
onnx_prog.save("tiny_vocoder.onnx")

# Chạy bằng ONNX Runtime, so sánh với PyTorch ở nhiều độ dài
opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
sess = ort.InferenceSession("tiny_vocoder.onnx", opts,
                            providers=["CPUExecutionProvider"])
for t in (8, 50, 333):
    x = torch.randn(1, 80, t)
    with torch.inference_mode():
        ref = model(x).numpy()                 # (1, 1, t*hop)
    out = sess.run(["wav"], {"mel": x.numpy()})[0]
    err = np.abs(out - ref).max()
    print(f"T={t:4d}  out={out.shape}  max|diff|={err:.1e}")
    assert out.shape == ref.shape and err < 1e-4
```

Vài điểm cần giải thích:

- Trong các phiên bản PyTorch gần đây, `torch.onnx.export` mặc định dùng bộ xuất dựa trên `torch.export` (tham số `dynamo=True`); trục động được khai báo bằng `dynamic_shapes` với `Dim`. Các phiên bản cũ hơn dùng tham số `dynamic_axes`. API này thay đổi khá nhanh, hãy kiểm tra tài liệu của phiên bản bạn dùng.
- Với tích chập chuyển vị có `kernel_size = 2r`, `stride = r`, `padding = r/2` (với $r$ chẵn), độ dài đầu ra đúng bằng $r$ lần đầu vào: $(T-1)r - 2 \cdot \tfrac{r}{2} + 2r = rT$. Nhờ vậy dạng sóng luôn dài đúng $T \times$ `hop` mẫu, điều kiện để vocoder khớp với mel ([Chương 26](../phan-6-vocoder/ch26-diffusion-vocoder-va-so-sanh.md)).
- Sai số giữa hai môi trường chạy ở mức $10^{-7}$, tức sai số làm tròn số thực dấu phẩy động 32 bit. Nếu bạn thấy sai số cỡ $10^{-2}$ trở lên, gần như chắc chắn có lỗi xuất (một phép toán bị dịch sai, một nhánh điều kiện bị "đóng băng" theo giá trị lúc xuất).
- **Luôn kiểm thử ở nhiều độ dài**, kể cả độ dài rất ngắn và độ dài khác hẳn lúc xuất. Lỗi phổ biến nhất là một kích thước bị ghi cứng thành hằng số trong đồ thị.

### Những thành phần khó xuất trong mô hình TTS

Mô hình TTS có vài đặc điểm làm việc xuất khó hơn mô hình phân loại ảnh:

- **Kích thước phụ thuộc dữ liệu.** Bộ điều chỉnh độ dài (*length regulator*) của FastSpeech ([Chương 22](../phan-5-mo-hinh-am-hoc/ch22-fastspeech.md)) lặp mỗi véc-tơ âm vị theo trường độ *dự đoán*, nên độ dài đầu ra phụ thuộc vào giá trị chứ không chỉ kích thước đầu vào. Các bộ xuất hiện đại có hỗ trợ một phần cho kích thước như vậy, nhưng cách an toàn và phổ biến là **tách mô hình thành hai đồ thị**: (1) bộ mã hóa văn bản + bộ dự đoán trường độ; (2) bộ giải mã nhận đặc trưng đã được mở rộng. Phép mở rộng ở giữa viết lại bằng các phép toán có kích thước tường minh (ví dụ dùng chỉ số tích lũy và phép `gather`) hoặc thực hiện ngoài đồ thị.
- **Vòng lặp.** Giải mã tự hồi quy (Tacotron, LM token) và các bước giải ODE của flow matching là vòng lặp có số lần lặp thay đổi. Cách làm chuẩn là **xuất một bước** (một bước giải mã với KV cache là đầu vào/đầu ra tường minh, hoặc một lần đánh giá trường vận tốc) và viết vòng lặp ở mã chủ.
- **Tính ngẫu nhiên.** Nhiễu khởi đầu của flow matching hay nhiễu trong VITS nên được truyền vào như một **đầu vào** của đồ thị thay vì sinh bên trong. Như vậy đầu ra tất định với cùng đầu vào, dễ kiểm thử và dễ tái lập lỗi.
- **STFT và iSTFT.** Tính đến thời điểm biên soạn, ONNX có toán tử STFT (từ opset 17) nhưng chưa có toán tử iSTFT chuẩn. Các vocoder dạng Vocos/iSTFTNet thường cài iSTFT bằng nhân ma trận với cơ sở DFT ngược cộng tích chập chuyển vị để chồng cộng (*overlap-add*), hoặc để iSTFT ngoài đồ thị.
- **Weight normalization.** Các vocoder GAN dùng weight norm khi huấn luyện ([Chương 25](../phan-6-vocoder/ch25-gan-vocoder.md)). Hãy gỡ bỏ nó (gộp vào trọng số thường) trước khi xuất để đồ thị gọn và nhanh hơn.
- **Dropout "cố ý" khi suy luận.** Pre-net của Tacotron 2 giữ dropout cả khi suy luận ([Chương 20](../phan-5-mo-hinh-am-hoc/ch20-tacotron.md)). Gọi `model.eval()` trước khi xuất sẽ tắt nó và làm thay đổi hành vi mô hình; phải cài dropout tường minh, luôn bật, nếu muốn giữ nguyên.

## 39.3 Lượng tử hóa

### Hình thức hóa

**Lượng tử hóa** (*quantization*) biểu diễn số thực bằng số nguyên ít bit để giảm bộ nhớ và tận dụng các phép toán số nguyên nhanh. Lượng tử hóa affine với $b$ bit ánh xạ số thực $x$ thành số nguyên

$$
q = \operatorname{clamp}\Big(\operatorname{round}\big(x / s\big) + z,\; q_{\min},\; q_{\max}\Big),
\qquad
\hat{x} = s\,(q - z),
$$

với **thang đo** $s > 0$ và **điểm không** $z$. Dạng **đối xứng** dùng $z = 0$, $q \in [-(2^{b-1} - 1),\, 2^{b-1} - 1]$ và $s = \max \lvert x \rvert / (2^{b-1} - 1)$; đây là lựa chọn phổ biến cho trọng số.

**Sai số lượng tử hóa.** Bỏ qua phần bị cắt (*clamp*), sai số làm tròn $e = \hat{x} - x$ nằm trong $[-s/2, s/2]$. Với giả định chuẩn rằng $e$ phân phối đều trên khoảng đó, phương sai là

$$
\operatorname{Var}(e) = \frac{1}{s} \int_{-s/2}^{s/2} e^2\, de = \frac{s^2}{12}.
$$

Giả sử $x$ phân phối đều trên $[-A, A]$ (phương sai $A^2/3$) và ta dùng toàn bộ $2^b$ mức cho khoảng đó, tức $s = 2A / 2^b$. Tỉ số tín hiệu trên nhiễu lượng tử hóa là

$$
\text{SQNR} = 10 \log_{10} \frac{A^2/3}{s^2/12} = 10 \log_{10} \frac{A^2/3}{A^2 / (3 \cdot 2^{2b})} = 10 \log_{10} 2^{2b} \approx 6{,}02\, b \;\text{dB}.
$$

Mỗi bit thêm vào cho khoảng 6 dB: int8 cho khoảng 48 dB, int4 chỉ còn khoảng 24 dB. Công thức cũng cho thấy điểm yếu chí mạng của lượng tử hóa: $s$ tỉ lệ với **giá trị lớn nhất**. Nếu có một giá trị ngoại lai lớn gấp 10 lần phần còn lại, $s$ tăng 10 lần và SQNR của phần còn lại giảm 20 dB. Mọi kỹ thuật lượng tử hóa nâng cao về cơ bản đều là cách đối phó với giá trị ngoại lai.

### Theo tensor hay theo kênh

Dùng một thang đo cho cả tensor trọng số (*per-tensor*) thì kênh có biên độ nhỏ bị lượng tử hóa rất thô. Dùng một thang đo cho mỗi kênh đầu ra (*per-channel*) khắc phục điều này với chi phí lưu thêm một số thực mỗi kênh. Đoạn mã sau (tiếp nối khối mã trước, dùng lại `model`) kiểm chứng quy tắc 6 dB/bit, rồi so sánh hai cách lượng tử hóa trên vocoder đồ chơi. Để mô phỏng mạng đã huấn luyện, ta nhân trọng số từng kênh của lớp đầu với một hệ số $s_c$ và chia đầu vào tương ứng của lớp sau cho $s_c$: vì LeakyReLU thuần nhất dương ($\operatorname{LReLU}(s_c u) = s_c \operatorname{LReLU}(u)$ với $s_c > 0$), hàm của mạng không đổi, nhưng biên độ trọng số giữa các kênh chênh lệch tới khoảng 100 lần.

```python
import copy

def fake_quant(w, n_bits=8, per_channel=False, ch_axis=0):
    """Lượng tử hóa đối xứng rồi giải lượng tử ngay (fake-quant) để đo sai số."""
    qmax = 2 ** (n_bits - 1) - 1                          # 127 với int8
    if per_channel:                                       # một thang đo cho mỗi kênh ra
        dims = [d for d in range(w.dim()) if d != ch_axis]
        amax = w.abs().amax(dim=dims, keepdim=True)
    else:                                                 # một thang đo cho cả tensor
        amax = w.abs().max()
    scale = amax.clamp(min=1e-12) / qmax
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)  # số nguyên trong [-127, 127]
    return q * scale

def snr_db(ref, est):
    return 10 * torch.log10(ref.pow(2).sum() / (ref - est).pow(2).sum()).item()

# (a) Kiểm chứng quy tắc "khoảng 6 dB mỗi bit" trên dữ liệu phân phối đều
u = torch.rand(200_000) * 2 - 1
for b in (4, 6, 8):
    print(f"{b} bit: SQNR = {snr_db(u, fake_quant(u, b)):5.1f} dB "
          f"(6.02*b = {6.02 * b:5.1f})")

# (b) Lượng tử hóa trọng số của vocoder đồ chơi, đo SNR của dạng sóng đầu ra
torch.manual_seed(1)
base = copy.deepcopy(model)
with torch.no_grad():
    # Nhân kênh c của lớp đầu với s_c và chia đầu vào tương ứng của lớp sau cho s_c.
    # LeakyReLU thuần nhất dương nên hàm của mạng KHÔNG đổi, nhưng biên độ trọng số
    # giữa các kênh chênh lệch tới ~100 lần (hay gặp ở mạng đã huấn luyện).
    s = 10 ** torch.empty(base.pre.out_channels).uniform_(-2.0, 0.0)
    base.pre.weight.mul_(s[:, None, None]); base.pre.bias.mul_(s)
    base.ups[1].weight.div_(s[:, None, None])  # ConvTranspose1d: (C_in, C_out, k)
assert torch.allclose(base(torch.ones(1, 80, 9)), model(torch.ones(1, 80, 9)), atol=1e-5)

def quantize_pre(m, per_channel):
    """Chỉ lượng tử hóa lớp đầu (Conv1d, trọng số (C_out, C_in, k)) để cô lập hiệu ứng."""
    m = copy.deepcopy(m)
    with torch.no_grad():
        m.pre.weight.copy_(fake_quant(m.pre.weight, 8, per_channel, ch_axis=0))
    return m

x = torch.randn(1, 80, 200)
with torch.inference_mode():
    y_ref = base(x)
    for pc in (False, True):
        y_q = quantize_pre(base, pc)(x)
        print(f"int8 per-{'channel' if pc else 'tensor '}: SNR đầu ra = "
              f"{snr_db(y_ref, y_q):5.1f} dB")
```

Trong lần chạy của chúng tôi, phần (a) cho 22,9 / 35,8 / 48,1 dB với 4 / 6 / 8 bit, rất sát công thức $6{,}02\,b$ (sai lệch nhỏ ở 4 bit vì dạng đối xứng chỉ dùng $2^b - 1$ mức). Phần (b) cho SNR đầu ra khoảng 32 dB với per-tensor so với khoảng 62 dB với per-channel: chênh lệch tới 30 dB, chỉ vì một lớp. Lý do: các kênh biên độ nhỏ bị lượng tử hóa thô, rồi lớp sau *khuếch đại* chúng lên gấp $1/s_c$ lần.

Hai chi tiết đáng chú ý. Thứ nhất, trục kênh đầu ra của `ConvTranspose1d` là **trục 1** (trọng số có dạng `(C_in, C_out, k)`), khác với `Conv1d` (trục 0); nhầm trục là lỗi thường gặp khi tự viết lượng tử hóa. Thứ hai, nếu bạn lượng tử hóa cả lớp `ConvTranspose1d` phía sau, per-channel theo trục ra không cứu được, vì sự chênh lệch giờ nằm ở **trục vào** của lớp đó. Trường hợp này cần cân bằng lại biên độ giữa hai lớp liên tiếp trước khi lượng tử hóa (*cross-layer equalization*; Nagel và cộng sự, 2019), hoặc chuyển bớt độ khó từ activation sang trọng số như SmoothQuant (Xiao và cộng sự, 2023) làm với Transformer (Bài tập 4).

### Động, tĩnh, huấn luyện có lượng tử hóa

- **Lượng tử hóa động** (*dynamic*): trọng số lượng tử hóa sẵn; activation được lượng tử hóa lúc chạy, thang đo tính trên từng batch. Không cần dữ liệu hiệu chỉnh. Hiệu quả nhất với mô hình nặng về nhân ma trận (Transformer, LSTM) trên CPU.
- **Lượng tử hóa tĩnh** (*static*, hay sau huấn luyện, PTQ): thang đo của activation được ước lượng trước trên một tập **hiệu chỉnh** (*calibration*) vài trăm mẫu. Nhanh hơn lượng tử hóa động vì không phải tính thang đo lúc chạy, phù hợp mạng tích chập; nhưng chất lượng phụ thuộc tập hiệu chỉnh có đại diện cho dữ liệu thật hay không.
- **Huấn luyện có lượng tử hóa** (*quantization-aware training*, QAT; Jacob và cộng sự, 2018): chèn phép lượng tử hóa giả vào lượt truyền thẳng khi fine-tune, dùng straight-through estimator cho gradient. Tốn công nhất, chất lượng tốt nhất, cần thiết khi xuống 4 bit hoặc với các lớp nhạy cảm.
- **Số thực 16 bit**: fp16 và bf16 trên GPU thường gần như không mất chất lượng và là bước đầu tiên nên thử. fp16 có miền giá trị hẹp (giá trị lớn nhất khoảng 65 504), nên một số vocoder có activation lớn bị tràn thành `inf`/`NaN`; bf16 có cùng miền mũ với fp32 nên an toàn hơn về tràn số, đổi lại độ chính xác phần định trị thấp hơn.
- **Chỉ lượng tử hóa trọng số cho phần LM**: với LM vài trăm triệu đến vài tỉ tham số trong TTS dựa trên LLM, giải mã từng token bị giới hạn bởi **băng thông bộ nhớ** (mỗi bước phải đọc toàn bộ trọng số) chứ không phải tính toán. Lượng tử hóa trọng số xuống int8 hoặc int4 (các phương pháp như LLM.int8(), GPTQ, AWQ; Dettmers và cộng sự, 2022; Frantar và cộng sự, 2023; Lin và cộng sự, 2024) giảm lượng dữ liệu phải đọc, nên tăng tốc gần tỉ lệ thuận, trong khi activation vẫn giữ 16 bit.

Tính đến thời điểm biên soạn (2026), các API lượng tử hóa cũ trong `torch.ao.quantization` của PyTorch đã bị đánh dấu sắp loại bỏ và PyTorch khuyến nghị chuyển sang thư viện `torchao`; ONNX Runtime và TensorRT có công cụ lượng tử hóa riêng.

### Lượng tử hóa không tự động nhanh hơn

Đoạn mã sau (tiếp nối hai khối mã ở trên và khối đo RTF ở Mục 39.5) dùng công cụ lượng tử hóa động của ONNX Runtime trên vocoder đồ chơi.

```python
from onnxruntime.quantization import QuantType, quantize_dynamic

# Lượng tử hóa động: trọng số int8 lưu sẵn, activation lượng tử hóa lúc chạy
quantize_dynamic("tiny_vocoder.onnx", "tiny_vocoder_int8.onnx", weight_type=QuantType.QInt8)
sess8 = ort.InferenceSession("tiny_vocoder_int8.onnx", opts, providers=["CPUExecutionProvider"])

feed = {"mel": x.numpy()}                         # x, audio_sec: từ khối mã đo RTF ở Mục 39.5
a, b = sess.run(["wav"], feed)[0], sess8.run(["wav"], feed)[0]
print(f"SNR fp32 so với int8: {10 * np.log10((a**2).sum() / ((a - b)**2).sum()):.1f} dB")
for name, s in [("ORT fp32", sess), ("ORT int8 động", sess8)]:
    r = benchmark_rtf(lambda: s.run(["wav"], feed), audio_sec)
    print(f"{name:14s} p50={r['p50_ms']:6.1f} ms  RTF={r['rtf_p50']:.3f}")
```

Trên máy của chúng tôi, SNR giữa hai phiên bản khoảng 44 dB (chấp nhận được), nhưng phiên bản int8 **chậm hơn gấp đôi** phiên bản fp32. Nhìn vào đồ thị sau lượng tử hóa sẽ thấy lý do: các lớp `Conv` được thay bằng `ConvInteger` cộng các phép lượng tử hóa/giải lượng tử hóa activation chạy liên tục, trong khi nhân tích chập số nguyên trên CPU này không đủ nhanh để bù lại, và lớp `ConvTranspose` vẫn ở fp32. Đây là một bài học rất thực tế: **lượng tử hóa động được thiết kế cho mô hình nặng về nhân ma trận; với mạng tích chập, hãy thử lượng tử hóa tĩnh dạng QDQ với dữ liệu hiệu chỉnh, và luôn đo**. ONNX Runtime cũng khuyến nghị chạy bước tiền xử lý mô hình (`python -m onnxruntime.quantization.preprocess`) trước khi lượng tử hóa.

### Lượng tử hóa và chất lượng âm thanh

SNR là thước đo số học, không phải thước đo cảm nhận. Với TTS, một số lưu ý thực tế:

- **Vocoder nhạy cảm nhất.** Sai số ở vocoder biến thành nhiễu tần số cao, tiếng rè hoặc tiếng "kim loại" nghe rõ ngay cả khi SNR có vẻ cao. Lớp đầu và lớp cuối của vocoder thường nên giữ độ chính xác cao.
- **Các tham số nhỏ đặc biệt** như tham số của hàm kích hoạt Snake trong BigVGAN, các lớp chuẩn hóa, hay embedding người nói nên giữ ở số thực.
- **Bộ giải ODE cộng dồn sai số** qua nhiều bước; lượng tử hóa trường vận tốc có thể ảnh hưởng nhiều hơn một lần gọi mạng đơn lẻ.
- **Kiểm tra lại toàn bộ bộ đánh giá** sau lượng tử hóa: CER bằng ASR, độ giống giọng, và với tiếng Việt, bộ câu kiểm tra thanh điệu ([Chương 38](../phan-9-thuc-tien/ch38-danh-gia-tts.md)). Cuối cùng là một bài nghe so sánh AB với phiên bản gốc.
