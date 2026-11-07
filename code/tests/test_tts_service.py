"""Kiểm thử cho dịch vụ TTS streaming (Dự án 5).

Chạy: ``cd code && python -m pytest tests/test_tts_service.py -q``.
Không cần mô hình: mọi kiểm thử dùng ``ToneSynthesizer`` (tất định).
"""
from __future__ import annotations

import asyncio
import io
import json
import socket
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # để import được tts_service

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from tts_service import text as text_mod  # noqa: E402
from tts_service.audio import (apply_fade, float_to_pcm16, iter_frames,  # noqa: E402
                               pcm16_to_float, to_wav_bytes, wav_header)
from tts_service.engine import TTSEngine  # noqa: E402
from tts_service.server import ServiceConfig, create_app  # noqa: E402
from tts_service.synthesizer import (CallableSynthesizer, OnnxVitsSynthesizer,  # noqa: E402
                                     ToneSynthesizer, intersperse, syllable_tone)
from tts_service.text import (BASIC_FRONTEND, ChunkConfig, IncrementalChunker,  # noqa: E402
                              basic_split_sentences, chunk_text, split_long)

SR = 16000
LONG_TEXT = ("Xin chào các bạn. Hôm nay chúng ta xây dựng một dịch vụ tổng hợp tiếng nói, "
             "có khả năng truyền âm thanh theo luồng, đo độ trễ và chịu tải tốt. "
             "Bạn đã sẵn sàng chưa? Bắt đầu nào!")


def make_app(**kw):
    synth = kw.pop("synth", None) or ToneSynthesizer(sample_rate=SR)
    cfg = ServiceConfig(warmup=False, **kw)
    return create_app(synth, cfg, frontend=BASIC_FRONTEND)


def words(chunks: list[str]) -> list[str]:
    return " ".join(chunks).split()


# ----------------------------------------------------------------------------- âm thanh
def test_pcm_roundtrip_and_clipping():
    x = np.array([0.0, 0.5, -0.5, 1.5, -2.0, np.nan], dtype=np.float32)
    pcm = float_to_pcm16(x)
    assert len(pcm) == 2 * len(x)
    y = pcm16_to_float(pcm)
    assert np.allclose(y[:3], x[:3], atol=1 / 32767)
    assert y[3] == pytest.approx(1.0) and y[4] == pytest.approx(-1.0)  # bị clip, không tràn số
    assert y[5] == 0.0  # NaN -> im lặng


def test_wav_header_and_file_are_valid():
    audio = 0.1 * np.sin(np.linspace(0, 100, SR // 2)).astype(np.float32)
    with wave.open(io.BytesIO(to_wav_bytes(audio, SR))) as wf:
        assert wf.getframerate() == SR and wf.getnframes() == len(audio)
    assert len(wav_header(SR, 10)) == 44
    with wave.open(io.BytesIO(wav_header(SR, len(audio)) + float_to_pcm16(audio))) as wf:
        assert wf.getnframes() == len(audio)


def test_iter_frames_and_fade():
    pcm = bytes(range(10)) * 10  # 100 byte = 50 mẫu
    frames = list(iter_frames(pcm, 16))
    assert b"".join(frames) == pcm and all(len(f) == 32 for f in frames[:-1])
    x = apply_fade(np.ones(1000, dtype=np.float32), SR, fade_ms=5)
    assert x[0] == pytest.approx(0.0) and x[-1] == pytest.approx(0.0) and x[500] == 1.0


# ----------------------------------------------------------------------------- văn bản
def test_basic_split_sentences_handles_abbreviations():
    s = basic_split_sentences("Ông ở TP. Hồ Chí Minh. Bà ở Hà Nội! Còn bạn?")
    assert s == ["Ông ở TP. Hồ Chí Minh.", "Bà ở Hà Nội!", "Còn bạn?"]


def test_chunk_text_limits_and_preserves_words():
    cfg = ChunkConfig(first_max_chars=40, max_chars=100)
    chunks = chunk_text(LONG_TEXT, cfg, BASIC_FRONTEND)
    assert words(chunks) == LONG_TEXT.split()  # không mất, không đảo thứ tự từ
    for k, c in enumerate(chunks):
        assert 0 < len(c) <= cfg.limit(k)
    assert len(chunks[0]) <= 40  # đoạn đầu ngắn để TTFB thấp


def test_split_long_prefers_phrase_boundaries_and_never_splits_words():
    s = "một hai ba bốn, năm sáu bảy tám, chín mười"
    assert split_long(s, 16) == ["một hai ba bốn,", "năm sáu bảy tám,", "chín mười"]
    assert split_long("https://example.com/rat-dai", 5) == ["https://example.com/rat-dai"]


def test_incremental_chunker_matches_words_and_waits_for_whitespace():
    ch = IncrementalChunker(ChunkConfig(first_max_chars=40, max_chars=100))
    assert ch.push("Giá là 3.") == []          # có thể là "3.5": phải chờ
    out = ch.push("5 triệu đồng. Rẻ")          # giờ mới chắc câu đầu đã xong
    assert out == ["Giá là 3.5 triệu đồng."]
    pieces = out + ch.push(" quá, ") + ch.flush()
    assert words(pieces) == "Giá là 3.5 triệu đồng. Rẻ quá,".split()


def test_incremental_chunker_token_stream_equals_text():
    ch = IncrementalChunker(ChunkConfig(first_max_chars=30, max_chars=80))
    out: list[str] = []
    for i in range(0, len(LONG_TEXT), 3):  # mô phỏng LLM nhả 3 ký tự mỗi lần
        out += ch.push(LONG_TEXT[i:i + 3])
    out += ch.flush()
    assert words(out) == LONG_TEXT.split()
    assert all(len(c) <= 80 for c in out)


def test_frontend_fallback_when_vn_frontend_missing(monkeypatch):
    monkeypatch.setattr(text_mod, "load_vn_frontend", lambda: None)
    fe = text_mod.default_frontend()
    assert fe.name == "basic"
    assert fe.normalize("  Hoà  ") == "Hoà"  # NFD -> NFC, gộp khoảng trắng


def test_default_frontend_works_with_or_without_vn_frontend():
    fe = text_mod.default_frontend()
    chunks = chunk_text("Xin chào. Tôi là trợ lý ảo.", frontend=fe)
    assert chunks and all(isinstance(c, str) and c for c in chunks)


# ----------------------------------------------------------------------------- synthesizer
def test_syllable_tone_detection():
    assert [syllable_tone(s) for s in ["ma", "mà", "má", "mả", "mã", "mạ", "mặt", "người"]] == \
        ["ngang", "huyen", "sac", "hoi", "nga", "nang", "nang", "huyen"]


def _f0_track(x: np.ndarray, sr: int, win: int = 400) -> np.ndarray:
    """Ước lượng F0 thô bằng số lần cắt qua 0 trên từng cửa sổ (đủ cho âm hài sạch)."""
    f0 = []
    for s in range(0, len(x) - win, win):
        seg = x[s:s + win]
        f0.append(np.sum(np.abs(np.diff(np.signbit(seg).astype(int)))) * sr / (2 * win))
    return np.array(f0)


def test_tone_synthesizer_is_deterministic_and_follows_tones():
    synth = ToneSynthesizer(sample_rate=SR, amplitude=0.5)
    a, b = synth.synthesize("má mà"), synth.synthesize("má mà")
    assert a.dtype == np.float32 and np.array_equal(a, b)
    assert np.abs(a).max() <= 1.0
    sac, huyen = _f0_track(synth._syllable("má", 180.0, 1.0), SR), _f0_track(synth._syllable("mà", 180.0, 1.0), SR)
    assert sac[-1] > sac[0] * 1.15      # thanh sắc đi lên
    assert huyen[-1] < huyen[0] * 0.9   # thanh huyền đi xuống
    # nói nhanh gấp đôi -> ngắn đi khoảng một nửa
    assert len(synth.synthesize("má mà", speed=2.0)) == pytest.approx(len(a) / 2, rel=0.02)


# ----------------------------------------------------------------------------- HTTP
def test_health_and_full_wav():
    with TestClient(make_app()) as client:
        h = client.get("/health").json()
        assert h["status"] == "ok" and h["sample_rate"] == SR
        r = client.post("/v1/tts", json={"text": "Xin chào Việt Nam."})
        assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
        with wave.open(io.BytesIO(r.content)) as wf:
            n = wf.getnframes()
            assert wf.getframerate() == SR
        expected = len(ToneSynthesizer(sample_rate=SR).synthesize("Xin chào Việt Nam."))
        assert n == expected
        assert float(r.headers["x-tts-rtf"]) >= 0.0


def test_http_validation_errors():
    with TestClient(make_app(max_text_chars=50)) as client:
        assert client.post("/v1/tts", json={"text": ""}).status_code == 422
        assert client.post("/v1/tts", json={"text": "..."}).status_code == 422  # không có chữ
        assert client.post("/v1/tts", json={"text": "a" * 51}).status_code == 413
        assert client.post("/v1/tts", json={"text": "chào", "speed": 5}).status_code == 422


def test_http_stream_equals_full():
    with TestClient(make_app()) as client:
        full = client.post("/v1/tts", json={"text": LONG_TEXT, "format": "pcm"}).content
        with client.stream("POST", "/v1/tts/stream", json={"text": LONG_TEXT, "format": "pcm"}) as r:
            assert r.status_code == 200 and r.headers["x-sample-rate"] == str(SR)
            parts = list(r.iter_bytes())
        assert b"".join(parts) == full
        with client.stream("POST", "/v1/tts/stream", json={"text": LONG_TEXT}) as r:
            body = b"".join(r.iter_bytes())
        assert body[:4] == b"RIFF" and body[44:] == full


def test_metrics_endpoint_counts_requests():
    with TestClient(make_app()) as client:
        client.post("/v1/tts", json={"text": "Một hai ba."})
        text = client.get("/metrics").text
        assert 'tts_requests_total{endpoint="full",status="ok"} 1' in text
        assert "tts_ttfb_seconds_count 1" in text and "tts_active_streams 0" in text


# ----------------------------------------------------------------------------- WebSocket
def _collect(ws) -> tuple[list[dict], bytes]:
    """Đọc cho tới khi gặp 'end' / 'cancelled' / 'error'."""
    events, pcm = [], bytearray()
    while True:
        msg = ws.receive()
        if msg.get("bytes") is not None:
            pcm += msg["bytes"]
            continue
        ev = json.loads(msg["text"])
        events.append(ev)
        if ev["type"] in ("end", "cancelled", "error"):
            return events, bytes(pcm)


def test_websocket_stream_protocol_and_stats():
    app = make_app(chunk=ChunkConfig(first_max_chars=40, max_chars=100))
    with TestClient(app) as client:
        full = client.post("/v1/tts", json={"text": LONG_TEXT, "format": "pcm"}).content
        with client.websocket_connect("/v1/tts/ws") as ws:
            ready = ws.receive_json()
            assert ready["type"] == "ready" and ready["sample_rate"] == SR
            ws.send_json({"type": "synthesize", "text": LONG_TEXT, "request_id": "r1"})
            events, pcm = _collect(ws)
    kinds = [e["type"] for e in events]
    assert kinds[0] == "start" and kinds[-1] == "end" and kinds.count("chunk") >= 3
    assert pcm == full  # streaming và không streaming cho cùng âm thanh
    chunk_samples = sum(e["samples"] for e in events if e["type"] == "chunk")
    assert 2 * chunk_samples == len(pcm)
    stats = events[-1]["stats"]
    assert stats["chunks"] == kinds.count("chunk")
    assert 0 <= stats["ttfb_ms"] <= 1000 * stats["wall_seconds"]
    assert stats["rtf"] > 0 and stats["audio_seconds"] == pytest.approx(len(pcm) / 2 / SR, abs=1e-3)


def test_websocket_cancel_then_reuse_connection():
    slow = ToneSynthesizer(sample_rate=SR, simulated_rtf=0.3)  # mỗi đoạn tốn thời gian thật
    app = make_app(synth=slow, chunk=ChunkConfig(first_max_chars=20, max_chars=40))
    n_planned = len(chunk_text(LONG_TEXT, ChunkConfig(first_max_chars=20, max_chars=40), BASIC_FRONTEND))
    with TestClient(app) as client, client.websocket_connect("/v1/tts/ws") as ws:
        ws.receive_json()  # ready
        ws.send_json({"type": "synthesize", "text": LONG_TEXT})
        assert ws.receive_json()["type"] == "start"
        assert ws.receive_json()["type"] == "chunk"
        ws.send_json({"type": "cancel"})  # ngắt lời (barge-in)
        events, _ = _collect(ws)
        assert events[-1]["type"] == "cancelled"
        assert sum(e["type"] == "chunk" for e in events) + 1 < n_planned
        ws.send_json({"type": "synthesize", "text": "Còn đây là câu mới."})
        events, pcm = _collect(ws)
        assert events[-1]["type"] == "end" and len(pcm) > 0


def test_websocket_incremental_text_like_llm():
    app = make_app(chunk=ChunkConfig(first_max_chars=30, max_chars=80))
    with TestClient(app) as client:
        full = client.post("/v1/tts", json={"text": LONG_TEXT, "format": "pcm"}).content
        with client.websocket_connect("/v1/tts/ws") as ws:
            ws.receive_json()
            for i in range(0, len(LONG_TEXT), 7):
                ws.send_json({"type": "text", "text": LONG_TEXT[i:i + 7]})
            ws.send_json({"type": "flush"})
            events, pcm = _collect(ws)
    assert events[-1]["type"] == "end"
    spoken = " ".join(e["text"] for e in events if e["type"] == "chunk")
    assert spoken.split() == LONG_TEXT.split()
    # Cách chia đoạn có thể khác chế độ trọn văn bản, nên chỉ so độ dài gần đúng
    assert len(pcm) == pytest.approx(len(full), rel=0.15)


def test_websocket_errors_keep_connection_alive():
    with TestClient(make_app()) as client, client.websocket_connect("/v1/tts/ws") as ws:
        ws.receive_json()
        ws.send_text("không phải JSON")
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "synthesize", "text": "chào", "speed": 9})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "hello"})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "synthesize", "text": "Vẫn hoạt động."})
        events, pcm = _collect(ws)
        assert events[-1]["type"] == "end" and pcm


# ----------------------------------------------------------------------------- engine & adapter
def test_engine_respects_max_concurrency():
    lock, state = threading.Lock(), {"now": 0, "peak": 0}

    def fn(text, speaker=None, speed=1.0):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.02)
        with lock:
            state["now"] -= 1
        return np.zeros(160, dtype=np.float32)

    engine = TTSEngine(CallableSynthesizer(fn, SR), max_concurrency=2, frontend=BASIC_FRONTEND)

    async def one():
        return [c async for c in engine.stream(["a", "b", "c"])]

    async def main():
        return await asyncio.gather(*(one() for _ in range(6)))

    results = asyncio.run(main())
    engine.close()
    assert all(len(r) == 3 for r in results)
    assert state["peak"] <= 2


class _FakeInput:
    def __init__(self, name):
        self.name = name


class _FakeSession:
    """Giả lập onnxruntime.InferenceSession theo quy ước đầu vào kiểu Piper."""

    def __init__(self):
        self.feeds = None

    def get_inputs(self):
        return [_FakeInput(n) for n in ("input", "input_lengths", "scales", "sid")]

    def run(self, _outputs, feeds):
        self.feeds = feeds
        n = int(feeds["input_lengths"][0])
        length = int(256 * n * feeds["scales"][1])  # mỗi ID ~ 256 mẫu nhân length_scale
        return [np.zeros((1, 1, length), dtype=np.float32)]


def test_onnx_adapter_feeds():
    sess = _FakeSession()
    synth = OnnxVitsSynthesizer(session=sess, text_to_ids=lambda t: [5, 6, 7], add_blank=True,
                                speaker_ids={"lan": 3}, sample_rate=SR)
    audio = synth.synthesize("bất kỳ", speaker="lan", speed=2.0)
    assert sess.feeds["input"].tolist() == [intersperse([5, 6, 7])] == [[0, 5, 0, 6, 0, 7, 0]]
    assert sess.feeds["input"].dtype == np.int64 and sess.feeds["sid"].tolist() == [3]
    assert sess.feeds["scales"].tolist() == pytest.approx([0.667, 0.5, 0.8])
    assert audio.shape == (int(256 * 7 * 0.5),)


def test_onnx_adapter_with_real_onnx_graph(tmp_path):
    """Đồ thị ONNX nhỏ dựng bằng onnx.helper: output = Cast(input) * 0.001, hình dạng (1, 1, N)."""
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper
    nodes = [
        helper.make_node("Cast", ["input"], ["x"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["x", "k"], ["y"]),
        helper.make_node("Unsqueeze", ["y", "axes"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes, "toy_vits",
        [helper.make_tensor_value_info("input", TensorProto.INT64, [1, "N"]),
         helper.make_tensor_value_info("input_lengths", TensorProto.INT64, [1]),
         helper.make_tensor_value_info("scales", TensorProto.FLOAT, [3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, "N"])],
        initializer=[helper.make_tensor("k", TensorProto.FLOAT, [], [0.001]),
                     helper.make_tensor("axes", TensorProto.INT64, [1], [0])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "toy.onnx"
    onnx.save(model, str(path))
    synth = OnnxVitsSynthesizer(str(path), text_to_ids=lambda t: [ord(c) for c in t], sample_rate=SR)
    audio = synth.synthesize("abc")
    assert np.allclose(audio, np.array([97, 98, 99]) * 0.001)
    assert ort.get_device() in ("CPU", "GPU")


# ----------------------------------------------------------------------------- client & thử tải thật
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def live_server():
    """Chạy uvicorn thật trong một luồng nền để thử client.py và loadtest.py."""
    uvicorn = pytest.importorskip("uvicorn")
    port = _free_port()
    app = make_app(synth=ToneSynthesizer(sample_rate=SR, simulated_rtf=0.02))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        if time.time() > deadline:
            pytest.skip("Không khởi động được máy chủ thử")
        time.sleep(0.02)
    yield port
    server.should_exit = True
    thread.join(timeout=5)


def test_async_clients_and_load_test(live_server):
    from tts_service.client import synthesize_http_stream, synthesize_ws
    from tts_service.loadtest import run_load_test
    port = live_server
    ws_url, http_url = f"ws://127.0.0.1:{port}/v1/tts/ws", f"http://127.0.0.1:{port}"
    res_ws = asyncio.run(synthesize_ws(ws_url, LONG_TEXT))
    res_http = asyncio.run(synthesize_http_stream(http_url, LONG_TEXT))
    assert res_ws.pcm == res_http.pcm and res_ws.sample_rate == SR
    assert 0 < res_ws.ttfb_s <= res_ws.total_s
    assert res_ws.server_stats["chunks"] == res_ws.n_chunks >= 2
    report = asyncio.run(run_load_test(ws_url, concurrency=3, requests=6))
    s = report.summary()
    assert s["ok"] == 6 and s["errors"] == 0
    assert s["ttfb_ms_p50"] <= s["ttfb_ms_p95"]
    report = asyncio.run(run_load_test(ws_url, requests=4, rate=20.0))
    assert report.summary()["ok"] == 4
