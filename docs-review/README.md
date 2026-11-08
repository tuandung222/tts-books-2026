# Danh sách điểm cần thẩm định

Bản đầu tiên của giáo trình (11/2026) được viết bởi nhiều tác giả AI song song và **chưa qua thẩm định độc lập**. Dưới đây là các khẳng định mà chính tác giả đánh dấu là chưa chắc chắn (đã được rào đón trong văn bản), cùng các chương chưa qua bước tự rà soát cuối. Người thẩm định nên bắt đầu từ danh sách này, sau đó rà toàn bộ toán, mã và trích dẫn.

## ch31–ch33 (author a10) — claims the author was unsure about (all hedged in text; verify or soften)
- AudioLM specifics: 25 Hz semantic tokens, 12 RVQ levels, 4 coarse levels.
- VALL-E size (12 layers, d=1024).
- DAC: codebook dim 8, quantizer-dropout prob 0.5.
- Mimi: 12.5 Hz, 8×2048 codebooks, 1.1 kbps.
- WavTokenizer 40–75 tokens/s; X-codec 2 ~65K FSQ codes at 50 Hz; CosyVoice token rate ~25 Hz.
- Voicebox masking 70–100%.
- E2 TTS X1/X2 variants.
- F5-TTS sway default s = −1; pinyin for Chinese.
- MaskGCT semantic tokens via VQ-VAE.
- XTTS 16 languages, Vietnamese not officially supported.
- Orpheus/SNAC details.
- Venues: WavTokenizer (ICLR 2025), Codec-SUPERB (Findings of ACL).

## ch34, ch35, ch39 (author a11) — claims the author was unsure about (hedged in text; verify or soften)
- CosyVoice 2: FSQ codebook 3^8 = 6561 at 25 Hz; text:speech interleave 5:15; four training masks in flow matching.
- CosyVoice 3 details (DiffRO, multi-task tokenizer); SenseVoice as ASR backbone; HiFTNet vocoder.
- Seed-TTS eval details and PPO/REINFORCE vs DPO comparison.
- MiniMax-Speech Flow-VAE; first-author names cited for Fish-Speech and Spark-TTS; IndexTTS cited as team.
- Moshi: 160 ms theoretical / ~200 ms practical latency; models the user stream.
- GLM-4-Voice 12.5 Hz; Kokoro 82M Apache 2.0; VibeVoice 7.5 Hz.
- ONNX has no ISTFT operator (check: ONNX opset 17 added STFT; ISTFT status?); Piper 2025 repo move / license change / Vietnamese voices; sherpa-onnx supported-model list.

## ch36, ch37, ch38, ch40 (author a12) — claims the author was unsure about (verify or soften)
- Dataset figures: HiFi-TTS ≈290 h; MLS ≈50k h; Libriheavy license; Common Voice vi size; InfoRe and VLSP terms; Emilia-Large terms. (viVoice, PhoAudiobook were checked on HF/web by the author.)
- ch37 hyperparameters: FastSpeech 2 batch size; VITS config details.
- Legal (author used secondary sources only): Vietnam laws 91/2025, 134/2025, 131/2025, 116/2025; Decree 356/2025 replacing Decree 13/2023; EU "Digital Omnibus" effect on AI Act Art. 50; US NO FAKES Act status; California 2024 laws.
- XTTS-v2 license; MUSHRA post-screening rule; Voicebox SIM-o/SIM-r definitions; Seed-TTS-eval composition.
- Citations: François & Boëffard 2001; Luong & Vu 2016 venue; Kirkland 2023 (SSW12); Chu & Alwan 2009; Owen 2007.

## ch20–ch23 (author a07)
- FastSpeech 2 mel loss L1 (hedged "common implementations"); FastSpeech 2 pitch CWT details.
- DCA prior details; GMMv2 parameterisation.
- Glow-TTS temperature ≈0.33; Grad-TTS β0=0.05, β1=20, τ=1.5, ~2 s training crops.
- Matcha keeps prior loss, σ_min=1e-4, evaluated 2–10 steps.
- Tacotron MOS 3.82; Tacotron 2 4.53 vs 4.58 ground truth; Deep Voice 3 >2,000 speakers.
- Existence of a Vietnamese MFA model (hedged).

## ch01–ch03 (author a01)
- Tacotron 2 / HiFi-GAN / FastSpeech 2 optimizer settings (checked only via secondary sources).
- VITS optimizer (AdamW, betas 0.8/0.99, lr 2e-4, per-epoch exponential decay) — from memory.
- HiFi-GAN official init N(0, 0.01²); Tacotron paper's pre-net "bottleneck" remark.
- micrograd dated 2020; Maas et al. 2013 as ICML workshop.

## Not self-reviewed (authors were interrupted by a rate limit before their final self-review pass)
- ch04–ch06, ch07–ch09, ch10–ch12, ch17–ch19, ch24–ch26, ch27–ch30 (drafts complete but not self-reviewed).

## ch13–ch16 (author a05)
- "About 22 initial consonants" (Đoàn Thiện Thuật) and 16 nuclei (13 monophthongs + 3 diphthongs).
- Chao tone values and their variants.
- Details of Brunelle (2009); Brunelle, Hạ & Grice (2012); Đỗ, Trần & Boulakia (1998).
- ParallelWaveGAN recipe mel settings (fmin 80, fmax 7600, log10); Tacotron 2 paper mel settings (125 Hz–7.6 kHz, 50 ms frame / 12.5 ms hop).
- HiFi-GAN training code normalizes audio peak to 0.95; librosa pad_mode default changed to "constant".
- "quốc" vs "cuốc" claimed near-homophones.

## Dự án 1–2 + code/vn_frontend, code/fastspeech2 (author a13)
- Kirby (2011) JIPA 41(3), 381–392; vPhon citation; PhoWhisper venue; JETS author list (Lim et al., 2022).
- G2P phonology choices: anh/ach → /ɛ/, glottal stop for vowel-initial syllables, simplified Southern rules.
- MFA Vietnamese models (hedged); HiFi-GAN fine-tune flags (--fine_tuning, --input_mels_dir), ft_dataset mel layout (80, T).
- FastSpeech 2 paper hyperparameters (4 FFT blocks, 256 hidden, kernel 9, filter 1024).

## Dự án 3–5 + code/tts_service (author a14)
- Upstream CLI names/behaviours (F5-TTS, Piper, original VITS) as of Nov 2026: MASTER_PORT='80000', bucket boundaries, max_text_len=190 filter, F5 `.get(c, 0)` vocab lookup.
- Count of 19/73 lowercase Vietnamese letters missing from F5TTS_v1_Base vocab (hand-transcribed; re-check).
- Emilia "about 95k hours"; F5 "8 A100s, over 1M updates"; F5-TTS code license; piper1-gpl license; Seed-TTS-eval (WavLM-based SIM); RFC author lists.
