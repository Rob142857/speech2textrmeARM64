#!/usr/bin/env python3
"""Native QNN Whisper Transcription (Rust tokenizer + mel + ONNX Runtime).

Greedy decode implementation:
1. Load tokenizer_native + mel_native from native/ (ensure in PYTHONPATH)
2. Build ONNX Runtime sessions (encoder / decoder_iter) with providers
   preference QNNExecutionProvider then CPUExecutionProvider
3. Generate mel from PCM (expects 16k mono WAV; minimal WAV loader)
4. Greedy loop until EOT or max tokens
5. Decode to text and print result

Limitations:
- No beam search / timestamps yet
- Assumes model export used convert_whisper_to_onnx.py script
"""
from __future__ import annotations
import argparse, wave, struct, sys, time
from pathlib import Path

import numpy as np

# Attempt to import native modules (expect built artifacts copied to native/)
NATIVE_DIR = Path(__file__).parent / 'native'
if NATIVE_DIR.exists() and str(NATIVE_DIR) not in sys.path:
    sys.path.insert(0, str(NATIVE_DIR))

try:
    import tokenizer_native  # type: ignore
    import mel_native  # type: ignore
except ImportError as e:
    print(f"❌ Native modules not found: {e}\nBuild with build_native.bat first.")
    sys.exit(1)

try:
    import onnxruntime as ort
except ImportError:
    print("❌ onnxruntime not installed")
    sys.exit(1)


def load_wav_mono_16k(path: Path) -> np.ndarray:
    with wave.open(path.as_posix(), 'rb') as w:
        sr = w.getframerate()
        if sr != 16000:
            print(f"⚠️ Sample rate {sr} != 16000; continuing (consider resampling externally)")
        ch = w.getnchannels()
        n = w.getnframes()
        sampwidth = w.getsampwidth()
        raw = w.readframes(n)
    if sampwidth == 2:
        fmt = '<' + 'h' * (len(raw)//2)
        data = np.array(struct.unpack(fmt, raw), dtype=np.float32) / 32768.0
    else:
        raise ValueError('Only 16-bit PCM supported')
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data.astype(np.float32)


def build_sessions(model_dir: Path, provider: str):
    enc = model_dir / ('encoder_int8.onnx' if (model_dir/'encoder_int8.onnx').exists() else 'encoder.onnx')
    dec = model_dir / ('decoder_iter_int8.onnx' if (model_dir/'decoder_iter_int8.onnx').exists() else 'decoder_iter.onnx')
    if not enc.exists() or not dec.exists():
        raise FileNotFoundError('Missing ONNX encoder/decoder files')
    providers = []
    if provider.upper() == 'QNN':
        providers.append('QNNExecutionProvider')
    providers.append('CPUExecutionProvider')
    so = ort.SessionOptions(); so.log_severity_level = 3
    t0 = time.time()
    enc_sess = ort.InferenceSession(enc.as_posix(), so, providers=providers)
    dec_sess = ort.InferenceSession(dec.as_posix(), so, providers=providers)
    load_time = time.time()-t0
    return enc_sess, dec_sess, providers, load_time


def get_initial_tokens(tokenizer_json: Path, language: str | None, task: str) -> list[int]:
    # Load special token ids via native interface (tokenizer file already loaded)
    specials = tokenizer_native.special_token_ids()
    # Fallback scanning for required tokens
    def find(tok: str):
        # specials keys are token content strings -> id
        return specials.get(tok) if hasattr(specials, 'get') else None
    start_id = find('<|startoftranscript|>')
    if start_id is None:
        raise RuntimeError('startoftranscript token missing')
    tokens = [start_id]
    if language:
        lang_tok = f'<|{language}|>'
        lid = find(lang_tok)
        if lid:
            tokens.append(lid)
    task_tok = '<|transcribe|>' if task == 'transcribe' else '<|translate|>'
    task_id = find(task_tok)
    if task_id:
        tokens.append(task_id)
    nt_id = find('<|notimestamps|>')
    if nt_id:
        tokens.append(nt_id)
    return tokens


def greedy_decode(enc_out: np.ndarray, dec_sess, tokens: list[int], eot_id: int | None, max_new: int) -> list[int]:
    for _ in range(max_new):
        inp = np.array([tokens], dtype=np.int64)
        logits = dec_sess.run(None, {'encoder_hidden_states': enc_out, 'decoder_input_ids': inp})[0]
        next_id = int(np.argmax(logits[0,-1]))
        tokens.append(next_id)
        if eot_id is not None and next_id == eot_id:
            break
    return tokens


def parse_args():
    ap = argparse.ArgumentParser(description='Native QNN Whisper Greedy Transcription')
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--audio', required=True)
    ap.add_argument('--language', default=None)
    ap.add_argument('--task', default='transcribe', choices=['transcribe','translate'])
    ap.add_argument('--provider', default='QNN', choices=['QNN','CPU'])
    ap.add_argument('--max-new', type=int, default=448)
    ap.add_argument('--n-fft', type=int, default=400)
    ap.add_argument('--hop-length', type=int, default=160)
    ap.add_argument('--n-mels', type=int, default=80)
    return ap.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    tok_path = model_dir / 'tokenizer.json'
    if not tok_path.exists():
        print('❌ tokenizer.json missing in model directory')
        return 1
    tokenizer_native.init_tokenizer(tok_path.as_posix())
    enc_sess, dec_sess, providers, load_time = build_sessions(model_dir, args.provider)
    print('Providers preference:', providers)
    print('Actual encoder providers:', enc_sess.get_providers())

    # Audio -> mel
    pcm = load_wav_mono_16k(Path(args.audio))
    mel = mel_native.pcm_to_mel(pcm.tolist(), 16000, args.n_fft, args.hop_length, args.n_mels)
    # shape (frames, 80) -> ONNX expects (1, frames, 80)
    mel_in = np.array(mel, dtype=np.float32)
    mel_in = mel_in.reshape(1, mel_in.shape[0], mel_in.shape[1])

    t_enc0 = time.time()
    enc_out = enc_sess.run(None, {'mel': mel_in})[0]
    t_enc = time.time() - t_enc0

    specials = tokenizer_native.special_token_ids()
    eot_id = specials.get('<|endoftext|>') if hasattr(specials,'get') else None
    tokens = get_initial_tokens(tok_path, args.language, args.task)
    t_dec0 = time.time()
    tokens = greedy_decode(enc_out, dec_sess, tokens, eot_id, args.max_new)
    t_dec = time.time() - t_dec0

    # Decode (drop BOS)
    text = tokenizer_native.decode(tokens)
    print('\n===== TRANSCRIPTION =====\n')
    print(text.strip())
    print('\n===== METRICS =====')
    print(f'Load time: {load_time:.2f}s  Encoder: {t_enc:.2f}s  Decode: {t_dec:.2f}s  Tokens: {len(tokens)}')
    if eot_id and tokens and tokens[-1] != eot_id:
        print('⚠️  Reached max tokens before EOT')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
