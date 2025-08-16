#!/usr/bin/env python3
"""Reusable native ONNX/QNN transcription backend.
Provides function native_transcribe for integration with GUI/CLI.
"""
from __future__ import annotations
from pathlib import Path
import time, wave, struct, sys
import numpy as np

# Inject native dir
NATIVE_DIR = Path(__file__).parent / 'native'
if NATIVE_DIR.exists() and str(NATIVE_DIR) not in sys.path:
    sys.path.insert(0, str(NATIVE_DIR))

try:
    import tokenizer_native  # type: ignore
    import mel_native  # type: ignore
    NATIVE_OK = True
except Exception:
    NATIVE_OK = False

try:
    import onnxruntime as ort  # type: ignore
except Exception:
    ort = None  # type: ignore


def _load_wav_16k(path: Path) -> np.ndarray:
    with wave.open(path.as_posix(), 'rb') as w:
        if w.getsampwidth() != 2:
            raise ValueError('Only 16-bit PCM supported')
        raw = w.readframes(w.getnframes())
        data = np.frombuffer(raw, dtype='<i2').astype('float32')/32768.0
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels()).mean(axis=1)
        if w.getframerate() != 16000:
            # Simple naive resample (nearest) to 16k fallback
            import math
            ratio = 16000 / w.getframerate()
            idx = (np.arange(int(len(data)*ratio))/ratio).astype(int)
            idx = np.clip(idx, 0, len(data)-1)
            data = data[idx]
        return data


def _build_sessions(model_dir: Path, prefer_qnn: bool = True):
    if ort is None:
        raise RuntimeError('onnxruntime not installed')
    enc = model_dir / ('encoder_int8.onnx' if (model_dir/'encoder_int8.onnx').exists() else 'encoder.onnx')
    dec = model_dir / ('decoder_iter_int8.onnx' if (model_dir/'decoder_iter_int8.onnx').exists() else 'decoder_iter.onnx')
    if not enc.exists() or not dec.exists():
        raise FileNotFoundError('encoder/decoder ONNX files missing')
    avail = ort.get_available_providers()
    providers = []
    if prefer_qnn and 'QNNExecutionProvider' in avail:
        providers.append('QNNExecutionProvider')
    providers.append('CPUExecutionProvider')
    so = ort.SessionOptions(); so.log_severity_level = 3
    t0 = time.time()
    enc_sess = ort.InferenceSession(enc.as_posix(), so, providers=providers)
    dec_sess = ort.InferenceSession(dec.as_posix(), so, providers=providers)
    return enc_sess, dec_sess, providers, time.time()-t0


def _initial_tokens(language: str | None, task: str):
    specials = tokenizer_native.special_token_ids()
    def find(tok: str):
        return specials.get(tok) if hasattr(specials, 'get') else None
    start = find('<|startoftranscript|>')
    if start is None:
        raise RuntimeError('startoftranscript token missing')
    toks = [start]
    if language:
        lid = find(f'<|{language}|>')
        if lid: toks.append(lid)
    task_tok = '<|transcribe|>' if task=='transcribe' else '<|translate|>'
    tid = find(task_tok)
    if tid: toks.append(tid)
    nt = find('<|notimestamps|>')
    if nt: toks.append(nt)
    eot = find('<|endoftext|>')
    return toks, eot


def native_transcribe(model_dir: str, audio_path: str, language: str | None = None, task: str = 'transcribe', max_new_tokens: int = 448, prefer_qnn: bool = True) -> dict:
    if not NATIVE_OK:
        raise RuntimeError('Native modules not available (build_native.bat)')
    model_dir_p = Path(model_dir)
    audio_p = Path(audio_path)
    tok_json = model_dir_p / 'tokenizer.json'
    if not tok_json.exists():
        raise FileNotFoundError('tokenizer.json missing in model directory')
    tokenizer_native.init_tokenizer(tok_json.as_posix())
    enc_sess, dec_sess, providers, load_time = _build_sessions(model_dir_p, prefer_qnn=prefer_qnn)
    pcm = _load_wav_16k(audio_p)
    mel = mel_native.pcm_to_mel(pcm.tolist(), 16000, 400, 160, 80)
    mel = np.array(mel, dtype=np.float32).reshape(1, -1, 80)
    t_enc0 = time.time(); enc_out = enc_sess.run(None, {'mel': mel})[0]; t_enc = time.time()-t_enc0
    toks, eot = _initial_tokens(language, task)
    t_dec0 = time.time()
    for _ in range(max_new_tokens):
        inp = np.array([toks], dtype=np.int64)
        logits = dec_sess.run(None, {'encoder_hidden_states': enc_out, 'decoder_input_ids': inp})[0]
        nxt = int(np.argmax(logits[0, -1]))
        toks.append(nxt)
        if eot is not None and nxt == eot:
            break
    t_dec = time.time()-t_dec0
    text = tokenizer_native.decode(toks)
    return {
        'text': text,
        'tokens': toks,
        'language': language or 'auto',
        'timings': {'load_s': load_time, 'encoder_s': t_enc, 'decode_s': t_dec},
        'providers': enc_sess.get_providers(),
        'npu_used': 'QNNExecutionProvider' in enc_sess.get_providers(),
        'eot_reached': (eot is None) or (toks and toks[-1]==eot)
    }
