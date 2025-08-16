#!/usr/bin/env python3
"""Benchmark native ONNX/QNN vs CPU for a short audio file.
Usage: python benchmark_native.py --model-dir models/whisper_large_v3_onnx --audio sample.wav
"""
from __future__ import annotations
import argparse, time, sys, json, statistics
from pathlib import Path
import numpy as np

NATIVE_DIR = Path(__file__).parent / 'native'
if NATIVE_DIR.exists() and str(NATIVE_DIR) not in sys.path:
    sys.path.insert(0, str(NATIVE_DIR))

try:
    import tokenizer_native, mel_native  # type: ignore
except Exception as e:
    print('Native modules missing:', e)
    raise SystemExit(1)

try:
    import onnxruntime as ort
except ImportError:
    print('onnxruntime missing')
    raise SystemExit(1)

import wave, struct

def load_pcm(path: Path):
    with wave.open(path.as_posix(), 'rb') as w:
        if w.getsampwidth()!=2: raise ValueError('expect 16-bit')
        raw = w.readframes(w.getnframes())
        data = np.frombuffer(raw, dtype='<i2').astype('float32')/32768.0
        if w.getnchannels()>1: data = data.reshape(-1,w.getnchannels()).mean(axis=1)
        return data

def run_pass(model_dir: Path, audio: Path, provider: str):
    importer = lambda name: model_dir / name
    enc = importer('encoder.onnx'); dec = importer('decoder_iter.onnx')
    if not enc.exists() or not dec.exists():
        raise FileNotFoundError('encoder/decoder missing')
    providers = [provider, 'CPUExecutionProvider'] if provider=='QNNExecutionProvider' else ['CPUExecutionProvider']
    so = ort.SessionOptions(); so.log_severity_level=3
    enc_sess = ort.InferenceSession(enc.as_posix(), so, providers=providers)
    dec_sess = ort.InferenceSession(dec.as_posix(), so, providers=providers)
    pcm = load_pcm(audio)
    mel = mel_native.pcm_to_mel(pcm.tolist(), 16000, 400, 160, 80)
    mel = np.array(mel, dtype=np.float32).reshape(1, -1, 80)
    t0 = time.time(); enc_out = enc_sess.run(None, {'mel': mel})[0]; t_enc = time.time()-t0
    toks = tokenizer_native.encode('test')  # warm decode path
    specials = tokenizer_native.special_token_ids()
    start = specials.get('<|startoftranscript|>')
    eot = specials.get('<|endoftext|>')
    seq = [start] if start is not None else []
    for _ in range(32):
        inp = np.array([seq], dtype=np.int64)
        logits = dec_sess.run(None, {'encoder_hidden_states': enc_out, 'decoder_input_ids': inp})[0]
        nxt = int(np.argmax(logits[0,-1])); seq.append(nxt)
        if eot and nxt==eot: break
    return {'enc_s': t_enc, 'tokens': len(seq), 'provider': enc_sess.get_providers()[0]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--audio', required=True)
    args = ap.parse_args()
    tokenizer_native.init_tokenizer(str(Path(args.model_dir)/'tokenizer.json'))
    results = {}
    for prov in ['CPUExecutionProvider','QNNExecutionProvider']:
        times = []
        for _ in range(3):
            r = run_pass(Path(args.model_dir), Path(args.audio), prov)
            times.append(r['enc_s'])
        results[prov] = {'encoder_mean_s': statistics.mean(times), 'encoder_min_s': min(times)}
    print(json.dumps(results, indent=2))

if __name__=='__main__':
    main()
