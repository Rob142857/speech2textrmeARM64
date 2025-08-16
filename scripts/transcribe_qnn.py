#!/usr/bin/env python3
"""ONNX Runtime (QNN) Whisper inference demo.
NOTE: This currently performs a dummy greedy decode and prints token IDs.
Next iteration: integrate tokenizer to produce text & timestamps.
"""
from __future__ import annotations
import argparse, time
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True)
    p.add_argument('--audio', required=True)
    p.add_argument('--provider', default='QNN', choices=['QNN','CPU'])
    p.add_argument('--max-tokens', type=int, default=64)
    p.add_argument('--no-quant', action='store_true', help='Force use of float ONNX even if *_int8 present')
    return p.parse_args()

def load_mel(audio_path: Path):
    import soundfile as sf, numpy as np, librosa
    audio, sr = sf.read(audio_path.as_posix())
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Standard Whisper mel parameters (approximate quick variant)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    mel_db = librosa.power_to_db(mel).T  # [frames, 80]
    return mel_db.astype('float32')

def main():
    args = parse_args()
    import onnxruntime as ort, numpy as np
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print('❌ Model dir missing:', model_dir)
        return 1
    enc_name = 'encoder_int8.onnx' if (not args.no_quant and (model_dir/'encoder_int8.onnx').exists()) else 'encoder.onnx'
    dec_name = 'decoder_iter_int8.onnx' if (not args.no_quant and (model_dir/'decoder_iter_int8.onnx').exists()) else 'decoder_iter.onnx'
    enc_path = model_dir / enc_name
    dec_path = model_dir / dec_name
    for pth in (enc_path, dec_path):
        if not pth.exists():
            print('❌ Missing model file:', pth)
            return 1
    providers = []
    if args.provider == 'QNN':
        providers.append('QNNExecutionProvider')
    providers.append('CPUExecutionProvider')
    print('System providers:', ort.get_available_providers())
    print('Preference order:', providers)
    so = ort.SessionOptions()
    t_load = time.time()
    enc_sess = ort.InferenceSession(enc_path.as_posix(), so, providers=providers)
    dec_sess = ort.InferenceSession(dec_path.as_posix(), so, providers=providers)
    print(f'Model sessions ready in {time.time()-t_load:.2f}s')

    # Load / prep audio
    t_audio = time.time()
    mel = load_mel(Path(args.audio))
    mel_in = mel[None, ...]  # [1, frames, 80]
    print(f'Mel shape {mel_in.shape} prepared in {time.time()-t_audio:.2f}s')

    # Encoder
    t0 = time.time()
    enc_out = enc_sess.run(None, {'mel': mel_in})[0]
    print(f'Encoder output {enc_out.shape} in {time.time()-t0:.2f}s (provider={enc_sess.get_providers()[0]})')

    # Greedy decode placeholder (token ids only)
    start_token = 50257  # placeholder; real start token differs
    seq = [start_token]
    t_dec = time.time()
    for _ in range(args.max_tokens):
        inp = np.array([seq], dtype='int64')
        logits = dec_sess.run(None, {'encoder_hidden_states': enc_out, 'decoder_input_ids': inp})[0]
        nxt = int(np.argmax(logits[0, -1]))
        seq.append(nxt)
        if nxt == start_token:
            break
    print(f'Decode time {time.time()-t_dec:.2f}s provider={dec_sess.get_providers()[0]}')
    print('Generated token ids:', seq)
    print('NOTE: Implement text decoding (tokenizer) in next iteration.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
