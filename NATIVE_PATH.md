# Native QNN Whisper Path (ARM64 Windows)

This document summarizes the new native accelerated path that removes several Python wheel constraints on Windows ARM64 (Surface Laptop 7 Snapdragon X Elite).

## Overview

Components:
- `rust_tokenizer` (pyo3): Loads `tokenizer.json`, provides `init_tokenizer`, `encode`, `decode`, `special_token_ids`.
- `rust_mel` (pyo3): Implements fast log-mel spectrogram (`pcm_to_mel`) using Rust FFT (`realfft`) + triangular mel filter bank.
- ONNX Runtime sessions (Encoder + Decoder iterative) with provider order: `QNNExecutionProvider` -> `CPUExecutionProvider`.
- Python orchestration script (next step: `transcribe_qnn_native.py`) performs greedy decode with stop on end-of-text token.

## Build

```
build_native.bat
```
Artifacts copied to `native/`.

Ensure Rust toolchain (aarch64-pc-windows-msvc) installed. Script auto-adds target.

## Tokenization
Load `tokenizer.json` from exported ONNX model directory. Minimal set of special tokens used:
- `<|startoftranscript|>` (BOS)
- `<|endoftext|>` (EOT)
- `<|transcribe|>` / `<|translate|>` depending on task
- `<|notimestamps|>` for disabling timestamps (optional)

## Mel Generation
`pcm_to_mel` expects:
- PCM mono float32 samples at 16 kHz (resample externally if needed)
- `n_fft` (default suggestion 400 or 1024) and `hop_length`
- Generates natural log mel energies (ln) per frame (basic; normalization refinements TBD)

## Decode Loop (planned)
1. Build initial prompt tokens (language + task + notimestamps) via tokenizer special IDs.
2. Iteratively append argmax(logits_last) until EOT or max length.
3. Decode token list (excluding BOS/EOT) to string.

Hooks reserved for future features: beam search, temperature sampling, timestamps, streaming chunking.

## Fallbacks
If native modules cannot import, Python path reverts to placeholder or PyTorch Whisper (if installed). Clear log messages indicate fallback reason.

## Next Steps
- Implement `transcribe_qnn_native.py` consuming native tokenizer + mel.
- Integrate into GUI auto-detect path (if `native/tokenizer_native.*` & `native/mel_native.*` present, prefer native pipeline).
- Add basic performance benchmark script comparing CPU vs QNN provider.

