## Snapdragon X Elite NPU Whisper Acceleration (QNN Execution Provider)

This guide shows how to run Whisper (or smaller / distilled variants) locally on the Surface Laptop 7 (Qualcomm Snapdragon X Elite) using the Qualcomm NPU through ONNX Runtime's QNN Execution Provider (EP).

### Acceleration Paths Overview

| Path | Difficulty | Accuracy | Performance | Notes |
|------|------------|----------|-------------|-------|
| A. ONNX Runtime + QNN EP (custom build) | High | Full Whisper | Highest | Requires building ORT with `--use_qnn` |
| B. Distil / Smaller Whisper + QNN | Medium | Slightly lower | Faster | Same pipeline, smaller weights |
| C. CPU (current fallback) | Low | Full | Slow | No build needed |
| D. DirectML fallback | Low | Full | Moderate | GPU/CPU; not NPU |

We provide scaffolding for Path A now; B is a smaller change (just different model id).

---
### 1. Prerequisites
1. Visual Studio 2022 Build Tools (Administrator):
   * Workload: Desktop development with C++
   * Components: MSVC v143 (ARM64), CMake, Ninja, Windows 11 SDK
2. Git (ARM64 build)
3. Python 3.11 ARM64 (already in project)
4. Rust (for tokenizers / tiktoken if later needed)
5. (Optional) Qualcomm QNN runtime DLLs (if separately distributed) placed on PATH or next to python.exe

If QNN DLLs are missing you can still build ONNX Runtime; provider may not initialize and will fall back to CPU.

---
### 2. Build ONNX Runtime with QNN EP
```pwsh
git clone --recursive https://github.com/microsoft/onnxruntime.git
cd onnxruntime
git checkout v1.19.0   # or a newer tag supporting QNN EP
./build.bat --config Release --arm64 --build_shared_lib `
  --parallel --enable_reduced_ops --skip_tests --use_qnn `
  --cmake_generator "Ninja" --cmake_extra_defines CMAKE_SYSTEM_VERSION=10.0.22621.0
```
Result wheel (example path):
```
onnxruntime\build\Windows\Release\Release\onnxruntime_py_wheel\onnxruntime-*-cp311-cp311-win_arm64.whl
```
Install:
```pwsh
pip install path\to\onnxruntime-*-win_arm64.whl
```
Verify provider availability:
```pwsh
python - <<'PY'
import onnxruntime as ort; print(ort.get_available_providers())
PY
```
Expect to see: `["QNNExecutionProvider", "CPUExecutionProvider"]`

If QNN missing: ensure build flag `--use_qnn` was applied and QNN runtime libraries are reachable.

---
### 3. Export Whisper / Distil-Whisper to ONNX
```pwsh
python scripts/convert_whisper_to_onnx.py \
  --model openai/whisper-large-v3 \
  --output-dir models/whisper_large_v3_onnx
```
Generated files:
```
encoder.onnx
decoder_iter.onnx
tokenizer.json
config.json
```
Optional dynamic quantization (reduces size / may improve NPU throughput):
```pwsh
python -m onnxruntime.quantization.quantize_dynamic \
  --model_input encoder.onnx --model_output encoder_int8.onnx --op_types_to_quantize MatMul
python -m onnxruntime.quantization.quantize_dynamic \
  --model_input decoder_iter.onnx --model_output decoder_iter_int8.onnx --op_types_to_quantize MatMul
```

Use a smaller model (example): `--model openai/whisper-small` or `--model distil-whisper/distil-large-v3` (Hugging Face id) for faster experiments.

---
### 4. Run Demo (Greedy Decode)
```pwsh
python scripts/transcribe_qnn.py --model-dir models/whisper_large_v3_onnx --audio sample.wav --provider QNN
```
The script will fall back to CPU automatically if QNN cannot initialise.

---
### 5. Planned GUI Integration
1. Auto-detect an ONNX model directory containing `encoder.onnx` & `decoder_iter.onnx`.
2. Show actual provider used (QNN vs CPU) in status bar.
3. Option to select quantized vs float model at runtime.
4. Real token -> text decoding with timestamps (current demo only outputs token ids placeholder).

---
### 6. Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| QNN provider absent | Build flag missing or DLLs not found | Rebuild with `--use_qnn`, place QNN libs on PATH |
| Build failure (compiler) | Missing MSVC ARM64 toolchain | Re-run VS Installer; ensure v143 + CMake + Ninja |
| Very slow decoding | Large model, no quantization | Use INT8 or smaller model (small / distil) |
| Out-of-memory | Large-v3 model on limited RAM | Switch to medium / small variant |
| Incorrect tokens | Demo greedy loop placeholder | Implement full decoding pipeline (planned) |

---
### 7. Roadmap
* Proper tokenizer-driven decoding (text output)
* Streaming / chunked audio processing
* Provider fallback chain: QNN -> DirectML -> CPU
* FP16 export path where beneficial
* Batch multi-file transcription

---
### 8. References
* QNN EP: https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
* Whisper: https://github.com/openai/whisper
* Distil-Whisper: https://huggingface.co/distil-whisper
* ONNX Runtime: https://github.com/microsoft/onnxruntime
