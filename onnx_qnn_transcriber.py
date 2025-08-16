#!/usr/bin/env python3
"""ONNX Runtime QNN Whisper minimal greedy transcription.

This module uses ONNX-exported Whisper encoder / iterative decoder graphs
produced by `scripts/convert_whisper_to_onnx.py` and runs them with the
Qualcomm NPU (QNNExecutionProvider) when available, falling back to CPU.

It purposely keeps the implementation lightweight: a simple greedy decode
loop with the Hugging Face `WhisperProcessor` handling feature extraction
and token decoding. Beam search, temperature sampling, and timestamps are
not implemented here (future extension) – the goal is to provide a fast
hardware-accelerated path for basic high‑quality text output that can then
be post‑processed (punctuation / paragraphing handled elsewhere).

Requirements:
  transformers, sentencepiece, soundfile, numpy, onnxruntime-qnn (for NPU)

Export reminder:
  python scripts/convert_whisper_to_onnx.py --model openai/whisper-large-v3 \
         --output-dir models/whisper_large_v3_onnx --simplify --quantize-dynamic
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import json

import numpy as np

try:
    import onnxruntime as ort  # type: ignore
except ImportError:  # pragma: no cover
    ort = None  # type: ignore

try:
    from transformers import WhisperProcessor
except ImportError:  # pragma: no cover
    WhisperProcessor = None  # type: ignore

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None  # type: ignore


class OnnxWhisperQNN:
    """Greedy ONNX Whisper inference with optional QNN acceleration."""

    def __init__(
        self,
        model_dir: str,
        model_id: str = "openai/whisper-large-v3",
        provider_preference: Optional[List[str]] = None,
        max_new_tokens: int = 448,
        language: Optional[str] = None,
        task: str = "transcribe",
        use_int8: bool = True,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.language = language
        self.task = task
        self.use_int8 = use_int8
        self.initialized = False
        self._enc_session = None
        self._dec_session = None
        self.providers_used: List[str] = []
        self.processor: Optional[WhisperProcessor] = None
        self.special_token_ids: Dict[str, int] = {}
        self._provider_preference = provider_preference or ["QNNExecutionProvider", "CPUExecutionProvider"]

        self._init()  # Eager init so failures surface early

    # ---------------------- Internal helpers ----------------------
    def _select_model_files(self) -> Dict[str, Path]:
        if not self.model_dir.exists():
            raise FileNotFoundError(f"ONNX model directory not found: {self.model_dir}")
        def pick(name: str) -> Path:
            return self.model_dir / name
        if self.use_int8 and pick("encoder_int8.onnx").exists():
            enc = pick("encoder_int8.onnx")
        else:
            enc = pick("encoder.onnx")
        if self.use_int8 and pick("decoder_iter_int8.onnx").exists():
            dec = pick("decoder_iter_int8.onnx")
        else:
            dec = pick("decoder_iter.onnx")
        if not enc.exists() or not dec.exists():
            raise FileNotFoundError("Missing encoder/decoder ONNX files – run export script first.")
        return {"encoder": enc, "decoder": dec}

    def _available_providers(self) -> List[str]:
        if ort is None:
            return []
        try:
            return ort.get_available_providers()
        except Exception:
            return []

    def _resolve_providers(self) -> List[str]:
        avail = self._available_providers()
        chosen = [p for p in self._provider_preference if p in avail]
        if not chosen:
            chosen = ["CPUExecutionProvider"]
        return chosen

    def _init_processor(self) -> None:
        if WhisperProcessor is None:
            raise RuntimeError("transformers / WhisperProcessor not installed")
        self.processor = WhisperProcessor.from_pretrained(self.model_id)
        # Common special ids
        tok = self.processor.tokenizer
        specials = [
            "<|startoftranscript|>", "<|endoftext|>", "<|translate|>",
            "<|transcribe|>", "<|notimestamps|>"
        ]
        for s in specials:
            if s in tok.get_vocab():
                self.special_token_ids[s] = tok.convert_tokens_to_ids(s)
        # Language tokens like <|en|>
        if self.language:
            lang_token = f"<|{self.language}|>"
            if lang_token in tok.get_vocab():
                self.special_token_ids[lang_token] = tok.convert_tokens_to_ids(lang_token)

    def _init_sessions(self) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime not installed")
        files = self._select_model_files()
        providers = self._resolve_providers()
        so = ort.SessionOptions()
        so.log_severity_level = 3
        t0 = time.time()
        self._enc_session = ort.InferenceSession(files["encoder"].as_posix(), so, providers=providers)
        self._dec_session = ort.InferenceSession(files["decoder"].as_posix(), so, providers=providers)
        self.providers_used = providers
        self._load_time = time.time() - t0

    def _init(self) -> None:
        self._init_processor()
        self._init_sessions()
        self.initialized = True

    # ---------------------- Public API ----------------------
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        if not self.initialized:
            raise RuntimeError("Model not initialized")
        if sf is None:
            raise RuntimeError("soundfile not installed")
        t_start = time.time()
        # Load audio
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        # Feature extraction (80 x 3000 typical) using processor for consistency
        feats = self.processor.feature_extractor(audio, sampling_rate=sr, return_tensors="np").input_features
        # HF returns shape (1, 80, 3000); export expects (1, 3000, 80)
        if feats.shape[1] == 80:  # transpose to frames-first
            feats_t = np.transpose(feats, (0, 2, 1)).astype("float32")
        else:
            feats_t = feats.astype("float32")
        # Encoder
        t_enc0 = time.time()
        enc_out = self._enc_session.run(None, {"mel": feats_t})[0]
        t_enc = time.time() - t_enc0
        # Build initial decoder prompt ids
        forced = self.processor.get_decoder_prompt_ids(language=self.language, task=self.task)
        # forced is list of (index, token_id)
        seq = [tid for _, tid in forced]
        # Append no timestamps token if available
        nots = self.special_token_ids.get("<|notimestamps|>")
        if nots and nots not in seq:
            seq.append(nots)
        end_id = self.special_token_ids.get("<|endoftext|>")
        # Greedy loop
        t_dec0 = time.time()
        for _ in range(self.max_new_tokens):
            inp = np.array([seq], dtype=np.int64)
            logits = self._dec_session.run(None, {
                "encoder_hidden_states": enc_out,
                "decoder_input_ids": inp
            })[0]
            next_id = int(np.argmax(logits[0, -1]))
            seq.append(next_id)
            if end_id is not None and next_id == end_id:
                break
        t_dec = time.time() - t_dec0
        # Decode (skip special tokens for clean text)
        text = self.processor.tokenizer.decode(seq, skip_special_tokens=True)
        total_time = time.time() - t_start
        return {
            "text": text,
            "tokens": seq,
            "language": self.language or "auto",
            "timings": {
                "encoder_s": t_enc,
                "decode_s": t_dec,
                "total_s": total_time,
                "load_s": getattr(self, "_load_time", None)
            },
            "npu_providers": self.providers_used,
            "model_dir": str(self.model_dir),
            "onnx": True,
            "npu_used": any(p.startswith("QNN") for p in self.providers_used),
        }

    def info(self) -> Dict[str, Any]:
        return {
            "initialized": self.initialized,
            "model_dir": str(self.model_dir),
            "model_id": self.model_id,
            "providers": self.providers_used,
            "max_new_tokens": self.max_new_tokens,
            "language": self.language,
            "task": self.task,
        }


if __name__ == "__main__":  # Simple manual test
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--language", default="en")
    ap.add_argument("--max-new", type=int, default=128)
    args = ap.parse_args()
    tr = OnnxWhisperQNN(args.model_dir, language=args.language, max_new_tokens=args.max_new)
    out = tr.transcribe(args.audio)
    print(json.dumps({k: v for k, v in out.items() if k != "tokens"}, indent=2))
    print("Tokens:", out["tokens"][:50], "...")
