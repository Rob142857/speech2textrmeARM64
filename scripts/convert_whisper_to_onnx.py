#!/usr/bin/env python3
"""Export Whisper (HF) to ONNX encoder/decoder graphs for QNN EP.
Prereqs: pip install torch transformers sentencepiece onnxruntime onnx onnxsim (optional) librosa soundfile
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='openai/whisper-large-v3', help='HuggingFace model id')
    p.add_argument('--output-dir', required=True, help='Destination directory')
    p.add_argument('--fp16', action='store_true', help='Attempt FP16 export (encoder hidden states still float)')
    p.add_argument('--simplify', action='store_true', help='Run onnx-simplifier')
    p.add_argument('--quantize-dynamic', action='store_true', help='Create *_int8.onnx copies')
    return p.parse_args()

def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except Exception as e:
        print('❌ Missing deps:', e)
        print('Install: pip install torch transformers sentencepiece')
        return 1

    print('Loading model:', args.model)
    t0 = time.time()
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    processor = WhisperProcessor.from_pretrained(args.model)
    print(f'Loaded in {time.time()-t0:.1f}s')
    model.eval()

    if args.fp16:
        try:
            model.to(dtype=torch.float16)
            print('Using FP16 weights where possible')
        except Exception as e:
            print('FP16 conversion failed:', e)

    # Dummy inputs
    import torch
    mel = torch.randn(1, 3000, 80)
    start_id = processor.tokenizer.convert_tokens_to_ids('<|startoftranscript|>')
    decoder_input_ids = torch.tensor([[start_id]])

    # Encoder export
    enc_path = out / 'encoder.onnx'
    print('Exporting encoder ->', enc_path)
    torch.onnx.export(
        model.model.encoder,
        mel,
        enc_path.as_posix(),
        input_names=['mel'],
        output_names=['encoder_hidden_states'],
        dynamic_axes={'mel': {0:'batch',1:'frames'}, 'encoder_hidden_states': {0:'batch',1:'frames'}},
        opset_version=17,
    )

    # Decoder step wrapper
    class DecoderStep(torch.nn.Module):
        def __init__(self, base):
            super().__init__(); self.base = base
        def forward(self, encoder_hidden_states, decoder_input_ids):
            out = self.base.model.decoder(decoder_input_ids, encoder_hidden_states=encoder_hidden_states)
            logits = self.base.lm_head(out.last_hidden_state)
            return logits

    step = DecoderStep(model)
    dec_path = out / 'decoder_iter.onnx'
    print('Exporting decoder step ->', dec_path)
    torch.onnx.export(
        step,
        (mel, decoder_input_ids),
        dec_path.as_posix(),
        input_names=['encoder_hidden_states','decoder_input_ids'],
        output_names=['logits'],
        dynamic_axes={'encoder_hidden_states': {0:'batch',1:'frames'}, 'decoder_input_ids': {0:'batch',1:'seq'}, 'logits': {0:'batch',1:'seq'}},
        opset_version=17,
    )

    # Save tokenizer + config
    if hasattr(processor.tokenizer, '_tokenizer'):
        (out/'tokenizer.json').write_text(processor.tokenizer._tokenizer.to_str())
    (out/'config.json').write_text(json.dumps(model.config.to_dict(), indent=2))
    print('Saved tokenizer.json & config.json')

    if args.simplify:
        try:
            import onnx, onnxsim
            for f in ['encoder.onnx','decoder_iter.onnx']:
                p = out/f; m = onnx.load(p.as_posix())
                sm, ok = onnxsim.simplify(m)
                if ok:
                    onnx.save(sm, p.as_posix())
                    print('Simplified', f)
        except Exception as e:
            print('Simplify skipped:', e)

    if args.quantize_dynamic:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            for f in ['encoder.onnx','decoder_iter.onnx']:
                src = out/f; dst = out / f.replace('.onnx','_int8.onnx')
                quantize_dynamic(src.as_posix(), dst.as_posix(), weight_type=QuantType.QInt8)
                print('Quantized', f)
        except Exception as e:
            print('Quantization skipped:', e)

    print('✅ DONE ->', out)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
