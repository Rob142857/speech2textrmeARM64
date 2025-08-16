#!/usr/bin/env python3
"""Quick sanity test for native tokenizer + mel modules.
Run after build_native.bat and model export.
"""
from pathlib import Path
import sys, math
import numpy as np

NATIVE_DIR = Path(__file__).parent / 'native'
if NATIVE_DIR.exists() and str(NATIVE_DIR) not in sys.path:
    sys.path.insert(0, str(NATIVE_DIR))

try:
    import tokenizer_native, mel_native  # type: ignore
except ImportError as e:
    print('❌ Native imports failed:', e)
    raise SystemExit(1)

print('✅ Native modules imported')

# Synthetic tokenizer test (needs real tokenizer.json path passed as arg)
if len(sys.argv) < 2:
    print('Usage: python test_native_pipeline.py path/to/tokenizer.json')
    raise SystemExit(0)

tok_path = Path(sys.argv[1])
if not tok_path.exists():
    print('Tokenizer file missing')
    raise SystemExit(1)

tokenizer_native.init_tokenizer(tok_path.as_posix())
ids = tokenizer_native.encode('Hello world')
print('Encoded length:', len(ids))
print('Decoded sample:', tokenizer_native.decode(ids)[:40])

# Mel test: generate 1 second 440 Hz tone at 16k
sr = 16000
samples = np.sin(2*math.pi*440*np.arange(sr)/sr).astype('float32')
mel = mel_native.pcm_to_mel(samples.tolist(), sr, 400, 160, 80)
print('Mel shape:', mel.shape)
print('✅ Native pipeline basic test complete')
