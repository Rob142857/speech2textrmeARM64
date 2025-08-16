#!/usr/bin/env python3
"""
Manual Test Script for ARM64 Whisper Transcription
Quick verification that all components work correctly
"""

import os
import sys
from pathlib import Path

def main():
    """Run manual tests for the transcription system."""
    
    print("🧪 ARM64 Whisper Transcription - Manual Test")
    print("=" * 50)
    print()
    
    # Test 1: Check imports
    print("1️⃣ Testing imports...")
    try:
        import whisper
        print("   ✅ OpenAI Whisper imported successfully")
        
        import torch
        print("   ✅ PyTorch imported successfully")
        
        from audio_processor import AudioProcessor
        print("   ✅ Audio Processor imported successfully")
        
        from whisper_npu import WhisperNPU
        print("   ✅ NPU module imported successfully")
        
        from document_generator import DocumentGenerator
        print("   ✅ Document Generator imported successfully")
        
        from transcribe import ARM64WhisperTranscriber
        print("   ✅ Main Transcriber imported successfully")
        
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    print()
    
    # Test 2: Check system info
    print("2️⃣ System Information...")
    import platform
    print(f"   Platform: {platform.system()} {platform.machine()}")
    print(f"   Python: {platform.python_version()}")
    
    # Check if ARM64
    if platform.machine() == 'ARM64':
        print("   ✅ Running on ARM64 architecture")
    else:
        print(f"   ⚠️  Not ARM64 (current: {platform.machine()})")
    
    print()
    
    # Test 3: NPU Detection
    print("3️⃣ NPU Detection...")
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"   Available providers: {providers}")
        
        if "QNNExecutionProvider" in providers:
            print("   ✅ NPU (QNN Provider) available")
        else:
            print("   ⚠️  NPU not available - will use CPU")
    except ImportError:
        print("   ❌ ONNX Runtime not available")
    
    print()
    
    # Test 4: Audio Processor
    print("4️⃣ Audio Processor...")
    try:
        processor = AudioProcessor()
        print("   ✅ Audio Processor initialized")
        print(f"   Supported formats: {len(processor.supported_formats)} formats")
        print(f"   Audio: {', '.join(sorted(processor.supported_audio))}")
        print(f"   Video: {', '.join(sorted(processor.supported_video))}")
    except Exception as e:
        print(f"   ❌ Audio Processor failed: {e}")
    
    print()
    
    # Test 5: Whisper Model (test download)
    print("5️⃣ Whisper Model Test...")
    try:
        print("   Testing model loading (this may take a moment)...")
        
        # Test with tiny model first (faster download)
        print("   Loading 'tiny' model for quick test...")
        model = whisper.load_model("tiny")
        print("   ✅ Whisper model loaded successfully")
        
        # Show model info
        print(f"   Model device: {next(model.parameters()).device}")
        print(f"   Model dtype: {next(model.parameters()).dtype}")
        
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        print("   This might be due to network issues or missing dependencies")
    
    print()
    
    # Test 6: Document Generator
    print("6️⃣ Document Generator...")
    try:
        doc_gen = DocumentGenerator()
        print("   ✅ Document Generator initialized")
        
        if doc_gen.docx_available:
            print("   ✅ DOCX support available")
        else:
            print("   ⚠️  DOCX support limited (python-docx not available)")
            
    except Exception as e:
        print(f"   ❌ Document Generator failed: {e}")
    
    print()
    
    # Test 7: Full transcriber initialization
    print("7️⃣ Full Transcriber Test...")
    try:
        print("   Initializing ARM64 Whisper Transcriber with large model...")
        # Note: This will try to download the large model
        transcriber = ARM64WhisperTranscriber(model_name="large", use_npu=True)
        print("   ✅ Transcriber initialized successfully")
        
        # Show transcriber info
        print(f"   Model: {transcriber.model_name}")
        print(f"   NPU enabled: {transcriber.use_npu}")
        print(f"   System: {transcriber.system_info}")
        
    except Exception as e:
        print(f"   ❌ Transcriber initialization failed: {e}")
        print("   This might be due to model download or dependency issues")
    
    print()
    print("🎯 Manual Testing Instructions:")
    print("-" * 30)
    print()
    print("To test with actual audio/video files:")
    print()
    print("1. GUI Mode:")
    print("   python gui.py")
    print("   - Browse for MP3/MP4/FLAC/WAV file")
    print("   - Ensure 'large' model is selected")
    print("   - Click 'Start Transcription'")
    print()
    print("2. Command Line:")
    print("   python transcribe.py \"path/to/your/audio.mp3\"")
    print("   python transcribe.py \"path/to/your/video.mp4\" --model large")
    print()
    print("Expected output:")
    print("   - Text file with formatted transcription")
    print("   - DOCX file with proper paragraphs and punctuation")
    print("   - JSON metadata file")
    print()
    print("✅ System appears ready for manual testing!")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
    
    print()
    input("Press Enter to exit...")
