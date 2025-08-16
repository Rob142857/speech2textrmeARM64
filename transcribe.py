#!/usr/bin/env python3
"""
ARM64 Whisper Transcription Engine
Optimized for Windows ARM64 with NPU acceleration via ONNX Runtime QNN Provider
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import platform
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import core libraries
try:
    # Temporarily disable Whisper imports until compilation issues are resolved
    # import whisper
    # import torch
    # import numpy as np
    from audio_processor import AudioProcessor
    # from whisper_npu import WhisperNPU
    from document_generator import DocumentGenerator
    WHISPER_AVAILABLE = False
    print("⚠️ Warning: Whisper not available, using placeholder functionality")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("📦 Please install requirements: pip install -r requirements.txt")
    sys.exit(1)


class ARM64WhisperTranscriber:
    """
    High-performance speech-to-text transcription engine optimized for ARM64 Windows
    with NPU acceleration via Qualcomm Snapdragon X Elite QNN Provider.
    """
    
    def __init__(self, model_name: str = "large", use_npu: bool = True, output_dir: str = "output"):
        """
        Initialize the ARM64 Whisper transcription engine.
        
        Args:
            model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            use_npu: Whether to attempt NPU acceleration
            output_dir: Directory for output files
        """
        self.model_name = model_name
        self.use_npu = use_npu
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # System information
        self.system_info = {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "is_arm64": platform.machine() == 'ARM64'
        }
        
        print(f"🚀 ARM64 Whisper Transcriber Initializing...")
        print(f"📱 Platform: {self.system_info['platform']} {self.system_info['architecture']}")
        print(f"🐍 Python: {self.system_info['python_version']}")
        
        # Initialize components
        self.audio_processor = AudioProcessor()
        # self.whisper_npu = WhisperNPU(use_npu=use_npu) if use_npu else None
        self.whisper_npu = None  # Temporarily disabled until compilation issues resolved
        self.document_generator = DocumentGenerator()
        
        # Load Whisper model (temporarily disabled)
        # self.model = self._load_whisper_model()
        self.model = None  # Placeholder until Whisper is available
        
        # NPU status
        # if self.whisper_npu and self.whisper_npu.is_npu_available:
        #     print(f"🧠 NPU Status: Available (QNN Provider)")
        print(f"⚠️ NPU Status: Disabled (Whisper not available)")
        print(f"💻 Processing Mode: Placeholder (needs Whisper compilation)")
            print(f"⚠️  NPU acceleration unavailable")
    
    def _load_whisper_model(self) -> whisper.Whisper:
        """Load and optimize Whisper model for ARM64."""
        print(f"📥 Loading Whisper model: {self.model_name}")
        
        try:
            # Load model with ARM64 optimizations
            model = whisper.load_model(
                self.model_name,
                device="cpu"  # We'll handle NPU acceleration separately
            )
            
            # Apply ARM64-specific optimizations
            if self.system_info["is_arm64"]:
                print("🔧 Applying ARM64 optimizations...")
                # Enable ARM64 NEON optimizations if available
                if hasattr(torch.backends, 'cpu') and hasattr(torch.backends.cpu, 'enable_neon'):
                    torch.backends.cpu.enable_neon = True
            
            print(f"✅ Model loaded successfully: {self.model_name}")
            return model
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("💡 Try running: python -c \"import whisper; whisper.load_model('base')\"")
            raise
    
    def transcribe_file(
        self,
        input_path: Union[str, Path],
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        temperature: float = 0.0,
        condition_on_previous_text: bool = True,
        verbose: bool = True,
        word_timestamps: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe an audio or video file.
        
        Args:
            input_path: Path to input file
            language: Language code (e.g., 'en', 'es', 'fr') or None for auto-detection
            initial_prompt: Optional text to guide transcription
            temperature: Sampling temperature (0.0 = deterministic)
            condition_on_previous_text: Use previous text as context
            verbose: Print progress information
            word_timestamps: Include word-level timestamps
            
        Returns:
            Dictionary with transcription results and metadata
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        print(f"\n🎯 Starting transcription pipeline...")
        print(f"📁 Input: {input_path.name}")
        
        start_time = time.time()
        
        # Step 1: Audio preprocessing
        print(f"🔧 Processing audio...")
        processed_audio_path = self.audio_processor.process_file(
            str(input_path),
            output_dir=str(self.output_dir / "temp")
        )
        
        if not processed_audio_path:
            raise RuntimeError("Audio processing failed")
        
        # Step 2: Transcription with NPU acceleration
        print(f"🧠 Running Whisper transcription...")
        
        transcription_options = {
            "language": language,
            "initial_prompt": initial_prompt,
            "temperature": temperature,
            "condition_on_previous_text": condition_on_previous_text,
            "verbose": verbose,
            "word_timestamps": word_timestamps
        }
        
        # Use NPU-accelerated transcription if available
        if self.whisper_npu and self.whisper_npu.is_npu_available:
            print("⚡ Using NPU acceleration...")
            result = self.whisper_npu.transcribe(
                processed_audio_path,
                self.model,
                **transcription_options
            )
        else:
            print("💻 Using CPU processing...")
            result = self.model.transcribe(
                processed_audio_path,
                **transcription_options
            )
        
        processing_time = time.time() - start_time
        
        # Step 3: Post-processing and formatting
        print(f"📝 Post-processing transcription...")
        formatted_text = self._format_transcription(result["text"])
        
        # Step 4: Generate output files
        base_name = input_path.stem
        output_files = {}
        
        # Text file
        txt_path = self.output_dir / f"{base_name}_transcription.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(formatted_text)
        output_files["txt"] = str(txt_path)
        
        # DOCX file
        print(f"📄 Generating DOCX document...")
        docx_path = self.document_generator.create_document(
            formatted_text,
            str(self.output_dir / f"{base_name}_transcription.docx"),
            title=f"Transcription: {input_path.name}",
            metadata={
                "source_file": str(input_path),
                "model": self.model_name,
                "processing_time": f"{processing_time:.2f} seconds",
                "npu_used": self.whisper_npu.is_npu_available if self.whisper_npu else False,
                "language": result.get("language", "auto-detected"),
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }
        )
        output_files["docx"] = docx_path
        
        # Metadata JSON
        metadata = {
            "input_file": str(input_path),
            "output_files": output_files,
            "processing_time_seconds": processing_time,
            "model_used": self.model_name,
            "npu_acceleration": self.whisper_npu.is_npu_available if self.whisper_npu else False,
            "system_info": self.system_info,
            "transcription_options": transcription_options,
            "detected_language": result.get("language", "unknown"),
            "segments_count": len(result.get("segments", [])),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_path = self.output_dir / f"{base_name}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        output_files["metadata"] = str(metadata_path)
        
        # Cleanup temporary files
        try:
            if Path(processed_audio_path).exists():
                Path(processed_audio_path).unlink()
        except:
            pass
        
        print(f"\n✅ Transcription completed successfully!")
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        print(f"📄 Text file: {txt_path}")
        print(f"📄 DOCX file: {docx_path}")
        print(f"📋 Metadata: {metadata_path}")
        
        return {
            "success": True,
            "transcription": formatted_text,
            "output_files": output_files,
            "metadata": metadata,
            "processing_time": processing_time
        }
    
    def _format_transcription(self, text: str) -> str:
        """
        Format transcription text with proper punctuation and paragraphing.
        Enhanced formatting for professional documents.
        
        Args:
            text: Raw transcription text
            
        Returns:
            Formatted text with proper structure, punctuation, and paragraphs
        """
        if not text or not text.strip():
            return ""
        
        # Basic cleanup and normalization
        text = text.strip()
        
        # Fix common transcription issues
        text = text.replace(' um ', ' ')  # Remove filler words
        text = text.replace(' uh ', ' ')
        text = text.replace(' ah ', ' ')
        text = text.replace('  ', ' ')  # Remove double spaces
        
        # Split into segments for better processing
        # Whisper often provides natural sentence boundaries
        sentences = []
        
        # Split on various sentence endings
        import re
        
        # Split on sentence endings but preserve them
        sentence_parts = re.split(r'([.!?]+)', text)
        
        current_sentence = ""
        for i, part in enumerate(sentence_parts):
            if re.match(r'^[.!?]+$', part):  # This is punctuation
                current_sentence += part
                sentences.append(current_sentence.strip())
                current_sentence = ""
            else:
                current_sentence += part
        
        # Add any remaining text
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # Clean and format sentences
        formatted_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Capitalize first letter
            if sentence:
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
            
            # Ensure proper ending punctuation
            if sentence and not re.search(r'[.!?]$', sentence):
                sentence += '.'
            
            # Fix spacing around punctuation
            sentence = re.sub(r'\s+([,.!?])', r'\1', sentence)  # Remove space before punctuation
            sentence = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', sentence)  # Add space after punctuation
            
            formatted_sentences.append(sentence)
        
        # Create paragraphs based on natural breaks and length
        paragraphs = []
        current_paragraph = []
        sentence_count = 0
        
        for sentence in formatted_sentences:
            current_paragraph.append(sentence)
            sentence_count += 1
            
            # Create paragraph breaks based on:
            # 1. Natural conversation markers
            # 2. Length (every 3-5 sentences)
            # 3. Topic shift indicators
            should_break = (
                sentence_count >= 4 or  # Every 4 sentences
                any(sentence.lower().startswith(marker) for marker in [
                    'and then', 'so then', 'but then', 'now', 'after that', 
                    'meanwhile', 'however', 'therefore', 'in addition',
                    'furthermore', 'moreover', 'on the other hand'
                ]) or
                any(phrase in sentence.lower() for phrase in [
                    'moving on', 'next topic', 'another thing', 'speaking of',
                    'by the way', 'incidentally'
                ])
            )
            
            if should_break and current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
                sentence_count = 0
        
        # Add any remaining sentences
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        # Join paragraphs with double line breaks
        formatted_text = '\n\n'.join(paragraphs)
        
        # Final cleanup
        formatted_text = re.sub(r'\n{3,}', '\n\n', formatted_text)  # Remove excessive line breaks
        formatted_text = re.sub(r' {2,}', ' ', formatted_text)  # Remove excessive spaces
        
        return formatted_text.strip()


def main():
    """Command-line interface for ARM64 Whisper transcription."""
    parser = argparse.ArgumentParser(
        description="ARM64 Whisper Speech-to-Text Transcription with NPU Acceleration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcribe.py audio.mp3
  python transcribe.py video.mp4 --model medium --language en
  python transcribe.py recording.wav --no-npu --output ./results
  python transcribe.py interview.m4a --model large --word-timestamps
        """
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        help="Input audio or video file path"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="large",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: large)"
    )
    
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g., en, es, fr) or auto-detect if not specified"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory (default: output)"
    )
    
    parser.add_argument(
        "--no-npu",
        action="store_true",
        help="Disable NPU acceleration (use CPU only)"
    )
    
    parser.add_argument(
        "--initial-prompt",
        type=str,
        default=None,
        help="Initial prompt to guide transcription"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = deterministic, default: 0.0)"
    )
    
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Include word-level timestamps"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize transcriber
        transcriber = ARM64WhisperTranscriber(
            model_name=args.model,
            use_npu=not args.no_npu,
            output_dir=args.output
        )
        
        # Perform transcription
        result = transcriber.transcribe_file(
            input_path=args.input_file,
            language=args.language,
            initial_prompt=args.initial_prompt,
            temperature=args.temperature,
            word_timestamps=args.word_timestamps,
            verbose=args.verbose
        )
        
        if result["success"]:
            print(f"\n🎉 Transcription completed successfully!")
            return 0
        else:
            print(f"\n❌ Transcription failed")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Transcription cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
