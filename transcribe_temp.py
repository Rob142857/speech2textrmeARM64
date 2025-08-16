#!/usr/bin/env python3
"""
ARM64 Whisper Transcription Engine - Enhanced Version
Now includes Windows Speech Recognition as functional fallback
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

# Import available libraries
try:
    from audio_processor import AudioProcessor
    from document_generator import DocumentGenerator
    from windows_speech import WindowsSpeechTranscriber
    DEPENDENCIES_AVAILABLE = True
    WINDOWS_SPEECH_AVAILABLE = True
except ImportError as e:
    print(f"❌ Import error: {e}")
    DEPENDENCIES_AVAILABLE = False
    WINDOWS_SPEECH_AVAILABLE = False

class ARM64WhisperTranscriber:
    """
    ARM64 Whisper Transcriber with Windows Speech Recognition fallback
    """
    
    def __init__(self, 
                 model_name: str = "large",
                 use_npu: bool = True,
                 output_dir: str = "output"):
        """
        Initialize ARM64 Whisper Transcriber
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
        
        # Initialize available components
        if DEPENDENCIES_AVAILABLE:
            self.audio_processor = AudioProcessor()
            self.document_generator = DocumentGenerator()
            
            if WINDOWS_SPEECH_AVAILABLE:
                self.windows_speech = WindowsSpeechTranscriber()
                print(f"🎤 Windows Speech Recognition available")
            else:
                self.windows_speech = None
                
            print(f"✅ Core components available")
        else:
            self.audio_processor = None
            self.document_generator = None
            self.windows_speech = None
            print(f"❌ Core dependencies not available")
        
        # Whisper model placeholder (until compilation issues resolved)
        self.model = None
        print(f"⚠️ Status: OpenAI Whisper not available - using Windows Speech Recognition when possible")
    
    def transcribe_file(self, 
                       audio_file_path: str,
                       output_format: str = "docx",
                       language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio file using available speech recognition methods
        """
        audio_path = Path(audio_file_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"📁 Processing: {audio_path.name}")
        print(f"📏 File size: {audio_path.stat().st_size / (1024*1024):.1f} MB")
        
        # Try Windows Speech Recognition if available
        if self.windows_speech:
            try:
                print(f"🎤 Attempting Windows Speech Recognition...")
                result = self.windows_speech.transcribe_audio_file(str(audio_path))
                
                if result and not result.get('error'):
                    print(f"✅ Speech recognition successful!")
                    
                    # Enhance the result with our metadata
                    result.update({
                        "model": f"Windows Speech Recognition (via {result.get('service', 'Unknown')})",
                        "transcriber": "ARM64WhisperTranscriber",
                        "npu_available": self.use_npu,
                        "processing_method": "Windows Speech API"
                    })
                    
                    # Generate output document
                    if self.document_generator and output_format.lower() == "docx":
                        output_path = self._generate_output_document(result, audio_path, output_format)
                        result["output_file"] = str(output_path)
                        print(f"📄 Document saved: {output_path}")
                    
                    return result
                else:
                    print(f"⚠️ Speech recognition failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"❌ Speech recognition error: {str(e)}")
        
        # Fallback to placeholder if speech recognition fails
        print(f"📝 Using placeholder transcription...")
        placeholder_text = self._generate_placeholder_text(audio_path, language)
        
        # Create result dictionary
        result = {
            "text": placeholder_text,
            "segments": [
                {
                    "start": 0.0,
                    "end": 10.0,
                    "text": placeholder_text.strip()
                }
            ],
            "language": language or "en",
            "model": f"{self.model_name} (placeholder - Whisper not available)",
            "transcriber": "ARM64WhisperTranscriber",
            "npu_available": self.use_npu,
            "processing_method": "Placeholder",
            "file_info": {
                "original_file": str(audio_path),
                "file_size_mb": audio_path.stat().st_size / (1024*1024),
                "processing_time": 0.1
            }
        }
        
        # Generate output document
        if self.document_generator and output_format.lower() == "docx":
            output_path = self._generate_output_document(result, audio_path, output_format)
            result["output_file"] = str(output_path)
            print(f"📄 Document saved: {output_path}")
        
        return result
    
    def _generate_placeholder_text(self, audio_path: Path, language: Optional[str]) -> str:
        """Generate placeholder text when speech recognition fails"""
        speech_status = "Available" if self.windows_speech else "Not Available"
        
        return f"""This is a transcription attempt for the file: {audio_path.name}

TRANSCRIPTION STATUS:
- Windows Speech Recognition: {speech_status}
- OpenAI Whisper: Not Available (compilation required)
- NPU Acceleration: {'Requested' if self.use_npu else 'Disabled'}

SYSTEM INFORMATION:
- Platform: {self.system_info['platform']} {self.system_info['architecture']}
- Python Version: {self.system_info['python_version']}
- Model Requested: {self.model_name}

TO ENABLE FULL TRANSCRIPTION:

Option 1 - OpenAI Whisper (Best Quality):
1. Install Visual Studio Build Tools with C++ workload
2. Install Rust compiler toolchain  
3. Run: pip install openai-whisper

Option 2 - Windows Speech Recognition (Working Now):
1. Convert audio to WAV format for best compatibility
2. Ensure good audio quality
3. Try again with supported audio format

File processed: {time.strftime('%Y-%m-%d %H:%M:%S')}
Audio format: {audio_path.suffix}
"""
    
    def _generate_output_document(self, result: Dict[str, Any], 
                                audio_file: Path, 
                                format_type: str) -> Path:
        """Generate output document"""
        if format_type.lower() == "docx" and self.document_generator:
            output_path = self.output_dir / f"{audio_file.stem}_transcription.docx"
            
            # Enhanced metadata
            metadata = {
                "source_file": str(audio_file),
                "model": result.get("model", "Unknown"),
                "language": result.get("language", "Unknown"),
                "processing_time": result["file_info"]["processing_time"],
                "file_size_mb": result["file_info"]["file_size_mb"],
                "transcriber": result.get("transcriber", "ARM64WhisperTranscriber"),
                "processing_method": result.get("processing_method", "Unknown"),
                "npu_available": result.get("npu_available", False)
            }
            
            self.document_generator.create_document(
                result["text"],
                str(output_path),
                metadata=metadata
            )
            return output_path
        else:
            # Fallback to text file
            output_path = self.output_dir / f"{audio_file.stem}_transcription.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result["text"])
            return output_path

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="ARM64 Whisper Transcription Tool with Windows Speech Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python transcribe_temp.py audio.wav
    python transcribe_temp.py video.mp4 --model large --language en
    python transcribe_temp.py recording.wav --output-format docx

Supported Methods:
    1. Windows Speech Recognition (Available Now)
    2. OpenAI Whisper (Requires compilation)
        """)
    
    parser.add_argument("input_file", help="Audio/video file to transcribe")
    parser.add_argument("--model", default="large", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size (when available)")
    parser.add_argument("--language", help="Audio language (auto-detect if not specified)")
    parser.add_argument("--output-format", default="docx", 
                       choices=["docx", "txt"],
                       help="Output format")
    parser.add_argument("--output-dir", default="output",
                       help="Output directory")
    parser.add_argument("--no-npu", action="store_true",
                       help="Disable NPU acceleration")
    
    args = parser.parse_args()
    
    # Initialize transcriber
    transcriber = ARM64WhisperTranscriber(
        model_name=args.model,
        use_npu=not args.no_npu,
        output_dir=args.output_dir
    )
    
    try:
        # Perform transcription
        print(f"\n🎙️ Starting transcription...")
        start_time = time.time()
        
        result = transcriber.transcribe_file(
            args.input_file,
            output_format=args.output_format,
            language=args.language
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Display results
        print(f"\n✅ Transcription completed!")
        print(f"🔧 Method: {result.get('processing_method', 'Unknown')}")
        print(f"⏱️ Processing time: {processing_time:.2f} seconds")
        if "output_file" in result:
            print(f"📄 Output saved to: {result['output_file']}")
        
        # Show preview of transcription
        text_preview = result['text'][:200]
        if len(result['text']) > 200:
            text_preview += "..."
        print(f"\n📝 Transcription Preview:")
        print(f"   {text_preview}")
        
    except KeyboardInterrupt:
        print("\n❌ Transcription cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during transcription: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
