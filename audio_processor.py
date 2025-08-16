#!/usr/bin/env python3
"""
Audio Processing Module for ARM64 Whisper Transcription
Handles multi-format audio/video input with FFmpeg integration
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Union
import platform

class AudioProcessor:
    """
    Audio processing engine optimized for ARM64 Windows with multi-format support.
    Uses bundled FFmpeg for audio extraction and preprocessing.
    """
    
    def __init__(self, ffmpeg_path: Optional[str] = None):
        """
        Initialize audio processor.
        
        Args:
            ffmpeg_path: Custom path to FFmpeg executable
        """
        self.ffmpeg_path = self._find_ffmpeg(ffmpeg_path)
        self.temp_dir = Path.cwd() / "temp_audio"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Supported formats
        self.supported_audio = {'.wav', '.mp3', '.m4a', '.aac', '.ogg', '.flac', '.wma'}
        self.supported_video = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.webm'}
        self.supported_formats = self.supported_audio | self.supported_video
        
        print(f"🔧 Audio Processor initialized")
        print(f"📦 FFmpeg: {'✅ Found' if self.ffmpeg_path else '❌ Not found'}")
        if self.ffmpeg_path:
            print(f"   Path: {self.ffmpeg_path}")
    
    def _find_ffmpeg(self, custom_path: Optional[str] = None) -> Optional[str]:
        """Find FFmpeg executable."""
        if custom_path and Path(custom_path).exists():
            return str(Path(custom_path))
        
        # Check bundled FFmpeg (same directory as script)
        script_dir = Path(__file__).parent
        bundled_ffmpeg = script_dir / "ffmpeg.exe"
        if bundled_ffmpeg.exists():
            return str(bundled_ffmpeg)
        
        # Check system PATH
        try:
            result = subprocess.run(
                ["where" if platform.system() == "Windows" else "which", "ffmpeg"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip().split('\n')[0]
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        return None
    
    def is_supported_format(self, file_path: str) -> bool:
        """Check if file format is supported."""
        return Path(file_path).suffix.lower() in self.supported_formats
    
    def get_audio_info(self, file_path: str) -> Dict[str, Any]:
        """Get audio file information using FFprobe."""
        if not self.ffmpeg_path:
            return {}
        
        ffprobe_path = str(Path(self.ffmpeg_path).parent / "ffprobe.exe")
        if not Path(ffprobe_path).exists():
            return {}
        
        try:
            cmd = [
                ffprobe_path,
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            import json
            info = json.loads(result.stdout)
            
            # Extract relevant audio information
            audio_streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
            
            if audio_streams:
                stream = audio_streams[0]
                return {
                    "duration": float(info.get("format", {}).get("duration", 0)),
                    "bitrate": int(info.get("format", {}).get("bit_rate", 0)),
                    "sample_rate": int(stream.get("sample_rate", 0)),
                    "channels": int(stream.get("channels", 0)),
                    "codec": stream.get("codec_name", "unknown")
                }
        
        except Exception as e:
            print(f"⚠️  Could not get audio info: {e}")
        
        return {}
    
    def process_file(self, input_path: Union[str, Path], output_dir: Optional[Union[str, Path]] = None) -> Optional[str]:
        """
        Process audio/video file for Whisper transcription.
        
        Args:
            input_path: Path to input file
            output_dir: Directory for temporary files
            
        Returns:
            Path to processed audio file or None if failed
        """
        input_path_obj = Path(input_path)
        
        if not input_path_obj.exists():
            print(f"❌ Input file not found: {input_path_obj}")
            return None
        
        if not self.is_supported_format(str(input_path_obj)):
            print(f"❌ Unsupported format: {input_path_obj.suffix}")
            return None
        
        if not self.ffmpeg_path:
            print(f"❌ FFmpeg not available - cannot process audio")
            return None
        
        # Set output directory
        if output_dir:
            output_dir_obj = Path(output_dir)
            output_dir_obj.mkdir(parents=True, exist_ok=True)
        else:
            output_dir_obj = self.temp_dir
        
        # Generate output path
        output_filename = f"{input_path_obj.stem}_processed.wav"
        output_path = output_dir_obj / output_filename
        
        print(f"🔧 Processing audio: {input_path_obj.name}")
        
        # Get input file info
        info = self.get_audio_info(str(input_path_obj))
        if info:
            duration = info.get("duration", 0)
            print(f"   Duration: {duration:.1f} seconds")
            print(f"   Sample rate: {info.get('sample_rate', 'unknown')} Hz")
            print(f"   Channels: {info.get('channels', 'unknown')}")
        
        try:
            # FFmpeg command for Whisper-optimized audio
            cmd = [
                self.ffmpeg_path,
                "-i", str(input_path_obj),
                "-ar", "16000",      # 16kHz sample rate (Whisper standard)
                "-ac", "1",          # Mono audio
                "-c:a", "pcm_s16le", # 16-bit PCM
                "-f", "wav",         # WAV format
                "-y",                # Overwrite output
                str(output_path)
            ]
            
            # Add progress reporting for long files
            if info.get("duration", 0) > 30:  # For files longer than 30 seconds
                print("⏳ Processing... (this may take a moment for long files)")
            
            # Run FFmpeg
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5-minute timeout
            )
            
            if result.returncode == 0:
                if output_path.exists():
                    file_size = output_path.stat().st_size
                    print(f"✅ Audio processed successfully")
                    print(f"   Output: {output_path.name} ({file_size // 1024} KB)")
                    return str(output_path)
                else:
                    print(f"❌ Output file not created")
                    return None
            else:
                print(f"❌ FFmpeg error: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"❌ Processing timeout (file too long or system too slow)")
            return None
        except Exception as e:
            print(f"❌ Processing error: {e}")
            return None
    
    def batch_process(self, input_paths: list, output_dir: Union[str, Path]) -> Dict[str, Optional[str]]:
        """
        Process multiple files in batch.
        
        Args:
            input_paths: List of input file paths
            output_dir: Directory for output files
            
        Returns:
            Dictionary mapping input paths to output paths (or None if failed)
        """
        results = {}
        output_dir_obj = Path(output_dir)
        output_dir_obj.mkdir(parents=True, exist_ok=True)
        
        print(f"📦 Batch processing {len(input_paths)} files...")
        
        for i, input_path in enumerate(input_paths, 1):
            print(f"\n[{i}/{len(input_paths)}] Processing: {Path(input_path).name}")
            
            result = self.process_file(input_path, str(output_dir_obj))
            results[input_path] = result
            
            if result:
                print(f"✅ Success: {Path(result).name}")
            else:
                print(f"❌ Failed: {Path(input_path).name}")
        
        successful = sum(1 for r in results.values() if r is not None)
        print(f"\n📊 Batch processing completed: {successful}/{len(input_paths)} successful")
        
        return results
    
    def cleanup_temp_files(self) -> int:
        """Clean up temporary audio files."""
        if not self.temp_dir.exists():
            return 0
        
        cleaned = 0
        for temp_file in self.temp_dir.glob("*_processed.wav"):
            try:
                temp_file.unlink()
                cleaned += 1
            except Exception as e:
                print(f"⚠️  Could not delete {temp_file}: {e}")
        
        if cleaned > 0:
            print(f"🧹 Cleaned up {cleaned} temporary audio files")
        
        return cleaned

def main():
    """Test the audio processor."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python audio_processor.py <input_file>")
        return 1
    
    processor = AudioProcessor()
    result = processor.process_file(sys.argv[1])
    
    if result:
        print(f"✅ Success: {result}")
        return 0
    else:
        print(f"❌ Failed to process audio")
        return 1

if __name__ == "__main__":
    sys.exit(main())
