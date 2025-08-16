#!/usr/bin/env python3
"""
Windows Speech Recognition Alternative
Uses native Windows Speech API as fallback until Whisper compilation is resolved
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import platform

try:
    import win32com.client
    import pythoncom
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

class WindowsSpeechTranscriber:
    """
    Windows Speech Recognition transcriber using native Windows APIs
    """
    
    def __init__(self, language: str = "en-US"):
        """Initialize Windows Speech transcriber"""
        self.language = language
        self.recognizer = None
        
        print(f"🎤 Windows Speech Transcriber Initializing...")
        print(f"📱 Platform: {platform.system()} {platform.machine()}")
        print(f"🔊 Language: {language}")
        
        # Check available speech recognition methods
        self.available_methods = []
        
        if SPEECH_RECOGNITION_AVAILABLE:
            self.recognizer = sr.Recognizer()
            self.available_methods.append("SpeechRecognition")
            print(f"✅ SpeechRecognition library available")
        
        if WIN32_AVAILABLE:
            self.available_methods.append("Windows SAPI")
            print(f"✅ Windows SAPI available")
        
        if not self.available_methods:
            print(f"❌ No speech recognition methods available")
            print(f"💡 Install: pip install SpeechRecognition pywin32")
    
    def transcribe_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio file using Windows Speech Recognition
        """
        audio_file = Path(audio_path)
        
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        print(f"🎵 Processing: {audio_file.name}")
        
        if not self.recognizer:
            return self._create_error_result("No speech recognition available", audio_file)
        
        try:
            # Load audio file
            with sr.AudioFile(str(audio_file)) as source:
                print(f"📖 Reading audio data...")
                audio_data = self.recognizer.record(source)
                
                # Attempt transcription with multiple services
                results = []
                
                # Try Windows Speech Recognition first
                try:
                    print(f"🎯 Attempting Windows Speech Recognition...")
                    text = self.recognizer.recognize_windows(audio_data, language=self.language)
                    results.append(("Windows Speech", text))
                    print(f"✅ Windows Speech Recognition successful")
                except Exception as e:
                    print(f"⚠️ Windows Speech Recognition failed: {e}")
                
                # Try Google Speech Recognition as fallback (requires internet)
                try:
                    print(f"🌐 Attempting Google Speech Recognition...")
                    text = self.recognizer.recognize_google(audio_data, language=self.language.split('-')[0])
                    results.append(("Google Speech", text))
                    print(f"✅ Google Speech Recognition successful")
                except Exception as e:
                    print(f"⚠️ Google Speech Recognition failed: {e}")
                
                if results:
                    # Use the first successful result
                    service, transcribed_text = results[0]
                    return self._create_success_result(transcribed_text, service, audio_file, results)
                else:
                    return self._create_error_result("All speech recognition services failed", audio_file)
                    
        except Exception as e:
            return self._create_error_result(f"Audio processing error: {str(e)}", audio_file)
    
    def _create_success_result(self, text: str, service: str, audio_file: Path, all_results: List) -> Dict[str, Any]:
        """Create successful transcription result"""
        # Format the text with basic punctuation
        formatted_text = self._format_text(text)
        
        return {
            "text": formatted_text,
            "segments": [
                {
                    "start": 0.0,
                    "end": 10.0,  # Placeholder - Windows Speech API doesn't provide timestamps
                    "text": formatted_text
                }
            ],
            "language": self.language,
            "model": f"Windows Speech Recognition ({service})",
            "service": service,
            "all_results": all_results,
            "file_info": {
                "original_file": str(audio_file),
                "file_size_mb": audio_file.stat().st_size / (1024*1024),
                "processing_time": 0.5  # Placeholder
            }
        }
    
    def _create_error_result(self, error_message: str, audio_file: Path) -> Dict[str, Any]:
        """Create error result"""
        return {
            "text": f"Error: {error_message}\n\nTo enable speech recognition:\n1. Install: pip install SpeechRecognition pywin32\n2. Ensure audio file is in supported format (WAV recommended)\n3. Check internet connection for Google Speech Recognition",
            "segments": [],
            "language": self.language,
            "model": "Error",
            "error": error_message,
            "file_info": {
                "original_file": str(audio_file),
                "file_size_mb": audio_file.stat().st_size / (1024*1024) if audio_file.exists() else 0,
                "processing_time": 0.0
            }
        }
    
    def _format_text(self, text: str) -> str:
        """Apply basic text formatting"""
        if not text:
            return ""
        
        # Capitalize first letter
        text = text.strip()
        if text:
            text = text[0].upper() + text[1:]
        
        # Add period if missing
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
    def test_speech_recognition(self):
        """Test speech recognition capabilities"""
        print(f"\n🧪 Testing Speech Recognition Capabilities")
        print(f"=" * 50)
        
        if not self.recognizer:
            print(f"❌ No recognizer available")
            return
        
        # Test with microphone if available
        try:
            with sr.Microphone() as source:
                print(f"🎤 Microphone detected")
                print(f"🔇 Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print(f"✅ Microphone ready")
        except Exception as e:
            print(f"⚠️ Microphone test failed: {e}")
        
        # Test available recognition services
        test_audio_data = None
        
        print(f"\n📋 Available Services:")
        for method in self.available_methods:
            print(f"  ✅ {method}")
        
        print(f"\n💡 To test with actual audio:")
        print(f"  1. Place a WAV file in this directory")
        print(f"  2. Run: python windows_speech.py <filename.wav>")

def main():
    """Main function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Windows Speech Recognition Test")
    parser.add_argument("audio_file", nargs="?", help="Audio file to transcribe")
    parser.add_argument("--language", default="en-US", help="Language code")
    parser.add_argument("--test", action="store_true", help="Run capability test")
    
    args = parser.parse_args()
    
    # Initialize transcriber
    transcriber = WindowsSpeechTranscriber(language=args.language)
    
    if args.test:
        transcriber.test_speech_recognition()
        return
    
    if args.audio_file:
        try:
            result = transcriber.transcribe_audio_file(args.audio_file)
            
            print(f"\n📄 Transcription Result:")
            print(f"=" * 50)
            print(f"Service: {result.get('model', 'Unknown')}")
            print(f"Language: {result.get('language', 'Unknown')}")
            print(f"\n📝 Text:")
            print(result['text'])
            
            if 'all_results' in result:
                print(f"\n🔄 All Attempts:")
                for service, text in result['all_results']:
                    print(f"  {service}: {text[:100]}...")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print(f"\n💡 Usage:")
        print(f"  python windows_speech.py audio.wav")
        print(f"  python windows_speech.py --test")

if __name__ == "__main__":
    main()
