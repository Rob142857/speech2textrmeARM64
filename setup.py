#!/usr/bin/env python3
"""
Setup and Installation Script for ARM64 Whisper Transcription
Handles environment setup and dependency installation for Windows ARM64
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import json
import time

class ARM64SetupManager:
    """Setup manager for ARM64 Whisper transcription environment."""
    
    def __init__(self):
        self.is_arm64 = platform.machine() == 'ARM64'
        self.is_windows = platform.system() == 'Windows'
        self.python_version = platform.python_version()
        self.project_root = Path(__file__).parent.absolute()
        
        print(f"🚀 ARM64 Whisper Transcription Setup")
        print(f"=" * 50)
        print(f"Platform: {platform.system()} {platform.machine()}")
        print(f"Python: {self.python_version}")
        print(f"Project: {self.project_root}")
        print()
        
        if not self.is_arm64:
            print("⚠️  WARNING: This setup is optimized for ARM64 architecture")
            print("   Some features may not work optimally on x64")
        
        if not self.is_windows:
            print("⚠️  WARNING: This setup is optimized for Windows")
            print("   Compatibility on other platforms may vary")
    
    def check_python_compatibility(self) -> bool:
        """Check if Python version is compatible."""
        major, minor = map(int, self.python_version.split('.')[:2])
        
        if major < 3 or (major == 3 and minor < 8):
            print(f"❌ Python {self.python_version} is not supported")
            print("   Minimum required: Python 3.8")
            return False
        
        if major == 3 and minor >= 12:
            print(f"⚠️  Python {self.python_version} may have compatibility issues")
            print("   Recommended: Python 3.9-3.11")
        
        print(f"✅ Python {self.python_version} is compatible")
        return True
    
    def check_system_requirements(self) -> dict:
        """Check system requirements and capabilities."""
        requirements = {
            "python_ok": self.check_python_compatibility(),
            "pip_available": self._check_pip(),
            "ffmpeg_needed": self._check_ffmpeg(),
            "npu_potential": self._check_npu_potential(),
            "memory_ok": self._check_memory(),
            "disk_space_ok": self._check_disk_space()
        }
        
        print("\n📊 System Requirements Check:")
        for requirement, status in requirements.items():
            status_icon = "✅" if status else "❌"
            req_name = requirement.replace('_', ' ').title()
            print(f"   {status_icon} {req_name}")
        
        return requirements
    
    def _check_pip(self) -> bool:
        """Check if pip is available."""
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"], 
                         capture_output=True, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is needed (we'll bundle it)."""
        ffmpeg_bundled = self.project_root / "ffmpeg.exe"
        return not ffmpeg_bundled.exists()  # True if we need to get it
    
    def _check_npu_potential(self) -> bool:
        """Check if NPU acceleration might be available."""
        if not self.is_arm64 or not self.is_windows:
            return False
        
        # Check for Qualcomm Snapdragon processor
        processor = platform.processor().lower()
        return "snapdragon" in processor or "qualcomm" in processor
    
    def _check_memory(self) -> bool:
        """Check available memory."""
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            return memory_gb >= 8  # Minimum 8GB recommended
        except ImportError:
            return True  # Assume OK if can't check
    
    def _check_disk_space(self) -> bool:
        """Check available disk space."""
        try:
            import shutil
            free_space_gb = shutil.disk_usage(self.project_root).free / (1024**3)
            return free_space_gb >= 5  # Minimum 5GB for models and temp files
        except:
            return True  # Assume OK if can't check
    
    def install_dependencies(self, requirements_file: str = "requirements.txt") -> bool:
        """Install Python dependencies."""
        requirements_path = self.project_root / requirements_file
        
        if not requirements_path.exists():
            print(f"❌ Requirements file not found: {requirements_path}")
            return False
        
        print(f"\n📦 Installing dependencies from {requirements_file}...")
        print("   This may take several minutes, especially for ARM64 builds")
        
        try:
            # Upgrade pip first
            print("🔧 Upgrading pip...")
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                         check=True)
            
            # Install requirements
            print("🔧 Installing packages...")
            cmd = [
                sys.executable, "-m", "pip", "install", 
                "-r", str(requirements_path),
                "--no-cache-dir"  # Avoid cache issues with ARM64 builds
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Dependencies installed successfully")
                return True
            else:
                print(f"❌ Dependency installation failed:")
                print(result.stderr)
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Installation error: {e}")
            return False
    
    def setup_ffmpeg(self) -> bool:
        """Set up FFmpeg (bundled approach)."""
        ffmpeg_path = self.project_root / "ffmpeg.exe"
        
        if ffmpeg_path.exists():
            print("✅ FFmpeg already available")
            return True
        
        print("\n🎥 Setting up FFmpeg...")
        print("   FFmpeg is required for audio/video processing")
        print(f"   Please download FFmpeg for Windows ARM64 and place ffmpeg.exe in:")
        print(f"   {self.project_root}")
        print("   ")
        print("   Download from: https://www.gyan.dev/ffmpeg/builds/")
        print("   Look for 'release-full' ARM64 builds")
        
        return False
    
    def verify_installation(self) -> bool:
        """Verify that the installation works."""
        print("\n🧪 Verifying installation...")
        
        tests = {
            "Core imports": self._test_core_imports,
            "Audio processing": self._test_audio_processor,
            "NPU detection": self._test_npu_detection,
            "Document generation": self._test_document_generator
        }
        
        all_passed = True
        for test_name, test_func in tests.items():
            try:
                result = test_func()
                status = "✅" if result else "❌"
                print(f"   {status} {test_name}")
                if not result:
                    all_passed = False
            except Exception as e:
                print(f"   ❌ {test_name}: {e}")
                all_passed = False
        
        return all_passed
    
    def _test_core_imports(self) -> bool:
        """Test core library imports."""
        try:
            import whisper
            import torch
            import numpy as np
            return True
        except ImportError:
            return False
    
    def _test_audio_processor(self) -> bool:
        """Test audio processor functionality."""
        try:
            from audio_processor import AudioProcessor
            processor = AudioProcessor()
            return True
        except ImportError:
            return False
    
    def _test_npu_detection(self) -> bool:
        """Test NPU detection."""
        try:
            from whisper_npu import WhisperNPU
            npu = WhisperNPU(use_npu=True)
            return True  # Don't require NPU to be available
        except ImportError:
            return False
    
    def _test_document_generator(self) -> bool:
        """Test document generator."""
        try:
            from document_generator import DocumentGenerator
            generator = DocumentGenerator()
            return True
        except ImportError:
            return False
    
    def create_desktop_shortcut(self) -> bool:
        """Create desktop shortcut for the GUI."""
        if not self.is_windows:
            return False
        
        try:
            import win32com.client
            
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            shortcut_path = os.path.join(desktop, "ARM64 Whisper Transcription.lnk")
            
            shell = win32com.client.Dispatch("WScript.Shell")
            shortcut = shell.CreateShortCut(shortcut_path)
            shortcut.Targetpath = sys.executable
            shortcut.Arguments = f'"{self.project_root / "gui.py"}"'
            shortcut.WorkingDirectory = str(self.project_root)
            shortcut.Description = "ARM64 Whisper Speech-to-Text Transcription"
            shortcut.save()
            
            print(f"✅ Desktop shortcut created: {shortcut_path}")
            return True
            
        except ImportError:
            print("⚠️  Could not create desktop shortcut (pywin32 not available)")
            return False
        except Exception as e:
            print(f"⚠️  Could not create desktop shortcut: {e}")
            return False
    
    def run_full_setup(self) -> bool:
        """Run complete setup process."""
        print("🚀 Starting full setup process...\n")
        
        # Check requirements
        requirements = self.check_system_requirements()
        
        if not requirements["python_ok"]:
            print("\n❌ Setup cannot continue due to Python compatibility issues")
            return False
        
        if not requirements["pip_available"]:
            print("\n❌ Setup cannot continue - pip is not available")
            return False
        
        # Install dependencies
        if not self.install_dependencies():
            print("\n❌ Setup failed during dependency installation")
            return False
        
        # Setup FFmpeg
        self.setup_ffmpeg()
        
        # Verify installation
        if not self.verify_installation():
            print("\n⚠️  Setup completed with some verification failures")
            print("   The application may still work, but some features might be limited")
        
        # Create desktop shortcut
        self.create_desktop_shortcut()
        
        print("\n🎉 Setup completed!")
        print("\nTo start the application:")
        print(f"   python {self.project_root / 'gui.py'}")
        print("\nOr run the transcription engine directly:")
        print(f"   python {self.project_root / 'transcribe.py'} <audio_file>")
        
        return True

def main():
    """Main setup function."""
    try:
        setup = ARM64SetupManager()
        success = setup.run_full_setup()
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⏹️  Setup cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
