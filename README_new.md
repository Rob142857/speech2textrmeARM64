# ARM64 Whisper Transcription Engine

🚀 **High-performance local speech-to-text transcription optimized for Windows ARM64 devices with NPU acceleration via Qualcomm Snapdragon X Elite.**

## ✨ Features

- **🧠 Local OpenAI Whisper Processing** - No cloud dependencies, complete privacy
- **⚡ NPU Acceleration** - Optimized for Qualcomm Snapdragon X Elite via ONNX Runtime QNN
- **🎵 Multi-format Support** - Audio/video files (MP3, MP4, WAV, M4A, AVI, MOV, etc.)
- **📄 Professional Output** - DOCX documents with proper punctuation and paragraphing
- **🖥️ Dual Interface** - Both GUI and command-line interfaces
- **🔧 ARM64 Optimized** - Native ARM64 Windows compatibility

## 🎯 Designed For

- **Surface Laptop 7** with Qualcomm Snapdragon X Elite NPU
- **Windows 11 ARM64** devices
- **Local processing** requirements (no internet needed)
- **Professional transcription** workflows

## 🚀 Quick Start

### 1. Prerequisites

- **Windows 11 ARM64**
- **Python 3.9-3.11** (3.10 recommended)
- **8GB+ RAM** (16GB recommended for large files)
- **5GB+ free disk space** (for models and temporary files)

### 2. Installation

```powershell
# Clone the repository
git clone https://github.com/Rob142857/speech2textrmeARM64.git
cd speech2textrmeARM64

# Run automated setup
python setup.py
```

### 3. Launch Application

**GUI Mode:**
```powershell
python gui.py
```

**Command Line:**
```powershell
python transcribe.py "path/to/audio.mp3" --model medium --language en
```

## 📋 System Requirements

### Minimum Requirements
- Windows 11 ARM64
- Python 3.8+
- 8GB RAM
- 5GB free disk space
- Qualcomm Snapdragon processor (for NPU)

### Recommended Configuration
- Surface Laptop 7 (Snapdragon X Elite)
- Python 3.10
- 16GB+ RAM
- 10GB+ free disk space
- High-speed SSD storage

## 🔧 Installation Guide

### Automated Setup
The `setup.py` script handles most configuration:

```powershell
python setup.py
```

This will:
- ✅ Check system compatibility
- 📦 Install ARM64-compatible dependencies
- 🎥 Guide FFmpeg setup
- 🧪 Verify installation
- 🖥️ Create desktop shortcuts

### Manual Installation

1. **Install Core Dependencies:**
```powershell
pip install -r requirements.txt
```

2. **Download FFmpeg for ARM64:**
   - Visit: https://www.gyan.dev/ffmpeg/builds/
   - Download ARM64 "release-full" build
   - Extract `ffmpeg.exe` to project directory

3. **Verify Installation:**
```powershell
python transcribe.py --help
```

## 🎮 Usage Examples

### GUI Interface
```powershell
python gui.py
```
- Browse and select audio/video files
- Choose Whisper model size
- Configure language and output options
- Monitor real-time progress
- Export to DOCX and text formats

### Command Line Interface

**Basic Transcription:**
```powershell
python transcribe.py "recording.mp3"
```

**Advanced Options:**
```powershell
python transcribe.py "interview.mp4" --model medium --language en --output ./results --word-timestamps --verbose
```

## ⚙️ Configuration Options

### Model Sizes
- **tiny** - Fastest, lowest accuracy (~39 MB)
- **base** - Balanced speed/accuracy (~74 MB)
- **small** - Good accuracy (~244 MB)
- **medium** - High accuracy (~769 MB)
- **large** - Best accuracy (~1550 MB)

### Language Support
Auto-detection or specify: `en`, `es`, `fr`, `de`, `it`, `pt`, `ru`, `ja`, `ko`, `zh`, etc.

### Output Formats
- **TXT** - Plain text transcription
- **DOCX** - Formatted Microsoft Word document
- **JSON** - Detailed metadata and results

## 🧠 NPU Acceleration

### NPU Detection
The system automatically detects and uses NPU acceleration when available:

```powershell
# Check NPU status
python -c "from whisper_npu import WhisperNPU; npu=WhisperNPU(); print(npu.get_npu_info())"
```

### Requirements for NPU
- Qualcomm Snapdragon X Elite processor
- ONNX Runtime QNN Provider
- Windows 11 ARM64
- QNN SDK (installed automatically)

### Performance Benefits
- **2-5x faster** transcription on compatible hardware
- **Lower CPU usage** during processing
- **Better battery life** on mobile devices

## 📁 Project Structure

```
speech2textrmeARM64/
├── transcribe.py           # Main transcription engine
├── gui.py                 # GUI interface
├── audio_processor.py     # Audio/video processing
├── whisper_npu.py         # NPU acceleration
├── document_generator.py  # DOCX output formatting
├── setup.py              # Installation and setup
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── .github/             # GitHub configuration
│   └── copilot-instructions.md
└── output/              # Default output directory
```

## 🔧 Troubleshooting

### Common Issues

**ImportError: No module named 'whisper'**
```powershell
pip install openai-whisper
```

**NPU not detected**
```powershell
pip install onnxruntime-qnn
# Ensure you're on ARM64 Windows with Snapdragon processor
```

**FFmpeg not found**
```powershell
# Download FFmpeg ARM64 build and place ffmpeg.exe in project directory
```

**Memory errors with large files**
- Use smaller model size (`tiny`, `base`)
- Process shorter audio segments
- Ensure sufficient free disk space

### Performance Optimization

**For better accuracy:**
- Use `medium` or `large` models
- Ensure good quality audio input
- Specify correct language

**For faster processing:**
- Use `tiny` or `base` models
- Enable NPU acceleration
- Process shorter segments

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI** - Whisper speech recognition models
- **Qualcomm** - Snapdragon X Elite NPU technology
- **Microsoft** - Windows ARM64 platform
- **ONNX Runtime** - NPU acceleration framework

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/Rob142857/speech2textrmeARM64/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Rob142857/speech2textrmeARM64/discussions)
- 📚 **Documentation**: [Wiki](https://github.com/Rob142857/speech2textrmeARM64/wiki)

---

**Built with ❤️ for the ARM64 Windows community**
