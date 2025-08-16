# ARM64 Whisper Transcription Tool

A high-performance speech-to-text transcription tool optimized for ARM64 Windows devices with NPU acceleration.

## Features

- 🧠 **Local OpenAI Whisper Processing** - No cloud dependencies
- 🚀 **NPU Acceleration** - Leverages Qualcomm Snapdragon X Elite NPU via ONNX Runtime QNN
- 🎵 **Multi-Format Support** - Audio and video files (MP3, MP4, WAV, M4A, etc.)
- 📄 **DOCX Output** - Properly formatted documents with punctuation and paragraphing
- 🖥️ **Dual Interface** - Both GUI and command-line interfaces
- 🏗️ **ARM64 Native** - Optimized for Windows ARM64 architecture

## Requirements

- Windows 11 ARM64 (tested on Surface Laptop 7)
- Python 3.9+ (ARM64 version)
- Qualcomm Snapdragon X Elite NPU (optional but recommended)

## Quick Start

### Installation

```powershell
# Clone and setup
git clone <repository-url>
cd speech2textrmeARM64

# Install dependencies
pip install -r requirements.txt

# Download Whisper models (first run)
python setup.py
```

### Usage

#### GUI Mode
```powershell
python gui_transcribe.py
```

#### Command Line
```powershell
# Basic transcription
python transcribe.py "path/to/audio.mp3"

# Advanced options
python transcribe.py "video.mp4" --model medium --use-npu --punctuate --output "output_folder"
```

## Architecture

### Core Components

- **`transcribe.py`** - Main transcription engine with NPU support
- **`gui_transcribe.py`** - Tkinter-based GUI interface
- **`audio_processor.py`** - FFmpeg audio processing pipeline
- **`whisper_npu.py`** - ONNX Runtime QNN integration for NPU acceleration
- **`document_generator.py`** - DOCX formatting and export
- **`setup.py`** - Model download and environment setup

### NPU Acceleration

The tool automatically detects and uses the Qualcomm Snapdragon X Elite NPU when available:

- **Primary**: ONNX Runtime QNN Provider for NPU acceleration
- **Fallback**: CPU-based processing when NPU unavailable
- **Optimization**: Model quantization for ARM64 performance

## Supported Formats

### Input
- **Audio**: MP3, WAV, M4A, AAC, FLAC, OGG
- **Video**: MP4, AVI, MOV, MKV, WMV, WebM

### Output
- **Text**: UTF-8 encoded plain text
- **DOCX**: Formatted Microsoft Word documents
- **Metadata**: JSON processing logs and timestamps

## Performance

Typical processing times on Surface Laptop 7 (Snapdragon X Elite):

| Audio Length | NPU Mode | CPU Mode |
|--------------|----------|----------|
| 1 minute     | ~10 sec  | ~30 sec  |
| 10 minutes   | ~45 sec  | ~4 min   |
| 1 hour       | ~4 min   | ~20 min  |

## Development

### Project Structure
```
speech2textrmeARM64/
├── transcribe.py          # Main transcription engine
├── gui_transcribe.py      # GUI interface  
├── audio_processor.py     # Audio preprocessing
├── whisper_npu.py        # NPU integration
├── document_generator.py  # DOCX generation
├── setup.py              # Environment setup
├── requirements.txt       # Dependencies
├── models/               # Downloaded Whisper models
├── temp/                 # Temporary processing files
└── output/               # Default output directory
```

### Dependencies

- **OpenAI Whisper**: Core transcription model
- **ONNX Runtime QNN**: NPU acceleration provider
- **PyTorch ARM64**: Neural network framework
- **FFmpeg**: Audio/video processing
- **python-docx**: Document generation
- **tkinter**: GUI framework

## Troubleshooting

### NPU Issues
```powershell
# Check NPU availability
python -c "import onnxruntime as ort; print('QNN Provider:', 'QNNExecutionProvider' in ort.get_available_providers())"
```

### Audio Processing Issues
```powershell
# Test FFmpeg
ffmpeg -version
```

### Model Download Issues
```powershell
# Manual model download
python -c "import whisper; whisper.load_model('base')"
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please ensure all changes are tested on ARM64 Windows devices.

---

**Platform**: Windows ARM64 | **Optimized for**: Surface Laptop 7 Snapdragon X Elite | **NPU**: Qualcomm QNN Provider
