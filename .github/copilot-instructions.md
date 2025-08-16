<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# ARM64 Whisper Transcription Project - Copilot Instructions

This project is specifically designed for ARM64 Windows devices (Surface Laptop 7 with Qualcomm Snapdragon X Elite) to provide local OpenAI Whisper speech-to-text transcription with NPU acceleration.

## Core Requirements

- **Platform**: Windows ARM64 (Qualcomm Snapdragon X Elite NPU)
- **AI Engine**: OpenAI Whisper (local processing, no cloud APIs)
- **Acceleration**: NPU via ONNX Runtime QNN Provider
- **Audio Processing**: FFmpeg for multi-format support
- **Output**: DOCX files with proper punctuation and paragraphing
- **Interface**: Both GUI (tkinter) and CLI

## Technical Constraints

- All dependencies must be ARM64-compatible
- Use ONNX Runtime QNN Provider for NPU acceleration
- Avoid x64-only binaries (whisper.cpp, some PyTorch versions)
- Prioritize native ARM64 Python packages
- Use OpenAI Whisper Python library with ONNX optimization

## Architecture Principles

1. **NPU First**: Always attempt to use QNNExecutionProvider when available
2. **Local Processing**: No external API dependencies (Google, Azure, etc.)
3. **Multi-format Support**: Handle video/audio files via FFmpeg
4. **Quality Output**: Proper sentence structure, punctuation, paragraphing
5. **Error Resilience**: Graceful fallbacks when NPU unavailable

## Key Dependencies to Focus On

- `openai-whisper`: Core transcription engine
- `onnxruntime-qnn`: NPU provider for ARM64
- `torch`: ARM64-compatible version
- `python-docx`: Document generation
- `tkinter`: GUI framework (built-in)
- `ffmpeg-python`: Audio processing

## Code Style

- Use type hints for better code clarity
- Include comprehensive error handling
- Add progress indicators for long operations  
- Document NPU usage and fallback strategies
- Follow PEP 8 naming conventions
