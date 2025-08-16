@echo off
REM ARM64 Whisper Transcription Launcher
REM Optimized for Windows ARM64 with NPU acceleration

title ARM64 Whisper Transcription

echo.
echo ================================================
echo   ARM64 Whisper Transcription Engine
echo   NPU-Accelerated Speech-to-Text
echo ================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found in PATH
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

REM Check if this is ARM64
for /f "tokens=*" %%i in ('python -c "import platform; print(platform.machine())"') do set ARCH=%%i
if not "%ARCH%"=="ARM64" (
    echo ⚠️  Warning: Not running on ARM64 architecture
    echo   NPU acceleration may not be available
    echo   Current architecture: %ARCH%
    echo.
)

REM Check if setup has been run
if not exist "whisper_npu.py" (
    echo ❌ Project files not found
    echo Please run setup first: python setup.py
    pause
    exit /b 1
)

REM Show menu
:MENU
echo Select an option:
echo.
echo   [1] Launch GUI Application
echo   [2] Run Setup/Installation
echo   [3] Test System (NPU check)
echo   [4] Command Line Help
echo   [5] Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto GUI
if "%choice%"=="2" goto SETUP
if "%choice%"=="3" goto TEST
if "%choice%"=="4" goto HELP
if "%choice%"=="5" goto EXIT
echo Invalid choice. Please try again.
goto MENU

:GUI
echo.
echo 🚀 Launching ARM64 Whisper GUI...
python gui.py
if %errorlevel% neq 0 (
    echo.
    echo ❌ GUI failed to start
    echo Check that dependencies are installed: python setup.py
    pause
)
goto MENU

:SETUP
echo.
echo 📦 Running setup and installation...
python setup.py
pause
goto MENU

:TEST
echo.
echo 🧪 Testing system capabilities...
echo.
echo Platform Information:
python -c "import platform; print(f'Platform: {platform.system()} {platform.machine()}'); print(f'Python: {platform.python_version()}')"
echo.
echo NPU Detection:
python -c "try: import onnxruntime as ort; print('Available providers:', ort.get_available_providers()); npu_available = 'QNNExecutionProvider' in ort.get_available_providers(); print('NPU Available:', npu_available); except Exception as e: print('Error:', e)"
echo.
echo Whisper Test:
python -c "try: import whisper; print('✅ OpenAI Whisper available'); except ImportError: print('❌ OpenAI Whisper not installed')"
echo.
pause
goto MENU

:HELP
echo.
echo 📚 Command Line Usage:
echo.
echo Basic transcription:
echo   python transcribe.py "path\to\audio.mp3"
echo.
echo Advanced options:
echo   python transcribe.py "video.mp4" --model medium --language en --output results
echo.
echo Available options:
echo   --model      Model size: tiny, base, small, medium, large
echo   --language   Language code: en, es, fr, de, etc. (auto if not specified)
echo   --output     Output directory
echo   --no-npu     Disable NPU acceleration
echo   --word-timestamps    Include word-level timestamps
echo   --verbose    Show detailed progress
echo.
pause
goto MENU

:EXIT
echo.
echo 👋 Thank you for using ARM64 Whisper Transcription!
exit /b 0
