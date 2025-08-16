@echo off
REM ARM64 Whisper Transcription Launcher
REM Optimized for Windows ARM64 with NPU acceleration

REM --- Short TEMP mitigation for Windows path length issues (onnxsim / deep build trees) ---
if not exist C:\s2t_tmp mkdir C:\s2t_tmp >nul 2>&1
set "TMP=C:\s2t_tmp"
set "TEMP=C:\s2t_tmp"
REM Warn if current path is long ( > 80 chars )
set "CURP=%cd%"
set /a LEN=0
for /f "delims=":" tokens=1*" %%A in ('cmd /c echo %CURP%') do set "LINE=%%A"
call :strlen LEN "%CURP%"
if %LEN% GTR 80 (
    echo ⚠️  Current path length %LEN% may cause pip build failures. Consider moving project to C:\s2t
)

goto :after_strlen_helper

:strlen
setlocal EnableDelayedExpansion
set "s=%~2"
set /a len=0
:strlen_loop
if defined s (if not "!s!"=="" (set "s=!s:~1!"& set /a len+=1 & goto strlen_loop))
endlocal & set "%1=%len%"
goto :eof

:after_strlen_helper

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

REM Check if project files exist
if not exist "gui.py" (
    echo ❌ Project files not found
    echo Please ensure you're in the correct directory
    pause
    exit /b 1
)

REM Show menu
:MENU
echo Select an option:
echo.
echo   [1] Launch GUI Application (Placeholder Mode)
echo   [2] Command Line Help  
echo   [3] Test Current Setup
echo   [4] Install Dependencies (needs C++ compiler)
echo   [5] QNN / NPU Guide
echo   [6] Convert Whisper -> ONNX
echo   [7] Run ONNX QNN Demo
echo   [8] Exit
echo.
set /p choice="Enter your choice (1-8): "

if "%choice%"=="1" goto GUI
if "%choice%"=="2" goto HELP
if "%choice%"=="3" goto TEST
if "%choice%"=="4" goto SETUP
if "%choice%"=="5" goto QNN
if "%choice%"=="6" goto CONVERT
if "%choice%"=="7" goto RUNQNN
if "%choice%"=="8" goto EXIT
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

:QNN
echo.
echo 📘 Opening QNN / NPU guide...
if exist "npu\README_QNN.md" (
    start npu\README_QNN.md
) else (
    echo File missing: npu\README_QNN.md
)
pause
goto MENU

:CONVERT
echo.
echo 🔄 Convert Whisper model to ONNX (encoder/decoder)...
set /p mdl="Model id (default openai/whisper-large-v3): "
if "%mdl%"=="" set mdl=openai/whisper-large-v3
set /p outdir="Output dir (default models\whisper_large_v3_onnx): "
if "%outdir%"=="" set outdir=models\whisper_large_v3_onnx
python scripts\convert_whisper_to_onnx.py --model %mdl% --output-dir %outdir%
pause
goto MENU

:RUNQNN
echo.
echo 🚀 Run ONNX QNN demo transcription
set /p modeldir="Model dir (e.g. models\whisper_large_v3_onnx): "
set /p audiofile="Audio wav file: "
if "%modeldir%"=="" goto MENU
if "%audiofile%"=="" goto MENU
python scripts\transcribe_qnn.py --model-dir %modeldir% --audio %audiofile%
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
