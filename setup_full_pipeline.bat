@echo on
REM ================================================================
REM  ARM64 Whisper Full Pipeline Setup & Launch (Verbose Logging)
REM  Tasks:
REM    1. Verify / locate MSVC build tools (link.exe, cl.exe)
REM    2. Verify / install Rust toolchain
REM    3. Ensure cargo in PATH
REM    4. Install Python deps (tokenizers/transformers/sentencepiece)
REM    5. Export Whisper Large-V3 to ONNX (encoder + decoder_iter)
REM    6. Build native Rust extensions (tokenizer + mel)
REM    7. Launch GUI (native ONNX/QNN path)
REM  Usage:  setup_full_pipeline.bat
REM  Log: setup_full_pipeline.log
REM ================================================================

setlocal enableextensions enabledelayedexpansion
set LOG=setup_full_pipeline.log
echo ===============================================================> %LOG%
echo  SETUP START %DATE% %TIME%>> %LOG%
echo ===============================================================>> %LOG%

REM Helper :run label for logging commands
goto :run_skip
:run
set DESC=%~1
shift
echo.>> %LOG%
echo [CMD] %DESC% >> %LOG%
echo [CMD] %DESC%
echo > tmp_run_cmd.bat @echo off
echo %*>> tmp_run_cmd.bat
call tmp_run_cmd.bat >> %LOG% 2>&1
set ERR=%ERRORLEVEL%
del tmp_run_cmd.bat >nul 2>&1
if not %ERR%==0 (
  echo [FAIL] %DESC% (exit %ERR%)
  echo [FAIL] %DESC% (exit %ERR%)>> %LOG%
) else (
  echo [OK] %DESC%
  echo [OK] %DESC%>> %LOG%
)
exit /b %ERR%
:run_skip

REM Step 0: Sanity
where python >nul 2>&1 || (echo Python not found. Install Python 3.11+ and re-run.& exit /b 1)
for /f "delims=" %%v in ('python -c "import platform;print(platform.machine())"') do set ARCH=%%v
echo Detected Python arch: %ARCH% >> %LOG%

REM Step 1: MSVC / ARM64 toolchain detection (robust)
echo --- MSVC / ARM64 detection --- >> %LOG%

REM Prefer vswhere if available
set VSWHERE="C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if exist %VSWHERE% (
  for /f "usebackq tokens=*" %%I in (`%VSWHERE% -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do set VSINSTALL=%%I
)

REM Fall back to manual probe if vswhere didn't yield a path
if not defined VSINSTALL (
  for %%E in (BuildTools Community Enterprise Professional) do (
    if exist "C:\Program Files\Microsoft Visual Studio\2022\%%E" set VSINSTALL=C:\Program Files\Microsoft Visual Studio\2022\%%E
  )
)

if not defined VSINSTALL (
  echo ERROR: Could not locate a Visual Studio 2022 installation >> %LOG%
  echo Visual Studio 2022 Build Tools not found. Install C++ Build Tools (Desktop C++ incl. ARM64) and re-run.
  exit /b 1
)
echo VSINSTALL=%VSINSTALL% >> %LOG%

REM Call vcvarsall to populate env (silence output into log)
set VCVARS="%VSINSTALL%\VC\Auxiliary\Build\vcvarsall.bat"
if exist %VCVARS% (
  call %VCVARS% amd64_arm64 >> %LOG% 2>&1
) else (
  echo WARN: vcvarsall.bat missing at %VCVARS% >> %LOG%
)

REM Search for latest MSVC tools version folder
set MSVC_TOOLS_ROOT=
for /f "delims=" %%D in ('dir /b /ad "%VSINSTALL%\VC\Tools\MSVC" 2^>nul ^| sort') do set MSVC_TOOLS_ROOT=%VSINSTALL%\VC\Tools\MSVC\%%D
if not defined MSVC_TOOLS_ROOT (
  echo ERROR: MSVC tools directory missing under VS install >> %LOG%
  exit /b 1
)
echo MSVC_TOOLS_ROOT=%MSVC_TOOLS_ROOT% >> %LOG%

REM Candidate bin paths (order matters)
set CANDIDATE_BINS=%MSVC_TOOLS_ROOT%\bin\HostARM64\arm64;%MSVC_TOOLS_ROOT%\bin\Hostarm64\arm64;%MSVC_TOOLS_ROOT%\bin\Hostx64\arm64;%MSVC_TOOLS_ROOT%\bin\HostX64\arm64
set FOUND_LINK=
for %%P in (%CANDIDATE_BINS%) do (
  if exist "%%P\link.exe" (
    set FOUND_LINK=%%P
    goto :found_link
  )
)
:found_link
if not defined FOUND_LINK (
  echo ERROR: ARM64 link.exe not found. Ensure "MSVC v143 - VS 2022 C++ ARM64 build tools" component is installed. >> %LOG%
  echo Missing ARM64 linker. Re-run VS Installer -> Modify -> Individual Components -> add:
  echo   - MSVC v143 - VS 2022 C++ ARM64 build tools
  echo   - MSVC v143 - VS 2022 C++ x64/x86 build tools (already likely present)
  echo   - Windows 11 SDK (10.0.22621.x)
  exit /b 1
)
echo Using ARM64 toolchain bin: %FOUND_LINK% >> %LOG%
set PATH=%FOUND_LINK%;%PATH%

where link >nul 2>&1 || (echo ERROR: link.exe still not discoverable after PATH update >> %LOG% & exit /b 1)
where cl >nul 2>&1 || echo WARN: cl.exe not found though link.exe present >> %LOG%
echo MSVC OK (ARM64 link.exe detected) >> %LOG%

REM Step 2: Rust
where cargo >nul 2>&1
if errorlevel 1 (
  echo Installing rustup ... >> %LOG%
  powershell -NoProfile -Command "Invoke-WebRequest https://win.rustup.rs/ -OutFile rustup-init.exe" >> %LOG% 2>&1
  rustup-init.exe -y --default-toolchain stable-aarch64-pc-windows-msvc --profile minimal >> %LOG% 2>&1
  del rustup-init.exe >nul 2>&1
)
set PATH=%USERPROFILE%\.cargo\bin;%PATH%
where cargo >nul 2>&1 || (echo ERROR: cargo not found after install >> %LOG% & exit /b 1)
echo Rust toolchain OK >> %LOG%

REM Step 3: Python deps
call :run "Upgrade pip" pip install --upgrade pip
call :run "Core deps" pip install numpy onnx onnxruntime-qnn python-docx ffmpeg-python soundfile
call :run "tokenizers" pip install --no-cache-dir tokenizers==0.21.4
call :run "transformers+sentencepiece" pip install --no-cache-dir transformers==4.55.2 sentencepiece==0.2.1

REM Step 4: Export ONNX large model
set MODEL_ID=openai/whisper-large-v3
set MODEL_DIR=models\whisper_large_onnx
if exist %MODEL_DIR% rmdir /s /q %MODEL_DIR% >> %LOG% 2>&1
call :run "Export ONNX" python scripts\convert_whisper_to_onnx.py --model %MODEL_ID% --output-dir %MODEL_DIR% --simplify
if not exist %MODEL_DIR%\encoder.onnx (
  echo ERROR: ONNX export failed (encoder.onnx missing) >> %LOG%
  echo ONNX export failed. See %LOG%. & exit /b 1
)
echo ONNX export OK >> %LOG%

REM Step 5: Build natives
call :run "Build natives" cmd /c build_native.bat

REM Step 6: Launch GUI
echo Launching GUI... >> %LOG%
python gui.py

echo ===============================================================>> %LOG%
echo  SETUP COMPLETE %DATE% %TIME%>> %LOG%
echo ===============================================================>> %LOG%
endlocal
exit /b 0
