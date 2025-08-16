@echo off
setlocal EnableDelayedExpansion
REM Relocate project to a short path (C:\s2t) to avoid Windows MAX_PATH build issues
set "TARGET=C:\s2t"
echo Relocating project to %TARGET% ...
if not exist "%TARGET%" mkdir "%TARGET%"
for /f "delims=" %%I in ('cd') do set "SRC=%%I"
echo Source: %SRC%
echo Copying files (this may take a moment)...
REM Use xcopy to avoid special robocopy parsing in PowerShell
xcopy "%SRC%" "%TARGET%" /E /I /Y >nul 2>&1
if errorlevel 1 (
  echo ⚠️  xcopy reported code %errorlevel%. Verify files.
)
echo ✅ Copy attempt complete.
echo ------------------------------------------------------------
echo NEXT STEPS:
echo 1. Close this VS Code window.
echo 2. Open new VS Code at: %TARGET%
echo 3. python -m venv .venv  (inside new location)
echo 4. .venv\Scripts\activate
echo 5. pip install --upgrade pip
echo 6. (Re)install minimal deps needed
echo ------------------------------------------------------------
echo TIP: After reopening, run run.bat (options 5-7 for QNN/ONNX).
pause
endlocal
