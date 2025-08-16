@echo off
REM Quick diagnostic for MSVC ARM64 toolchain (simplified: search only)
echo === MSVC ARM64 Diagnostic (search mode) ===
set ROOTVS=C:\Program Files\Microsoft Visual Studio\2022
if exist "%ROOTVS%" (
  echo Scanning %ROOTVS% for MSVC toolchains...
) else (
  echo Visual Studio 2022 root not found at %ROOTVS%
)

echo --- Candidate ARM64 linker paths ---
set FOUND=
for /f "delims=" %%F in ('dir /s /b "%ROOTVS%\*\VC\Tools\MSVC\*\bin\Host*\arm64\link.exe" 2^>nul') do (
  echo %%F
  set FOUND=1
)
if not defined FOUND echo (none found)

echo --- Candidate cl.exe (compiler) paths ---
set FOUNDCL=
for /f "delims=" %%F in ('dir /s /b "%ROOTVS%\*\VC\Tools\MSVC\*\bin\Host*\arm64\cl.exe" 2^>nul') do (
  echo %%F
  set FOUNDCL=1
)
if not defined FOUNDCL echo (none found)

echo --- PATH link/cl (current session) ---
where link 2>nul
where cl 2>nul

echo --- Result ---
if defined FOUND (
  echo ARM64 linker present. OK.
) else (
  echo ARM64 linker NOT found. Install / add components:
  echo   * MSVC v143 - VS 2022 C++ ARM64 build tools
  echo   * Windows 11 SDK (10.0.22621.*)
  echo Then re-run.
)

echo ====================================
exit /b 0

echo --- Searching for ARM64 link.exe ---
set FOUND=
for /f "delims=" %%F in ('dir /s /b "C:\Program Files\Microsoft Visual Studio\2022\*\VC\Tools\MSVC\*\bin\Host*\arm64\link.exe" 2^>nul') do (
  echo %%F
  set FOUND=1
)
if not defined FOUND echo (none found)

echo --- PATH link/cl ---
where link 2>nul
where cl 2>nul

echo --- Recommendation ---
echo If no ARM64 link.exe paths listed, open Visual Studio Installer -> Modify -> Individual Components and add:
echo   * MSVC v143 - VS 2022 C++ ARM64 build tools
echo   * Windows 11 SDK (10.0.22621.*)
echo   * C++ CMake tools for Windows (optional but helpful)
echo Then re-run setup_full_pipeline.bat
echo ====================================
exit /b 0
