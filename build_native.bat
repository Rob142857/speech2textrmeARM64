@echo off
setlocal
REM Build Rust native extensions (tokenizer + mel) for ARM64 Windows

echo [1] Start batch

if not exist rust_tokenizer\Cargo.toml (echo rust_tokenizer crate missing & exit /b 1)
if not exist rust_mel\Cargo.toml (echo rust_mel crate missing & exit /b 1)
echo [2] Crates found

REM Ensure Rust toolchain
where rustup >nul 2>&1
echo [3] rustup presence check complete

rustup target add aarch64-pc-windows-msvc >nul 2>&1
echo [4] Target added

set RUSTFLAGS=-Ctarget-feature=+neon
echo [5] RUSTFLAGS set

set CARGO_PATH=%USERPROFILE%\.cargo\bin\cargo.exe
if not exist "%CARGO_PATH%" (
	echo cargo.exe not found at %CARGO_PATH%
	exit /b 1
)
echo Using cargo at %CARGO_PATH%
for %%C in (rust_tokenizer rust_mel) do (
	echo Building %%C ...
	pushd %%C
	"%CARGO_PATH%" build --release --target aarch64-pc-windows-msvc || goto :err
	popd
)
echo Builds done

if not exist native mkdir native
echo [9] Native folder ready
copy /y rust_tokenizer\target\aarch64-pc-windows-msvc\release\tokenizer_native.* native\ >nul 2>&1
copy /y rust_mel\target\aarch64-pc-windows-msvc\release\mel_native.* native\ >nul 2>&1
echo [10] Copied artifacts

echo Native builds complete. Artifacts in native\
exit /b 0

:err
echo Build failed
exit /b 1
