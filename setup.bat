@echo off
setlocal enabledelayedexpansion

echo.
echo ==========================================
echo   Flux 2 Multi-Model API Server Setup
echo ==========================================
echo.

:: Check if Python 3.11 is available
where py >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python launcher 'py' not found!
    echo Please install Python 3.11 from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: Check Python version
for /f "tokens=*" %%i in ('py -3.11 --version 2^>nul') do set PYVER=%%i
if "%PYVER%"=="" (
    echo [ERROR] Python 3.11 not found!
    echo Please install Python 3.11 from https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [OK] Found %PYVER%

:: Check for NVIDIA GPU
nvidia-smi >nul 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] nvidia-smi not found. Make sure you have an NVIDIA GPU with drivers installed.
) else (
    echo [OK] NVIDIA GPU detected
)

echo.
echo Step 1/4: Detecting GPU architecture...
echo.

:: Try to detect GPU architecture for correct PyTorch version
set "CUDA_VERSION=cu121"
for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul') do (
    echo Detected GPU: %%i
    echo %%i | findstr /i "5090 5080 5070 5060 Blackwell" >nul
    if !errorlevel! equ 0 (
        set "CUDA_VERSION=cu128"
        echo [INFO] Blackwell GPU detected - using CUDA 12.8 nightly
    )
)

echo.
echo Step 2/4: Installing PyTorch with CUDA...
echo.

if "%CUDA_VERSION%"=="cu128" (
    echo Installing PyTorch nightly for Blackwell GPUs...
    py -3.11 -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
) else (
    echo Installing PyTorch stable for Ampere/Ada GPUs...
    py -3.11 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
)

if %errorlevel% neq 0 (
    echo [ERROR] Failed to install PyTorch
    pause
    exit /b 1
)
echo [OK] PyTorch installed

echo.
echo Step 3/4: Installing dependencies...
echo.

py -3.11 -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install requirements
    pause
    exit /b 1
)
echo [OK] Base dependencies installed

echo.
echo Step 4/4: Installing diffusers from git (required for Flux 2)...
echo.

py -3.11 -m pip install git+https://github.com/huggingface/diffusers
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install diffusers from git
    pause
    exit /b 1
)
echo [OK] Diffusers installed

echo.
echo ==========================================
echo   Setup Complete!
echo ==========================================
echo.
echo Next steps:
echo 1. Copy .env.example to .env and configure your API key
echo 2. Run start.bat to start the server
echo 3. First startup will download models (~10-15GB each)
echo.
echo Models will be cached in: %USERPROFILE%\.cache\huggingface\hub
echo.

:: Create .env from example if it doesn't exist
if not exist ".env" (
    if exist ".env.example" (
        echo Creating .env from .env.example...
        copy .env.example .env >nul
        echo [OK] Created .env - please edit it with your settings
    )
)

pause
