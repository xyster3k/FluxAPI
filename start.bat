@echo off
echo.
echo ==========================================
echo   Flux 2 Multi-Model API Server
echo ==========================================
echo.

:: Check if .env exists
if not exist ".env" (
    echo [WARNING] .env file not found!
    echo.
    if exist ".env.example" (
        echo Creating .env from .env.example...
        copy .env.example .env >nul
        echo [OK] Created .env
        echo.
        echo IMPORTANT: Edit .env to set your API key before using the server.
        echo.
    ) else (
        echo Please run setup.bat first, or create .env from .env.example
        pause
        exit /b 1
    )
)

echo Starting server...
echo.
echo First startup will download models from Hugging Face (~10-25GB).
echo Models are cached and won't be re-downloaded.
echo.
echo Press Ctrl+C to stop the server.
echo.

py -3.11 server.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Server stopped with error code %errorlevel%
    echo.
    echo Common issues:
    echo - Run setup.bat if you haven't installed dependencies
    echo - Check that Python 3.11 is installed
    echo - Ensure NVIDIA drivers are up to date
    echo.
)

pause
