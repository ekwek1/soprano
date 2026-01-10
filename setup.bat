@echo off
title Soprano TTS Setup
color 0A

echo ================================================
echo        SOPRANO TTS SETUP
echo ================================================
echo.
echo This script will:
echo 1. Install the Soprano TTS package
echo 2. Install/fix PyTorch with CUDA support
echo 3. Verify the installation
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause >nul

echo.
echo Installing Soprano TTS package...
echo.

REM Install the package in editable mode
pip install -e .

if %errorlevel% neq 0 (
    echo Error occurred during installation. Attempting to fix...
    goto fix_pytorch
)

echo.
echo Installing PyTorch with CUDA support...
echo.

:fix_pytorch
REM Uninstall current PyTorch
pip uninstall -y torch torchvision torchaudio

REM Install PyTorch with CUDA 12.6 support
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

if %errorlevel% neq 0 (
    echo Warning: PyTorch CUDA installation failed. Installing CPU version...
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
)

echo.
echo Verifying installation...
echo.

REM Test the installation
python -c "import soprano; from soprano import SopranoTTS; print('Soprano TTS imported successfully'); print('Installation verified successfully!')"

if %errorlevel% neq 0 (
    echo Warning: Verification failed, but installation may still be OK.
)

echo.
echo ================================================
echo Setup completed!
echo.
echo To use Soprano TTS:
echo - Run Soprano.bat to access the main menu
echo - Or run individual components as needed
echo ================================================

pause