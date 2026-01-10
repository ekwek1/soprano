@echo off
title Soprano TTS API Server
echo Starting Soprano TTS API Server...
echo.

REM Activate virtual environment if present
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist env\Scripts\activate.bat (
    call env\Scripts\activate.bat
)

REM Start the API server with performance optimizations
echo Starting server on http://localhost:8000
python -c "import uvicorn; import torch; print('Starting Soprano TTS API Server...'); device = 'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'; print(f'Available device: {device}'); uvicorn.run('soprano.server.api:app', host='localhost', port=8000, workers=1, log_level='info')"

pause