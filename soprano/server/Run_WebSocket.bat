@echo off
title Soprano TTS WebSocket Server
echo Starting Soprano TTS WebSocket Server...
echo.

REM Activate virtual environment if present
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist env\Scripts\activate.bat (
    call env\Scripts\activate.bat
)

REM Start the WebSocket server
echo Starting WebSocket server on ws://localhost:8001/ws/tts
python -c "
import uvicorn
import torch
print('Starting Soprano TTS WebSocket Server...')
device = 'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'
print(f'Available device: {device}')
uvicorn.run(
    'soprano.server.websocket:app', 
    host='localhost', 
    port=8001, 
    log_level='info'
)
"

pause