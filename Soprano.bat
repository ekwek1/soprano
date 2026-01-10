@echo off
title Soprano TTS Launcher
color 0A

:menu
cls
echo ================================================
echo           SOPRANO TTS LAUNCHER
echo ================================================
echo.
echo Select an option:
echo.
echo 1. Launch API Server
echo 2. Launch CLI Interface
echo 3. Launch API Test Client
echo 4. Launch WebSocket Server
echo 5. Launch WebSocket Test Client
echo 6. Exit
echo.
set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto api_server
if "%choice%"=="2" goto cli
if "%choice%"=="3" goto api_test
if "%choice%"=="4" goto websocket_server
if "%choice%"=="5" goto websocket_test
if "%choice%"=="6" goto exit
goto invalid_choice

:invalid_choice
echo.
echo Invalid choice. Please enter 1, 2, 3, or 4.
timeout /t 2 /nobreak >nul
goto menu

:api_server
echo.
echo Starting Soprano TTS API Server...
echo.
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist env\Scripts\activate.bat (
    call env\Scripts\activate.bat
)
cd soprano\server
call Run_API.bat
cd ..\..
goto end

:cli
echo.
echo Starting Soprano TTS CLI Interface...
echo.
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist env\Scripts\activate.bat (
    call env\Scripts\activate.bat
)
python -m CLI.soprano_cli
goto end

:api_test
echo.
echo Starting Soprano TTS API Test Client...
echo.
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist env\Scripts\activate.bat (
    call env\Scripts\activate.bat
)
cd soprano\server
call Run_API_Test.bat "Hello, this is a test of the Soprano TTS API. The system is working properly."
cd ..\..
goto end

:websocket_server
echo.
echo Starting Soprano TTS WebSocket Server...
echo.
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist env\Scripts\activate.bat (
    call env\Scripts\activate.bat
)
cd soprano\server
call Run_WebSocket.bat
cd ..\..
goto end

:websocket_test
echo.
echo Starting Soprano TTS WebSocket Test Client...
echo.
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist env\Scripts\activate.bat (
    call env\Scripts\activate.bat
)
cd soprano\server
call Run_WebSocket_Test.bat "Hello, this is a test of the WebSocket TTS system. Audio is streaming in real-time."
cd ..\..
goto end

:exit
exit /b

:end
pause