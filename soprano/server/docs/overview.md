# Soprano TTS API Documentation

## Overview

The Soprano TTS API is a high-performance text-to-speech service that converts text to realistic audio using advanced neural models. The API follows OpenAI's speech endpoint format for compatibility and ease of use.

## Features

- **High-Quality Audio**: Uses state-of-the-art neural models for realistic speech synthesis
- **GPU Acceleration**: Automatically utilizes CUDA when available for faster processing
- **Error Handling**: Comprehensive error handling with circuit breaker and retry mechanisms
- **File Management**: Automatic sequential file naming and cleanup of old files
- **OpenAI Compatible**: Follows OpenAI's speech endpoint format

## Architecture

The API is built using FastAPI and follows a modular architecture:

- **API Layer**: FastAPI endpoints with request/response handling
- **Business Logic**: TTSManager with singleton pattern and resource management
- **Model Layer**: SopranoTTS with neural model integration
- **Utilities**: Audio processing, file management, and error handling components

## Requirements

- Python 3.10+
- CUDA-compatible GPU (optional, CPU fallback available)
- Required Python packages (see pyproject.toml)

## Installation

1. Clone the repository
2. Install dependencies: `pip install soprano-tts`
3. Run the API server

## API Server Execution

To start the API server, run:
```
python soprano\server\api.py
```

The server will start on `http://localhost:8000` and automatically detect CUDA availability.