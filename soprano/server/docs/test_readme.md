# Soprano TTS Test Clients

This directory contains test clients for both the API and WebSocket servers.

## API Test Client

The API test client allows you to test the REST API server functionality.

### Usage
```bash
python -m soprano.server.test_api "Your text here"
```

### Features
- Tests the main TTS endpoint
- Includes health check functionality
- Saves received audio to audio_output directory
- Handles connection errors gracefully
- Uses aiohttp for async HTTP requests

## WebSocket Test Client

The WebSocket test client allows you to test the WebSocket streaming server functionality.

### Usage
```bash
python -m soprano.server.test_websocket "Your text here"
```

### Features
- Tests WebSocket connection and streaming
- Real-time audio playback using PyAudio
- Connection testing with ping/pong
- Proper audio stream management
- Comprehensive error handling

## Prerequisites

- For API tests: `pip install aiohttp`
- For WebSocket tests: `pip install websockets pyaudio`
- Running API server on http://localhost:8000
- Running WebSocket server on ws://localhost:8001/ws/tts