# Soprano TTS Server

This directory contains the server components for the Soprano TTS system, including the API server and test utilities.

## Components

- **API Server** (`api.py`): OpenAI-compatible text-to-speech API server
- **WebSocket Server** (`websocket.py`): Real-time streaming TTS via WebSocket
- **API Test Client** (`test_api.py`): Test client for the API server
- **WebSocket Test Client** (`test_websocket.py`): Test client for the WebSocket server
- **Documentation**: README files explaining usage and integration

## API Compatibility

The API server implements OpenAI-compatible endpoints, making it easy to integrate with existing applications and services that expect OpenAI's speech API format.

## Quick Start

1. Start the API server: `python -m soprano.server.api`
2. Start the WebSocket server: `python -m soprano.server.websocket`
3. Test with the clients: `python -m soprano.server.test_api` or `python -m soprano.server.test_websocket`
4. Or use directly with HTTP requests to `http://localhost:8000/v1/audio/speech`

## Integration Ready

The server is designed for seamless integration with:
- Workflow automation tools (like n8n)
- Web and mobile applications
- Voice-enabled systems
- Any system capable of making HTTP requests or WebSocket connections