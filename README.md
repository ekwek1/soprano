# Soprano TTS

Soprano is an ultra-realistic Text-to-Speech system that provides both REST API and WebSocket streaming capabilities.

## Features

- **High Quality Audio**: Generates ultra-realistic speech using advanced TTS models
- **Multiple Interfaces**: REST API and WebSocket streaming options
- **OpenAI Compatible**: Follows OpenAI's speech endpoint format
- **Real-time Streaming**: WebSocket support for real-time audio streaming
- **Configurable Parameters**: Supports temperature, top_p, repetition_penalty, and min_text_length controls

## Components

- **API Server**: RESTful API with OpenAI-compatible endpoints
- **WebSocket Server**: Real-time audio streaming via WebSocket
- **CLI Interface**: Interactive command-line interface
- **Test Clients**: Dedicated test clients for both API and WebSocket

## Quick Start

### Using the Launcher
Run `Soprano.bat` to access the main menu with options to launch any component.

### API Server
Start the API server and send requests to `http://localhost:8000/v1/audio/speech`

### WebSocket Server
Start the WebSocket server and connect to `ws://localhost:8001/ws/tts`

## Endpoints

### API
- `POST /v1/audio/speech` - Generate speech from text
- `GET /health` - Health check endpoint
- `GET /` - Root endpoint with API information

### WebSocket
- `ws://localhost:8001/ws/tts` - Real-time TTS streaming

## Integration

The API is designed for easy integration with workflow automation platforms like n8n, Zapier, and other systems that can make HTTP requests.

## License

Licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.