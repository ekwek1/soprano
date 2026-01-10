# Soprano TTS API

The Soprano TTS API provides a high-quality, ultra-realistic text-to-speech service with OpenAI-compatible endpoints. This API allows you to convert text to natural-sounding speech using the Soprano model.

## Features

- **OpenAI Compatible**: Follows OpenAI's speech endpoint format for easy integration
- **High Quality Audio**: Generates ultra-realistic speech using advanced TTS models
- **Configurable Parameters**: Supports temperature, top_p, repetition_penalty, and min_text_length controls
- **Fast Processing**: Model loaded once at startup for optimal performance
- **Production Ready**: Includes health checks and error handling

## Endpoints

### Generate Speech
- **URL**: `POST /v1/audio/speech`
- **Description**: Convert text to speech
- **Request Body**:
  ```json
  {
    "input": "Text to synthesize (required, 1-1000 chars)",
    "model": "Model to use (optional, ignored)",
    "voice": "Voice to use (optional, ignored)", 
    "response_format": "Response format (optional, default: 'wav')",
    "speed": "Speech speed (optional, not implemented)",
    "temperature": "Generation temperature (optional, default: 0.3, range: 0.0-1.0)",
    "top_p": "Top-p sampling parameter (optional, default: 1.0, range: 0.0-1.0)",
    "repetition_penalty": "Repetition penalty (optional, default: 1.2, range: 0.1-2.0)",
    "min_text_length": "Minimum text length for processing (optional, default: 30, range: 1-1000)"
  }
  ```
- **Response**: WAV audio file as binary data

### Health Check
- **URL**: `GET /health`
- **Description**: Check if the server and TTS model are running properly
- **Response**: Status and device information

### Root Endpoint
- **URL**: `GET /`
- **Description**: API information and available endpoints

## Integration

This API is designed for easy integration with various systems including:
- Automation platforms (like n8n)
- Web applications
- Mobile applications
- Voice assistants
- Any system that can make HTTP requests

## Performance

- Model is loaded once at startup for optimal performance
- Efficient audio processing with minimal overhead
- Designed for concurrent requests with proper error handling

## Usage Example

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is a test.",
    "temperature": 0.3,
    "top_p": 1.0,
    "repetition_penalty": 1.2
  }' \
  --output output.wav
```