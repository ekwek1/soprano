# API Endpoints

## Base URL
All endpoints are relative to `http://localhost:8000`

## Available Endpoints

### Root Endpoint
- **URL**: `GET /`
- **Description**: Provides information about the Soprano TTS API
- **Response**: 
  ```json
  {
    "message": "Soprano TTS API",
    "version": "1.0.0",
    "description": "Ultra-realistic Text-to-Speech API based on Soprano model",
    "endpoints": {
      "tts": "/v1/audio/speech",
      "health": "/health"
    }
  }
  ```

### Text-to-Speech Generation
- **URL**: `POST /v1/audio/speech`
- **Description**: Generate speech from input text following OpenAI's Speech endpoint format
- **Request Body**: 
  ```json
  {
    "input": "string (required, min 1, max 1000 characters)",
    "model": "string (optional)",
    "voice": "string (optional)",
    "response_format": "string (optional, default: 'wav')",
    "temperature": "float (optional, default: 0.3)",
    "top_p": "float (optional, default: 0.95)",
    "repetition_penalty": "float (optional, default: 1.2)"
  }
  ```
- **Response**: WAV audio file
- **Headers**: 
  - `Content-Disposition: attachment; filename="output_N.wav"`
  - `Content-Length: {size}`

### Health Check
- **URL**: `GET /health`
- **Description**: Check if the server and TTS model are running properly
- **Response**: 
  ```json
  {
    "status": "healthy",
    "device": "cuda" or "cpu"
  }
  ```

### API Documentation
- **URL**: `GET /docs`
- **Description**: Interactive API documentation with Swagger UI

- **URL**: `GET /redoc`
- **Description**: Alternative API documentation with ReDoc

- **URL**: `GET /openapi.json`
- **Description**: OpenAPI schema specification

## Request Examples

### cURL Example
```bash
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world, this is a test of the Soprano TTS system.",
    "temperature": 0.3,
    "top_p": 0.95,
    "repetition_penalty": 1.2
  }' \
  --output output.wav
```

### Python Example
```python
import requests

url = "http://localhost:8000/v1/audio/speech"
payload = {
    "input": "Hello world, this is a test of the Soprano TTS system.",
    "temperature": 0.3,
    "top_p": 0.95,
    "repetition_penalty": 1.2
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    with open("output.wav", "wb") as f:
        f.write(response.content)
    print("Audio saved successfully")
else:
    print(f"Request failed with status {response.status_code}")
```