# Soprano TTS API Documentation

Welcome to the comprehensive documentation for the Soprano TTS API. This documentation provides detailed information about the API, its usage, configuration, and implementation details.

## Table of Contents

1. [Overview](overview.md) - Introduction to the Soprano TTS API
2. [API Endpoints](endpoints.md) - Detailed information about all API endpoints
3. [Configuration](configuration.md) - Setup and configuration instructions
4. [Usage Examples](usage_examples.md) - Practical examples and use cases
5. [Architecture](architecture.md) - Technical architecture and implementation details
6. [Error Handling & Troubleshooting](errors_and_troubleshooting.md) - Error handling and troubleshooting guide

## Quick Start

### Prerequisites
- Python 3.10 or higher
- CUDA-compatible GPU (optional, CPU fallback available)

### Installation
```
pip install soprano-tts
```

### Running the API Server
```
python soprano\server\api.py
```

The server will start on `http://localhost:8000` and automatically detect available hardware.

### Making Your First Request
```
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world, this is a test of the Soprano TTS system."
  }' \
  --output output.wav
```

## Support

For support, please refer to the troubleshooting section or create an issue in the project repository.