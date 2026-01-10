# Soprano TTS - Complete Implementation

This repository contains a complete implementation of the Soprano TTS system with:

- REST API server with OpenAI-compatible endpoints
- WebSocket server for real-time audio streaming
- Comprehensive test clients for both interfaces
- Proper documentation and launch scripts
- Production-ready architecture with error handling

## Features

- **API Server**: OpenAI-compatible `/v1/audio/speech` endpoint
- **WebSocket Server**: Real-time streaming TTS with backpressure handling
- **Configurable Parameters**: temperature, top_p, repetition_penalty, min_text_length
- **Robust Architecture**: Circuit breakers, retry mechanisms, graceful shutdown
- **Workflow Integration**: Ready for n8n, Zapier, and other automation platforms