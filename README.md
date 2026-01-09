<div align="center">

# Soprano TTS

**Ultra-realistic Text-to-Speech System**

[![License](https://img.shields.io/github/license/ekwek1/soprano)](LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/soprano-tts)](https://pypi.org/project/soprano-tts/)
[![PyPI Version](https://img.shields.io/pypi/v/soprano-tts)](https://pypi.org/project/soprano-tts/)
[![GitHub](https://img.shields.io/badge/GitHub-Original%20Repo-blue?logo=github)](https://github.com/ekwek1/soprano)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange)](https://huggingface.co/ekwek1/soprano)
[![Gradio](https://img.shields.io/badge/Demo-Live%20on%20HF-green)](https://huggingface.co/spaces/ekwek1/soprano)

*Soprano delivers high-quality, natural-sounding speech synthesis with minimal latency using cutting-edge deep learning techniques.*

</div>

## Key Features

- **High-fidelity audio** - Crystal-clear speech synthesis
- **GPU acceleration** - Support for both CPU and CUDA
- **Multiple backends** - Transformers & LMDeploy with auto-selection
- **REST API** - Easy HTTP integration
- **Interactive CLI** - Command-line interface for quick usage
- **Streaming support** - Real-time audio generation capabilities

## Installation

```bash
pip install soprano-tts
```

## Quick Start

### Using the CLI

```bash
# Launch the interactive CLI
soprano

# Customize backend and cache size
soprano --backend transformers --cache-size 50
```

### Using the Python API

```python
from soprano import SopranoTTS

# Initialize the TTS model
tts = SopranoTTS(device='cuda')  # Use 'cpu' if CUDA is not available

# Generate speech
audio = tts.infer("Hello, welcome to Soprano TTS!")

# Save to file
tts.infer("Hello world!", out_path="output.wav")
```

### Using the REST API

```bash
# Start the API server
cd soprano/server
python api.py

# Make requests to the API
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, this is Soprano TTS!", "model": "soprano"}' \
  --output output.wav
```

## Architecture

Soprano TTS is built with a modular architecture:

- **Core Engine** - Advanced synthesis engine for text processing
- **Adaptive Backends** - Multiple inference options with smart selection
- **Custom Decoder** - Specialized vocoder for high-quality audio
- **RESTful API** - FastAPI-powered HTTP interface
- **Interactive CLI** - User-friendly command-line experience

## Future Roadmap

We're constantly enhancing Soprano TTS with innovative features:

### WebUI for Visual Voice Management

An intuitive web-based interface for comprehensive voice control:

- **Visual Parameter Adjustment** - Drag-and-drop controls for pitch, speed, and tone
- **Real-time Previews** - Instant playback of voice modifications
- **Profile Management** - Save and share custom voice configurations
- **Advanced Editing Tools** - Format and segment text with ease
- **Batch Processing** - Handle multiple text inputs simultaneously
- **Multi-format Export** - Download in various audio formats

### WebSocket Integration for Real-Time Streaming

Low-latency audio streaming for interactive applications:

- **Ultra-low Latency** - Sub-millisecond response times for live streaming
- **Bidirectional Communication** - Full-duplex interaction capabilities
- **Streaming Synthesis** - Continuous audio generation for long texts
- **Real-time Feedback** - Dynamic adjustments during playback
- **Optimized Buffering** - Consistent quality with intelligent caching
- **Resilient Connections** - Automatic recovery from network interruptions

These enhancements will extend the existing REST API in `soprano/server/api.py`, providing both traditional HTTP endpoints and real-time streaming capabilities for diverse use cases.

## Contributing

We welcome contributions! Please see our contributing guidelines for details on how to participate in the project.

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

---

<div align="center">

For the open-source community

[GitHub](https://github.com/ekwek1/soprano) • [Issues](https://github.com/ekwek1/soprano/issues)

</div>
