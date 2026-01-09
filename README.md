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



## Architecture

Soprano TTS is built with a modular architecture:

- **Core Engine** - Advanced synthesis engine for text processing
- **Adaptive Backends** - Multiple inference options with smart selection
- **Custom Decoder** - Specialized vocoder for high-quality audio
- **RESTful API** - FastAPI-powered HTTP interface
- **Interactive CLI** - User-friendly command-line experience

## Future Roadmap
- [ ] Web Socket for real time streaming and audio playback.
- [ ] Web UI for User Interaction
- [ ] LLM Intregation


These features will build upon the recently added REST API located in `soprano/server/api.py`.

## Contributing

We welcome contributions! Please see our contributing guidelines for details on how to participate in the project.

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

---

<div align="center">

For the open-source community

[GitHub](https://github.com/ekwek1/soprano) • [Issues](https://github.com/ekwek1/soprano/issues)

</div>
