Here is the updated, professional `README.md` file. I have removed all emojis, incorporated the new installation/startup instructions, and organized the menu options as requested. 

***

# Soprano TTS

Soprano is an ultra-realistic Text-to-Speech system that provides REST API, WebSocket streaming capabilities, and a user-friendly Web UI. It is designed to be lightweight yet high-fidelity, offering OpenAI-compatible endpoints for seamless integration into existing workflows.

> **Note:** Soprano uses **LMDeploy** to accelerate inference by default. If LMDeploy cannot be installed in your environment, Soprano can fall back to the HuggingFace **transformers** backend (with slower performance). To enable this, pass `backend='transformers'` when creating the TTS model

## Features

- **High Quality Audio:** Generates ultra-realistic speech at 32 kHz using advanced TTS models.
- **Multiple Interfaces:** Includes REST API, WebSocket streaming, Web UI, and CLI.
- **OpenAI Compatible:** Follows OpenAI's speech endpoint format for drop-in replacement.
- **Real-time Streaming:** WebSocket support for real-time audio streaming with <15 ms latency.
- **Configurable Parameters:** Supports temperature, top_p, repetition_penalty, and min_text_length controls.
- **Interactive Launcher:** Easy-to-use batch script for managing services.

## Installation and Setup

### Prerequisites
Ensure you have Git and Python installed on your system.

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/biswas445/soprano.git
   ```
2. Navigate to the project directory:
   ```bash
   cd soprano
   ```
3. Run the setup script and follow the prompts:
   ```bat
   setup.bat
   ```

## Quick Start

To start the application, run the `start.bat` file located in the root directory:

```bat
start.bat
```

This will launch the interactive menu where you can choose the desired component:

1.  **API Server:** Starts the RESTful API server.
2.  **Test API:** Launches the API server and automatically runs the API test client to verify functionality.
3.  **Real-time Assistant:** Launches a voice-to-voice AI assistant demo featuring real-time audio streaming.
4.  **WebSocket Test:** Launches the WebSocket server and the corresponding test client.
5.  *(Reserved)*
6.  **Web UI:** Starts the browser-based interface for standard users.
7.  **CLI:** Starts the interactive Command Line Interface for testing purposes.


## Technical Architecture

### 1. High-fidelity 32 kHz Audio
Soprano synthesizes speech at **32 kHz**, delivering quality that is perceptually indistinguishable from 44.1/48 kHz audio and significantly sharper than the 24 kHz output used by many existing TTS models.

### 2. Vocoder-based Neural Decoder
Instead of slow diffusion decoders, Soprano uses a **vocoder-based decoder** with a Vocos architecture. This enables **orders-of-magnitude faster** waveform generation while maintaining comparable perceptual quality.

### 3. Seamless Streaming
Soprano leverages the decoder's finite receptive field to losslessly stream audio with ultra-low latency. The streamed output is acoustically identical to offline synthesis, and streaming can begin after generating just 5 audio tokens, enabling **<15 ms latency**.

### 4. State-of-the-art Neural Audio Codec
Speech is represented using a **neural codec** that compresses audio to **~15 tokens/sec** at just **0.2 kbps**, allowing extremely fast generation and efficient memory usage without sacrificing quality.

### 5. Sentence-level Streaming
Each sentence is generated independently, enabling **effectively infinite generation length** while maintaining stability and real-time performance for long-form generation.

## Project Status

The core infrastructure, including the OpenAI-compatible API and various interfaces, is complete.

**Current Focus Areas:**
1.  **Backend Strengthening:** Improving the robustness of the inference engine.
2.  **Text Normalization:** Enhancing the handling of numbers, abbreviations, and special characters to improve pronunciation accuracy.

## Limitations

Soprano was optimized purely for speed and was pretrained on approximately 1000 hours of audio. Consequently:
*   Numbers and special characters may occasionally be mispronounced (phonetic conversion is recommended).
*   Voice cloning and style controls are currently not implemented.
*   Stability and quality are expected to improve with future training on larger datasets.

## Acknowledgements

Soprano uses and/or is inspired by the following projects:

*   [Vocos](https://github.com/gemelo-ai/vocos)
*   [XTTS](https://github.com/coqui-ai/TTS)
*   [LMDeploy](https://github.com/InternLM/lmdeploy)

## License

Licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
