# Soprano TTS CLI Documentation

## Overview

Soprano TTS is an ultra-realistic text-to-speech system that generates high-quality audio from text input. The CLI provides an interactive interface to utilize the Soprano TTS engine with customizable voice parameters for naturalistic speech synthesis.

## Features

- Interactive menu-driven interface
- Real-time audio playback without file saving
- File-based audio generation with customizable output paths
- Adjustable voice parameters for naturalistic speech
- Automatic device selection (CUDA fallback to CPU)
- Progress indicators during audio playback

## Installation

```bash
pip install soprano-tts
```

## Usage

Run the CLI with default settings:

```bash
python soprano_cli.py
```

With optional parameters:

```bash
python soprano_cli.py --model-path /path/to/model --backend auto --cache-size 10
```

### Command Line Arguments

- `--model-path` or `-m`: Path to local model directory (optional, defaults to Hugging Face model)
- `--backend`: Backend to use for inference (options: auto, transformers, lmdeploy; default: auto)
- `--cache-size` or `-c`: Cache size in MB for lmdeploy backend (default: 10)

## Interactive Menu Options

### Option 1: Input Text for Synthesis (with file saving)

Generates audio from input text and saves it to a WAV file in the `audio_output` directory. The system automatically creates this directory if it doesn't exist and uses incremental naming:
- First file: `output_audio.wav`
- Second file: `output_audio1.wav`
- And so on...

### Option 2: Real-time Audio Playback (no file saving)

Generates audio from input text and plays it directly without saving to disk. This option:
- Generates audio in real-time
- Plays audio through system speakers
- Waits for complete playback before returning to menu

### Option 3: View Saved Audio Files

Displays a list of all audio files saved in the `audio_output` directory with their filenames.

### Option 4: Exit

Terminates the CLI application.

## Visual Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Soprano TTS CLI                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   User Input    │  │  Model Load   │  │  Device Check   │ │
│  │   & Validation  │  │    & Init     │  │    & Select     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│              │                   │                   │         │
│              ▼                   ▼                   ▼         │
│  ┌─────────────────────────────────────────────────────────────┤
│  │                Main Menu Loop                               │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  │ Option 1: Save  │  │ Option 2: Play  │  │ Option 3:   │ │
│  │  │   to File       │  │   to Speaker    │  │View Audio   │ │
│  │  └─────────────────┘  └─────────────────┘  │   Files     │ │
│  └─────────────────────────────────────────────────────────────┤
│              │                   │                   │         │
│              ▼                   ▼                   ▼         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┤
│  │   Text Pre-     │  │   Text Pre-     │  │   List        │
│  │   processing    │  │   processing    │  │   Audio       │
│  └─────────────────┘  └─────────────────┘  │   Files       │
│              │                   │          └─────────────────┤
│              ▼                   ▼                   │         │
│  ┌─────────────────┐  ┌─────────────────┐           │         │
│  │   Model Inference│  │   Model Inference│          │         │
│  │   (Generate     │  │   (Generate     │          │         │
│  │    Audio Data)  │  │    Audio Data)  │          │         │
│  └─────────────────┘  └─────────────────┘          │         │
│              │                   │                   │         │
│              ▼                   ▼                   │         │
│  ┌─────────────────┐  ┌─────────────────┐           │         │
│  │   Save to       │  │   Audio         │           │         │
│  │   File (.wav)   │  │   Playback      │           │         │
│  │in audio_output  │  │(real-time)      │           │         │
│  │   directory     │  │                 │           │         │
│  └─────────────────┘  └─────────────────┘           │         │
│                                                      │         │
│                                                      ▼         │
│                                            ┌─────────────────┤
│                                            │   Return to     │
│                                            │   Main Menu     │
│                                            └─────────────────┤
└─────────────────────────────────────────────────────────────────┘
```

## Voice Characteristics

The system uses optimized default parameters for naturalistic speech:

- **Temperature**: 0.7 (provides natural variation and creativity)
- **Top-p**: 0.9 (balances coherent speech with natural variation)
- **Repetition Penalty**: 1.05 (minimizes repetition while maintaining quality)

These parameters are built-in and optimized for the most natural, human-like voice output.

## Technical Details

### Audio Specifications
- Sample Rate: 32,000 Hz
- Format: WAV (for saved files)
- Real-time playback through system audio

### Model Architecture
- Uses Soprano-80M model by default
- Vocos-based decoder for high-quality audio synthesis
- Support for both LMDeploy and Transformers backends

### Supported Platforms
- Windows, macOS, Linux
- CUDA-compatible GPUs (recommended) or CPU
- Python 3.10+

## Troubleshooting

### Common Issues:

1. **No audio output**: Ensure `sounddevice` is installed:
   ```bash
   pip install sounddevice
   ```

2. **CUDA unavailable**: The system will automatically fall back to CPU

3. **Long text processing**: Text is limited to 1000 characters per input

4. **Model loading errors**: Check internet connection for downloading models from Hugging Face

## License

This project is licensed under the terms specified in the LICENSE file.