# Configuration and Setup

## System Requirements

### Hardware Requirements
- **CPU**: Modern multi-core processor
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: Sufficient space for model files and audio output

### Software Requirements
- **Operating System**: Windows, macOS, or Linux
- **Python**: Version 3.10 or higher
- **CUDA**: Version 11.3 or higher (for GPU acceleration)

## Installation

### Prerequisites
1. Install Python 3.10 or higher
2. Install pip package manager

### Installation Steps
1. Install the Soprano TTS package:
   ```
   pip install soprano-tts
   ```

2. The installation will automatically handle all dependencies

## Running the API Server

### Starting the Server
To start the API server, execute:
```
python soprano\server\api.py
```

The server will:
- Detect available hardware (CUDA/CPU)
- Load the TTS model
- Start on `http://localhost:8000`
- Initialize file cleanup processes

### Environment Variables
The API does not require specific environment variables, but you can configure:

- **CUDA_VISIBLE_DEVICES**: To specify which GPU(s) to use
- **TORCH_DEVICE**: To force a specific device (though the API will auto-detect)

## Configuration Options

### Model Configuration
The API uses the Soprano-80M model by default, which will be downloaded automatically on first use.

### File Output Configuration
- **Output Directory**: `audio_output` (created automatically)
- **File Naming**: Sequential (output_1.wav, output_2.wav, etc.)
- **File Cleanup**: Automatic cleanup of files older than 24 hours

## Performance Optimization

### GPU Usage
The API automatically detects and uses CUDA when available:
- Prioritizes GPU for faster processing
- Falls back to CPU if GPU is not available
- Uses appropriate backend (lmdeploy for GPU, transformers for CPU)

### Memory Management
- Model loaded once at startup
- Efficient tensor processing
- Automatic cleanup of temporary resources