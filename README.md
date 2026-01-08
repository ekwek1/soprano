# Soprano TTS

Soprano TTS is an ultra-realistic text-to-speech system that generates high-quality audio from text input. This enhanced CLI version provides an interactive interface with improved functionality and naturalistic voice synthesis.

## Key Improvements

- **Interactive Menu System**: Streamlined user experience with intuitive menu options
- **Automatic File Management**: Creates `audio_output` directory and manages incremental file naming
- **Persistent Session**: Continues running until explicitly exited, allowing multiple operations
- **Optimized Voice Parameters**: Built-in naturalistic defaults (temperature=0.7, top_p=0.9, repetition_penalty=1.05)
- **Robust Audio Playback**: Reliable real-time audio playback without interruption
- **Clean Interface**: Suppressed verbose model loading messages for better UX

## Features

- Real-time audio playback without file saving
- File-based audio generation with automatic naming
- Audio file management and listing
- Naturalistic voice synthesis with optimized parameters

## Future Roadmap

- [ ] **Web UI**: Develop a browser-based interface for easy access
- [ ] **Server/API**: Create RESTful API endpoints for integration into applications
- [ ] **LLM Integration**: Connect with Large Language Models for end-to-end conversation capabilities