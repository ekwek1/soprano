# Soprano TTS WebSocket

The Soprano TTS WebSocket provides real-time streaming text-to-speech functionality. This WebSocket server allows you to generate audio in real-time and stream it to clients as it's produced.

## Features

- **Real-time Streaming**: Generate and stream audio in real-time
- **Raw PCM Frames**: Outputs raw PCM frames suitable for playback via PyAudio
- **Metadata Support**: Sends audio format metadata at the start
- **Small Chunks**: Streams audio in small chunks (~1024 samples) for low latency
- **End Signal**: Sends "end" signal when synthesis finishes

## Connection

- **Endpoint**: `ws://localhost:8001/ws/tts`
- **Protocol**: WebSocket with JSON control messages and binary audio frames

## Message Format

### Client to Server
```json
{
  "type": "synthesize",
  "text": "Your text here",
  "stream": true,
  "min_text_length": 30
}
```

### Server to Client
- **Metadata** (JSON):
```json
{
  "type": "metadata",
  "sample_rate": 32000,
  "channels": 1,
  "format": "int16"
}
```

- **Audio Data** (Binary): Raw PCM audio frames
- **End Signal** (JSON):
```json
{
  "type": "end"
}
```

- **Error** (JSON):
```json
{
  "type": "error",
  "message": "Error description"
}
```

## Integration

The WebSocket server is ideal for:
- Real-time voice assistants
- Interactive applications
- Live broadcasting systems
- Gaming applications
- Any system requiring immediate audio feedback

## Usage Example

```javascript
const ws = new WebSocket('ws://localhost:8001/ws/tts');

ws.onopen = () => {
    ws.send(JSON.stringify({
        type: "synthesize",
        text: "Hello, this is a real-time audio stream",
        stream: true,
        min_text_length: 30
    }));
};

ws.onmessage = (event) => {
    if (typeof event.data === 'string') {
        const message = JSON.parse(event.data);
        if (message.type === 'metadata') {
            // Handle audio format info
        } else if (message.type === 'end') {
            // Streaming finished
        }
    } else {
        // Binary audio data - play with audio API
        playAudioChunk(event.data);
    }
};
```

## Performance

- Implements backpressure handling to manage slow clients
- Uses asyncio queues to decouple audio generation from network transmission
- Supports graceful shutdown with proper task cancellation
- Optimized for real-time performance with minimal latency