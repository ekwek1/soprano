# Soprano TTS

Soprano is an ultra-realistic Text-to-Speech system that provides both REST API and WebSocket streaming capabilities.

> **Note**: Soprano uses **LMDeploy** to accelerate inference by default. If LMDeploy cannot be installed in your environment, Soprano can fall back to the HuggingFace **transformers** backend (with slower performance). To enable this, pass `backend='transformers'` when creating the TTS model.

---

## Features

- **High Quality Audio**: Generates ultra-realistic speech using advanced TTS models
- **Multiple Interfaces**: REST API and WebSocket streaming options
- **OpenAI Compatible**: Follows OpenAI's speech endpoint format for easy integration
- **Real-time Streaming**: WebSocket support for real-time audio streaming
- **Configurable Parameters**: Supports temperature, top_p, repetition_penalty, and min_text_length controls

## Components

- **API Server**: RESTful API with OpenAI-compatible endpoints for workflow integration
- **WebSocket Server**: Real-time audio streaming via WebSocket
- **CLI Interface**: Interactive command-line interface
- **Test Clients**: Dedicated test clients for both API and WebSocket

## Usage

```python
from soprano import SopranoTTS

model = SopranoTTS(backend='auto', device='cuda', cache_size_mb=100, decoder_batch_size=1)
```

> **Tip**: You can increase cache_size_mb and decoder_batch_size to increase inference speed at the cost of higher memory usage.

### Basic inference

```python
out = model.infer("Soprano is an extremely lightweight text to speech model.") # can achieve 2000x real-time with sufficiently long input!
```

### Save output to a file

```python
out = model.infer("Soprano is an extremely lightweight text to speech model.", "out.wav")
```

### Custom sampling parameters

```python
out = model.infer(
    "Soprano is an extremely lightweight text to speech model.",
    temperature=0.3,
    top_p=0.95,
    repetition_penalty=1.2,
)
```

### Batched inference

```python
out = model.infer_batch(["Soprano is an extremely lightweight text to speech model."] * 10) # can achieve 2000x real-time with sufficiently large input size!
```

#### Save batch outputs to a directory

```python
out = model.infer_batch(["Soprano is an extremely lightweight text to speech model."] * 10, "/dir")
```

### Streaming inference

```python
import torch

stream = model.infer_stream("Soprano is an extremely lightweight text to speech model.", chunk_size=1)

# Audio chunks can be accessed via an iterator
chunks = []
for chunk in stream:
    chunks.append(chunk) # first chunk arrives in <15 ms!

out = torch.cat(chunks)
```

### Serve endpoint

```
uvicorn soprano.server:app --host 0.0.0.0 --port 8000
```

Compatible with OpenAI speech API. Use the endpoint like this:

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "The quick brown fox jumped over the lazy dog."
  }' \
  --output speech.wav
```

## Usage tips:

* Soprano works best when each sentence is between 2 and 15 seconds long.
* Although Soprano recognizes numbers and some special characters, it occasionally mispronounces them. Best results can be achieved by converting these into their phonetic form. (1+1 -> one plus one, etc)
* If Soprano produces unsatisfactory results, you can easily regenerate it for a new, potentially better generation. You may also change the sampling settings for more varied results.
* Avoid improper grammar such as not using contractions, multiple spaces, etc.

---

## Key Features

### 1. High‑fidelity 32 kHz audio

Soprano synthesizes speech at **32 kHz**, delivering quality that is perceptually indistinguishable from 44.1/48 kHz audio and significantly sharper and clearer than the 24 kHz output used by many existing TTS models.

### 2. Vocoder‑based neural decoder

Instead of slow diffusion decoders, Soprano uses a **vocoder‑based decoder** with a Vocos architecture, enabling **orders‑of‑magnitude faster** waveform generation while maintaining comparable perceptual quality.

### 3. Seamless Streaming

Soprano leverages the decoder’s finite receptive field to losslessly stream audio with ultra‑low latency. The streamed output is acoustically identical to offline synthesis, and streaming can begin after generating just 5 audio tokens, enabling **<15 ms latency**.

### 4. State‑of‑the‑art neural audio codec

Speech is represented using a **neural codec** that compresses audio to **~15 tokens/sec** at just **0.2 kbps**, allowing extremely fast generation and efficient memory usage without sacrificing quality.

### 5. Sentence‑level streaming for infinite context

Each sentence is generated independently, enabling **effectively infinite generation length** while maintaining stability and real‑time performance for long‑form generation.

---

## Quick Start

### Using the Launcher
Run `Soprano.bat` to access the main menu with options to launch any component.

### API Server
Start the API server and send requests to `http://localhost:8000/v1/audio/speech`

### WebSocket Server
Start the WebSocket server and connect to `ws://localhost:8001/ws/tts`

## Endpoints

### API
- `POST /v1/audio/speech` - Generate speech from text
- `GET /health` - Health check endpoint
- `GET /` - Root endpoint with API information

### WebSocket
- `ws://localhost:8001/ws/tts` - Real-time TTS streaming

## Integration

The API is designed for easy integration with workflow automation platforms like n8n, Zapier, and other systems that can make HTTP requests.

## Limitations

I’m a second-year undergrad who’s just started working on TTS models, so I wanted to start small. Soprano was only pretrained on 1000 hours of audio (~100x less than other TTS models), so its stability and quality will improve tremendously as I train it on more data. Also, I optimized Soprano purely for speed, which is why it lacks bells and whistles like voice cloning, style control, and multilingual support. Now that I have experience creating TTS models, I have a lot of ideas for how to make Soprano even better in the future, so stay tuned for those!

---

## Roadmap

* [x] Add model and inference code
* [x] Seamless streaming
* [x] Batched inference
* [x] Command-line interface (CLI)
* [x] CPU support
* [x] Server / API inference
* [ ] Additional LLM backends
* [ ] Voice cloning
* [ ] Multilingual support

---

## Acknowledgements

Soprano uses and/or is inspired by the following projects:

* [Vocos](https://github.com/gemelo-ai/vocos)
* [XTTS](https://github.com/coqui-ai/TTS)
* [LMDeploy](https://github.com/InternLM/lmdeploy)

---

## License

Licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.