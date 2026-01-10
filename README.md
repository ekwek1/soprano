<!-- Version 0.0.2 -->
<div align="center">
  
  # Soprano: Instant, Ultra‑Realistic Text‑to‑Speech 

  [![Alt Text](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/ekwek/Soprano-80M)
  [![Alt Text](https://img.shields.io/badge/HuggingFace-Demo-yellow?logo=huggingface)](https://huggingface.co/spaces/ekwek/Soprano-TTS)
</div>

https://github.com/user-attachments/assets/525cf529-e79e-4368-809f-6be620852826

---

## Overview

**Soprano** is an ultra‑lightweight, open‑source text‑to‑speech (TTS) model designed for real‑time, high‑fidelity speech synthesis at unprecedented speed, all while remaining compact and easy to deploy at **under 1 GB VRAM usage**.

With only **80M parameters**, Soprano achieves a real‑time factor (RTF) of **~2000×**, capable of generating **10 hours of audio in under 20 seconds**. Soprano uses a **seamless streaming** technique that enables true real‑time synthesis in **<15 ms**, multiple orders of magnitude faster than existing TTS pipelines.

---

## Installation

**Requirements**: Linux or Windows, Python >= 3.10, CUDA or ROCm enabled GPU

Soprano supports multiple PyTorch variants depending on your hardware:

#### CUDA (NVIDIA GPUs)
```bash
pip install -e .[cuda]
# or if you use uv:
uv pip install -e .[cuda]
```

#### ROCm (AMD GPUs)
```bash
pip install -e .[rocm]
# or if you use uv:
uv pip install -e .[rocm]

# Install lmdeploy for ROCm (recommended for best performance)
LMDEPLOY_TARGET_DEVICE=rocm pip install git+https://github.com/InternLM/lmdeploy.git
# or with uv:
LMDEPLOY_TARGET_DEVICE=rocm uv pip install git+https://github.com/InternLM/lmdeploy.git
```

> **Note**:
> - **ROCm users**: You may need to set the `HSA_OVERRIDE_GFX_VERSION` environment variable to match your GPU architecture, for example:
>   ```bash
>   export HSA_OVERRIDE_GFX_VERSION=11.0.0
>   ```
> - **ROCm backends**: Both `lmdeploy` and `transformers` backends are fully supported on ROCm:
>   - `lmdeploy` (default with `backend='auto'`): Faster inference, recommended for production
>   - `transformers`: Alternative backend, useful for debugging or compatibility
> - **ROCm Triton compatibility**: Torch compilation must be disabled on ROCm. Set this before running:
>   ```bash
>   export TORCH_COMPILE_DISABLE=1
>   ```
>   This resolves compatibility issues between pytorch-triton-rocm and PyTorch's inductor backend.

---

## Web Interface (Gradio)

For a simple web interface, you can use the included Gradio app:

```bash
# Install Gradio 6.2.0
pip install gradio==6.2.0
# or if you use uv:
uv pip install gradio==6.2.0

# ROCm users: set environment variables before running
# export HSA_OVERRIDE_GFX_VERSION=11.0.0
# export TORCH_COMPILE_DISABLE=1
# Run the web interface (accessible at 0.0.0.0:7860)
python gradio_app.py
```

Then open your browser at `http://localhost:7860` (or `http://<your-server-ip>:7860` if running on a remote server)

The Gradio interface provides:
- Text input with advanced sampling controls (temperature, top_p, repetition penalty)
- Real-time audio generation and playback
- Example prompts to get started
- Simple, user-friendly interface

---

## Usage

```python
from soprano import SopranoTTS

# CUDA (NVIDIA) and ROCm (AMD)
model = SopranoTTS(backend='auto', device='cuda', cache_size_mb=10, decoder_batch_size=1)

# Or explicitly specify backend
model = SopranoTTS(backend='lmdeploy', device='cuda', cache_size_mb=10, decoder_batch_size=1)  # Faster
model = SopranoTTS(backend='transformers', device='cuda', cache_size_mb=10, decoder_batch_size=1)  # Alternative
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

## Limitations

I’m a second-year undergrad who’s just started working on TTS models, so I wanted to start small. Soprano was only pretrained on 1000 hours of audio (~100x less than other TTS models), so its stability and quality will improve tremendously as I train it on more data. Also, I optimized Soprano purely for speed, which is why it lacks bells and whistles like voice cloning, style control, and multilingual support. Now that I have experience creating TTS models, I have a lot of ideas for how to make Soprano even better in the future, so stay tuned for those!

---

## Roadmap

* [x] Add model and inference code
* [x] Seamless streaming
* [x] Batched inference
* [ ] Command-line interface (CLI)
* [ ] Server / API inference
* [ ] Additional LLM backends
* [ ] CPU support
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

This project is licensed under the **Apache-2.0** license. See `LICENSE` for details.
