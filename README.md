<!-- Version 0.1.0 -->
<div align="center">
  
  # Soprano: Instant, Ultra‑Realistic Text‑to‑Speech

  [![Alt Text](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/ekwek/Soprano-1.1-80M)
  [![Alt Text](https://img.shields.io/badge/HuggingFace-Demo-yellow?logo=huggingface)](https://huggingface.co/spaces/ekwek/Soprano-TTS)
  
  <img width="640" height="320" alt="soprano-github" src="https://github.com/user-attachments/assets/4d612eac-23b8-44e6-8c59-d7ac14ebafd1" />
</div>

### 📰 News
**2026.01.14 - [Soprano-1.1-80M](https://huggingface.co/ekwek/Soprano-1.1-80M) released! 95% fewer hallucinations and a 63% preference rate over Soprano-80M.**  
2026.01.13 - [Soprano-Factory](https://github.com/ekwek1/soprano-factory) released! You can now train/fine-tune your own Soprano models.  
2025.12.22 - Soprano-80M released! [Model](https://huggingface.co/ekwek/Soprano-80M) | [Demo](https://huggingface.co/spaces/ekwek/Soprano-TTS)

---

## Overview

**Soprano** is an ultra‑lightweight, on-device text‑to‑speech (TTS) model designed for expressive, high‑fidelity speech synthesis at unprecedented speed. Soprano was designed with the following features:
- Up to **20x** real-time generation on CPU and **2000x** real-time on GPU
- **Lossless streaming** with **<250 ms** latency on CPU, **<15 ms** on GPU
- **<1 GB** memory usage with a compact 80M parameter architecture
- **Infinite generation length** with automatic text splitting
- Highly expressive, crystal clear audio generation at **32kHz**
- Widespread support for CUDA, CPU, and MPS devices on Windows, Linux, and Mac
- Supports WebUI, CLI, and OpenAI-compatible endpoint for easy and production-ready inference

https://github.com/user-attachments/assets/525cf529-e79e-4368-809f-6be620852826

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [WebUI](#webui)
  - [CLI](#cli)
  - [OpenAI-compatible endpoint](#openai-compatible-endpoint)
  - [Python script](#python-script)
- [Usage tips](#usage-tips)
- [Roadmap](#roadmap)

## Installation

### Install with wheel (CUDA)

```bash
pip install soprano-tts[lmdeploy]
```

### Install with wheel (CPU/MPS)

```bash
pip install soprano-tts
```

To get the latest features, you can install from source instead.

### Install from source (CUDA)

```bash
git clone https://github.com/ekwek1/soprano.git
cd soprano
pip install -e .[lmdeploy]
```

### Install from source (CPU/MPS)

```bash
git clone https://github.com/ekwek1/soprano.git
cd soprano
pip install -e .
```

> ### ⚠️ Warning: Windows CUDA users
> 
> On Windows with CUDA, `pip` will install a CPU-only PyTorch build. To ensure CUDA support works as expected, reinstall PyTorch explicitly with the correct CUDA wheel **after** installing Soprano:
> 
> ```bash
> pip uninstall -y torch
> pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
> ```

---

## Usage

### WebUI

Start WebUI:

```bash
soprano-webui # hosted on http://127.0.0.1:7860 by default
```
> **Tip:** You can increase cache size and decoder batch size to increase inference speed at the cost of higher memory usage. For example:
> ```bash
> soprano-webui --cache-size 1000 --decoder-batch-size 4
> ```

### CLI

```
soprano "Soprano is an extremely lightweight text to speech model."

optional arguments:
  --output, -o                  Output audio file path (non-streaming only). Defaults to 'output.wav'
  --model-path, -m              Path to local model directory (optional)
  --device, -d                  Device to use for inference. Supported: auto, cuda, cpu, mps. Defaults to 'auto'
  --backend, -b                 Backend to use for inference. Supported: auto, transformers, lmdeploy. Defaults to 'auto'
  --cache-size, -c              Cache size in MB (for lmdeploy backend). Defaults to 100
  --decoder-batch-size, -bs     Decoder batch size. Defaults to 1
  --streaming, -s               Enable streaming playback to speakers
```
> **Tip:** You can increase cache size and decoder batch size to increase inference speed at the cost of higher memory usage.

> **Note:** The CLI will reload the model every time it is called. As a result, inference speed will be slower than other methods.

### OpenAI-compatible endpoint

Start server:

```bash
uvicorn soprano.server:app --host 0.0.0.0 --port 8000
```

Use the endpoint like this:

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Soprano is an extremely lightweight text to speech model."
  }' \
  --output speech.wav
```

> **Note:** Currently, this endpoint only supports nonstreaming output.

### Python script

```python
from soprano import SopranoTTS

model = SopranoTTS(backend='auto', device='auto', cache_size_mb=100, decoder_batch_size=1)
```

> **Tip:** You can increase cache_size_mb and decoder_batch_size to increase inference speed at the cost of higher memory usage.

```python
# Basic inference
out = model.infer("Soprano is an extremely lightweight text to speech model.") # can achieve 2000x real-time with sufficiently long input!

# Save output to a file
out = model.infer("Soprano is an extremely lightweight text to speech model.", "out.wav")

# Custom sampling parameters
out = model.infer(
    "Soprano is an extremely lightweight text to speech model.",
    temperature=0.3,
    top_p=0.95,
    repetition_penalty=1.2,
)


# Batched inference
out = model.infer_batch(["Soprano is an extremely lightweight text to speech model."] * 10) # can achieve 2000x real-time with sufficiently large input size!

# Save batch outputs to a directory
out = model.infer_batch(["Soprano is an extremely lightweight text to speech model."] * 10, "/dir")


# Streaming inference
from soprano.utils.streaming import play_stream
stream = model.infer_stream("Soprano is an extremely lightweight text to speech model.", chunk_size=1)
play_stream(stream) # plays audio with <15 ms latency!
```

## Usage tips:

* Soprano works best when each sentence is between 2 and 30 seconds long.
* Although Soprano recognizes numbers and some special characters, it occasionally mispronounces them. Best results can be achieved by converting these into their phonetic form. (1+1 -> one plus one, etc)
* If Soprano produces unsatisfactory results, you can easily regenerate it for a new, potentially better generation. You may also change the sampling settings for more varied results.
* Avoid improper grammar such as not using contractions, multiple spaces, etc.

---

## Roadmap

* [x] Add model and inference code
* [x] Seamless streaming
* [x] Batched inference
* [x] Command-line interface (CLI)
* [x] CPU support
* [x] Server / API inference
* [ ] ROCm support (see [#29](/../../issues/29))
* [ ] Additional LLM backends
* [ ] Voice cloning
* [ ] Multilingual support

---

## Limitations

Soprano is currently English-only and does not support voice cloning. In addition, Soprano was trained on only 1,000 hours of audio (~100x less than other TTS models), so mispronunciation of uncommon words may occur. This is expected to diminish as Soprano is trained on more data.

---

## Acknowledgements

Soprano uses and/or is inspired by the following projects:

* [Vocos](https://github.com/gemelo-ai/vocos)
* [XTTS](https://github.com/coqui-ai/TTS)
* [LMDeploy](https://github.com/InternLM/lmdeploy)

---

## License

This project is licensed under the **Apache-2.0** license. See `LICENSE` for details.
