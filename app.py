import gradio as gr
import torch
import numpy as np
from soprano import SopranoTTS
from scipy.io.wavfile import write as wav_write
import tempfile
import os

# Detect device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load model once - works on both CUDA and CPU
model = SopranoTTS(
    backend="auto",  # Will automatically choose best backend for device
    device=DEVICE,
    cache_size_mb=100,  # Only relevant for CUDA
    decoder_batch_size=1,
)

SAMPLE_RATE = 32000

# Remove @spaces.GPU decorator - not needed for CPU support
def tts_stream(text, temperature, top_p, repetition_penalty, state):
    if not text.strip():
        yield None, state
        return
    
    out = model.infer(
        text,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    
    audio_np = out.cpu().numpy()
    yield (SAMPLE_RATE, audio_np), audio_np

def save_audio(state):
    if state is None or len(state) == 0:
        return None
    
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    wav_write(path, SAMPLE_RATE, state)
    return path

with gr.Blocks() as demo:
    state_audio = gr.State(None)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                f"# Soprano Demo\n\n"
                f"**Running on: {DEVICE.upper()}**\n\n"
                "Soprano is an ultra‑lightweight, open‑source text‑to‑speech (TTS) model designed for real‑time, "
                "high‑fidelity speech synthesis at unprecedented speed. Soprano can achieve **<15 ms streaming latency** "
                "and up to **2000x real-time generation**, all while being easy to deploy at **<1 GB VRAM usage**.\n\n"
                "Github: https://github.com/ekwek1/soprano\n\n"
                "Model Weights: https://huggingface.co/ekwek/Soprano-80M"
            )
            
            text_in = gr.Textbox(
                label="Input Text",
                placeholder="Enter text to synthesize...",
                value="Soprano is an extremely lightweight text to speech model designed to produce highly realistic speech at unprecedented speed.",
                lines=4,
            )
            
            with gr.Accordion("Advanced options", open=False):
                temperature = gr.Slider(
                    0.0, 1.0, value=0.3, step=0.05, label="Temperature"
                )
                top_p = gr.Slider(
                    0.0, 1.0, value=0.95, step=0.01, label="Top-p"
                )
                repetition_penalty = gr.Slider(
                    1.0, 2.0, value=1.2, step=0.05, label="Repetition penalty"
                )
            
            gen_btn = gr.Button("Generate")
        
        with gr.Column():
            audio_out = gr.Audio(
                label="Output Audio",
                autoplay=True,
                streaming=False,
            )
            
            download_btn = gr.Button("Download")
            file_out = gr.File(label="Download file")
            
            gr.Markdown(
                "Usage tips:\n\n"
                "- Soprano works best when each sentence is between 2 and 15 seconds long.\n"
                "- Although Soprano recognizes numbers and some special characters, it occasionally mispronounces them. "
                "Best results can be achieved by converting these into their phonetic form. (1+1 -> one plus one, etc)\n"
                "- If Soprano produces unsatisfactory results, you can easily regenerate it for a new, potentially better generation. "
                "You may also change the sampling settings for more varied results.\n"
                "- Avoid improper grammar such as not using contractions, multiple spaces, etc.\n\n"
                f"**Note:** {'GPU acceleration active' if DEVICE == 'cuda' else 'Running on CPU - generation may be slower'}"
            )
    
    gen_btn.click(
        fn=tts_stream,
        inputs=[text_in, temperature, top_p, repetition_penalty, state_audio],
        outputs=[audio_out, state_audio],
    )
    
    download_btn.click(
        fn=save_audio,
        inputs=[state_audio],
        outputs=[file_out],
    )

demo.queue()
demo.launch()
