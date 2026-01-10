#!/usr/bin/env python3
"""
Gradio Web Interface for Soprano TTS
"""

import gradio as gr
import torch
from soprano import SopranoTTS
import numpy as np
import socket
import time

# Detect device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model
print("Loading Soprano TTS model...")
model = SopranoTTS(
    backend="auto",
    device=DEVICE,
    cache_size_mb=100,
    decoder_batch_size=1,
)
print("Model loaded successfully!")

SAMPLE_RATE = 32000


def generate_speech(
    text: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> tuple:
    if not text.strip():
        return None, "Please enter some text to generate speech."

    try:
        start_time = time.perf_counter()

        audio = model.infer(
            text,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        gen_time = time.perf_counter() - start_time

        audio_np = audio.cpu().numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)

        audio_seconds = len(audio_np) / SAMPLE_RATE
        rtf = audio_seconds / gen_time if gen_time > 0 else float("inf")

        status = (
            f"✓ Generated {audio_seconds:.2f} s audio | "
            f"Generation time: {gen_time:.3f} s "
            f"({rtf:.2f}x realtime)"
        )

        return (SAMPLE_RATE, audio_int16), status

    except Exception as e:
        return None, f"✗ Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Soprano TTS") as demo:

    gr.Markdown(
        f"""
# 🎵 Soprano TTS

**Running on: {DEVICE.upper()}**

Soprano is an ultra-lightweight, open-source text-to-speech (TTS) model designed for real-time,
high-fidelity speech synthesis at unprecedented speed. Soprano can achieve **<15 ms streaming latency**
and up to **2000x real-time generation**, all while being easy to deploy at **<1 GB VRAM usage**.

**GitHub:** https://github.com/ekwek1/soprano  
**Model Demo:** https://huggingface.co/spaces/ekwek/Soprano-TTS  
**Model Weights:** https://huggingface.co/ekwek/Soprano-80M
"""
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Text to Synthesize",
                placeholder="Enter text here...",
                value="Soprano is an extremely lightweight text to speech model designed to produce highly realistic speech at unprecedented speed.",
                lines=5,
                max_lines=10,
            )

            with gr.Accordion("Advanced Settings", open=False):
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.5,
                    value=0.3,
                    step=0.05,
                    label="Temperature",
                )

                top_p = gr.Slider(
                    minimum=0.5,
                    maximum=1.0,
                    value=0.95,
                    step=0.05,
                    label="Top P",
                )

                repetition_penalty = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    value=1.2,
                    step=0.1,
                    label="Repetition Penalty",
                )

            generate_btn = gr.Button("Generate Speech", variant="primary", size="lg")

        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="Generated Speech",
                type="numpy",
                autoplay=True,
            )

            status_output = gr.Textbox(
                label="Status",
                interactive=False,
                lines=3,
                max_lines=10
            )

    gr.Examples(
        examples=[
            ["Soprano is an extremely lightweight text to speech model.", 0.3, 0.95, 1.2],
            ["Hello! Welcome to Soprano text to speech.", 0.3, 0.95, 1.2],
            ["The quick brown fox jumps over the lazy dog.", 0.3, 0.95, 1.2],
            ["Artificial intelligence is transforming the world.", 0.5, 0.90, 1.2],
        ],
        inputs=[text_input, temperature, top_p, repetition_penalty],
        label="Example Prompts",
    )

    generate_btn.click(
        fn=generate_speech,
        inputs=[text_input, temperature, top_p, repetition_penalty],
        outputs=[audio_output, status_output],
    )
    gr.Markdown(
        f"""
### Usage tips:

- Soprano works best when each sentence is between 2 and 15 seconds long.
- Although Soprano recognizes numbers and some special characters, it occasionally mispronounces them.
  Best results can be achieved by converting these into their phonetic form.
  (1+1 -> one plus one, etc)
- If Soprano produces unsatisfactory results, you can easily regenerate it for a new, potentially better generation.
  You may also change the sampling settings for more varied results.
- Avoid improper grammar such as not using contractions, multiple spaces, etc.
"""
    )


def find_free_port(start_port=7860, max_tries=100):
    for port in range(start_port, start_port + max_tries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    raise OSError("Could not find a free port")

def main():
    port = find_free_port(7860)
    print(f"Starting Gradio interface on port {port}")
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        theme=gr.themes.Soft(primary_hue="green"),
        css="""
a {
    color: var(--primary-600);
}
a:hover {
    color: var(--primary-700);
}
"""
    )

if __name__ == "__main__":
    main()