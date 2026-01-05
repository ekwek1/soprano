#!/usr/bin/env python3
"""
Gradio Web Interface for Soprano TTS
"""

import gradio as gr
import torch
from soprano import SopranoTTS
import numpy as np
import socket

# Initialize model
print("Loading Soprano TTS model...")
model = SopranoTTS(
    backend='auto',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    cache_size_mb=10,
    decoder_batch_size=1
)
print("Model loaded successfully!")

def generate_speech(
    text: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float
) -> tuple:
    """
    Generate speech from text using Soprano TTS

    Args:
        text: Input text to synthesize
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter
        repetition_penalty: Penalty for repeating tokens

    Returns:
        Tuple of (sample_rate, audio_array) for Gradio Audio component
    """
    if not text.strip():
        return None, "Please enter some text to generate speech."

    try:
        # Generate audio
        audio = model.infer(
            text,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Convert to numpy array for Gradio
        audio_np = audio.cpu().numpy()

        # Convert to int16 format (Gradio expects 16-bit PCM audio)
        audio_int16 = (audio_np * 32767).astype(np.int16)

        # Return sample rate and audio
        return (32000, audio_int16), f"✓ Generated {len(audio_np) / 32000:.2f} seconds of audio"

    except Exception as e:
        return None, f"✗ Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Soprano TTS") as demo:
    gr.Markdown(
        """
        # 🎵 Soprano TTS

        Ultra-lightweight, high-fidelity text-to-speech at 32 kHz.

        **Usage Tips:**
        - Best results with sentences between 2-15 seconds long
        - Convert numbers to words (1+1 → "one plus one")
        - Use proper grammar and avoid multiple spaces
        - Regenerate if results are unsatisfactory
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Text to Synthesize",
                placeholder="Enter text here... (e.g., 'Soprano is an extremely lightweight text to speech model.')",
                lines=5,
                max_lines=10
            )

            with gr.Accordion("Advanced Settings", open=False):
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.5,
                    value=0.3,
                    step=0.05,
                    label="Temperature",
                    info="Higher values = more variation"
                )

                top_p = gr.Slider(
                    minimum=0.5,
                    maximum=1.0,
                    value=0.95,
                    step=0.05,
                    label="Top P",
                    info="Nucleus sampling threshold"
                )

                repetition_penalty = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    value=1.2,
                    step=0.1,
                    label="Repetition Penalty",
                    info="Penalty for repeating sounds"
                )

            generate_btn = gr.Button("Generate Speech", variant="primary", size="lg")

        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="Generated Speech",
                type="numpy"
            )

            status_output = gr.Textbox(
                label="Status",
                interactive=False
            )

    # Examples
    gr.Examples(
        examples=[
            ["Soprano is an extremely lightweight text to speech model.", 0.3, 0.95, 1.2],
            ["Hello! Welcome to Soprano text to speech.", 0.3, 0.95, 1.2],
            ["The quick brown fox jumps over the lazy dog.", 0.3, 0.95, 1.2],
            ["Artificial intelligence is transforming the world.", 0.5, 0.90, 1.2],
        ],
        inputs=[text_input, temperature, top_p, repetition_penalty],
        label="Example Prompts"
    )

    # Connect the button
    generate_btn.click(
        fn=generate_speech,
        inputs=[text_input, temperature, top_p, repetition_penalty],
        outputs=[audio_output, status_output]
    )

def find_free_port(start_port=7860, max_tries=100):
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + max_tries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise OSError(f"Could not find free port in range {start_port}-{start_port + max_tries}")

if __name__ == "__main__":
    port = find_free_port(7860)
    print(f"Starting Gradio interface on port {port}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        theme=gr.themes.Soft()
    )
