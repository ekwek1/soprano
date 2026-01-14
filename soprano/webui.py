#!/usr/bin/env python3
"""
Gradio Web Interface for Soprano TTS
"""

import argparse
import socket
import time
import gradio as gr
import numpy as np
from soprano import SopranoTTS
from soprano.utils.streaming import play_stream


parser = argparse.ArgumentParser(description='Soprano Text-to-Speech Gradio WebUI')
parser.add_argument('--model-path', '-m',
                    help='Path to local model directory (optional)')
parser.add_argument('--device', '-d', default='auto',
                    choices=['auto', 'cuda', 'cpu', 'mps'],
                    help='Device to use for inference')
parser.add_argument('--backend', '-b', default='auto',
                    choices=['auto', 'transformers', 'lmdeploy'],
                    help='Backend to use for inference')
parser.add_argument('--cache-size', '-c', type=int, default=100,
                    help='Cache size in MB (for lmdeploy backend)')
parser.add_argument('--decoder-batch-size', '-bs', type=int, default=1,
                    help='Batch size when decoding audio')
args = parser.parse_args()

# Initialize model
print("Loading Soprano TTS model...")
model = SopranoTTS(
    backend=args.backend,
    device=args.device,
    cache_size_mb=args.cache_size,
    decoder_batch_size=args.decoder_batch_size,
    model_path=args.model_path
)
device = model.device
backend = model.backend
print("Model loaded successfully!")

SAMPLE_RATE = 32000


def generate_speech(
    text: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    chunk_size: int,
    streaming: bool,
):
    if not text.strip():
        yield None, "Please enter some text to generate speech."
        return

    try:
        if streaming:
            stream = model.infer_stream(
                text,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                chunk_size=chunk_size,
            )
            yield None, "⏳ Streaming..."

            latency = play_stream(stream)

            yield None, (
                f"✓ Streaming complete | "
                f"{latency*1000:.2f} ms latency"
            )
            return

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

        yield (SAMPLE_RATE, audio_int16), status
        return

    except Exception as e:
        yield None, f"✗ Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Soprano TTS") as demo:
    gr.Markdown(
        f"""# 🗣️ Soprano TTS

<div align="center">
<img width="300" height="300" alt="soprano-github" src="https://github.com/user-attachments/assets/4d612eac-23b8-44e6-8c59-d7ac14ebafd1" />
</div>

**Device:** {device.upper()} | **Backend:** {backend}

**Model Weights:** https://huggingface.co/ekwek/Soprano-1.1-80M  
**Model Demo:** https://huggingface.co/spaces/ekwek/Soprano-TTS  
**GitHub:** https://github.com/ekwek1/soprano  
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
            streaming = gr.Checkbox(
                label="Stream Audio",
                value=False,
                info="Note: This bypasses the Gradio interface and streams audio directly to your speaker."
            )
            with gr.Accordion("Advanced Settings", open=False):
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
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
                chunk_size = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=1,
                    step=1,
                    precision=0,
                    label="Chunk Size (Streaming only)",
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
            ["Soprano is an extremely lightweight text to speech model.", 0.0, 0.95, 1.2],
            ["Artificial intelligence is transforming the world.", 0.0, 0.95, 1.2],
            ["I'm so excited, I can't even wait!", 0.0, 0.95, 1.2],
            ["Why don't you go ahead and try it?", 0.0, 0.95, 1.2],
        ],
        inputs=[text_input, temperature, top_p, repetition_penalty],
        label="Example Prompts",
    )
    generate_btn.click(
        fn=generate_speech,
        inputs=[text_input, temperature, top_p, repetition_penalty, chunk_size, streaming],
        outputs=[audio_output, status_output],
    )
    gr.Markdown(
        f"""
### Usage tips:

- Soprano works best when each sentence is between 2 and 30 seconds long.
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
    # Start Gradio interface
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