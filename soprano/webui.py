#!/usr/bin/env python3
"""
Gradio Web Interface for Soprano TTS
"""

import gradio as gr
import torch
import sys
import os
# Add the parent directory to the Python path to resolve import issues when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soprano import SopranoTTS
import numpy as np
import socket
import time
import threading

# Try to import pyaudio, handle if not available
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("PyAudio not found. Install it with 'pip install pyaudio' for real-time audio streaming.")

# Global variables for PyAudio management
current_stream = None
current_pyaudio_instance = None
stream_lock = threading.Lock()

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


async def generate_speech(
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


async def speak_realtime(
    text: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> str:
    if not text.strip():
        return "Please enter some text to speak."

    if not PYAUDIO_AVAILABLE:
        return "PyAudio is not available. Install it with 'pip install pyaudio' for real-time audio streaming."

    # Use the lock to prevent concurrent access to the audio stream
    with stream_lock:
        global current_stream, current_pyaudio_instance

        # Check if there's already an active stream
        if current_stream is not None:
            try:
                current_stream.stop_stream()
                current_stream.close()
            except:
                pass  # Stream might already be closed

        if current_pyaudio_instance is not None:
            try:
                current_pyaudio_instance.terminate()
            except:
                pass  # Instance might already be terminated

        try:
            # Initialize PyAudio
            p = pyaudio.PyAudio()
            current_pyaudio_instance = p

            # Open stream
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                output=True
            )
            current_stream = stream

            # Start streaming inference
            start_time = time.perf_counter()

            # Use the streaming inference method from the model
            stream_gen = model.infer_stream(
                text,
                chunk_size=1,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            total_samples = 0

            # Process audio chunks in real-time
            for audio_chunk in stream_gen:
                # Check if stream is still active
                if current_stream is None or not current_stream.is_active():
                    break

                # Convert tensor to numpy array
                audio_np = audio_chunk.cpu().numpy()

                # Ensure values are in the range [-1, 1] and convert to int16
                audio_np = np.clip(audio_np, -1.0, 1.0)
                audio_int16 = (audio_np * 32767).astype(np.int16)

                # Play the audio chunk directly
                stream.write(audio_int16.tobytes())
                total_samples += len(audio_int16)

            # Close stream and terminate PyAudio
            stream.stop_stream()
            stream.close()
            p.terminate()

            # Reset globals after successful playback
            current_stream = None
            current_pyaudio_instance = None

            gen_time = time.perf_counter() - start_time
            audio_seconds = total_samples / SAMPLE_RATE
            rtf = audio_seconds / gen_time if gen_time > 0 else float("inf")

            status = (
                f"✓ Finished speaking {audio_seconds:.2f} s audio | "
                f"Playback time: {gen_time:.3f} s "
                f"({rtf:.2f}x realtime)"
            )

            return status

        except Exception as e:
            # Ensure cleanup in case of error
            if current_stream:
                try:
                    current_stream.stop_stream()
                    current_stream.close()
                except:
                    pass
                current_stream = None

            if current_pyaudio_instance:
                try:
                    current_pyaudio_instance.terminate()
                except:
                    pass
                current_pyaudio_instance = None

            return f"✗ Error during real-time playback: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Soprano TTS") as demo:

    # State variable to track which function is active
    active_function = gr.State(value=None)  # Can be "generate", "speak", or None

    gr.Markdown(
        f"""
# 🎵 Soprano TTS

**Running on: {DEVICE.upper()}**

Soprano is an ultra-lightweight, open-source text-to-speech (TTS) model designed for real-time,
high-fidelity speech synthesis at unprecedented speed. Soprano can achieve **<15 ms streaming latency**
and up to **2000x real-time generation**, all while being easy to deploy at **<1 GB VRAM usage**.

<br>

<div style="display: flex; justify-content: center; gap: 20px; margin: 15px 0;">
    <a href="https://github.com/ekwek1/soprano" target="_blank" style="text-decoration: none; color: inherit;">Git Hub</a>
    <a href="https://huggingface.co/spaces/ekwek/Soprano-TTS" target="_blank" style="text-decoration: none; color: inherit;">Model Demo</a>
    <a href="https://huggingface.co/ekwek/Soprano-80M" target="_blank" style="text-decoration: none; color: inherit;"> Model Weights </a>
</div>
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

            with gr.Row():
                generate_btn = gr.Button("Generate Speech", variant="primary", size="lg")
                speak_btn = gr.Button("Speak", variant="primary", size="lg")
                clear_btn = gr.Button("Clear", variant="secondary", size="lg")

        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="Generated Speech",
                type="numpy",
                autoplay=True,
                streaming=True
            )

            status_output = gr.Textbox(
                label="Status",
                interactive=False,
                lines=3,
                max_lines=10
            )

    gr.Examples(
        examples=[
            ["Soprano is an extremely lightweight text to speech model designed to produce highly realistic speech at unprecedented speed."],
            ["Hello! Welcome to Soprano text to speech. This is a short example."],
            ["The quick brown fox jumps over the lazy dog. This sentence contains all letters of the alphabet."],
            ["Artificial intelligence is transforming the world in ways we never imagined. It's revolutionizing industries and changing how we interact with technology."],
            ["In a distant future, humanity has colonized the stars. Advanced AI systems govern interstellar travel, ensuring safety and efficiency across vast cosmic distances. Explorers venture into uncharted territories, seeking new worlds and civilizations."],
            ["To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles And by opposing end them. To die—to sleep, no more; and by a sleep to say we end The heart-ache and the thousand natural shocks That flesh is heir to: 'tis a consummation Devoutly to be wish'd. To die, to sleep; To sleep, perchance to dream—ay, there's the rub: For in that sleep of death what dreams may come, When we have shuffled off this mortal coil, Must give us pause—there's the respect That makes calamity of so long life."],
        ],
        inputs=[text_input],
        label="Examples",
    )

    async def check_and_set_active_generate(active_func, *args):
        if active_func is not None:
            return None, f"Error: Please press Clear first. Current operation: {active_func}", active_func
        # Call the actual generate function
        result = await generate_speech(args[0], args[1], args[2], args[3])
        return result[0], result[1], "generate"

    async def check_and_set_active_speak(active_func, *args):
        if active_func is not None:
            return f"Error: Please press Clear first. Current operation: {active_func}", active_func
        # Call the actual speak function
        result = await speak_realtime(args[0], args[1], args[2], args[3])
        return result, "speak"

    def clear_active_state():
        return None

    generate_btn.click(
        fn=check_and_set_active_generate,
        inputs=[active_function, text_input, temperature, top_p, repetition_penalty],
        outputs=[audio_output, status_output, active_function]
    )

    speak_btn.click(
        fn=check_and_set_active_speak,
        inputs=[active_function, text_input, temperature, top_p, repetition_penalty],
        outputs=[status_output, active_function]
    )

    def clear_inputs(active_func):
        # Reset the active function state
        return "", None, "Ready for input...", None

    clear_btn.click(
        fn=clear_inputs,
        inputs=[active_function],
        outputs=[text_input, audio_output, status_output, active_function]
    )

    gr.Markdown(
        """

<br>

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
    demo.queue(max_size=20).launch(
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