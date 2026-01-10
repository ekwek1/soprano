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

# Global model instance (lazy loaded)
model = None
backend_info = None
current_backend = None

def get_backend_name(model):
    """Get the backend name from the model"""
    if hasattr(model, 'pipeline'):
        # Check if it's lmdeploy
        pipeline_type = type(model.pipeline).__name__
        if 'lmdeploy' in pipeline_type.lower() or 'AsyncEngine' in pipeline_type:
            return 'lmdeploy'
        else:
            return 'transformers'
    else:
        return 'transformers'

def get_model(backend='auto'):
    """Lazy load the model to avoid multiprocessing issues"""
    global model, backend_info, current_backend

    # Check if we need to reload because of backend change
    if model is not None and current_backend is not None and current_backend != backend:
        print(f"Backend changed from {current_backend} to {backend}, reloading model...")
        del model
        model = None
        backend_info = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if model is None:
        print(f"Loading Soprano TTS model with backend={backend}...")
        
        # Check for GPU availability
        cuda_available = torch.cuda.is_available()
        device_type = 'cuda' if cuda_available else 'cpu'
        
        model = SopranoTTS(
            backend=backend,
            device=device_type,
            cache_size_mb=10,
            decoder_batch_size=1
        )

        # Get backend info
        backend_name = get_backend_name(model)
        
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            is_rocm = 'Radeon' in device_name or 'AMD' in device_name
            gpu_tag = "(ROCm)" if is_rocm else "(CUDA)"
            backend_info = f"**Backend:** {backend_name} | **Device:** {device_name} {gpu_tag}"
        else:
            # Clean display for CPU
            backend_info = f"**Backend:** {backend_name} | **Device:** CPU"

        current_backend = backend
        print(f"Model loaded successfully! Using {backend_name} backend on {device_type.upper()}")
        
    return model

def reload_model(backend_choice):
    """Reload the model with a different backend"""
    global model, backend_info, current_backend

    # Clear existing model
    if model is not None:
        print(f"Unloading existing model...")
        del model
        model = None
        backend_info = None
        current_backend = None

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Load new model
    print(f"Reloading model with backend={backend_choice}...")
    get_model(backend=backend_choice)

    return f"✓ Model reloaded with backend: {backend_choice}", get_system_info()

def get_system_info():
    """Get system and backend information"""
    if backend_info:
        return backend_info
    else:
        return "**Backend:** Not loaded yet"

def generate_speech(
    text: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    backend_choice: str
) -> tuple:
    """
    Generate speech from text using Soprano TTS

    Args:
        text: Input text to synthesize
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter
        repetition_penalty: Penalty for repeating tokens
        backend_choice: Backend to use (auto, lmdeploy, or transformers)

    Returns:
        Tuple of (sample_rate, audio_array, status_message, system_info) for Gradio components
    """
    if not text.strip():
        return None, "Please enter some text to generate speech.", get_system_info()

    try:
        # Start timing
        start_time = time.time()

        # Get model instance (will use backend_choice if model not loaded yet)
        tts_model = get_model(backend=backend_choice)

        # Generate audio
        audio = tts_model.infer(
            text,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Convert to numpy array for Gradio
        audio_np = audio.cpu().numpy()

        # Convert to int16 format (Gradio expects 16-bit PCM audio)
        audio_int16 = (audio_np * 32767).astype(np.int16)

        # Calculate generation time
        generation_time = time.time() - start_time
        audio_duration = len(audio_np) / 32000

        # Return sample rate, audio, status, and system info
        status = f"✓ Generated {audio_duration:.2f}s of audio in {generation_time:.2f}s (RTF: {audio_duration/generation_time:.1f}x)"
        return (32000, audio_int16), status, get_system_info()

    except Exception as e:
        return None, f"✗ Error: {str(e)}", get_system_info()

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

    # System info display
    system_info_display = gr.Markdown(
        value=get_system_info(),
        elem_id="system-info"
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
                backend_choice = gr.Radio(
                    choices=["auto", "lmdeploy", "transformers"],
                    value="auto",
                    label="Backend",
                    info="Choose inference backend (auto=lmdeploy on CUDA/ROCm, transformers=compatibility mode)"
                )

                reload_btn = gr.Button("Reload Model with Selected Backend", size="sm")
                reload_status = gr.Textbox(
                    label="Reload Status",
                    interactive=False,
                    visible=False
                )

                gr.Markdown("---")

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
            ["Soprano is an extremely lightweight text to speech model.", 0.3, 0.95, 1.2, "auto"],
            ["Hello! Welcome to Soprano text to speech.", 0.3, 0.95, 1.2, "auto"],
            ["The quick brown fox jumps over the lazy dog.", 0.3, 0.95, 1.2, "auto"],
            ["Artificial intelligence is transforming the world.", 0.5, 0.90, 1.2, "auto"],
        ],
        inputs=[text_input, temperature, top_p, repetition_penalty, backend_choice],
        label="Example Prompts"
    )

    # Connect the generate button
    generate_btn.click(
        fn=generate_speech,
        inputs=[text_input, temperature, top_p, repetition_penalty, backend_choice],
        outputs=[audio_output, status_output, system_info_display]
    )

    # Connect the reload button
    reload_btn.click(
        fn=reload_model,
        inputs=[backend_choice],
        outputs=[reload_status, system_info_display]
    ).then(
        fn=lambda: gr.update(visible=True),
        inputs=None,
        outputs=reload_status
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
