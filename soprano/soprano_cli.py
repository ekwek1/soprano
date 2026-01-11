"""
Soprano TTS Command Line Interface
"""
import argparse
import sys
import os
import torch
# Add the parent directory to the Python path to resolve import issues when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soprano import SopranoTTS

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

def get_device():
    """Determine the best available device (CUDA if available, otherwise CPU)"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def play_audio(audio_tensor):
    """Play audio tensor using sounddevice"""
    if not SOUNDDEVICE_AVAILABLE:
        print("Error: sounddevice library not available. Install it with 'pip install sounddevice'")
        return

    import numpy as np
    audio_np = audio_tensor.cpu().numpy() if isinstance(audio_tensor, torch.Tensor) else audio_tensor

    duration = len(audio_np) / 32000
    print(f"Playing audio ({duration:.2f}s)...")

    sample_rate = 32000
    sd.play(audio_np, samplerate=sample_rate)

    import time
    time.sleep(duration + 0.5)

    try:
        if sd.get_status().playing:
            sd.wait()
    except:
        time.sleep(0.5)

def validate_text(text):
    """Validate input text"""
    stripped_text = text.strip() if text else ""
    if not stripped_text:
        print("Error: Text cannot be empty.")
        return False
    if len(stripped_text) > 1000:
        print("Error: Text is too long (max 1000 characters).")
        return False
    return True

def get_validated_input(prompt, validator_func, error_msg=None):
    """Get validated input from user"""
    while True:
        user_input = input(prompt).strip()
        if validator_func(user_input):
            return user_input
        else:
            if error_msg:
                print(error_msg)
            else:
                print("Invalid input, please try again.")

def get_next_filename(base_name="output_audio", ext=".wav"):
    """Generate next available filename with incremental numbering"""
    import os

    output_dir = "audio_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    counter = 0
    while True:
        if counter == 0:
            filename = f"{base_name}{ext}"
        else:
            filename = f"{base_name}{counter}{ext}"

        full_path = os.path.join(output_dir, filename)
        if not os.path.exists(full_path):
            return full_path
        counter += 1

def main():
    parser = argparse.ArgumentParser(description='Soprano Text-to-Speech CLI')
    parser.add_argument('--model-path', '-m', help='Path to local model directory (optional)')
    parser.add_argument('--backend', '-b', default='auto',
                       choices=['auto', 'transformers', 'lmdeploy'],
                       help='Backend to use for inference')
    parser.add_argument('--cache-size', '-c', type=int, default=10,
                       help='Cache size in MB (for lmdeploy backend)')

    args = parser.parse_args()

    device = get_device()

    try:
        import io
        import contextlib

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            tts = SopranoTTS(
                backend=args.backend,
                device=device,
                cache_size_mb=args.cache_size,
                model_path=args.model_path
            )
    except Exception as e:
        print(f"Error initializing model: {e}")
        sys.exit(1)

    print("Soprano TTS is ready. Starting interactive menu...")

    while True:
        print("\n" + "="*50)
        print("           SOPRANO TTS MENU")
        print("="*50)
        print("1. Input text for synthesis (with file saving)")
        print("2. Real-time audio playback (no file saving)")
        print("3. View saved audio files")
        print("4. Exit")
        print("="*50)

        choice = input("Enter your choice (1-4): ").strip()

        if choice == '1':
            text = get_validated_input(
                "Enter text to synthesize: ",
                validate_text,
                "Text must not be empty and must be under 1000 characters."
            )

            output_path = get_next_filename()
            print(f"Using output path: {output_path}")

            print(f"Generating speech for: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            try:
                tts.infer(text, out_path=output_path)
                print(f"✓ Audio saved to: {output_path}")
            except Exception as e:
                print(f"✗ Error generating audio: {e}")

        elif choice == '2':
            text = get_validated_input(
                "Enter text for real-time playback: ",
                validate_text,
                "Text must not be empty and must be under 1000 characters."
            )

            print(f"Generating real-time audio for: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            try:
                audio_tensor = tts.infer(text)
                print("Playing audio...")
                play_audio(audio_tensor)
                print("✓ Playback finished.")
            except Exception as e:
                print(f"✗ Error during playback: {e}")

        elif choice == '3':
            import os
            output_dir = "audio_output"
            if os.path.exists(output_dir):
                files = [f for f in os.listdir(output_dir) if f.lower().endswith('.wav')]
                if files:
                    print(f"Found {len(files)} audio file(s) in {output_dir}/:")
                    for i, file in enumerate(sorted(files), 1):
                        print(f"  {i}. {file}")
                else:
                    print(f"No audio files found in {output_dir}/")
            else:
                print(f"No {output_dir}/ directory exists yet.")

        elif choice == '4':
            print("Thank you for using Soprano TTS. Goodbye!")
            break

        else:
            print("✗ Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()