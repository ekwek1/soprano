#!/usr/bin/env python3
"""
Soprano TTS Command Line Interface
"""
import argparse
import os
from soprano import SopranoTTS

def main():
    parser = argparse.ArgumentParser(description='Soprano Text-to-Speech CLI')
    parser.add_argument('text', help='Text to synthesize')
    parser.add_argument('--output', '-o', default='output.wav', help='Output audio file path')
    parser.add_argument('--model-path', '-m', help='Path to local model directory (optional)')
    parser.add_argument('--device', '-d', default='cpu', choices=['cuda', 'cpu'], 
                       help='Device to use for inference')
    parser.add_argument('--backend', '-b', default='auto', 
                       choices=['auto', 'transformers', 'lmdeploy'],
                       help='Backend to use for inference')
    parser.add_argument('--cache-size', '-c', type=int, default=10,
                       help='Cache size in MB (for lmdeploy backend)')
    
    args = parser.parse_args()
    
    # Initialize TTS
    tts = SopranoTTS(
        backend=args.backend,
        device=args.device,
        cache_size_mb=args.cache_size,
        model_path=args.model_path
    )
    
    # Generate speech
    print(f"Generating speech for: '{args.text}'")
    tts.infer(args.text, out_path=args.output)
    print(f"Audio saved to: {args.output}")

if __name__ == "__main__":
    main()