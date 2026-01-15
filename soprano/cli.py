#!/usr/bin/env python3
"""
Soprano TTS Command Line Interface
"""
import argparse
from soprano import SopranoTTS
from soprano.utils.streaming import play_stream

def main():
    parser = argparse.ArgumentParser(description='Soprano Text-to-Speech CLI')
    parser.add_argument('text', help='Text to synthesize')
    parser.add_argument('--output', '-o', default='output.wav',
                        help='Output audio file path (non-streaming only)')
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
    parser.add_argument('--streaming', '-s', action='store_true',
                        help='Enable streaming playback to speakers')
    
    args = parser.parse_args()
    
    # Initialize TTS
    tts = SopranoTTS(
        backend=args.backend,
        device=args.device,
        cache_size_mb=args.cache_size,
        decoder_batch_size=args.decoder_batch_size,
        model_path=args.model_path
    )
    
    print(f"Generating speech for: '{args.text}'")
    if args.streaming:
        stream = tts.infer_stream(args.text, chunk_size=1)
        play_stream(stream)
    else:
        tts.infer(args.text, out_path=args.output)
        print(f"Audio saved to: {args.output}")

if __name__ == "__main__":
    main()