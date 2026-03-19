#!/usr/bin/env python3
"""Soprano TTS Command Line Interface"""
import sys
import argparse

from soprano._constants import MODEL_ID

SUBCOMMANDS = {'tts', 'export', 'test-e2e'}


def cmd_tts(args):
    from soprano import SopranoTTS
    from soprano.utils.streaming import play_stream

    tts = SopranoTTS(
        backend=args.backend,
        device=args.device,
        cache_size_mb=args.cache_size,
        decoder_batch_size=args.decoder_batch_size,
        model_path=args.model_path,
    )
    print(f"Generating speech for: '{args.text}'")
    if args.streaming:
        stream = tts.infer_stream(args.text, chunk_size=1)
        play_stream(stream)
    else:
        tts.infer(args.text, out_path=args.output)
        print(f"Audio saved to: {args.output}")


def cmd_export(args):
    try:
        from soprano.onnx_export import export_all, export_backbone, export_decoder
    except ImportError:
        print("ONNX export requires extra dependencies. Install with:")
        print("  uv pip install soprano-tts[onnx]")
        sys.exit(1)

    shared = dict(
        output_dir=args.output_dir,
        model_id=args.model_id,
        validate=args.validate,
        convert_f16=args.f16,
        keep_f32=args.keep_f32,
    )
    if args.component == 'all':
        result = export_all(
            **shared,
            run_e2e=args.e2e,
            e2e_sentences=args.e2e_sentences,
            e2e_compare_baseline=args.compare_baseline,
            e2e_snr_pass_threshold=args.snr_threshold,
        )
    elif args.component == 'backbone':
        result = export_backbone(**shared)
    elif args.component == 'decoder':
        result = export_decoder(**shared)
    else:
        raise ValueError(f"Unknown component: {args.component}")

    paths = [v for k, v in result.items() if k.endswith('_path') and v]
    if paths:
        print(f"\nExported: {', '.join(paths)}")


def cmd_test_e2e(args):
    try:
        from soprano.onnx_export import test_e2e
    except ImportError:
        print("ONNX e2e test requires extra dependencies. Install with:")
        print("  uv pip install soprano-tts[onnx]")
        sys.exit(1)

    test_e2e(
        onnx_dir=args.onnx_dir,
        model_id=args.model_id,
        test_sentences=args.sentences,
        use_f16=not args.f32,
        max_new_tokens=args.max_new_tokens,
        compare_baseline=args.compare_baseline,
        snr_pass_threshold=args.snr_threshold,
        snr_marginal_threshold=args.snr_marginal_threshold,
        save_audio=args.save_audio,
    )


def main():
    # Backwards compat: if first arg isn't a subcommand, assume 'tts'.
    # Note: text that exactly matches a subcommand name (e.g. "export") will be
    # interpreted as that subcommand rather than TTS input.
    argv = sys.argv[1:]
    if argv and argv[0] not in SUBCOMMANDS and not argv[0].startswith('-'):
        argv = ['tts'] + argv

    parser = argparse.ArgumentParser(description='Soprano Text-to-Speech CLI')
    subparsers = parser.add_subparsers(dest='command')

    # --- tts ---
    tts_parser = subparsers.add_parser('tts', help='Synthesize speech from text')
    tts_parser.add_argument('text', help='Text to synthesize')
    tts_parser.add_argument('--output', '-o', default='output.wav',
                            help='Output audio file path (non-streaming only)')
    tts_parser.add_argument('--model-path', '-m',
                            help='Path to local model directory (optional)')
    tts_parser.add_argument('--device', '-d', default='auto',
                            choices=['auto', 'cuda', 'cpu', 'mps'],
                            help='Device to use for inference')
    tts_parser.add_argument('--backend', '-b', default='auto',
                            choices=['auto', 'transformers', 'lmdeploy'],
                            help='Backend to use for inference')
    tts_parser.add_argument('--cache-size', '-c', type=int, default=100,
                            help='Cache size in MB (for lmdeploy backend)')
    tts_parser.add_argument('--decoder-batch-size', '-bs', type=int, default=1,
                            help='Batch size when decoding audio')
    tts_parser.add_argument('--streaming', '-s', action='store_true',
                            help='Enable streaming playback to speakers')
    tts_parser.set_defaults(func=cmd_tts)

    # --- export ---
    export_parser = subparsers.add_parser('export', help='Export models to ONNX')
    export_parser.add_argument('--component', choices=['all', 'backbone', 'decoder'],
                               default='all', help='Which component to export')
    export_parser.add_argument('--output-dir', default='onnx_models',
                               help='Directory to save ONNX models')
    export_parser.add_argument('--model-id', default=MODEL_ID,
                               help='HuggingFace model ID')
    export_parser.add_argument('--no-validate', action='store_false', dest='validate',
                               help='Skip validation against PyTorch baselines')
    export_parser.add_argument('--no-f16', action='store_false', dest='f16',
                               help='Skip float16 conversion')
    export_parser.add_argument('--keep-f32', action='store_true',
                               help='Keep f32 model after f16 conversion')
    export_parser.add_argument('--no-e2e', action='store_false', dest='e2e',
                               help='Skip end-to-end test after export (only for --component all)')
    export_parser.add_argument('--e2e-sentences', nargs='+',
                               help='Custom test sentences for e2e test')
    export_parser.add_argument('--no-compare-baseline', action='store_false', dest='compare_baseline',
                               help='Skip PyTorch baseline comparison in e2e test')
    export_parser.add_argument('--snr-threshold', type=float, default=30.0,
                               help='Min SNR (dB) for e2e test PASS')
    export_parser.set_defaults(func=cmd_export)

    # --- test-e2e ---
    e2e_parser = subparsers.add_parser('test-e2e', help='Run ONNX end-to-end inference test')
    e2e_parser.add_argument('--onnx-dir', default='onnx_models',
                            help='Directory containing exported ONNX models')
    e2e_parser.add_argument('--model-id', default=MODEL_ID,
                            help='HuggingFace model ID')
    e2e_parser.add_argument('--sentences', nargs='+',
                            help='Custom test sentences')
    e2e_parser.add_argument('--f32', action='store_true',
                            help='Use f32 ONNX models instead of f16')
    e2e_parser.add_argument('--max-new-tokens', type=int, default=512,
                            help='Max tokens for autoregressive generation')
    e2e_parser.add_argument('--no-compare-baseline', action='store_false', dest='compare_baseline',
                            help='Skip PyTorch baseline comparison')
    e2e_parser.add_argument('--snr-threshold', type=float, default=30.0,
                            help='Min SNR (dB) for PASS status')
    e2e_parser.add_argument('--snr-marginal-threshold', type=float, default=20.0,
                            help='Min SNR (dB) for MARGINAL status')
    e2e_parser.add_argument('--save-audio', metavar='DIR',
                            help='Directory to save generated .wav files')
    e2e_parser.set_defaults(func=cmd_test_e2e)

    args = parser.parse_args(argv)
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
