"""
End-to-end ONNX inference for Soprano TTS.
Runs backbone (autoregressive with KV cache) + decoder (hidden states -> audio)
entirely via ONNX Runtime. Generates PyTorch baselines on the fly for comparison.
"""
import os
import time

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, AutoConfig

from soprano._constants import MODEL_ID
from soprano.onnx_export._utils import DEFAULT_OUTPUT_DIR, compute_snr

SAMPLE_RATE = 32000
MAX_NEW_TOKENS = 512

TEST_SENTENCES = [
    "Hello world, this is a test of the soprano text to speech system.",
    "The quick brown fox jumps over the lazy dog.",
    "Testing one two three.",
]


def get_model_config(model_id=MODEL_ID):
    """Read architecture constants from model config instead of hardcoding."""
    config = AutoConfig.from_pretrained(model_id)
    return {
        'num_layers': config.num_hidden_layers,
        'num_kv_heads': getattr(config, 'num_key_value_heads', config.num_attention_heads),
        'head_dim': config.hidden_size // config.num_attention_heads,
        'eos_token_id': config.eos_token_id,
    }


def load_pytorch_models(model_id=MODEL_ID):
    """Load PyTorch backbone and decoder once for baseline generation."""
    import torch
    from transformers import AutoModelForCausalLM
    from huggingface_hub import hf_hub_download
    from soprano.vocos.decoder import SopranoDecoder

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, device_map='cpu')
    model.eval()

    decoder = SopranoDecoder()
    decoder_path = hf_hub_download(repo_id=model_id, filename='decoder.pth')
    decoder.load_state_dict(torch.load(decoder_path, map_location='cpu'))
    decoder.eval()

    return model, decoder


def generate_pytorch_baseline(text, tokenizer, model, decoder, max_new_tokens=MAX_NEW_TOKENS):
    """Generate baseline audio using PyTorch (greedy decoding)."""
    import torch

    prompt = f'[STOP][TEXT]{text}[START]'
    inputs = tokenizer(prompt, return_tensors='pt')

    with torch.no_grad():
        gen_outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

    eos_token_id = model.config.eos_token_id
    seq = gen_outputs.sequences[0]
    hidden_states = []
    num_output_tokens = len(gen_outputs.hidden_states)
    for j in range(num_output_tokens):
        token = seq[j + seq.size(0) - num_output_tokens]
        if token != eos_token_id:
            hidden_states.append(gen_outputs.hidden_states[j][-1][0, -1, :])

    hs = torch.stack(hidden_states)
    decoder_input = hs.unsqueeze(0).transpose(1, 2).float()
    with torch.no_grad():
        audio = decoder(decoder_input)

    return audio[0].numpy(), hs.numpy(), len(hidden_states)


def load_sessions(use_f16=True, onnx_dir=DEFAULT_OUTPUT_DIR):
    """Load backbone and decoder ONNX sessions."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    suffix = '_f16' if use_f16 else ''
    backbone_path = os.path.join(onnx_dir, f'soprano_backbone_kv{suffix}.onnx')
    decoder_path = os.path.join(onnx_dir, f'soprano_decoder{suffix}.onnx')

    if not os.path.exists(backbone_path) or not os.path.exists(decoder_path):
        print("ONNX models not found, exporting first...")
        from soprano.onnx_export import export_all
        export_all(output_dir=onnx_dir, convert_f16=use_f16, run_e2e=False)

    print(f"Loading backbone: {os.path.basename(backbone_path)}")
    backbone = ort.InferenceSession(backbone_path, sess_options=opts, providers=['CPUExecutionProvider'])

    print(f"Loading decoder: {os.path.basename(decoder_path)}")
    decoder = ort.InferenceSession(decoder_path, providers=['CPUExecutionProvider'])

    return backbone, decoder


def run_backbone_autoregressive(backbone, input_ids, model_cfg, max_new_tokens=MAX_NEW_TOKENS):
    """
    Run autoregressive generation with the backbone ONNX model.
    Uses greedy decoding for deterministic comparison.
    """
    num_layers = model_cfg['num_layers']
    num_kv_heads = model_cfg['num_kv_heads']
    head_dim = model_cfg['head_dim']
    eos_token_id = model_cfg['eos_token_id']

    seq_len = input_ids.shape[1]
    generated_ids = list(input_ids[0])
    hidden_states = []

    kv_cache = {}
    for i in range(num_layers):
        kv_cache[f'past_key_values.{i}.key'] = np.zeros((1, num_kv_heads, 0, head_dim), dtype=np.float32)
        kv_cache[f'past_key_values.{i}.value'] = np.zeros((1, num_kv_heads, 0, head_dim), dtype=np.float32)

    # Prefill
    feeds = {
        'input_ids': input_ids.astype(np.int64),
        'attention_mask': np.ones((1, seq_len), dtype=np.int64),
        'position_ids': np.arange(seq_len, dtype=np.int64).reshape(1, -1),
        **kv_cache,
    }
    results = backbone.run(None, feeds)
    logits = results[0]
    last_hidden = results[1]
    for i in range(num_layers):
        kv_cache[f'past_key_values.{i}.key'] = results[2 + 2 * i]
        kv_cache[f'past_key_values.{i}.value'] = results[2 + 2 * i + 1]

    next_token = int(np.argmax(logits[0, -1, :]))

    # Autoregressive loop
    for step in range(max_new_tokens):
        if next_token == eos_token_id:
            break

        hidden_states.append(last_hidden[0, -1, :])
        generated_ids.append(next_token)

        past_len = kv_cache[f'past_key_values.0.key'].shape[2]
        feeds = {
            'input_ids': np.array([[next_token]], dtype=np.int64),
            'attention_mask': np.ones((1, past_len + 1), dtype=np.int64),
            'position_ids': np.array([[past_len]], dtype=np.int64),
            **kv_cache,
        }
        results = backbone.run(None, feeds)
        logits = results[0]
        last_hidden = results[1]
        for i in range(num_layers):
            kv_cache[f'past_key_values.{i}.key'] = results[2 + 2 * i]
            kv_cache[f'past_key_values.{i}.value'] = results[2 + 2 * i + 1]

        next_token = int(np.argmax(logits[0, -1, :]))

    return hidden_states, generated_ids[seq_len:]


def run_decoder(decoder, hidden_states):
    """Run decoder: hidden states -> audio."""
    hs = np.stack(hidden_states, axis=0)
    decoder_input = hs[np.newaxis].transpose(0, 2, 1).astype(np.float32)
    return decoder.run(None, {'hidden_states': decoder_input})[0].flatten()


def compute_metrics(audio_onnx, audio_baseline):
    """Compute SNR and max error."""
    min_len = min(len(audio_onnx), len(audio_baseline))
    a, b = audio_onnx[:min_len], audio_baseline[:min_len]
    max_err = float(np.max(np.abs(a - b)))
    snr = compute_snr(a, b)
    return snr, max_err


def test_e2e(
    onnx_dir=DEFAULT_OUTPUT_DIR,
    model_id=MODEL_ID,
    test_sentences=None,
    use_f16=True,
    max_new_tokens=MAX_NEW_TOKENS,
    compare_baseline=True,
    snr_pass_threshold=30.0,
    snr_marginal_threshold=20.0,
    save_audio=None,
):
    """
    Run end-to-end ONNX inference test for Soprano TTS.

    Args:
        onnx_dir: Directory containing exported ONNX models.
        model_id: HuggingFace model ID for tokenizer and PyTorch baseline.
        test_sentences: List of sentences to test. Defaults to built-in set.
        use_f16: Whether to use f16 ONNX models.
        max_new_tokens: Max tokens for autoregressive generation.
        compare_baseline: Whether to generate PyTorch baselines for comparison.
        snr_pass_threshold: Min SNR (dB) to consider a sentence PASS.
        snr_marginal_threshold: Min SNR (dB) to consider MARGINAL (below = FAIL).
        save_audio: Directory to save generated .wav files. None to skip.

    Returns:
        list of dicts, one per sentence, with keys: 'text', 'n_tokens',
        'audio_duration', 'backbone_time', 'decoder_time', 'rtf',
        and optionally 'snr', 'max_err', 'hs_snr', 'status'.
    """
    if test_sentences is None:
        test_sentences = TEST_SENTENCES

    model_cfg = get_model_config(model_id)
    backbone, decoder = load_sessions(use_f16=use_f16, onnx_dir=onnx_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"\n{'='*60}")
    print(f"E2E ONNX Inference ({'f16' if use_f16 else 'f32'}) {'vs PyTorch Baseline' if compare_baseline else ''}")
    print(f"{'='*60}")

    if save_audio:
        from scipy.io import wavfile as _wavfile
        os.makedirs(save_audio, exist_ok=True)

    baselines = {}
    if compare_baseline:
        print("\nGenerating PyTorch baselines...")
        pt_model, pt_decoder = load_pytorch_models(model_id)
        for i, text in enumerate(test_sentences):
            print(f"  Sentence {i}: \"{text[:40]}...\"")
            audio, hs, n_tokens = generate_pytorch_baseline(
                text, tokenizer, pt_model, pt_decoder, max_new_tokens=max_new_tokens,
            )
            baselines[i] = {'audio': audio, 'hidden_states': hs, 'n_tokens': n_tokens}
            print(f"    {n_tokens} tokens, {len(audio)} samples")
            if save_audio:
                wav_path = os.path.join(save_audio, f'sentence_{i}_pytorch.wav')
                audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
                _wavfile.write(wav_path, SAMPLE_RATE, audio_int16)
                print(f"    Saved: {wav_path}")
        del pt_model, pt_decoder

    results = []
    for i, text in enumerate(test_sentences):
        print(f"\n--- Sentence {i}: \"{text[:50]}\" ---")

        prompt = f'[STOP][TEXT]{text}[START]'
        inputs = tokenizer(prompt, return_tensors='np')
        input_ids = inputs['input_ids'].astype(np.int64)

        t0 = time.time()
        hidden_states, gen_tokens = run_backbone_autoregressive(
            backbone, input_ids, model_cfg, max_new_tokens=max_new_tokens,
        )
        backbone_time = time.time() - t0

        if len(hidden_states) == 0:
            print("  WARNING: No hidden states generated!")
            results.append({'text': text, 'n_tokens': 0, 'status': 'NO_OUTPUT'})
            continue

        t0 = time.time()
        audio_onnx = run_decoder(decoder, hidden_states)
        decoder_time = time.time() - t0

        total_time = backbone_time + decoder_time
        audio_dur = len(audio_onnx) / SAMPLE_RATE
        rtf = audio_dur / total_time

        if save_audio:
            wav_path = os.path.join(save_audio, f'sentence_{i}_onnx.wav')
            audio_int16 = np.clip(audio_onnx * 32767, -32768, 32767).astype(np.int16)
            _wavfile.write(wav_path, SAMPLE_RATE, audio_int16)
            print(f"  Saved: {wav_path}")

        print(f"  Tokens: {len(gen_tokens)} ({backbone_time:.2f}s, {len(gen_tokens)/backbone_time:.1f} tok/s)")
        print(f"  Audio: {audio_dur:.2f}s, decoder: {decoder_time:.3f}s, RTF: {rtf:.2f}x")

        entry = {
            'text': text,
            'n_tokens': len(gen_tokens),
            'audio_duration': audio_dur,
            'backbone_time': backbone_time,
            'decoder_time': decoder_time,
            'rtf': rtf,
        }

        if compare_baseline and i in baselines:
            bl = baselines[i]
            snr, max_err = compute_metrics(audio_onnx, bl['audio'])
            hs_onnx = np.stack(hidden_states, axis=0)
            hs_bl = bl['hidden_states']
            min_t = min(len(hs_onnx), len(hs_bl))
            hs_snr = compute_snr(hs_onnx[:min_t], hs_bl[:min_t])

            if snr > snr_pass_threshold:
                status = 'PASS'
            elif snr > snr_marginal_threshold:
                status = 'MARGINAL'
            else:
                status = 'FAIL'

            print(f"  vs PyTorch: SNR={snr:.1f} dB, max_err={max_err:.4f}")
            print(f"  Hidden states: tokens={len(hs_onnx)} vs {len(hs_bl)}, SNR={hs_snr:.1f} dB")
            print(f"  >>> {status}")

            entry.update({'snr': snr, 'max_err': max_err, 'hs_snr': hs_snr, 'status': status})

        results.append(entry)

    return results


def main():
    test_e2e()


if __name__ == '__main__':
    main()
