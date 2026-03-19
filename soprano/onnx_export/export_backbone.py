"""
Export Soprano LLM backbone to ONNX with KV cache and hidden states output.
Generates baselines, validates, converts to f16, and cleans up.

Model: ekwek/Soprano-1.1-80M (Qwen3, 17 layers, 4 heads, 512 hidden dim)
KV cache: 17 layers x 2 (key/value), each [batch, 1, seq_len, 128]
"""
import os

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from soprano._constants import MODEL_ID
from soprano.onnx_export._utils import DEFAULT_OUTPUT_DIR, convert_weights_to_f16


class BackboneWithHiddenStates(nn.Module):
    """
    Wraps the causal LM to output last_hidden_state alongside logits and KV cache.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config
        self.num_layers = model.config.num_hidden_layers

    def forward(self, input_ids, attention_mask, position_ids, *past_key_values_flat):
        past_key_values = DynamicCache()
        for i in range(self.num_layers):
            k = past_key_values_flat[2 * i]
            v = past_key_values_flat[2 * i + 1]
            past_key_values.update(k, v, i)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )

        present_kv = []
        cache = outputs.past_key_values
        for i in range(self.num_layers):
            present_kv.append(cache.layers[i].keys)
            present_kv.append(cache.layers[i].values)

        return (outputs.logits, outputs.hidden_states[-1], *present_kv)


def make_onnx_feeds(input_ids_np, num_layers, num_kv_heads, head_dim, past_len=0):
    """Build ONNX Runtime feed dict for backbone."""
    seq_len = input_ids_np.shape[1]
    feeds = {
        'input_ids': input_ids_np.astype(np.int64),
        'attention_mask': np.ones((1, past_len + seq_len), dtype=np.int64),
        'position_ids': np.arange(past_len, past_len + seq_len, dtype=np.int64).reshape(1, -1),
    }
    for i in range(num_layers):
        feeds[f'past_key_values.{i}.key'] = np.zeros((1, num_kv_heads, past_len, head_dim), dtype=np.float32)
        feeds[f'past_key_values.{i}.value'] = np.zeros((1, num_kv_heads, past_len, head_dim), dtype=np.float32)
    return feeds


def export_backbone(
    output_dir=DEFAULT_OUTPUT_DIR,
    model_id=MODEL_ID,
    test_prompt='[STOP][TEXT]Hello world.[START]',
    validate=True,
    convert_f16=True,
    keep_f32=False,
    f32_logits_threshold=0.01,
    f32_hidden_threshold=0.01,
    f16_logits_threshold=1.0,
    f16_hidden_threshold=0.1,
):
    """
    Export Soprano LLM backbone to ONNX with KV cache and hidden states output.

    Args:
        output_dir: Directory to save exported ONNX models.
        model_id: HuggingFace model ID to export.
        test_prompt: Prompt used for baseline generation and validation.
        validate: Whether to run validation against PyTorch baselines.
        convert_f16: Whether to convert weights to float16.
        keep_f32: Whether to keep the f32 model after f16 conversion.
        f32_logits_threshold: Max allowed logits error for f32 validation.
        f32_hidden_threshold: Max allowed hidden states error for f32 validation.
        f16_logits_threshold: Max allowed logits error for f16 validation.
        f16_hidden_threshold: Max allowed hidden states error for f16 validation.

    Returns:
        dict with keys: 'f32_path', 'f16_path' (None if not converted),
        and 'validation' results if validate=True.
    """
    os.makedirs(output_dir, exist_ok=True)
    f32_path = os.path.join(output_dir, 'soprano_backbone_kv.onnx')
    f16_path = os.path.join(output_dir, 'soprano_backbone_kv_f16.onnx')
    result = {'f32_path': f32_path, 'f16_path': None, 'validation': {}}

    # --- Load model and generate baselines ---
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, device_map='cpu')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    print(f"Config: {num_layers} layers, {num_kv_heads} KV heads, {head_dim} head_dim")

    baseline_logits, baseline_hidden, baseline_input_ids = None, None, None
    if validate:
        print("Generating PyTorch baseline...")
        inputs = tokenizer(test_prompt, return_tensors='pt')
        input_ids = inputs['input_ids']
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        baseline_logits = outputs.logits[:, -1, :].numpy()
        baseline_hidden = outputs.hidden_states[-1][:, -1, :].numpy()
        baseline_input_ids = input_ids.numpy()
        print(f"  Baseline logits: {baseline_logits.shape}, hidden: {baseline_hidden.shape}")

    # --- Export ---
    print("\nExporting backbone...")
    wrapper = BackboneWithHiddenStates(model)
    wrapper.eval()

    # Use past_len > 0 so tracer captures KV cache path
    batch, seq_len, past_len = 1, 1, 5
    dummy_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
    dummy_mask = torch.ones(batch, past_len + seq_len, dtype=torch.long)
    dummy_pos = torch.arange(past_len, past_len + seq_len).unsqueeze(0)
    past_kv_flat = []
    for _ in range(num_layers):
        past_kv_flat.append(torch.zeros(batch, num_kv_heads, past_len, head_dim))
        past_kv_flat.append(torch.zeros(batch, num_kv_heads, past_len, head_dim))

    input_names = ['input_ids', 'attention_mask', 'position_ids']
    output_names = ['logits', 'last_hidden_state']
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'past_sequence_length + sequence_length'},
        'position_ids': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size', 1: 'sequence_length'},
        'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'},
    }
    for i in range(num_layers):
        input_names += [f'past_key_values.{i}.key', f'past_key_values.{i}.value']
        output_names += [f'present.{i}.key', f'present.{i}.value']
        dynamic_axes[f'past_key_values.{i}.key'] = {0: 'batch_size', 2: 'past_sequence_length'}
        dynamic_axes[f'past_key_values.{i}.value'] = {0: 'batch_size', 2: 'past_sequence_length'}
        dynamic_axes[f'present.{i}.key'] = {0: 'batch_size', 2: 'past_sequence_length + sequence_length'}
        dynamic_axes[f'present.{i}.value'] = {0: 'batch_size', 2: 'past_sequence_length + sequence_length'}

    dummy_inputs = (dummy_ids, dummy_mask, dummy_pos, *past_kv_flat)

    print(f"  {len(input_names)} inputs, {len(output_names)} outputs")
    torch.onnx.export(
        wrapper, dummy_inputs, f32_path,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=18, dynamo=False,
    )
    print(f"  f32 export: {os.path.getsize(f32_path) / 1024 / 1024:.1f} MB")
    del model, wrapper

    # --- Validate f32 ONNX ---
    if validate:
        print("\nValidating f32 ONNX...")
        session = ort.InferenceSession(f32_path, providers=['CPUExecutionProvider'])
        feeds = make_onnx_feeds(baseline_input_ids, num_layers, num_kv_heads, head_dim)
        results = session.run(None, feeds)
        logits_err = float(np.max(np.abs(results[0][:, -1, :] - baseline_logits)))
        hidden_err = float(np.max(np.abs(results[1][:, -1, :] - baseline_hidden)))
        print(f"  Logits max error: {logits_err:.6f}")
        print(f"  Hidden max error: {hidden_err:.6f}")
        f32_passed = logits_err < f32_logits_threshold and hidden_err < f32_hidden_threshold
        result['validation']['f32'] = {
            'logits_error': logits_err,
            'hidden_error': hidden_err,
            'passed': f32_passed,
        }
        print(f"  >>> {'PASS' if f32_passed else 'FAIL'}")
        if not f32_passed:
            raise AssertionError(
                f"f32 ONNX validation failed! logits_err={logits_err:.6f}, hidden_err={hidden_err:.6f}"
            )
        del session

    # --- Convert to f16 (weight-only, computation stays f32) ---
    # NOTE: We intentionally use manual conversion + Cast nodes here instead of
    # onnxruntime's convert_float_to_float16(keep_io_types=True), which converts
    # all ops to f16. The backbone needs f32 computation for KV cache accuracy.
    if convert_f16:
        print("\nConverting to f16 (weight-only)...")
        model = onnx.load(f32_path)
        converted, cast_count = convert_weights_to_f16(model)
        onnx.save(model, f16_path)
        result['f16_path'] = f16_path
        print(f"  Converted {converted} weights, added {cast_count} Cast nodes")
        print(f"  f16 export: {os.path.getsize(f16_path) / 1024 / 1024:.1f} MB")
        del model

        # --- Validate f16 ONNX ---
        if validate:
            print("Validating f16 ONNX...")
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            session = ort.InferenceSession(f16_path, sess_options=opts, providers=['CPUExecutionProvider'])
            feeds = make_onnx_feeds(baseline_input_ids, num_layers, num_kv_heads, head_dim)
            results = session.run(None, feeds)
            logits_err = float(np.max(np.abs(results[0][:, -1, :] - baseline_logits)))
            hidden_err = float(np.max(np.abs(results[1][:, -1, :] - baseline_hidden)))
            top5_baseline = np.argsort(baseline_logits[0])[-5:][::-1]
            top5_f16 = np.argsort(results[0][0, -1, :])[-5:][::-1]
            print(f"  Logits max error: {logits_err:.6f}")
            print(f"  Hidden max error: {hidden_err:.6f}")
            print(f"  Top-5 match: {set(top5_baseline) == set(top5_f16)}")
            f16_passed = logits_err < f16_logits_threshold and hidden_err < f16_hidden_threshold
            result['validation']['f16'] = {
                'logits_error': logits_err,
                'hidden_error': hidden_err,
                'top5_match': set(top5_baseline) == set(top5_f16),
                'passed': f16_passed,
            }
            print(f"  >>> {'PASS' if f16_passed else 'FAIL'}")
            if not f16_passed:
                raise AssertionError(
                    f"f16 ONNX validation failed! logits_err={logits_err:.6f}, hidden_err={hidden_err:.6f}"
                )

        # --- Clean up f32 model ---
        if not keep_f32:
            print("\nCleaning up f32 model...")
            os.remove(f32_path)
            data_path = f32_path + '.data'
            if os.path.exists(data_path):
                os.remove(data_path)
            result['f32_path'] = None

    print("\nDone!")
    return result


def main():
    export_backbone()


if __name__ == '__main__':
    main()
