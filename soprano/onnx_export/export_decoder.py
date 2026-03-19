"""
Export SopranoDecoder to ONNX with full ISTFT baked in.
Uses dynamo exporter (default) + opset 18 for DFT and Col2Im support.
Patches in-place complex ops that break dynamo tracing.
Generates baselines, validates, converts to f16, and cleans up.
"""
import os

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from soprano.vocos.decoder import SopranoDecoder
from soprano.vocos.spectral_ops import ISTFT

from soprano._constants import MODEL_ID
from soprano.onnx_export._utils import DEFAULT_OUTPUT_DIR, compute_snr, convert_weights_to_f16


class PatchedISTFT(nn.Module):
    """
    ONNX-exportable ISTFT using explicit F.fold (maps to Col2Im at opset 18).
    Avoids:
    - torch.istft (decomposes to ScatterND, causes type errors)
    - In-place complex mutation (breaks dynamo tracing)
    Works for both "center" and "same" padding modes.
    """
    def __init__(self, original_istft: ISTFT):
        super().__init__()
        self.n_fft = original_istft.n_fft
        self.hop_length = original_istft.hop_length
        self.win_length = original_istft.win_length
        self.padding = original_istft.padding
        self.register_buffer("window", original_istft.window.clone())

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # Zero first/last freq bins via masking (no in-place mutation)
        N = spec.shape[1]
        mask = torch.ones(N, device=spec.device, dtype=torch.float32)
        mask[0] = 0.0
        mask[-1] = 0.0
        spec = spec * mask[None, :, None]

        B, N, T = spec.shape

        # Construct full spectrum from half-spectrum using conjugate symmetry,
        # then use ifft (not irfft). This avoids ONNX DFT's invalid
        # onesided=1 + inverse=1 combination.
        spec_flip = torch.flip(spec[:, 1:-1, :], dims=[1])
        spec_full = torch.cat([spec, spec_flip.conj()], dim=1)
        ifft = torch.fft.ifft(spec_full, dim=1, norm="backward").real
        ifft = ifft * self.window[None, :, None]

        # Overlap-add via F.fold (maps to Col2Im in ONNX opset 18)
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft, output_size=(1, output_size),
            kernel_size=(1, self.win_length), stride=(1, self.hop_length),
        )[:, 0, 0, :]

        # Window envelope normalization
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq, output_size=(1, output_size),
            kernel_size=(1, self.win_length), stride=(1, self.hop_length),
        ).squeeze()
        y = y / window_envelope.clamp(min=1e-11)

        # Trim based on padding mode
        if self.padding == "center":
            pad = self.n_fft // 2
            y = y[:, pad:-pad]
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
            y = y[:, pad:-pad]

        return y


def load_decoder(model_id=MODEL_ID):
    """Load original unpatched decoder."""
    decoder = SopranoDecoder()
    decoder_path = hf_hub_download(repo_id=model_id, filename='decoder.pth')
    decoder.load_state_dict(torch.load(decoder_path, map_location='cpu'))
    decoder.eval()
    return decoder


def load_and_patch_decoder(model_id=MODEL_ID):
    """Load decoder and patch ISTFT for ONNX export."""
    decoder = load_decoder(model_id=model_id)
    decoder.head.istft = PatchedISTFT(decoder.head.istft)
    return decoder


def export_decoder(
    output_dir=DEFAULT_OUTPUT_DIR,
    model_id=MODEL_ID,
    test_seq_lengths=(64, 42, 24),
    validate=True,
    convert_f16=True,
    keep_f32=False,
    f32_snr_threshold=40.0,
    f16_snr_threshold=30.0,
    patch_threshold=1e-4,
):
    """
    Export SopranoDecoder to ONNX with full ISTFT baked in.

    Args:
        output_dir: Directory to save exported ONNX models.
        model_id: HuggingFace model ID to download decoder weights from.
        test_seq_lengths: Sequence lengths for test inputs used in validation.
        validate: Whether to run validation against PyTorch baselines.
        convert_f16: Whether to convert to float16.
        keep_f32: Whether to keep the f32 model after f16 conversion.
        f32_snr_threshold: Min SNR (dB) for f32 validation to pass.
        f16_snr_threshold: Min SNR (dB) for f16 validation to pass.
        patch_threshold: Max allowed diff between patched and original decoder.

    Returns:
        dict with keys: 'f32_path', 'f16_path' (None if not converted),
        and 'validation' results if validate=True.
    """
    os.makedirs(output_dir, exist_ok=True)
    f32_path = os.path.join(output_dir, 'soprano_decoder.onnx')
    f16_path = os.path.join(output_dir, 'soprano_decoder_f16.onnx')
    result = {'f32_path': f32_path, 'f16_path': None, 'validation': {}}

    test_inputs = [torch.randn(1, 512, s) for s in test_seq_lengths]
    baselines = []

    if validate:
        # --- Generate baselines from original decoder ---
        print("Generating PyTorch baselines...")
        original = load_decoder(model_id=model_id)
        for inp in test_inputs:
            with torch.no_grad():
                baselines.append(original(inp).numpy())
        del original

    # --- Load and validate patched decoder ---
    print("Loading and patching decoder...")
    decoder = load_and_patch_decoder(model_id=model_id)

    if validate:
        print("Validating patch vs original...")
        for idx, inp in enumerate(test_inputs):
            with torch.no_grad():
                patched_audio = decoder(inp).numpy()
            diff = np.abs(patched_audio - baselines[idx]).max()
            print(f"  Input seq_len={inp.shape[2]}: max_diff={diff:.8f}")
            if diff >= patch_threshold:
                raise AssertionError(f"Patch diverged! max_diff={diff}")

    # --- Export to ONNX ---
    print("\nExporting with dynamo exporter, opset 18...")
    dummy = torch.randn(1, 512, 30)
    torch.onnx.export(
        decoder, dummy, f32_path,
        input_names=["hidden_states"],
        output_names=["audio"],
        dynamic_axes={"hidden_states": {2: "seq_len"}, "audio": {1: "audio_len"}},
        opset_version=18,
    )
    print(f"  f32 export: {os.path.getsize(f32_path) / 1024 / 1024:.1f} MB")

    # --- Validate f32 ONNX ---
    if validate:
        print("\nValidating f32 ONNX...")
        session = ort.InferenceSession(f32_path, providers=['CPUExecutionProvider'])
        f32_results = []
        for idx, inp in enumerate(test_inputs):
            inp_np = inp.numpy()
            audio_onnx = session.run(None, {'hidden_states': inp_np})[0]
            baseline = baselines[idx]
            min_len = min(audio_onnx.shape[1], baseline.shape[1])
            snr = compute_snr(audio_onnx[0, :min_len], baseline[0, :min_len])
            print(f"  seq_len={test_inputs[idx].shape[2]}: SNR={snr:.1f} dB")
            f32_results.append({'seq_len': inp.shape[2], 'snr': snr})
            if snr < f32_snr_threshold:
                raise AssertionError(f"f32 ONNX quality too low: SNR={snr:.1f}")
        result['validation']['f32'] = f32_results
        del session

    # --- Convert to f16 (weight-only, computation stays f32) ---
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
            f16_results = []
            for idx, inp in enumerate(test_inputs):
                inp_np = inp.numpy()
                audio_onnx = session.run(None, {'hidden_states': inp_np})[0]
                baseline = baselines[idx]
                min_len = min(audio_onnx.shape[1], baseline.shape[1])
                snr = compute_snr(audio_onnx[0, :min_len], baseline[0, :min_len])
                print(f"  seq_len={test_inputs[idx].shape[2]}: SNR={snr:.1f} dB")
                f16_results.append({'seq_len': inp.shape[2], 'snr': snr})
                if snr < f16_snr_threshold:
                    raise AssertionError(f"f16 ONNX quality too low: SNR={snr:.1f}")
            result['validation']['f16'] = f16_results

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
    export_decoder()


if __name__ == '__main__':
    main()
