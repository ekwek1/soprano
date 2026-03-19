"""
ONNX export API for Soprano TTS.

Usage:
    from soprano.onnx_export import export_all, export_backbone, export_decoder, test_e2e

    # Export both backbone and decoder, then run e2e test
    result = export_all()

    # Export individually
    result = export_backbone()
    result = export_decoder()

    # Export without validation (faster)
    result = export_all(validate=False)

    # Export + e2e test with custom sentences
    result = export_all(e2e_sentences=["Hello world.", "Test sentence."])

    # Run e2e test only (models already exported)
    results = test_e2e()
    results = test_e2e(compare_baseline=False)  # skip PyTorch comparison
"""
from soprano._constants import MODEL_ID
from ._utils import DEFAULT_OUTPUT_DIR
from .export_backbone import export_backbone
from .export_decoder import export_decoder
from .onnx_e2e_inference import test_e2e


def export_all(
    output_dir=DEFAULT_OUTPUT_DIR,
    model_id=MODEL_ID,
    validate=True,
    convert_f16=True,
    keep_f32=False,
    backbone_kwargs=None,
    decoder_kwargs=None,
    run_e2e=True,
    e2e_sentences=None,
    e2e_compare_baseline=True,
    e2e_snr_pass_threshold=30.0,
):
    """
    Export both backbone and decoder to ONNX, then optionally run e2e test.

    Args:
        output_dir: Directory to save exported ONNX models.
        model_id: HuggingFace model ID.
        validate: Whether to run per-component validation during export.
        convert_f16: Whether to convert weights to float16.
        keep_f32: Whether to keep f32 models after f16 conversion.
        backbone_kwargs: Extra kwargs passed to export_backbone.
        decoder_kwargs: Extra kwargs passed to export_decoder.
        run_e2e: Whether to run e2e inference test after export.
        e2e_sentences: Test sentences for e2e test (None = defaults).
        e2e_compare_baseline: Whether to compare against PyTorch baseline in e2e.
        e2e_snr_pass_threshold: Min SNR (dB) for e2e PASS.

    Returns:
        dict with keys: 'backbone', 'decoder' (export results),
        and 'e2e' (list of per-sentence results) if run_e2e=True.
    """
    shared = dict(
        output_dir=output_dir,
        model_id=model_id,
        validate=validate,
        convert_f16=convert_f16,
        keep_f32=keep_f32,
    )

    print("=" * 60)
    print("Exporting backbone...")
    print("=" * 60)
    backbone_result = export_backbone(**shared, **(backbone_kwargs or {}))

    print("\n" + "=" * 60)
    print("Exporting decoder...")
    print("=" * 60)
    decoder_result = export_decoder(**shared, **(decoder_kwargs or {}))

    result = {
        'backbone': backbone_result,
        'decoder': decoder_result,
    }

    if run_e2e:
        print("\n" + "=" * 60)
        print("Running E2E test...")
        print("=" * 60)
        e2e_results = test_e2e(
            onnx_dir=output_dir,
            model_id=model_id,
            test_sentences=e2e_sentences,
            use_f16=convert_f16,
            compare_baseline=e2e_compare_baseline,
            snr_pass_threshold=e2e_snr_pass_threshold,
        )
        result['e2e'] = e2e_results

    return result


__all__ = ['export_all', 'export_backbone', 'export_decoder', 'test_e2e']
