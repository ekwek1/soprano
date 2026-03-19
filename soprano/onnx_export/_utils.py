import numpy as np
import onnx
from onnx import numpy_helper

DEFAULT_OUTPUT_DIR = 'onnx_models'


def compute_snr(test, reference):
    """Compute SNR (dB) between test and reference signals."""
    noise_power = np.mean((test - reference) ** 2)
    signal_power = np.mean(reference ** 2)
    if noise_power == 0:
        return float('inf')
    if signal_power == 0:
        return float('-inf')
    return float(10 * np.log10(signal_power / noise_power))


def convert_weights_to_f16(model):
    """
    Convert all f32 initializers to f16 with Cast nodes so graph computation stays f32.
    Modifies the model in-place.

    Returns:
        (converted_count, cast_count) tuple.
    """
    converted = 0
    for initializer in model.graph.initializer:
        if initializer.data_type == onnx.TensorProto.FLOAT:
            arr = numpy_helper.to_array(initializer).astype(np.float16)
            initializer.CopyFrom(numpy_helper.from_array(arr, name=initializer.name))
            converted += 1

    init_names = {init.name for init in model.graph.initializer}
    for vi in model.graph.input:
        if vi.name in init_names and vi.type.tensor_type.elem_type == onnx.TensorProto.FLOAT:
            vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT16

    nodes_to_add = []
    for initializer in model.graph.initializer:
        if initializer.data_type == onnx.TensorProto.FLOAT16:
            cast_out = initializer.name + '_f32'
            nodes_to_add.append(onnx.helper.make_node(
                'Cast', inputs=[initializer.name], outputs=[cast_out],
                to=onnx.TensorProto.FLOAT,
            ))
            for node in model.graph.node:
                for j, inp in enumerate(node.input):
                    if inp == initializer.name:
                        node.input[j] = cast_out
    for node in nodes_to_add:
        model.graph.node.insert(0, node)

    return converted, len(nodes_to_add)
