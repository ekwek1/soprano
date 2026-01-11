import sounddevice as sd
import torch
import time


def play_stream(stream, sample_rate=32000):
    """
    Play streamed audio chunks to speakers in real time.
    """
    with sd.OutputStream(
        samplerate=sample_rate,
        channels=1,
        dtype='float32',
        blocksize=0
    ) as out_stream:
        start = time.time()
        latency = None
        first = True
        for chunk in stream:
            if first:
                latency = time.time()-start
                first = False

            if isinstance(chunk, torch.Tensor):
                chunk = chunk.detach().cpu()

            # Ensure shape (N, 1)
            if chunk.dim() == 1:
                chunk = chunk.unsqueeze(1)
            elif chunk.dim() == 2 and chunk.shape[0] == 1:
                chunk = chunk.transpose(0, 1)

            out_stream.write(chunk.numpy())
    return latency
