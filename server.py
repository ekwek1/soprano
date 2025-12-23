import base64
import io
import json
from typing import Generator

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from scipy.io.wavfile import write
from torch import Tensor

from soprano.tts import SopranoTTS

# Load model at startup
tts = SopranoTTS()

app = FastAPI(title="Soprano TTS API")

def _tensor_to_wav_bytes(tensor: Tensor) -> bytes:
    """
    Convert a 1D fp32 torch tensor to a WAV byte stream.
    """
    # convert to int16
    audio_int16 = (np.clip(tensor.numpy(), -1.0, 1.0) * 32767).astype(np.int16)

    wav_io = io.BytesIO()
    write(wav_io, 32000, audio_int16) # 32kHz sample rate
    wav_io.seek(0)
    return wav_io.read()


def _sse_generator(text: str) -> Generator[bytes, None, None]:
    """
    Yield Server Sent Events compatible with OpenAI's `stream_format=audio`.
    Each event contains a base64-encoded chunk of WAV audio.
    """
    try:
        for chunk_tensor in tts.infer_stream(text):
            wav_bytes = _tensor_to_wav_bytes(chunk_tensor)
            b64_audio = base64.b64encode(wav_bytes).decode("utf-8")
            event = json.dumps({
                "audio": b64_audio,
                "type": "speech.audio.delta",
            })
            # SSE format: `data: <json>\n\n`
            yield f"data: {event}\n\n".encode("utf-8")
    except Exception as exc:
        # send a final error event on exception
        error_event = json.dumps({"error": str(exc), "type": "error"})
        yield f"data: {error_event}\n\n".encode("utf-8")


@app.post("/v1/audio/speech")
async def create_speech(payload: dict):
    """
    Minimal implementation of OpenAI's Speech endpoint.
    Fields:
      - input: string - text to synthesize
      - model, voice, etc. are accepted but ignored.
      - stream: bool - if true, returns an SSE audio stream, else 'audio/wav' blob.
      - response_format: str - ignored, only support wav.
    """
    text = payload.get("input")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="`input` field must be a non-empty string.")

    stream = payload.get("stream", False)
    if stream:
        generator = _sse_generator(text)
        return StreamingResponse(generator, media_type="text/event-stream")
    else:
        audio_tensor = tts.infer(text)
        wav_bytes = _tensor_to_wav_bytes(audio_tensor)
        return Response(content=wav_bytes, media_type="audio/wav", headers={"Content-Disposition": 'attachment; filename="speech.wav"'})
