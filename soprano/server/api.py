import asyncio
import io
import logging
import os
import time
from typing import Optional, Dict, Any, AsyncGenerator
import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import Response
from pydantic import BaseModel, Field
from scipy.io.wavfile import write
from torch import Tensor
import torch
from contextlib import asynccontextmanager

from soprano.tts import SopranoTTS


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Circuit breaker implementation to handle external dependency failures.
    """
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is in OPEN state and not accepting requests")

        if self.state == "HALF_OPEN":
            try:
                result = func(*args, **kwargs)
                self._success()
                return result
            except Exception as e:
                self._failure()
                raise e

        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            self._failure()
            raise e

    def _failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

    def _success(self):
        self.failure_count = 0
        self.state = "CLOSED"


def retry(func, retries=3, delay=1, backoff=2):
    """
    Retry decorator with exponential backoff for transient failures.
    """
    def wrapper(*args, **kwargs):
        current_delay = delay
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == retries - 1:  # Last attempt
                    raise e
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay} seconds...")
                time.sleep(current_delay)
                current_delay *= backoff
        return None
    return wrapper


class SpeechRequest(BaseModel):
    """
    Request model for text-to-speech conversion following OpenAI API format.
    """
    input: str = Field(..., min_length=1, max_length=1000, description="Text to synthesize")
    model: Optional[str] = Field(None, description="Model to use (ignored, using default model)")
    voice: Optional[str] = Field(None, description="Voice to use (ignored, using default voice)")
    response_format: Optional[str] = Field("wav", description="Response format (only wav supported)")
    speed: Optional[float] = Field(None, ge=0.1, le=2.0, description="Speech speed (not implemented yet)")
    temperature: Optional[float] = Field(0.3, ge=0.0, le=1.0, description="Generation temperature")
    top_p: Optional[float] = Field(0.95, ge=0.0, le=1.0, description="Top-p sampling parameter")
    repetition_penalty: Optional[float] = Field(1.2, ge=0.1, le=2.0, description="Repetition penalty")


class TTSManager:
    """
    Singleton manager for TTS model lifecycle and inference.
    """
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.tts: Optional[SopranoTTS] = None
            # Prioritize CUDA, fallback to CPU only if CUDA is not available
            if torch.cuda.is_available():
                self.device = 'cuda'
                logger.info("CUDA is available, using GPU for TTS processing")
            else:
                self.device = 'cpu'
                logger.info("CUDA is not available, falling back to CPU for TTS processing")
            self.circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
            logger.info(f"Initializing TTS on device: {self.device}")

    async def initialize_model(self):
        """
        Initialize the TTS model asynchronously to avoid blocking the event loop.
        """
        async with self._lock:
            if self.tts is None:
                logger.info("Loading Soprano TTS model...")
                try:
                    # Run model initialization in a thread pool to avoid blocking
                    loop = asyncio.get_event_loop()

                    # Use retry mechanism for model loading
                    def load_model():
                        return SopranoTTS(
                            cache_size_mb=100,
                            device=self.device
                        )

                    self.tts = await loop.run_in_executor(
                        None,
                        retry(load_model, retries=3, delay=2, backoff=2)
                    )
                    logger.info("Soprano TTS model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load Soprano TTS model: {e}", exc_info=True)
                    raise RuntimeError(f"Failed to initialize TTS model: {str(e)}") from e

    def get_model(self) -> SopranoTTS:
        """
        Get the initialized TTS model instance.
        """
        if self.tts is None:
            raise RuntimeError("TTS model not initialized. Call initialize_model() first.")
        return self.tts

    def generate_audio(self, text: str, top_p: float, temperature: float, repetition_penalty: float):
        """
        Generate audio with circuit breaker protection and retry mechanism.
        """
        def _generate():
            return self.tts.infer(
                text=text,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty
            )

        # Use circuit breaker to protect against repeated failures
        return self.circuit_breaker.call(
            retry(_generate, retries=2, delay=1, backoff=2)
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan event handler for startup and shutdown events.
    """
    logger.info("Starting up Soprano TTS API server...")
    try:
        tts_manager = TTSManager()
        await tts_manager.initialize_model()

        # Perform initial cleanup of old files
        output_dir = "audio_output"
        os.makedirs(output_dir, exist_ok=True)
        cleanup_old_files(output_dir)

        logger.info("Soprano TTS API server started successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to start Soprano TTS API server: {e}", exc_info=True)
        raise
    finally:
        logger.info("Shutting down Soprano TTS API server...")


# Create FastAPI app with metadata
app = FastAPI(
    title="Soprano TTS API",
    description="Ultra-realistic Text-to-Speech API based on Soprano model",
    version="1.0.0",
    contact={
        "name": "Soprano TTS",
        "url": "https://github.com/ekwek1/soprano",
    },
    lifespan=lifespan
)


def _tensor_to_wav_bytes(tensor: Tensor) -> bytes:
    """
    Convert a 1D fp32 torch tensor to a WAV byte stream.
    """
    # Convert to numpy array
    audio_np = tensor.cpu().numpy()

    # Normalize to int16 range if needed
    if audio_np.dtype != np.int16:
        # Ensure values are in the range [-1, 1]
        audio_np = np.clip(audio_np, -1.0, 1.0)
        # Convert to int16
        audio_np = (audio_np * 32767).astype(np.int16)

    # Create in-memory WAV file
    wav_io = io.BytesIO()
    write(wav_io, 32000, audio_np)  # 32kHz sample rate
    wav_io.seek(0)
    return wav_io.read()


def cleanup_old_files(directory: str, max_age_hours: int = 24, max_files: int = 100):
    """
    Clean up old files in the specified directory to prevent unlimited growth.

    Args:
        directory: Directory to clean up
        max_age_hours: Maximum age of files in hours
        max_files: Maximum number of files to keep
    """
    import time
    try:
        files = []
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                files.append((filepath, os.path.getctime(filepath)))

        # Sort by creation time (oldest first)
        files.sort(key=lambda x: x[1])

        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)

        # Remove files older than cutoff time
        removed_count = 0
        for filepath, creation_time in files:
            if creation_time < cutoff_time:
                try:
                    os.remove(filepath)
                    removed_count += 1
                    logger.info(f"Removed old file: {filepath}")
                except OSError as e:
                    logger.error(f"Failed to remove old file {filepath}: {e}")

        # If still too many files, remove oldest ones beyond the limit
        remaining_files = len(files) - removed_count
        if remaining_files > max_files:
            excess_count = remaining_files - max_files
            for i in range(excess_count):
                if i < len(files) - removed_count:
                    filepath = files[i][0]
                    try:
                        os.remove(filepath)
                        removed_count += 1
                        logger.info(f"Removed excess file: {filepath}")
                    except OSError as e:
                        logger.error(f"Failed to remove excess file {filepath}: {e}")

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old files from {directory}")
    except Exception as e:
        logger.error(f"Error during file cleanup: {e}")




@app.post("/v1/audio/speech",
          response_class=Response,
          summary="Generate speech from text",
          description="Convert input text to audio using Soprano TTS model")
async def create_speech(request: SpeechRequest):
    """
    Generate speech from input text following OpenAI's Speech endpoint format.
    """
    try:
        # Validate input text
        if not request.input or not request.input.strip():
            raise HTTPException(
                status_code=400,
                detail="`input` field must be a non-empty string."
            )

        # Check text length
        if len(request.input) > 1000:
            raise HTTPException(
                status_code=400,
                detail="Input text exceeds maximum length of 1000 characters."
            )

        # Get TTS manager and generate audio using circuit breaker and retry
        tts_manager = TTSManager()

        logger.info(f"Processing TTS request for text: '{request.input[:50]}{'...' if len(request.input) > 50 else ''}'")

        try:
            # Generate audio with circuit breaker and retry mechanism
            audio_tensor = tts_manager.generate_audio(
                text=request.input,
                top_p=request.top_p,
                temperature=request.temperature,
                repetition_penalty=request.repetition_penalty
            )
        except Exception as e:
            logger.error(f"Circuit breaker or retry mechanism failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=503,
                detail=f"Service temporarily unavailable due to TTS processing error: {str(e)}"
            )

        # Convert tensor to WAV bytes
        wav_bytes = _tensor_to_wav_bytes(audio_tensor)

        # Create audio_output directory if it doesn't exist
        output_dir = "audio_output"
        os.makedirs(output_dir, exist_ok=True)

        # Generate unique filename with sequential numbering
        file_counter = 1
        while True:
            filename = f"output_{file_counter}.wav"
            filepath = os.path.join(output_dir, filename)
            if not os.path.exists(filepath):
                break
            file_counter += 1

        # Prevent path traversal by ensuring filename is safe
        if '..' in filename or '/' in filename or '\\' in filename:
            raise HTTPException(
                status_code=400,
                detail="Invalid characters in text that could lead to path traversal."
            )

        # Save the audio file to the audio_output directory with error handling
        try:
            with open(filepath, 'wb') as f:
                f.write(wav_bytes)
            logger.info(f"Audio saved to: {filepath}")
        except OSError as e:
            logger.error(f"Failed to save audio file: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save audio file: {str(e)}"
            )

        logger.info(f"TTS generation completed successfully.")

        # Return WAV response
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(wav_bytes))
            }
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error during TTS generation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during TTS generation: {str(e)}"
        )


@app.get("/",
         summary="Root endpoint",
         description="Provides information about the Soprano TTS API")
async def root():
    """
    Root endpoint to provide API information.
    """
    return {
        "message": "Soprano TTS API",
        "version": "1.0.0",
        "description": "Ultra-realistic Text-to-Speech API based on Soprano model",
        "endpoints": {
            "tts": "/v1/audio/speech",
            "health": "/health"
        }
    }


@app.get("/health",
         summary="Health check endpoint",
         description="Check if the server and TTS model are running properly")
async def health_check():
    """
    Health check endpoint to verify the server and model are operational.
    """
    try:
        tts_manager = TTSManager()
        tts = tts_manager.get_model()
        return {"status": "healthy", "device": tts.device}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable")


if __name__ == "__main__":
    import uvicorn
    import torch

    print("Starting Soprano TTS API Server...")
    print(f"Available device: {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}")

    # Start the server
    print("Server starting on http://localhost:8000")
    uvicorn.run(
        "soprano.server.api:app",
        host="localhost",
        port=8000,
        reload=False
    )
