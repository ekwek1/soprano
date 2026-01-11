import asyncio
import json
import logging
from typing import AsyncGenerator
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import torch
from contextlib import asynccontextmanager
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
from asyncio import Queue, QueueEmpty

# Handle import when running from within the server directory
try:
    from soprano.tts import SopranoTTS
except ImportError:
    import sys
    import os
    # Add the parent directory to the Python path to resolve import issues
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from soprano.tts import SopranoTTS


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TTSWebSocketManager:
    """
    Manager for WebSocket TTS streaming functionality.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.tts: SopranoTTS = None
            self._lock = asyncio.Lock()  # Move lock to instance level
            # Track active streaming tasks for graceful shutdown
            self.active_tasks = set()
            # Prioritize CUDA, fallback to CPU only if CUDA is not available
            if torch.cuda.is_available():
                self.device = 'cuda'
                logger.info("CUDA is available, using GPU for TTS processing")
            else:
                self.device = 'cpu'
                logger.info("CUDA is not available, falling back to CPU for TTS processing")
            logger.info(f"Initializing TTS on device: {self.device}")

    async def initialize_model(self):
        """
        Initialize the TTS model asynchronously to avoid blocking the event loop.
        """
        async with self._lock:
            if self.tts is None:
                logger.info("Loading Soprano TTS model for WebSocket streaming...")
                try:
                    # Run model initialization in a thread pool to avoid blocking
                    loop = asyncio.get_running_loop()  # Use get_running_loop instead of get_event_loop

                    def load_model():
                        # Import here in case it's needed in the executor
                        try:
                            from soprano.tts import SopranoTTS
                        except ImportError:
                            import sys
                            import os
                            # Add the parent directory to the Python path to resolve import issues
                            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                            from soprano.tts import SopranoTTS
                        return SopranoTTS(
                            cache_size_mb=100,
                            device=self.device
                        )

                    self.tts = await loop.run_in_executor(None, load_model)
                    logger.info("Soprano TTS model loaded successfully for WebSocket streaming")
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

    async def stream_audio_with_backpressure(self, websocket: WebSocket, text: str, min_text_length: int = 30):
        """
        Stream audio in real-time from the TTS model with backpressure handling.
        Uses a queue to decouple TTS generation from WebSocket sending.
        """
        # Check if streaming is supported (only available on GPU with LMDeploy backend)
        try:
            tts = self.get_model()
            # Check if we're using transformers backend which doesn't support streaming
            from soprano.backends.transformers import TransformersModel
            if isinstance(tts.pipeline, TransformersModel):
                # Send error message to client
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Real-time streaming is not supported on CPU. Only generate speech is supported for CPU."
                }))
                logger.warning("Streaming requested but not supported on CPU")
                return
        except Exception as e:
            logger.warning(f"Could not determine backend type: {e}")
            # Continue with original logic if there's an issue checking the backend

        # Create a queue to decouple generation from sending
        audio_queue = Queue(maxsize=10)  # Limit queue size to prevent memory buildup

        async def producer():
            """Generate audio chunks and put them in the queue."""
            try:
                # Get the TTS model
                tts = self.get_model()

                # Use the streaming inference method from the TTS model
                logger.info(f"Starting streaming TTS for text: '{text[:50]}{'...' if len(text) > 50 else ''}'")

                # Use the infer_stream method which is designed for streaming
                for audio_chunk in tts.infer_stream(
                    text=text,
                    chunk_size=1,
                    top_p=1.0,  # Using the default value we set
                    temperature=0.3,  # Using the default value we set
                    repetition_penalty=1.2,  # Using the default value we set
                    min_text_length=min_text_length  # Use the passed value instead of hardcoded 1000
                ):
                    # Convert tensor to numpy array
                    audio_np = audio_chunk.cpu().numpy()

                    # Ensure values are in the range [-1, 1] and convert to int16
                    audio_np = np.clip(audio_np, -1.0, 1.0)
                    audio_np = (audio_np * 32767).astype(np.int16)

                    # Convert to bytes
                    audio_bytes = audio_np.tobytes()

                    # Put audio chunk in queue, with timeout to handle slow consumers
                    try:
                        await asyncio.wait_for(audio_queue.put(audio_bytes), timeout=5.0)
                    except asyncio.TimeoutError:
                        logger.warning("Audio queue timeout - client may be slow")
                        break

                # Put None to signal end of stream
                await audio_queue.put(None)
            except NotImplementedError as e:
                logger.error(f"Streaming not supported: {str(e)}")
                # Send error message to client
                try:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Real-time streaming is not supported on CPU. Only generate speech is supported for CPU."
                    }))
                except:
                    pass
                try:
                    await audio_queue.put(None)  # Signal error to consumer
                except:
                    pass
            except Exception as e:
                logger.error(f"Error in audio producer: {str(e)}", exc_info=True)
                try:
                    await audio_queue.put(None)  # Signal error to consumer
                except:
                    pass

        async def consumer():
            """Take audio chunks from the queue and send them via WebSocket."""
            try:
                while True:
                    # Get audio chunk from queue with timeout
                    try:
                        audio_bytes = await asyncio.wait_for(audio_queue.get(), timeout=10.0)
                    except asyncio.TimeoutError:
                        logger.warning("Timeout waiting for audio data")
                        break

                    # If None, it means the producer is done
                    if audio_bytes is None:
                        break

                    # Send the audio chunk as binary data
                    await websocket.send_bytes(audio_bytes)
            except Exception as e:
                logger.error(f"Error in audio consumer: {str(e)}", exc_info=True)
                # Don't re-raise here as we want to ensure cleanup happens

        # Create tasks for producer and consumer
        producer_task = asyncio.create_task(producer())
        consumer_task = asyncio.create_task(consumer())

        # Add tasks to active tasks set for graceful shutdown
        self.active_tasks.add(producer_task)
        self.active_tasks.add(consumer_task)

        try:
            # Wait for both tasks to complete
            await asyncio.gather(producer_task, consumer_task, return_exceptions=True)
        finally:
            # Remove tasks from active tasks set
            self.active_tasks.discard(producer_task)
            self.active_tasks.discard(consumer_task)

            # Cancel tasks if they're still running
            if not producer_task.done():
                producer_task.cancel()
            if not consumer_task.done():
                consumer_task.cancel()

    async def stream_audio(self, websocket: WebSocket, text: str, min_text_length: int = 30):
        """
        Stream audio in real-time from the TTS model.
        """
        try:
            # Send metadata at the start
            metadata = {
                "type": "metadata",
                "sample_rate": 32000,
                "channels": 1,
                "format": "int16",
                "model_info": "soprano"
            }
            await websocket.send_text(json.dumps(metadata))

            # Use the backpressure-aware streaming method
            # This will return early if streaming is not supported
            await self.stream_audio_with_backpressure(websocket, text, min_text_length)

            # Only send end signal if streaming was not terminated early due to unsupported backend
            # The function will return before reaching here if streaming is not supported

        except Exception as e:
            logger.error(f"Error during streaming: {str(e)}", exc_info=True)
            try:
                await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
            except:
                pass  # If we can't send the error, just continue


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan event handler for startup and shutdown events.
    """
    logger.info("Starting up Soprano TTS WebSocket server...")
    tts_manager = TTSWebSocketManager()

    try:
        await tts_manager.initialize_model()
        logger.info("Soprano TTS WebSocket server started successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to start Soprano TTS WebSocket server: {e}", exc_info=True)
        raise
    finally:
        logger.info("Shutting down Soprano TTS WebSocket server...")
        # Cancel any active streaming tasks for graceful shutdown
        if hasattr(tts_manager, 'active_tasks'):
            for task in tts_manager.active_tasks.copy():  # Use copy to avoid modification during iteration
                if not task.done():
                    logger.info("Cancelling active streaming task...")
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass  # Expected when cancelling tasks
        logger.info("Soprano TTS WebSocket server shut down completed")


# Create FastAPI app with WebSocket support
app = FastAPI(
    title="Soprano TTS WebSocket API",
    description="Real-time streaming Text-to-Speech via WebSocket",
    version="1.0.0",
    lifespan=lifespan
)


@app.websocket("/ws/tts")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time TTS streaming.
    Supports multiple synthesize requests over the same connection.
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    try:
        tts_manager = TTSWebSocketManager()
        
        # Keep the connection open to handle multiple requests
        while True:
            # Wait for a message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "synthesize":
                text = message.get("text", "")
                stream = message.get("stream", True)
                # Allow client to specify min_text_length, default to 30
                min_text_length = message.get("min_text_length", 30)
                
                if not text or not text.strip():
                    await websocket.send_text(json.dumps({
                        "type": "error", 
                        "message": "Text cannot be empty"
                    }))
                    continue  # Continue to listen for more messages
                
                if stream:
                    # Start streaming audio
                    await tts_manager.stream_audio(websocket, text, min_text_length)
                else:
                    # For non-streaming, we could implement a regular synthesis
                    # but the requirement is for streaming, so we'll focus on that
                    await websocket.send_text(json.dumps({
                        "type": "error", 
                        "message": "Only streaming is supported in this endpoint"
                    }))
            elif message.get("type") == "ping":
                # Simple ping/pong for connection health
                await websocket.send_text(json.dumps({"type": "pong"}))
            else:
                await websocket.send_text(json.dumps({
                    "type": "error", 
                    "message": "Invalid message type. Use 'synthesize' or 'ping'."
                }))
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except json.JSONDecodeError:
        logger.error("Invalid JSON received")
        try:
            await websocket.send_text(json.dumps({
                "type": "error", 
                "message": "Invalid JSON format"
            }))
        except:
            pass
    except ConnectionClosedOK:
        logger.info("WebSocket connection closed normally")
    except ConnectionClosedError:
        logger.info("WebSocket connection closed with error")
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket: {str(e)}", exc_info=True)
        try:
            await websocket.send_text(json.dumps({
                "type": "error", 
                "message": f"Server error: {str(e)}"
            }))
        except:
            pass
    finally:
        try:
            if hasattr(websocket, 'client_state') and websocket.client_state.name != 'DISCONNECTED':
                await websocket.close()
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    import torch

    print("Starting Soprano TTS WebSocket Server...")
    print(f"Available device: {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}")

    # Start the server
    print("WebSocket server starting on ws://localhost:8001/ws/tts")
    uvicorn.run(
        app,
        host="localhost",
        port=8001,  # Using port 8001 to avoid conflict with the regular API
        reload=False
    )