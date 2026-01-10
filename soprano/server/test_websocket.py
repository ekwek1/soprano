import asyncio
import websockets
import json
import pyaudio
import sys

class SopranoWSClient:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        # Default settings (will be updated by metadata from server)
        self.rate = 32000 
        self.channels = 1

    def open_stream(self, rate, channels):
        """Opens the audio device for live playback."""
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            output=True
        )

    async def start_test(self, text):
        uri = "ws://localhost:8001/ws/tts"
        
        try:
            async with websockets.connect(uri) as ws:
                # 1. Send the synthesis request
                payload = {
                    "type": "synthesize",
                    "text": text,
                    "stream": True,
                    "min_text_length": 30
                }
                await ws.send(json.dumps(payload))
                print(f">>> Sent text to server. Waiting for audio...")

                # 2. Listen for chunks
                while True:
                    message = await ws.recv()

                    # Handle Audio Bytes
                    if isinstance(message, bytes):
                        if self.stream:
                            self.stream.write(message)
                            print(".", end="", flush=True)

                    # Handle JSON Messages
                    else:
                        data = json.loads(message)
                        if data["type"] == "metadata":
                            print(f"\n[Metadata] Rate: {data['sample_rate']}Hz")
                            self.open_stream(data['sample_rate'], data['channels'])
                        
                        elif data["type"] == "end":
                            print("\n[Finished] Server signaled end of stream.")
                            break
                        
                        elif data["type"] == "error":
                            print(f"\n[Error] {data['message']}")
                            break

        except Exception as e:
            print(f"Connection Error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

if __name__ == "__main__":
    input_text = "Testing the live websocket stream. I should hear this almost immediately."
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
    
    client = SopranoWSClient()
    asyncio.run(client.start_test(input_text))