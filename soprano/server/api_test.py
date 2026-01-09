import asyncio
import aiohttp
import time
from pathlib import Path

async def send_tts_request(text):
    """
    Send a TTS request to the API server with custom text
    """
    base_url = "http://localhost:8000"

    payload = {
        "input": text,
        "temperature": 0.3,
        "top_p": 0.95,
        "repetition_penalty": 1.2
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{base_url}/v1/audio/speech", json=payload) as response:
                status = response.status
                if status == 200:
                    audio_content = await response.read()

                    # The API already saves the file, so we just confirm success
                    print(f"Audio generated successfully. Check the audio_output folder for the file.")
                    return True
                else:
                    error_text = await response.text()
                    print(f"Request failed with status {status}")
                    print(f"Error: {error_text}")
                    return False
    except Exception as e:
        print(f"Request failed with error: {e}")
        return False

async def main():
    print("Soprano TTS API Request Sender")
    print("Make sure the API server is running on http://localhost:8000 before executing this.")
    print()

    while True:
        text = input("Enter text to convert to speech (or 'quit' to exit): ")
        if text.lower() == 'quit':
            break

        if not text.strip():
            print("Text cannot be empty. Please enter some text.")
            continue

        print(f"Sending request with text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        success = await send_tts_request(text)

        if success:
            print("Request completed successfully!")
        else:
            print("Request failed!")

        print()

if __name__ == "__main__":
    # Check if required packages are available
    try:
        import aiohttp
    except ImportError:
        print("Error: aiohttp is not installed. Please install it with: pip install aiohttp")
        exit(1)

    asyncio.run(main())