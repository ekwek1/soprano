import asyncio
import aiohttp
import json
import sys
import os
from pathlib import Path

async def test_api(text="Hello, this is a test of the Soprano TTS API."):
    """
    Test the Soprano TTS API endpoint
    """
    base_url = "http://localhost:8000"
    endpoint = "/v1/audio/speech"

    payload = {
        "input": text,
        "temperature": 0.3,
        "top_p": 1.0,
        "repetition_penalty": 1.2,
        "min_text_length": 30
    }

    print(f"Testing API at {base_url}{endpoint}")
    print(f"Sending text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{base_url}{endpoint}", json=payload) as response:
                status = response.status
                print(f"Response status: {status}")
                
                if status == 200:
                    # Read the audio content
                    audio_content = await response.read()
                    print(f"Received audio data: {len(audio_content)} bytes")
                    
                    # Save the audio to a file
                    output_dir = "audio_output"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Generate unique filename
                    file_counter = 1
                    while True:
                        filename = f"api_test_output_{file_counter}.wav"
                        filepath = os.path.join(output_dir, filename)
                        if not os.path.exists(filepath):
                            break
                        file_counter += 1
                    
                    with open(filepath, 'wb') as f:
                        f.write(audio_content)
                    
                    print(f"Audio saved to: {filepath}")
                    return True
                else:
                    error_text = await response.text()
                    print(f"Request failed with status {status}")
                    print(f"Error: {error_text}")
                    return False
    except aiohttp.ClientConnectorError:
        print("Error: Could not connect to API server. Make sure it's running on http://localhost:8000")
        return False
    except Exception as e:
        print(f"Request failed with error: {e}")
        return False


async def test_health():
    """
    Test the health check endpoint
    """
    base_url = "http://localhost:8000"
    endpoint = "/health"

    print(f"Testing health endpoint at {base_url}{endpoint}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}{endpoint}") as response:
                status = response.status
                print(f"Health check status: {status}")
                
                if status == 200:
                    health_data = await response.json()
                    print(f"Health check result: {health_data}")
                    return True
                else:
                    error_text = await response.text()
                    print(f"Health check failed with status {status}")
                    print(f"Error: {error_text}")
                    return False
    except Exception as e:
        print(f"Health check failed with error: {e}")
        return False


async def main():
    print("Soprano TTS API Test Client")
    print("Make sure the API server is running on http://localhost:8000 before executing this.")
    print()

    # Test health endpoint first
    print("Testing health endpoint...")
    health_ok = await test_health()
    if not health_ok:
        print("Health check failed. Exiting.")
        return

    print()

    # Get text from command line arguments or use default
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = "Hello, this is a test of the Soprano TTS API. The system is working properly."

    if not text.strip():
        print("Text cannot be empty. Please enter some text.")
        return

    print("Testing TTS API...")
    success = await test_api(text)

    if success:
        print("API test completed successfully!")
    else:
        print("API test failed!")


if __name__ == "__main__":
    # Check if required packages are available
    try:
        import aiohttp
    except ImportError:
        print("Error: aiohttp is not installed. Please install it with: pip install aiohttp")
        exit(1)

    asyncio.run(main())