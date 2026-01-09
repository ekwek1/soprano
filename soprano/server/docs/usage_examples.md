# Usage Examples

## Basic Usage

### Command Line Interface
The API can be tested using command line tools:

#### Using cURL
```bash
# Basic text-to-speech conversion
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world, this is a test of the Soprano TTS system."
  }' \
  --output output.wav

# With custom parameters
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "This is a test with custom parameters.",
    "temperature": 0.5,
    "top_p": 0.9,
    "repetition_penalty": 1.1
  }' \
  --output custom_output.wav
```

#### Using Python requests
```python
import requests

# Basic request
url = "http://localhost:8000/v1/audio/speech"
payload = {
    "input": "Hello world, this is a test of the Soprano TTS system."
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    with open("output.wav", "wb") as f:
        f.write(response.content)
    print("Audio saved successfully")
else:
    print(f"Request failed with status {response.status_code}")
    print(response.text)
```

## Advanced Usage

### Custom Parameters
The API supports various parameters to customize the output:

```python
import requests

url = "http://localhost:8000/v1/audio/speech"
payload = {
    "input": "This is a test with custom parameters.",
    "temperature": 0.3,          # Controls randomness (0.0-1.0)
    "top_p": 0.95,              # Controls diversity (0.0-1.0)
    "repetition_penalty": 1.2    # Controls repetition (0.1-2.0)
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    with open("custom_output.wav", "wb") as f:
        f.write(response.content)
    print("Custom audio saved successfully")
```

### Batch Processing
To process multiple texts:

```python
import requests
import time

def process_texts(texts):
    url = "http://localhost:8000/v1/audio/speech"
    
    for i, text in enumerate(texts):
        payload = {
            "input": text,
            "temperature": 0.3,
            "top_p": 0.95,
            "repetition_penalty": 1.2
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            filename = f"batch_output_{i+1}.wav"
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"Saved {filename}")
        else:
            print(f"Failed to process text {i+1}: {response.status_code}")
        
        # Optional: Add delay between requests
        time.sleep(1)

# Example usage
texts = [
    "This is the first text.",
    "This is the second text.",
    "This is the third text."
]

process_texts(texts)
```

## Integration Examples

### Web Application Integration
Example of integrating with a web application:

```python
from flask import Flask, request, send_file
import requests
import tempfile
import os

app = Flask(__name__)

@app.route('/tts', methods=['POST'])
def text_to_speech():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return {'error': 'Text is required'}, 400
    
    # Call the Soprano TTS API
    tts_url = "http://localhost:8000/v1/audio/speech"
    payload = {
        "input": text,
        "temperature": 0.3,
        "top_p": 0.95,
        "repetition_penalty": 1.2
    }
    
    response = requests.post(tts_url, json=payload)
    
    if response.status_code == 200:
        # Return the audio file
        return send_file(
            io.BytesIO(response.content),
            mimetype='audio/wav',
            as_attachment=True,
            download_name='output.wav'
        )
    else:
        return {'error': 'TTS generation failed'}, response.status_code

if __name__ == '__main__':
    app.run(debug=True)
```

### Using with the Test Client
The provided test.py file allows for interactive usage:

1. Start the API server
2. Run `python test.py`
3. Enter text when prompted
4. Check the audio_output folder for generated files

## File Management

### Output Files
- Files are saved in the `audio_output` directory
- Files use sequential naming: `output_1.wav`, `output_2.wav`, etc.
- Old files are automatically cleaned up after 24 hours

### File Access
Generated files can be accessed directly from the `audio_output` directory or through the API response.

## Best Practices

### Input Validation
Always validate input text:
- Ensure text is not empty
- Keep text under 1000 characters
- Avoid special characters that might cause path traversal issues

### Error Handling
Implement proper error handling in your client applications:

```python
import requests

def safe_tts_request(text):
    url = "http://localhost:8000/v1/audio/speech"
    payload = {"input": text}
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            return response.content
        elif response.status_code == 422:
            print(f"Validation error: {response.text}")
            return None
        elif response.status_code == 503:
            print("Service temporarily unavailable")
            return None
        else:
            print(f"Request failed with status {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None

# Example usage
audio_data = safe_tts_request("Hello world")
if audio_data:
    with open("output.wav", "wb") as f:
        f.write(audio_data)
    print("Audio saved successfully")
```

### Performance Considerations
- The first request after startup may take longer due to model loading
- Subsequent requests will be faster
- Consider the computational requirements for longer texts
- Monitor resource usage during heavy usage