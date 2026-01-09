"""
API Server for Soprano TTS
This file contains the API endpoints for the Soprano TTS service.
"""

from flask import Flask, request, jsonify
import os
import sys

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route('/tts', methods=['POST'])
def generate_speech():
    """Generate speech from text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        # Placeholder for TTS generation
        # Actual implementation would depend on the backend
        
        return jsonify({
            "message": "Speech generated successfully",
            "text": text
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)