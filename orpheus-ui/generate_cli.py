#!/usr/bin/env python3
"""
Command-line audio generation script for the UI
This runs in a separate process to avoid GPU memory conflicts
"""

import sys
import json
import wave
import argparse

# Add parent directory to path
sys.path.append('..')

def generate_audio(text, voice, output_file):
    """Generate audio and save to file"""
    from orpheus_tts import OrpheusModel
    
    # Initialize model
    model = OrpheusModel(model_name="canopylabs/orpheus-3b-0.1-ft")
    
    # Generate speech
    generator = model.generate_speech(prompt=text, voice=voice)
    
    # Collect audio bytes
    audio_bytes = b''
    chunk_count = 0
    
    for chunk in generator:
        chunk_count += 1
        if isinstance(chunk, bytes):
            audio_bytes += chunk
        # Print progress
        if chunk_count % 10 == 0:
            print(f"Progress: {chunk_count} chunks, {len(audio_bytes)} bytes", file=sys.stderr)
    
    # Save to file
    with wave.open(output_file, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)   # 16-bit
        wav_file.setframerate(24000)  # 24kHz
        wav_file.writeframes(audio_bytes)
    
    # Return info as JSON
    result = {
        "success": True,
        "chunks": chunk_count,
        "bytes": len(audio_bytes),
        "duration": (len(audio_bytes) / 2) / 24000
    }
    
    print(json.dumps(result))
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="Text to convert to speech")
    parser.add_argument("--voice", default="leo", help="Voice to use")
    parser.add_argument("--output", required=True, help="Output WAV file path")
    
    args = parser.parse_args()
    
    try:
        return generate_audio(args.text, args.voice, args.output)
    except Exception as e:
        result = {
            "success": False,
            "error": str(e)
        }
        print(json.dumps(result))
        return 1

if __name__ == "__main__":
    sys.exit(main())