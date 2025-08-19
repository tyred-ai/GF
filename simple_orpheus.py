#!/usr/bin/env python3
"""
Simple Orpheus TTS script - Generate speech from text
"""

import sys
import wave
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_speech(text, voice="leo", output_file="output.wav"):
    """Generate speech from text using Orpheus TTS"""
    
    print("Initializing Orpheus TTS...")
    from orpheus_tts import OrpheusModel
    
    # Initialize model
    model = OrpheusModel(model_name="canopylabs/orpheus-3b-0.1-ft")
    print("✅ Model loaded")
    
    # Generate speech
    print(f"Generating speech with voice '{voice}'...")
    print(f"Text: {text}")
    
    audio_generator = model.generate_speech(prompt=text, voice=voice)
    
    # Collect audio chunks
    audio_bytes = b''
    chunk_count = 0
    for chunk in audio_generator:
        chunk_count += 1
        if isinstance(chunk, bytes):
            audio_bytes += chunk
        else:
            audio_bytes += bytes(chunk)
        
        # Show progress
        if chunk_count % 10 == 0:
            print(f"  Processing chunk {chunk_count}...", end="\r")
    
    print(f"\nCollected {chunk_count} chunks ({len(audio_bytes)} bytes)")
    
    # Save to file
    if audio_bytes:
        with wave.open(output_file, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(24000)  # 24kHz
            wav_file.writeframes(audio_bytes)
        
        # Calculate duration
        duration = (len(audio_bytes) / 2) / 24000
        size_kb = len(audio_bytes) / 1024
        
        print(f"\n✅ Success!")
        print(f"   Output: {output_file}")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Size: {size_kb:.1f} KB")
        print(f"\nPlay with: aplay {output_file}")
        return True
    else:
        print("❌ No audio generated")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate speech from text using Orpheus TTS")
    parser.add_argument("text", help="Text to convert to speech")
    parser.add_argument("-v", "--voice", default="leo", 
                       choices=["zoe", "zac", "jess", "leo", "mia", "julia", "leah", "tara"],
                       help="Voice to use (default: leo)")
    parser.add_argument("-o", "--output", default="output.wav",
                       help="Output file name (default: output.wav)")
    
    args = parser.parse_args()
    
    try:
        success = generate_speech(args.text, args.voice, args.output)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()