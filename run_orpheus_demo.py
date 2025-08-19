#!/usr/bin/env python3
"""
Orpheus TTS Demo with vLLM backend
Demonstrates text-to-speech generation using the Orpheus model
"""

import os
import sys
import wave
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    print("=" * 70)
    print("Orpheus TTS Demo - Running on RTX 5090 with vLLM")
    print("=" * 70)
    
    # Import Orpheus
    from orpheus_tts import OrpheusModel
    
    # Available voices
    voices = ["zoe", "zac", "jess", "leo", "mia", "julia", "leah", "tara"]
    
    print(f"\nAvailable voices: {', '.join(voices)}")
    
    # Initialize model (will use cached model after first run)
    print("\nInitializing Orpheus model...")
    model = OrpheusModel(model_name="canopylabs/orpheus-3b-0.1-ft")
    print("✅ Model ready!")
    
    # Demo texts with different voices
    demos = [
        ("leo", "Hello! Welcome to Orpheus text to speech. This system can generate natural sounding speech."),
        ("zoe", "I can speak in different voices. Each voice has its own unique characteristics."),
        ("tara", "You can also add emotions like <laugh> or <sigh> to make speech more expressive."),
    ]
    
    for i, (voice, text) in enumerate(demos, 1):
        print(f"\n{'='*50}")
        print(f"Demo {i}: Voice '{voice}'")
        print(f"Text: {text}")
        print("-" * 50)
        
        try:
            # Generate speech with unique request ID
            print("Generating speech...")
            request_id = f"demo-{i}-{voice}"
            audio_generator = model.generate_speech(prompt=text, voice=voice, request_id=request_id)
            
            # Collect audio chunks (they come as bytes)
            audio_bytes = b''
            chunk_count = 0
            for chunk in audio_generator:
                chunk_count += 1
                # The chunks are already audio bytes from the decoder
                if isinstance(chunk, bytes):
                    audio_bytes += chunk
                elif isinstance(chunk, np.ndarray):
                    # If it's a numpy array, convert to bytes
                    if chunk.dtype != np.int16:
                        chunk = chunk.astype(np.int16)
                    audio_bytes += chunk.tobytes()
                else:
                    # Try to convert to bytes
                    audio_bytes += bytes(chunk)
            
            print(f"  Collected {chunk_count} chunks, {len(audio_bytes)} bytes")
            
            # Save to file
            if audio_bytes:
                output_file = f"demo_{i}_{voice}.wav"
                save_audio_bytes(audio_bytes, output_file)
                
                # Calculate duration from byte length
                # 16-bit mono at 24kHz = 2 bytes per sample
                num_samples = len(audio_bytes) / 2
                duration = num_samples / 24000
                size_kb = os.path.getsize(output_file) / 1024
                
                print(f"✅ Generated {duration:.2f}s of audio")
                print(f"   Saved to: {output_file} ({size_kb:.1f} KB)")
            else:
                print("❌ No audio generated")
                
        except Exception as e:
            print(f"❌ Error generating audio for voice '{voice}': {e}")
            continue
    
    print(f"\n{'='*70}")
    print("Demo complete! Generated audio files:")
    for i, (voice, _) in enumerate(demos, 1):
        print(f"  - demo_{i}_{voice}.wav")
    print("=" * 70)

def save_audio_bytes(audio_bytes, filename, sample_rate=24000):
    """Save audio bytes directly to WAV file"""
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)