#!/usr/bin/env python3
"""
Test Orpheus TTS with the new vLLM setup
"""

import os
import sys
import wave
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_orpheus():
    print("=" * 60)
    print("Testing Orpheus TTS with vLLM")
    print("=" * 60)
    
    try:
        # Import Orpheus from the correct module
        from orpheus_tts import OrpheusModel
        print("✅ OrpheusModel imported successfully from orpheus_tts")
        
        # Initialize model
        print("\nInitializing Orpheus model...")
        print("This will download the model on first run (may take a few minutes)")
        
        # Available voices: zoe, zac, jess, leo, mia, julia, leah, tara
        model = OrpheusModel(
            model_name="canopylabs/orpheus-3b-0.1-ft"
        )
        
        print("✅ Model loaded successfully")
        
        # Test text
        test_text = "Hello! This is a test of Orpheus text to speech system running with vLLM on an RTX 5090."
        
        print(f"\nGenerating speech for: '{test_text}'")
        print(f"Using voice: leo")
        print("-" * 40)
        
        # Generate audio with leo voice (returns a generator)
        audio_generator = model.generate_speech(prompt=test_text, voice="leo")
        
        # Collect all audio chunks (they come as bytes)
        audio_bytes = b''
        print("Generating audio chunks...")
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
            print(f"  Chunk {chunk_count}: {len(audio_bytes)} total bytes", end="\r")
        
        print(f"\nGenerated {chunk_count} chunks, {len(audio_bytes)} total bytes")
        
        if not audio_bytes:
            print("No audio generated!")
            return False
        
        # Save audio
        output_file = "test_output.wav"
        save_audio_bytes(audio_bytes, output_file)
        
        # Calculate statistics
        num_samples = len(audio_bytes) / 2  # 16-bit = 2 bytes per sample
        duration = num_samples / 24000
        
        print(f"✅ Audio generated successfully!")
        print(f"✅ Saved to: {output_file}")
        print(f"   - Sample rate: 24000 Hz")
        print(f"   - Total samples: {int(num_samples)}")
        print(f"   - Duration: {duration:.2f} seconds")
        print(f"   - Size: {os.path.getsize(output_file) / 1024:.2f} KB")
        
        return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_audio_bytes(audio_bytes, filename, sample_rate=24000):
    """Save audio bytes directly to WAV file"""
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)

if __name__ == "__main__":
    success = test_orpheus()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed! Orpheus TTS is working correctly.")
        print("   You can play the generated audio with:")
        print("   $ aplay test_output.wav  # Linux")
        print("   $ afplay test_output.wav # macOS")
    else:
        print("❌ Tests failed. Please check the error messages above.")
    print("=" * 60)
    
    sys.exit(0 if success else 1)