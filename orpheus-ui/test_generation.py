#!/usr/bin/env python3
"""
Test direct generation to debug the UI issue
"""

import sys
sys.path.append('..')

from orpheus_tts import OrpheusModel
import time

print("Initializing model...")
model = OrpheusModel(model_name="canopylabs/orpheus-3b-0.1-ft")
print("Model loaded!")

text = "hello"
voice = "leo"

print(f"\nGenerating speech for: '{text}' with voice: {voice}")
print("-" * 40)

# Test generation
start = time.time()
generator = model.generate_speech(prompt=text, voice=voice)

audio_bytes = b''
chunk_count = 0

try:
    print("Starting generation...")
    for i, chunk in enumerate(generator):
        chunk_count += 1
        if isinstance(chunk, bytes):
            audio_bytes += chunk
        print(f"Chunk {chunk_count}: {len(audio_bytes)} bytes", end="\r")
        
        # Break after 10 chunks for testing
        if chunk_count >= 10:
            print(f"\nReceived {chunk_count} chunks, stopping for test")
            break
            
    elapsed = time.time() - start
    print(f"\nGeneration took {elapsed:.2f} seconds")
    print(f"Total audio bytes: {len(audio_bytes)}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()