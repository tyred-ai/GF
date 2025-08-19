#!/usr/bin/env python3
"""
Test if SNAC decoder can run on CPU
"""

import os
import sys

# Force SNAC to use CPU before importing
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide GPU from SNAC
os.environ['SNAC_DEVICE'] = 'cpu'

print("Environment set to use CPU for SNAC decoder")
print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES'))
print("SNAC_DEVICE:", os.environ.get('SNAC_DEVICE'))

try:
    # Now import Orpheus - SNAC should load on CPU
    print("\nImporting orpheus_tts...")
    from orpheus_tts import OrpheusModel
    print("✅ Import successful!")
    
    # Try to initialize model
    print("\nInitializing model...")
    model = OrpheusModel(model_name="canopylabs/orpheus-3b-0.1-ft")
    print("✅ Model loaded!")
    
    # Test generation
    print("\nTesting generation...")
    generator = model.generate_speech(prompt="hello", voice="leo")
    
    audio_bytes = b''
    for i, chunk in enumerate(generator):
        if isinstance(chunk, bytes):
            audio_bytes += chunk
        print(f"Chunk {i+1}: {len(audio_bytes)} bytes", end="\r")
        if i >= 5:  # Just get a few chunks for testing
            break
    
    print(f"\n✅ Generated {len(audio_bytes)} bytes successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()