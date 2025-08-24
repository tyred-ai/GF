#!/usr/bin/env python3
"""
Test simple vLLM generation with proper tokens
"""

import os
import sys
import time

# Force settings
os.environ['VLLM_ENABLE_CHUNKED_PREFILL'] = '0'
os.environ['VLLM_MAX_MODEL_LEN'] = '16384'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orpheus_tts import OrpheusModel

def test_generation():
    """Test with simple text"""
    
    print("Initializing Orpheus model...")
    model = OrpheusModel(
        model_name="canopylabs/orpheus-3b-0.1-ft"
    )
    
    # Simple test text
    text = "Aunt Amy was out on the front porch, rocking back and forth in the chair."
    voice = "tara"
    
    print(f"\n📝 Testing generation:")
    print(f"   Text: {text}")
    print(f"   Voice: {voice}")
    
    # Generate with correct stop tokens
    chunks = []
    token_count = 0
    
    for chunk in model.generate_speech(
        prompt=text,
        voice=voice,
        temperature=0.4,
        top_p=0.9,
        repetition_penalty=1.1,
        max_tokens=5000,
        stop_token_ids=[128258],  # Correct EOS
        request_id="test-001"
    ):
        if chunk:
            chunks.append(chunk)
            token_count += 1
            if token_count % 100 == 0:
                print(f"   Generated {token_count} chunks...")
    
    print(f"\n✅ Generated {len(chunks)} audio chunks")
    
    if chunks:
        # Save audio
        audio_data = b"".join(chunks)
        
        import wave
        with wave.open("test_vllm_output.wav", "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(audio_data)
        
        print(f"✅ Saved to test_vllm_output.wav ({len(audio_data)} bytes)")
        
        # Analyze the audio to check if first words are present
        import numpy as np
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Check if audio starts with silence or speech
        first_second = audio_array[:24000] if len(audio_array) > 24000 else audio_array
        max_amplitude = np.max(np.abs(first_second))
        
        print(f"\n📊 First second analysis:")
        print(f"   Max amplitude: {max_amplitude}")
        print(f"   Has speech: {max_amplitude > 1000}")
        
        # Find first significant audio
        threshold = max_amplitude * 0.1
        for i in range(0, min(len(audio_array), 24000), 2400):  # Check every 100ms
            segment = audio_array[i:i+2400]
            if np.max(np.abs(segment)) > threshold:
                print(f"   First speech at: {i/24000:.2f}s")
                break

if __name__ == "__main__":
    test_generation()