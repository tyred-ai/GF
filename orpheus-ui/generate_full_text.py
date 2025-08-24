#!/usr/bin/env python3
"""
Alternative: Force continuation by appending "Continue speaking:" prompts
"""

import os
import sys
import time

# Set environment variables before import
os.environ['VLLM_ENABLE_CHUNKED_PREFILL'] = '0'
os.environ['VLLM_MAX_MODEL_LEN'] = '16384'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orpheus_tts import OrpheusModel

def generate_with_continuation(text, voice="tara", max_attempts=3):
    """Generate audio with continuation prompts if needed"""
    
    print("Initializing Orpheus model...")
    model = OrpheusModel(
        model_name="canopylabs/orpheus-3b-0.1-ft"
    )
    
    # Split into sentences
    sentences = text.replace('\n\n', ' ').replace('\n', ' ').split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    
    print(f"Total sentences: {len(sentences)}")
    
    all_chunks = []
    current_text = ""
    sentences_processed = 0
    
    # Process in batches
    batch_size = 3  # Process 3 sentences at a time
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        batch_text = ' '.join(batch)
        
        # Add continuation prompt for middle batches
        if i > 0:
            batch_text = "Continue: " + batch_text
        
        if i + batch_size < len(sentences):
            batch_text += " [More to follow]"
        
        print(f"\n📝 Processing batch {i//batch_size + 1}:")
        print(f"   Sentences {i+1}-{min(i+batch_size, len(sentences))}")
        print(f"   Text: {batch_text[:100]}...")
        
        try:
            # Generate audio for this batch
            chunks = []
            for chunk in model.generate_speech(
                prompt=batch_text,
                voice=voice,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.05,
                max_tokens=5000
            ):
                if chunk:
                    chunks.append(chunk)
            
            if chunks:
                all_chunks.extend(chunks)
                sentences_processed += len(batch)
                print(f"   ✅ Generated {len(chunks)} chunks")
            else:
                print(f"   ⚠️ No chunks generated")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n✅ Processed {sentences_processed}/{len(sentences)} sentences")
    print(f"✅ Total chunks: {len(all_chunks)}")
    
    # Combine all chunks
    if all_chunks:
        audio_data = b"".join(all_chunks)
        
        # Save to file
        import wave
        with wave.open("full_text_output.wav", "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(audio_data)
        
        print(f"✅ Saved to full_text_output.wav ({len(audio_data)} bytes)")
        return audio_data
    
    return None

# Test with the full text
if __name__ == "__main__":
    full_text = """Aunt Amy was out on the front porch, rocking back and forth in the high­backed chair and fanning herself, when Bill Soames rode his bicycle up the road and stopped in front of the house. Perspiring under the afternoon 'sun,' Bill lifted the box of groceries out of the big basket over the front wheel of the bike, and came up the front walk.

Little Anthony was sitting on the lawn, playing with a rat. He had caught the rat down in the base­ment – he had made it think that it smelled cheese, the most rich-smelling and crumbly-delicious cheese a rat had ever thought it smelled, and it had come out of its hole, and now Anthony had hold of it with his mind and was making it do tricks.

When the rat saw Bill Soames coming, it tried to run, but Anthony thought at it, and it turned a flip-flop on the grass, and lay trembling, its eyes gleaming in small black terror.

Bill Soames hurried past Anthony and reached the front steps, mumbling. He always mumbled when he came to the Fremont house, or passed by it, or even thought of it. Everybody did. They thought about silly things, things that didn't mean very much, like two-and-two-is-four-and-twice-is-eight and so on; they tried to jumble up their thoughts to keep them skipping back and forth, so Anthony couldn't read their minds. The mumbling helped. Because if Anthony got anything strong out of your thoughts, he might take a notion to do something about it – like curing your wife's sick headaches or your kid's mumps, or getting your old milk cow back on schedule, or fixing the privy. And while Anthony mightn't actually mean any harm, he couldn't be expected to have much notion of what was the right thing to do in such cases."""
    
    print("=" * 60)
    print("🚀 Generating Full Text with Continuation")
    print("=" * 60)
    
    generate_with_continuation(full_text, voice="tara")