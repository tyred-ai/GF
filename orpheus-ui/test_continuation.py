#!/usr/bin/env python3
"""
Test continuation approach - split text into chunks and generate sequentially
"""

import requests
import json
import time

# Split the text into smaller chunks that vLLM can handle
paragraphs = [
    """Aunt Amy was out on the front porch, rocking back and forth in the high­backed chair and fanning herself, when Bill Soames rode his bicycle up the road and stopped in front of the house. Perspiring under the afternoon 'sun,' Bill lifted the box of groceries out of the big basket over the front wheel of the bike, and came up the front walk.""",
    
    """Little Anthony was sitting on the lawn, playing with a rat. He had caught the rat down in the base­ment – he had made it think that it smelled cheese, the most rich-smelling and crumbly-delicious cheese a rat had ever thought it smelled, and it had come out of its hole, and now Anthony had hold of it with his mind and was making it do tricks.""",
    
    """When the rat saw Bill Soames coming, it tried to run, but Anthony thought at it, and it turned a flip-flop on the grass, and lay trembling, its eyes gleaming in small black terror.""",
    
    """Bill Soames hurried past Anthony and reached the front steps, mumbling. He always mumbled when he came to the Fremont house, or passed by it, or even thought of it. Everybody did. They thought about silly things, things that didn't mean very much, like two-and-two-is-four-and-twice-is-eight and so on; they tried to jumble up their thoughts to keep them skipping back and forth, so Anthony couldn't read their minds. The mumbling helped. Because if Anthony got anything strong out of your thoughts, he might take a notion to do something about it – like curing your wife's sick headaches or your kid's mumps, or getting your old milk cow back on schedule, or fixing the privy. And while Anthony mightn't actually mean any harm, he couldn't be expected to have much notion of what was the right thing to do in such cases."""
]

print("=" * 60)
print("🧪 Testing Chunked Generation Approach")
print("=" * 60)
print(f"\nTotal paragraphs: {len(paragraphs)}")

url = "http://localhost:8000/api/generate"
total_tokens = 0
audio_files = []

for i, paragraph in enumerate(paragraphs, 1):
    print(f"\n📤 Generating paragraph {i}/{len(paragraphs)}")
    print(f"   Length: {len(paragraph)} chars")
    print(f"   Preview: {paragraph[:50]}...")
    
    # Add continuation marker for middle paragraphs
    if i < len(paragraphs):
        text = paragraph + " [CONTINUE]"
    else:
        text = paragraph
    
    payload = {
        "text": text,
        "voice": "tara",
        "temperature": 0.7,
        "top_p": 0.95,
        "repetition_penalty": 1.05,
        "max_tokens": 10000  # Lower per chunk
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=60)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            tokens = result.get('tokens', 0)
            total_tokens += tokens
            audio_files.append(result.get('audio_url', ''))
            
            print(f"   ✅ Generated in {elapsed:.1f}s")
            print(f"   Tokens: {tokens}")
            print(f"   Duration: {result.get('duration', 0):.1f}s")
        else:
            print(f"   ❌ Error {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Small delay between paragraphs
    if i < len(paragraphs):
        time.sleep(1)

print("\n" + "=" * 60)
print("📊 SUMMARY:")
print("=" * 60)
print(f"✅ Successfully generated {len(audio_files)} audio files")
print(f"✅ Total tokens: {total_tokens}")
print("\nAudio files generated:")
for i, audio_url in enumerate(audio_files, 1):
    print(f"  {i}. {audio_url}")

print("\n💡 To combine audio files:")
print("  You can use ffmpeg or sox to concatenate the WAV files")
print("  Example: sox file1.wav file2.wav file3.wav file4.wav combined.wav")
print("=" * 60)