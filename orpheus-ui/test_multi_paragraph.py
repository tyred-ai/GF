#!/usr/bin/env python3
"""
Test multi-paragraph generation with fixes
"""

import requests
import json
import time

# Test with full multi-paragraph text
test_text = """Aunt Amy was out on the front porch, rocking back and forth in the high­backed chair and fanning herself, when Bill Soames rode his bicycle up the road and stopped in front of the house. Perspiring under the afternoon 'sun,' Bill lifted the box of groceries out of the big basket over the front wheel of the bike, and came up the front walk.

Little Anthony was sitting on the lawn, playing with a rat. He had caught the rat down in the base­ment – he had made it think that it smelled cheese, the most rich-smelling and crumbly-delicious cheese a rat had ever thought it smelled, and it had come out of its hole, and now Anthony had hold of it with his mind and was making it do tricks.

When the rat saw Bill Soames coming, it tried to run, but Anthony thought at it, and it turned a flip-flop on the grass, and lay trembling, its eyes gleaming in small black terror.

Bill Soames hurried past Anthony and reached the front steps, mumbling. He always mumbled when he came to the Fremont house, or passed by it, or even thought of it. Everybody did. They thought about silly things, things that didn't mean very much, like two-and-two-is-four-and-twice-is-eight and so on; they tried to jumble up their thoughts to keep them skipping back and forth, so Anthony couldn't read their minds. The mumbling helped. Because if Anthony got anything strong out of your thoughts, he might take a notion to do something about it – like curing your wife's sick headaches or your kid's mumps, or getting your old milk cow back on schedule, or fixing the privy. And while Anthony mightn't actually mean any harm, he couldn't be expected to have much notion of what was the right thing to do in such cases."""

print("=" * 60)
print("🧪 Testing Multi-Paragraph Generation")
print("=" * 60)
print(f"\nTest text has {len(test_text)} characters")
print(f"Number of paragraphs: 4")
print(f"First 50 chars: {test_text[:50]}...")
print(f"Last 50 chars: ...{test_text[-50:]}")

# Make request to the API
url = "http://localhost:8000/api/generate"
payload = {
    "text": test_text,
    "voice": "tara",
    "temperature": 0.7,
    "top_p": 0.95,
    "repetition_penalty": 1.05,
    "max_tokens": 30000  # Explicit high value
}

print("\n📤 Sending request with:")
print(f"  - max_tokens: {payload['max_tokens']}")
print(f"  - repetition_penalty: {payload['repetition_penalty']}")

try:
    start_time = time.time()
    response = requests.post(url, json=payload, timeout=120)
    elapsed = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ SUCCESS in {elapsed:.1f}s")
        print(f"  - Audio URL: {result.get('audio_url', 'N/A')}")
        print(f"  - Duration: {result.get('duration', 0):.1f}s")
        print(f"  - Tokens: {result.get('tokens', 0)}")
        print(f"  - TPS: {result.get('tps', 0):.1f}")
        
        # Check if enough tokens were generated for both paragraphs
        expected_min_tokens = len(test_text) * 2  # Rough estimate
        actual_tokens = result.get('tokens', 0)
        
        if actual_tokens < expected_min_tokens:
            print(f"\n⚠️ WARNING: Only {actual_tokens} tokens generated")
            print(f"  Expected at least {expected_min_tokens} for full text")
            print("  Second paragraph may have been cut off!")
        else:
            print(f"\n✅ Generated sufficient tokens for both paragraphs")
            
    else:
        print(f"\n❌ Error {response.status_code}: {response.text}")
        
except requests.exceptions.Timeout:
    print("\n❌ Request timed out after 120 seconds")
except Exception as e:
    print(f"\n❌ Error: {e}")

print("\n" + "=" * 60)
print("📊 Check server logs for [DEBUG] messages showing:")
print("  - Preprocessed text (should have 'Begin.' prefix)")
print("  - Generation kwargs (should show max_tokens=30000)")
print("=" * 60)