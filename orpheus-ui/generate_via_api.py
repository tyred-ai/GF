#!/usr/bin/env python3
"""
Generate full text by calling the API with proper chunking
"""

import requests
import time
import wave
import os

def combine_wav_files(files, output_file):
    """Combine multiple WAV files into one"""
    data = []
    for file in files:
        with wave.open(file, 'rb') as w:
            data.append(w.readframes(w.getnframes()))
    
    # Write combined file
    with wave.open(output_file, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        for d in data:
            w.writeframes(d)
    
    print(f"✅ Combined {len(files)} files into {output_file}")

def generate_full_text():
    # Split text into manageable chunks
    chunks = [
        "Aunt Amy was out on the front porch, rocking back and forth in the high­backed chair and fanning herself, when Bill Soames rode his bicycle up the road and stopped in front of the house. Perspiring under the afternoon 'sun,' Bill lifted the box of groceries out of the big basket over the front wheel of the bike, and came up the front walk.",
        
        "Little Anthony was sitting on the lawn, playing with a rat. He had caught the rat down in the base­ment – he had made it think that it smelled cheese, the most rich-smelling and crumbly-delicious cheese a rat had ever thought it smelled, and it had come out of its hole, and now Anthony had hold of it with his mind and was making it do tricks.",
        
        "When the rat saw Bill Soames coming, it tried to run, but Anthony thought at it, and it turned a flip-flop on the grass, and lay trembling, its eyes gleaming in small black terror.",
        
        "Bill Soames hurried past Anthony and reached the front steps, mumbling. He always mumbled when he came to the Fremont house, or passed by it, or even thought of it. Everybody did. They thought about silly things, things that didn't mean very much, like two-and-two-is-four-and-twice-is-eight and so on; they tried to jumble up their thoughts to keep them skipping back and forth, so Anthony couldn't read their minds.",
        
        "The mumbling helped. Because if Anthony got anything strong out of your thoughts, he might take a notion to do something about it – like curing your wife's sick headaches or your kid's mumps, or getting your old milk cow back on schedule, or fixing the privy. And while Anthony mightn't actually mean any harm, he couldn't be expected to have much notion of what was the right thing to do in such cases."
    ]
    
    print("=" * 60)
    print("🚀 Generating Full Text via API (Chunked)")
    print("=" * 60)
    
    # First, check if server is running
    try:
        r = requests.get("http://localhost:8000/api/health", timeout=2)
        if r.status_code != 200:
            print("❌ Server not running. Start it with:")
            print("   cd /home/lightning/Documents/Stream/vllm-orpheus-setup/orpheus-ui")
            print("   source ../vllm-env/bin/activate && python app.py")
            return
    except:
        print("❌ Server not running. Start it with:")
        print("   cd /home/lightning/Documents/Stream/vllm-orpheus-setup/orpheus-ui")
        print("   source ../vllm-env/bin/activate && python app.py")
        return
    
    print("✅ Server is running\n")
    
    audio_files = []
    total_duration = 0
    
    for i, chunk in enumerate(chunks, 1):
        print(f"📝 Generating chunk {i}/{len(chunks)}")
        print(f"   Length: {len(chunk)} chars")
        print(f"   Preview: {chunk[:50]}...")
        
        payload = {
            "text": chunk,
            "voice": "tara",
            "temperature": 0.7,
            "top_p": 0.95,
            "repetition_penalty": 1.05,
            "max_tokens": 10000
        }
        
        try:
            r = requests.post("http://localhost:8000/api/generate", json=payload, timeout=60)
            if r.status_code == 200:
                result = r.json()
                audio_url = result.get('audio_url', '')
                duration = result.get('duration', 0)
                
                # Download the audio file
                if audio_url:
                    audio_file = f"chunk_{i}.wav"
                    audio_path = f"http://localhost:8000{audio_url}"
                    audio_r = requests.get(audio_path)
                    with open(audio_file, 'wb') as f:
                        f.write(audio_r.content)
                    audio_files.append(audio_file)
                    total_duration += duration
                    print(f"   ✅ Generated {duration:.1f}s of audio")
                else:
                    print(f"   ⚠️ No audio URL returned")
            else:
                print(f"   ❌ Error: {r.status_code}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Small delay between chunks
        if i < len(chunks):
            time.sleep(2)
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY:")
    print(f"✅ Generated {len(audio_files)}/{len(chunks)} chunks")
    print(f"✅ Total duration: {total_duration:.1f} seconds")
    
    if audio_files:
        # Combine all audio files
        output_file = "full_story.wav"
        combine_wav_files(audio_files, output_file)
        print(f"\n🎵 Full audio saved to: {output_file}")
        
        # Clean up individual chunks
        for f in audio_files:
            os.remove(f)
        print("🧹 Cleaned up temporary files")
    
    print("=" * 60)

if __name__ == "__main__":
    generate_full_text()