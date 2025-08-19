# Orpheus TTS with vLLM on RTX 5090

High-performance text-to-speech system using Orpheus TTS with vLLM backend, optimized for NVIDIA RTX 5090.

## 🎯 Features

- **Ultra-low latency**: ~200ms time-to-first-byte
- **Multiple voices**: 8 different voice options (leo, tara, zoe, zac, jess, mia, julia, leah)
- **Emotion support**: Add tags like `<laugh>`, `<sigh>`, `<cough>` for expressive speech
- **Streaming generation**: Real-time audio streaming capabilities
- **RTX 5090 optimized**: Fully utilizes 32GB VRAM with CUDA 12.8

## 📋 Prerequisites

- **GPU**: NVIDIA RTX 5090 (or other CUDA 12.8 compatible GPU)
- **OS**: Ubuntu/Linux (tested on Ubuntu 22.04)
- **Python**: 3.11.9 (exact version required for vLLM compatibility)
- **CUDA**: 12.8
- **RAM**: 16GB+ recommended
- **Storage**: ~10GB for models and dependencies

## 🚀 Complete Installation Guide

### Step 1: Install UV Package Manager

UV is a fast Python package manager that ensures proper environment setup:

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (add this to your .bashrc for permanent setup)
source $HOME/.local/bin/env
```

### Step 2: Install Python 3.11.9

vLLM requires Python 3.11.9 specifically:

```bash
# Install Python 3.11.9 using UV
uv python install 3.11.9
```

### Step 3: Create Virtual Environment

```bash
# Create virtual environment with Python 3.11.9
uv venv vllm-env --python 3.11.9

# Activate the environment
source vllm-env/bin/activate
```

### Step 4: Install pip in Virtual Environment

```bash
# Ensure pip is installed in the virtual environment
python -m ensurepip
python -m pip install --upgrade pip
```

### Step 5: Install vLLM with CUDA 12.8 Support

**CRITICAL**: Use the CUDA 12.8 PyTorch index for RTX 5090 compatibility:

```bash
# Install vLLM with CUDA 12.8 support
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
```

This will install:
- PyTorch 2.7.1+cu128
- vLLM 0.10.0
- All CUDA 12.8 dependencies

### Step 6: Install Orpheus TTS

```bash
# Install Orpheus TTS and dependencies
pip install orpheus-speech python-dotenv
```

### Step 7: Set up HuggingFace Token (Optional)

If you want to use private models or avoid rate limits:

```bash
# Create .env file
echo "HUGGINGFACE_TOKEN=your_token_here" > .env

# Or export directly
export HUGGINGFACE_TOKEN=your_token_here
```

### Step 8: Verify Installation

```bash
# Test vLLM and GPU detection
python test_vllm.py

# Test Orpheus TTS
python test_orpheus.py
```

## 🎮 Server Management

### Starting the Web UI Server

```bash
# Navigate to the UI directory
cd /home/lightning/Documents/Stream/vllm-orpheus-setup/orpheus-ui

# Start the server
./launch.sh

# Or start with custom settings
VLLM_GPU_MEM_UTIL=0.8 ./launch.sh
```

### Stopping the Server

```bash
# While the server is running, press:
Ctrl + C

# If the server is stuck or frozen:
Ctrl + Z  # Suspend the process
kill %1   # Kill the suspended job

# If you closed the terminal, find and kill the process:
ps aux | grep "python app.py"  # Find the process ID
kill <PID>  # Replace <PID> with the actual process ID

# Force kill if needed:
kill -9 <PID>
```

### Running in Background

```bash
# Start server in background
nohup ./launch.sh > server.log 2>&1 &

# Check if it's running
ps aux | grep "python app.py"

# Monitor the logs
tail -f server.log

# Stop background server
pkill -f "python app.py"
```

### Quick Server Commands

```bash
# Check if server is running
curl http://localhost:8000/api/health

# Test generation from command line
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","voice":"leo"}'

# View server logs (if running with nohup)
tail -f orpheus-ui/server.log
```

### Troubleshooting Server Issues

```bash
# Check GPU memory usage
nvidia-smi

# Kill all Python processes using GPU (careful!)
nvidia-smi | grep python | awk '{print $5}' | xargs -r kill -9

# Clear GPU memory cache
python -c "import torch; torch.cuda.empty_cache()"

# Check which process is using port 8000
lsof -i :8000

# Force kill process on port 8000
lsof -t -i:8000 | xargs -r kill -9
```

## 💻 Usage

### Simple Command-Line Usage

```bash
# Generate speech with default voice (leo)
python simple_orpheus.py "Hello, world!" -o output.wav

# Use different voice
python simple_orpheus.py "Hello, world!" -v tara -o tara_voice.wav

# Available voices: zoe, zac, jess, leo, mia, julia, leah, tara
```

### Python API Usage

```python
from orpheus_tts import OrpheusModel

# Initialize model
model = OrpheusModel(model_name="canopylabs/orpheus-3b-0.1-ft")

# Generate speech
audio_generator = model.generate_speech(
    prompt="Hello, this is Orpheus TTS!", 
    voice="leo"
)

# Collect audio bytes
audio_bytes = b''
for chunk in audio_generator:
    if isinstance(chunk, bytes):
        audio_bytes += chunk

# Save to file
import wave
with wave.open("output.wav", 'wb') as wav_file:
    wav_file.setnchannels(1)  # Mono
    wav_file.setsampwidth(2)   # 16-bit
    wav_file.setframerate(24000)  # 24kHz
    wav_file.writeframes(audio_bytes)
```

### Adding Emotions

```python
text = "This is amazing <laugh> I can't believe it works <sigh>"
audio_generator = model.generate_speech(prompt=text, voice="zoe")
```

## 📁 Project Structure

```
.
├── README.md                # This file
├── test_vllm.py            # Verify vLLM installation
├── test_orpheus.py         # Test Orpheus TTS
├── simple_orpheus.py       # Command-line TTS tool
├── run_orpheus_demo.py     # Multi-voice demo script
├── test_inference.py       # vLLM inference test
└── .gitignore             # Git ignore file
```

## 🔧 Troubleshooting

### Common Issues and Solutions

1. **CUDA Not Found**
   ```bash
   # Verify CUDA installation
   nvidia-smi
   # Should show RTX 5090 with CUDA 12.8
   ```

2. **Python Version Mismatch**
   ```bash
   # Ensure Python 3.11.9 is active
   python --version  # Should show 3.11.9
   ```

3. **vLLM Import Error**
   ```bash
   # Reinstall with CUDA 12.8 support
   pip uninstall vllm torch
   pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
   ```

4. **Out of Memory**
   - The model requires ~7GB VRAM
   - Close other GPU applications
   - RTX 5090's 32GB should be more than sufficient

5. **Slow First Run**
   - First run downloads the model (~6GB)
   - Subsequent runs will be much faster

## 🔬 Technical Details

### System Specifications
- **GPU**: NVIDIA GeForce RTX 5090
- **VRAM**: 32GB GDDR7
- **Compute Capability**: 12.0 (Ada Lovelace)
- **CUDA Cores**: 21,760
- **Tensor Cores**: 680

### Software Stack
- **Python**: 3.11.9
- **PyTorch**: 2.7.1+cu128
- **vLLM**: 0.10.0
- **CUDA**: 12.8
- **Model**: canopylabs/orpheus-3b-0.1-ft (3B parameters)

### Performance Metrics
- **Time to First Byte**: ~200ms
- **Real-time Factor**: >1.0x (generates faster than playback)
- **Audio Quality**: 24kHz, 16-bit mono
- **Model Loading**: ~15 seconds (first run)
- **Inference Speed**: ~83 tokens/second

## 📚 Additional Resources

- [Orpheus TTS Paper](https://arxiv.org/abs/orpheus)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Baseten Orpheus API](https://www.baseten.co/library/orpheus-tts/)
- [HuggingFace Model](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft)

## 🙏 Acknowledgments

- Canopy Labs for Orpheus TTS
- vLLM team for the inference engine
- NVIDIA for CUDA and driver support

## 📄 License

This project uses Orpheus TTS which has its own licensing terms. Please refer to the original model's license for usage restrictions.

---

**Note**: This setup has been tested and verified on RTX 5090 with 32GB VRAM. Performance may vary on other GPUs.