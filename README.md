# Orpheus TTS with vLLM - Complete Setup Guide

## 🚀 Quick Overview

Production-ready **Orpheus TTS** system with modern web UI, optimized for NVIDIA RTX 5090 (32GB VRAM) using vLLM for ~200ms latency speech synthesis. Features real-time voice generation with 8 personas, emotion tags, and comprehensive performance monitoring.

## 📋 System Requirements

### Hardware
- **GPU**: NVIDIA RTX 4090/5090 or similar (minimum 16GB VRAM, 24GB+ recommended)
- **RAM**: 32GB minimum
- **Storage**: 50GB free space
- **CPU**: 8+ cores recommended

### Software
- **OS**: Ubuntu 24.04 LTS (tested) or Ubuntu 25.04
- **CUDA**: 12.1 or higher (12.8 tested)
- **Python**: 3.11.9 (REQUIRED - vLLM dependency)
- **Git**: For cloning repository

## 🛠️ Complete Installation Guide

### Step 1: Install System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y build-essential software-properties-common

# Install Python development packages
sudo apt install -y python3-dev python3-pip python3-venv

# Install system libraries
sudo apt install -y git curl wget ffmpeg libsndfile1 portaudio19-dev

# Install CUDA (if not already installed)
# For Ubuntu 24.04 with CUDA 12.8:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-8

# Add CUDA to PATH (add to ~/.bashrc)
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA installation
nvcc --version
nvidia-smi
```

### Step 2: Install UV Package Manager (CRITICAL)

```bash
# Install UV - Required for proper Python version management
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Verify UV installation
uv --version
```

### Step 3: Clone and Setup Repository

```bash
# Clone the repository
git clone https://github.com/tyred-ai/GF.git orpheus-tts-setup
cd orpheus-tts-setup

# Install Python 3.11.9 (REQUIRED for vLLM)
uv python install 3.11.9

# Create virtual environment with Python 3.11.9
uv venv vllm-env --python 3.11.9

# Activate the environment
source vllm-env/bin/activate

# Verify Python version (MUST be 3.11.9)
python --version
```

### Step 4: Install vLLM with Correct Configuration

```bash
# CRITICAL: Install vLLM with CUDA 12.8 support
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128

# Install PyTorch with CUDA 12.8 (if needed separately)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify vLLM installation
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
```

### Step 5: Install Orpheus TTS and Dependencies

```bash
# Install Orpheus TTS
pip install orpheus-speech

# Install web UI dependencies
pip install fastapi uvicorn[standard] pydantic python-multipart

# Install additional dependencies
pip install numpy scipy soundfile librosa
pip install Pillow  # For image handling in UI
pip install python-dotenv  # For environment variables

# Install HuggingFace dependencies
pip install transformers accelerate datasets
```

### Step 6: Configure Environment Variables

Create `.env` file in project root:

```bash
cat > .env << 'EOF'
# HuggingFace Token (get from https://huggingface.co/settings/tokens)
HUGGINGFACE_TOKEN=your_token_here

# Orpheus TTS Configuration
ORPHEUS_MODEL_ID=canopylabs/orpheus-3b-0.1-ft
ORPHEUS_VOICE=tara
ORPHEUS_PREWARM=1

# vLLM Optimization Settings for RTX 5090
VLLM_GPU_MEM_UTIL=0.85
VLLM_USE_V1=0
VLLM_MAX_NUM_SEQS=1
VLLM_ENFORCE_EAGER=0
VLLM_ENABLE_CHUNKED_PREFILL=1
VLLM_ENABLE_PREFIX_CACHING=0
VLLM_KV_CACHE_DTYPE=auto
VLLM_MAX_MODEL_LEN=8192

# Audio Padding Settings (prevents word cutoffs)
ORPHEUS_LEAD_PAD_MS=50
ORPHEUS_TRAIL_PAD_MS=150
EOF
```

### Step 7: Download Model (First Time Only)

```bash
# Login to HuggingFace (if using private models)
huggingface-cli login --token $HUGGINGFACE_TOKEN

# Pre-download model (optional, happens automatically on first run)
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('canopylabs/orpheus-3b-0.1-ft')"
```

## 🎯 Running the Application

### Start the Web UI Server

```bash
# Navigate to UI directory
cd orpheus-ui

# Start the server
python app.py

# Access the UI at:
# http://localhost:8000
```

### Stop the Server

```bash
# Press Ctrl+C in terminal
# OR from another terminal:
pkill -f "python app.py"

# Force kill if stuck:
lsof -i :8000 | grep LISTEN | awk '{print $2}' | xargs -r kill -9
```

### Run in Background

```bash
# Start in background
nohup python app.py > server.log 2>&1 &

# Monitor logs
tail -f server.log

# Stop background server
pkill -f "python app.py"
```

## 📁 Project Structure

```
orpheus-tts-setup/
├── orpheus-ui/               # Web UI application
│   ├── app.py               # FastAPI server
│   ├── static/              # CSS, JS, images
│   │   ├── css/            # Stylesheets
│   │   ├── js/             # JavaScript
│   │   └── images/         # Voice avatars
│   └── templates/           # HTML templates
├── Icons/                   # Original voice photos
├── .env                     # Environment configuration
├── README.md               # This file
└── vllm-env/               # Python virtual environment
```

## 🎨 Features

### Voices Available
- **Female**: Tara (default), Zoe, Jess, Mia, Julia, Leah
- **Male**: Leo, Zac

### Emotion Tags Supported
```
<laugh>   - Laughter
<sigh>    - Sighing
<cough>   - Coughing
<chuckle> - Chuckling
<sniffle> - Sniffling
<groan>   - Groaning
<yawn>    - Yawning
<gasp>    - Gasping
```

### UI Features
- Real-time TTS generation (~200ms latency)
- Performance metrics (tokens/sec, generation time, RTF)
- Voice selection with avatar photos
- Advanced settings (temperature, top-p, repetition penalty)
- Session history with audio playback
- Modern dark theme with neon yellow accents
- Large text input area for multi-paragraph content

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce GPU memory utilization in .env
   VLLM_GPU_MEM_UTIL=0.7
   
   # Clear GPU memory
   python -c "import torch; torch.cuda.empty_cache()"
   ```

2. **Port 8000 Already in Use**
   ```bash
   # Kill existing process
   lsof -i :8000 | grep LISTEN | awk '{print $2}' | xargs -r kill -9
   ```

3. **Model Download Issues**
   ```bash
   # Clear HuggingFace cache and retry
   rm -rf ~/.cache/huggingface/
   huggingface-cli login
   ```

4. **Python Version Mismatch**
   ```bash
   # Must use Python 3.11.9
   uv python install 3.11.9
   uv venv vllm-env --python 3.11.9
   source vllm-env/bin/activate
   python --version  # Should show 3.11.9
   ```

5. **vLLM Import Error**
   ```bash
   # Reinstall with correct CUDA support
   pip uninstall vllm torch
   pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
   ```

## 🚀 Performance Optimization

### For RTX 5090 (32GB VRAM)

```bash
# Optimal settings (already in .env)
VLLM_GPU_MEM_UTIL=0.85      # Use 85% of VRAM
VLLM_MAX_NUM_SEQS=1         # Single sequence for TTS
VLLM_KV_CACHE_DTYPE=auto    # Automatic optimization
```

### For RTX 4090 (24GB VRAM)

```bash
# Adjust in .env
VLLM_GPU_MEM_UTIL=0.8       # Use 80% of VRAM
VLLM_MAX_MODEL_LEN=4096     # Reduce context length
```

### Monitor Performance

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Check VRAM usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Monitor server health
curl http://localhost:8000/api/health
```

## 📦 Backup and Transfer

### Creating Portable Package

```bash
# Create backup (excludes venv and generated files)
tar -czf orpheus-tts-portable.tar.gz \
  --exclude='vllm-env' \
  --exclude='*.pyc' \
  --exclude='__pycache__' \
  --exclude='.git' \
  --exclude='orpheus-ui/static/audio/*.wav' \
  .

echo "Backup created: orpheus-tts-portable.tar.gz"
```

### Restore on New System

```bash
# Extract backup
tar -xzf orpheus-tts-portable.tar.gz
cd orpheus-tts-setup

# Follow installation steps 1-6 above
# Then start the application
cd orpheus-ui && python app.py
```

## 📝 Development Commands

### Testing TTS Generation

```python
# test_generation.py
from orpheus_tts import OrpheusModel
import wave

# Initialize model
model = OrpheusModel("canopylabs/orpheus-3b-0.1-ft")

# Generate speech
audio_generator = model.generate_speech(
    prompt="Hello! This is a test of Orpheus TTS.",
    voice="tara"
)

# Collect audio
audio_bytes = b''
for chunk in audio_generator:
    if isinstance(chunk, bytes):
        audio_bytes += chunk

# Save to file
with wave.open("test.wav", "wb") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(24000)
    f.writeframes(audio_bytes)

print("Audio saved to test.wav")
```

### API Usage

```bash
# Test generation via API
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world! This is Orpheus TTS.",
    "voice": "tara",
    "temperature": 0.4,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
    "max_tokens": 4096
  }'

# Get available voices
curl http://localhost:8000/api/voices

# Check system health
curl http://localhost:8000/api/health
```

## 🔐 Security Notes

- Never commit `.env` file with real tokens
- Use environment variables for sensitive data
- Run behind reverse proxy (nginx) in production
- Enable HTTPS for public deployment
- Regularly update dependencies

## 📄 License & Credits

- **Orpheus TTS** by Canopy Labs
- **vLLM** inference engine
- **UI Design**: Modern dark theme with neon accents
- Optimized for NVIDIA RTX GPUs

## 🆘 Support

For issues:
1. Check this README thoroughly
2. Verify Python 3.11.9 is active
3. Ensure CUDA is properly installed
4. Check GPU memory with `nvidia-smi`
5. Review `.env` configuration

## 📊 Performance Metrics

- **Time to First Byte**: ~200ms
- **Real-time Factor**: >2x (generates faster than playback)
- **Audio Quality**: 24kHz, 16-bit mono
- **Model Loading**: ~15 seconds (first run)
- **Tokens/Second**: 80-120 (varies by input)
- **GPU Memory**: ~7GB for model

---

**Last Updated**: August 2025
**Tested On**: Ubuntu 24.04, RTX 5090, CUDA 12.8, Python 3.11.9