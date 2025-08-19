# vLLM Installation for RTX 5090

This directory contains a working vLLM installation with CUDA 12.8 support for NVIDIA RTX 5090.

## Setup Details

- **Python Version**: 3.11.9 (required for vLLM compatibility)
- **CUDA Version**: 12.8
- **PyTorch Version**: 2.7.1+cu128
- **vLLM Version**: 0.10.0
- **GPU**: NVIDIA GeForce RTX 5090 (32GB VRAM, Compute Capability 12.0)

## Installation Steps Used

1. Install Python 3.11.9:
   ```bash
   uv python install 3.11.9
   ```

2. Create virtual environment:
   ```bash
   uv venv vllm-env --python 3.11.9
   ```

3. Activate environment:
   ```bash
   source vllm-env/bin/activate
   ```

4. Install vLLM with CUDA 12.8 support:
   ```bash
   pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
   ```

## Test Scripts

- `test_vllm.py` - Verifies installation and GPU detection
- `test_inference.py` - Runs a simple inference test with a small model

## Running Tests

```bash
# Activate the environment
source vllm-env/bin/activate

# Test installation
python test_vllm.py

# Test inference (will download a small model on first run)
python test_inference.py
```

## Next Steps for Orpheus TTS

Now that vLLM is working, you can install Orpheus TTS:

```bash
# Activate the environment if not already activated
source vllm-env/bin/activate

# Install Orpheus TTS
pip install orpheus-speech

# Or install from source if you have the code
# pip install -e ./orpheus_tts_pypi
```

## Important Notes

- The RTX 5090 uses compute capability 12.0 (Ada Lovelace architecture)
- PyTorch 2.7.1 with CUDA 12.8 fully supports this GPU
- vLLM 0.10.0 works correctly with this configuration
- Always use `--extra-index-url https://download.pytorch.org/whl/cu128` when installing PyTorch-based packages to ensure CUDA 12.8 compatibility