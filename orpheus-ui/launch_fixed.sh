#!/bin/bash
# Launch Orpheus TTS with all fixes for word skipping

echo "🚀 Launching Orpheus TTS with fixes for word skipping..."

# Navigate to script directory
cd "$(dirname "$0")"

# Activate virtual environment
source ../vllm-env/bin/activate

# CRITICAL: Set all environment variables to prevent word skipping
export VLLM_ENABLE_CHUNKED_PREFILL=0
export VLLM_CHUNKED_PREFILL_ENABLED=false
export ENABLE_CHUNKED_PREFILL=false
export VLLM_MAX_MODEL_LEN=16384  # Keep under 32k to prevent auto-enable
export VLLM_ENABLE_PREFIX_CACHING=0
export VLLM_USE_V1=0  # Use stable V0

# Additional settings
export VLLM_GPU_MEM_UTIL=0.85
export VLLM_MAX_NUM_SEQS=1
export CUDA_VISIBLE_DEVICES=0

echo "✅ Environment configured:"
echo "  - Chunked prefill: DISABLED"
echo "  - Max model length: 16384 (prevents auto-enable)"
echo "  - Prefix caching: DISABLED"
echo "  - Using V0 engine"

# Launch with explicit flags to override any defaults
python app.py \
    --enable-chunked-prefill=false \
    --max-model-len=16384 \
    2>&1 | tee orpheus.log