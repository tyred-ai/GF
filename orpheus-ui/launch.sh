#!/bin/bash

# Orpheus TTS Enterprise UI Launcher
# Ensures the server runs with the correct environment

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "           Orpheus TTS Enterprise UI Launcher"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check if we're in the correct directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Please run this script from the orpheus-ui directory."
    exit 1
fi

# Check for virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  No virtual environment detected."
    echo "Looking for vllm-env in parent directory..."
    
    if [ -d "../vllm-env" ]; then
        echo "✅ Found vllm-env, activating..."
        source ../vllm-env/bin/activate
    else
        echo "❌ Virtual environment not found. Please activate your environment first."
        echo "   Run: source ../vllm-env/bin/activate"
        exit 1
    fi
else
    echo "✅ Virtual environment active: $VIRTUAL_ENV"
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "📍 Python version: $PYTHON_VERSION"

if [[ ! "$PYTHON_VERSION" == 3.11.* ]]; then
    echo "⚠️  Warning: Python 3.11.x is recommended (found $PYTHON_VERSION)"
fi

# Check for required modules
echo ""
echo "Checking dependencies..."

# Check if Orpheus TTS is installed
if python -c "import orpheus_tts" 2>/dev/null; then
    echo "✅ Orpheus TTS found"
else
    echo "❌ Orpheus TTS not found. Installing..."
    pip install orpheus-speech
fi

# Check if FastAPI is installed
if python -c "import fastapi" 2>/dev/null; then
    echo "✅ FastAPI found"
else
    echo "📦 Installing UI dependencies..."
    pip install -r requirements.txt
fi

# Check GPU availability
echo ""
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    echo "✅ GPU detected: $GPU_INFO"
else
    echo "⚠️  No GPU detected. Performance may be limited."
fi

# Load environment variables if .env exists
if [ -f "../.env" ]; then
    echo "✅ Loading environment variables from .env"
    export $(cat ../.env | grep -v '^#' | xargs)
elif [ -f ".env" ]; then
    echo "✅ Loading environment variables from .env"
    export $(cat .env | grep -v '^#' | xargs)
fi

# Create audio directory if it doesn't exist
if [ ! -d "static/audio" ]; then
    echo "📁 Creating audio directory..."
    mkdir -p static/audio
fi

# Set host and port
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "🚀 Starting Orpheus TTS Enterprise UI Server"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "📍 Local access:    http://localhost:$PORT"
echo "📍 Network access:  http://$(hostname -I | awk '{print $1}'):$PORT"
echo ""
echo "Press Ctrl+C to stop the server"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check if generate_cli.py exists
if [ ! -f "generate_cli.py" ]; then
    echo "❌ Error: generate_cli.py not found!"
    echo "   This file is required for audio generation."
    exit 1
fi

echo "✅ Using subprocess isolation for GPU memory management"

# Launch the server
exec python app.py