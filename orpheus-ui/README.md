# Orpheus TTS Enterprise Web UI

A modern, enterprise-grade web interface for Orpheus TTS with real-time streaming capabilities.

## Features

- 🎨 **Modern Dark Theme UI** - Professional enterprise design
- 🎤 **8 Voice Options** - Choose from leo, tara, zoe, zac, jess, mia, julia, leah
- 🔄 **Real-time Streaming** - WebSocket support for live audio generation
- 💾 **Session Management** - Save and reload previous generations
- 📊 **Live Statistics** - Character count, word count, estimated duration
- 🎵 **Emotion Tags** - Support for `<laugh>`, `<sigh>`, `<cough>` and more
- 📥 **Download Audio** - Save generated speech as WAV files
- 🎛️ **Advanced Controls** - Temperature, top-p, and max tokens settings

## Quick Start

### Prerequisites

- Python 3.11.9 with vLLM environment activated
- Orpheus TTS installed (`pip install orpheus-speech`)
- RTX 5090 or compatible GPU (recommended)

### Installation & Launch

```bash
# Navigate to UI directory
cd orpheus-ui

# Launch the server (auto-installs dependencies)
./launch.sh
```

The launcher script will:
- Check for virtual environment
- Install missing dependencies
- Verify GPU availability
- Start the server on http://localhost:8000

### Manual Installation

```bash
# Activate your virtual environment
source ../vllm-env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

## Usage

1. **Open the UI**: Navigate to http://localhost:8000
2. **Enter Text**: Type or paste your text in the input area
3. **Select Voice**: Choose from 8 available voices
4. **Generate**: Click "Generate Speech" for full generation or "Stream Real-time" for streaming
5. **Download**: Save the generated audio using the download button

### Sample Texts

Click "Load Sample" to try pre-written examples:
- Professional announcement
- Conversational dialogue
- Narrative storytelling
- Technical documentation

### Advanced Settings

- **Temperature** (0.0-1.0): Controls randomness (lower = more consistent)
- **Top-P** (0.0-1.0): Nucleus sampling parameter
- **Max Tokens**: Maximum length of generated audio

## API Endpoints

### REST API

- `GET /` - Main UI interface
- `GET /api/voices` - List available voices
- `POST /api/generate` - Generate speech
- `GET /api/sessions` - List all sessions
- `GET /api/session/{id}` - Get session details
- `DELETE /api/session/{id}` - Delete session
- `GET /api/health` - Health check

### WebSocket

- `WS /ws` - Real-time streaming endpoint

### Example API Usage

```python
import requests

response = requests.post("http://localhost:8000/api/generate", json={
    "text": "Hello, world!",
    "voice": "leo",
    "temperature": 0.6,
    "top_p": 0.8,
    "max_tokens": 1200
})

result = response.json()
print(f"Audio URL: {result['audio_url']}")
print(f"Duration: {result['duration']}s")
```

## File Structure

```
orpheus-ui/
├── app.py              # FastAPI server
├── requirements.txt    # Python dependencies
├── launch.sh          # Launch script
├── README.md          # This file
├── templates/
│   └── index.html     # Main UI template
├── static/
│   ├── css/
│   │   └── style.css  # Enterprise styling
│   ├── js/
│   │   └── main.js    # UI interactions
│   └── audio/         # Generated audio files
```

## Configuration

### Environment Variables

Create a `.env` file in the parent directory:

```env
# HuggingFace token (optional)
HUGGINGFACE_TOKEN=your_token_here

# Server settings
HOST=0.0.0.0
PORT=8000
```

### Custom Settings

Modify `app.py` to change:
- Default voice
- Audio parameters (sample rate, bit depth)
- Model name
- Session storage

## Troubleshooting

### Server won't start
- Ensure virtual environment is activated
- Check Python version (3.11.9 required)
- Verify Orpheus TTS is installed

### No audio generation
- Check GPU availability with `nvidia-smi`
- Verify model is loaded (check health endpoint)
- Check console for error messages

### Slow performance
- First generation loads the model (~15 seconds)
- Ensure GPU is being used (not CPU)
- Check VRAM usage

## Performance

With RTX 5090:
- Model loading: ~15 seconds
- Time to first byte: ~200ms
- Real-time factor: >1.0x
- Concurrent sessions: Multiple supported

## Security Notes

- Default binding is to all interfaces (0.0.0.0)
- No authentication by default
- Consider proxy/firewall for production use
- Audio files are stored locally

## License

This UI is provided as-is for use with Orpheus TTS. Please refer to the Orpheus TTS license for model usage terms.