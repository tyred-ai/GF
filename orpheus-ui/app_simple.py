#!/usr/bin/env python3
"""
Simplified Orpheus TTS Web UI Server
Handles the GPU memory issue by simplifying the architecture
"""

import os
import sys
import json
import wave
import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
import subprocess
import base64

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Orpheus TTS Enterprise UI", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Storage for generated audio sessions
audio_sessions: Dict[str, Dict] = {}

# Available voices
VOICES = ["leo", "tara", "zoe", "zac", "jess", "mia", "julia", "leah"]

# Request models
class TTSRequest(BaseModel):
    text: str
    voice: str = "leo"
    session_id: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    status: str
    message: str
    audio_url: Optional[str] = None
    duration: Optional[float] = None
    file_size: Optional[int] = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main UI"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "voices": VOICES
    })

@app.get("/api/voices")
async def get_voices():
    """Get list of available voices"""
    return {"voices": VOICES}

@app.get("/api/sessions")
async def get_sessions():
    """Get all audio generation sessions"""
    sessions_list = []
    for session_id, session_data in audio_sessions.items():
        sessions_list.append({
            "session_id": session_id,
            "text": session_data.get("text", "")[:100] + "...",
            "voice": session_data.get("voice", "unknown"),
            "timestamp": session_data.get("timestamp", ""),
            "duration": session_data.get("duration", 0),
            "status": session_data.get("status", "unknown")
        })
    return {"sessions": sorted(sessions_list, key=lambda x: x["timestamp"], reverse=True)}

@app.post("/api/generate")
async def generate_speech(request: TTSRequest):
    """Generate speech from text using subprocess to avoid memory issues"""
    try:
        # Create session ID
        session_id = request.session_id or str(uuid.uuid4())
        
        # Store session info
        audio_sessions[session_id] = {
            "text": request.text,
            "voice": request.voice,
            "timestamp": datetime.now().isoformat(),
            "status": "processing"
        }
        
        # Generate audio using the standalone script
        print(f"Generating speech for session {session_id}")
        print(f"Text: {request.text[:100]}...")
        print(f"Voice: {request.voice}")
        
        # Create a temporary script to generate audio
        script_content = f'''
import sys
sys.path.append('..')
from orpheus_tts import OrpheusModel
import wave

model = OrpheusModel(model_name="canopylabs/orpheus-3b-0.1-ft")
generator = model.generate_speech(prompt="{request.text}", voice="{request.voice}")

audio_bytes = b''
for chunk in generator:
    if isinstance(chunk, bytes):
        audio_bytes += chunk

# Save to file
with wave.open("static/audio/audio_{session_id}.wav", 'wb') as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(24000)
    wav_file.writeframes(audio_bytes)

print(len(audio_bytes))
'''
        
        # Write script to temp file
        temp_script = Path(f"temp_gen_{session_id}.py")
        temp_script.write_text(script_content)
        
        try:
            # Run the script in a subprocess
            result = subprocess.run(
                [sys.executable, str(temp_script)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(Path(__file__).parent)
            )
            
            if result.returncode != 0:
                raise Exception(f"Generation failed: {result.stderr}")
            
            # Get the audio size from output
            try:
                audio_size = int(result.stdout.strip().split('\n')[-1])
            except:
                audio_size = 0
            
            # Clean up temp script
            temp_script.unlink()
            
            audio_filename = f"audio_{session_id}.wav"
            audio_path = Path("static/audio") / audio_filename
            
            if not audio_path.exists():
                raise Exception("Audio file was not created")
            
            # Get actual file size
            file_size = audio_path.stat().st_size
            
            # Calculate duration
            # WAV header is 44 bytes, rest is audio data
            audio_data_size = file_size - 44
            duration = (audio_data_size / 2) / 24000  # 16-bit mono at 24kHz
            
            # Update session info
            audio_sessions[session_id].update({
                "status": "completed",
                "audio_file": audio_filename,
                "duration": duration,
                "file_size": file_size
            })
            
            return SessionResponse(
                session_id=session_id,
                status="completed",
                message=f"Generated {duration:.2f}s of audio",
                audio_url=f"/static/audio/{audio_filename}",
                duration=duration,
                file_size=file_size
            )
            
        finally:
            # Clean up temp script if it exists
            if temp_script.exists():
                temp_script.unlink()
        
    except Exception as e:
        print(f"Error generating speech: {e}")
        if session_id in audio_sessions:
            audio_sessions[session_id]["status"] = "error"
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get session details"""
    if session_id not in audio_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = audio_sessions[session_id]
    return {
        "session_id": session_id,
        "text": session_data.get("text", ""),
        "voice": session_data.get("voice", ""),
        "timestamp": session_data.get("timestamp", ""),
        "status": session_data.get("status", ""),
        "duration": session_data.get("duration", 0),
        "file_size": session_data.get("file_size", 0),
        "audio_url": f"/static/audio/{session_data.get('audio_file', '')}" if session_data.get("audio_file") else None
    }

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its audio file"""
    if session_id not in audio_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = audio_sessions[session_id]
    
    # Delete audio file if exists
    if "audio_file" in session_data:
        audio_path = Path("static/audio") / session_data["audio_file"]
        if audio_path.exists():
            audio_path.unlink()
    
    # Remove from sessions
    del audio_sessions[session_id]
    
    return {"message": "Session deleted successfully"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    # Test if we can import the model
    can_import = False
    try:
        import orpheus_tts
        can_import = True
    except:
        pass
    
    return {
        "status": "healthy",
        "model_available": can_import,
        "sessions_count": len(audio_sessions)
    }

if __name__ == "__main__":
    # Create audio directory if it doesn't exist
    Path("static/audio").mkdir(parents=True, exist_ok=True)
    
    # Run the server
    print("🚀 Starting Orpheus TTS Enterprise UI Server (Simplified)")
    print("📍 Access at: http://localhost:8000")
    print("⚠️  Note: Using subprocess isolation to avoid GPU memory conflicts")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)