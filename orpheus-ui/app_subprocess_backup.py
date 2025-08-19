#!/usr/bin/env python3
"""
Orpheus TTS Web UI Server - Subprocess Version
Uses subprocess to avoid GPU memory conflicts between vLLM and SNAC decoder
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

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
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
        
        # Generate audio
        print(f"Generating speech for session {session_id}")
        print(f"Text: {request.text[:100]}...")
        print(f"Voice: {request.voice}")
        
        # Save audio path
        audio_filename = f"audio_{session_id}.wav"
        audio_path = Path("static/audio") / audio_filename
        
        # Use CLI to generate audio (avoids GPU memory conflicts)
        cmd = [
            sys.executable,
            "generate_cli.py",
            "--text", request.text,
            "--voice", request.voice,
            "--output", str(audio_path)
        ]
        
        print(f"Running generation command...")
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        proc = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(Path(__file__).parent)
            )
        )
        
        if proc.returncode != 0:
            error_msg = proc.stderr or proc.stdout
            raise Exception(f"Generation failed: {error_msg}")
        
        # Parse result JSON
        try:
            gen_info = json.loads(proc.stdout)
            if not gen_info.get("success"):
                raise Exception(gen_info.get("error", "Unknown error"))
                
            print(f"Generated {gen_info['chunks']} chunks, {gen_info['bytes']} bytes")
            duration = gen_info.get("duration", 0)
            
        except json.JSONDecodeError:
            print(f"Warning: Could not parse generation output: {proc.stdout}")
            # Fall back to file-based calculation
            if audio_path.exists():
                file_size = audio_path.stat().st_size
                audio_data_size = file_size - 44  # WAV header
                duration = (audio_data_size / 2) / 24000
            else:
                raise Exception("Audio file was not created")
        
        # Check if file was created
        if not audio_path.exists():
            raise Exception("Audio file was not created")
        
        # Get file size
        file_size = audio_path.stat().st_size
        
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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time streaming"""
    await websocket.accept()
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            if data.get("type") == "generate":
                text = data.get("text", "")
                voice = data.get("voice", "leo")
                session_id = str(uuid.uuid4())
                
                # Send initial status
                await websocket.send_json({
                    "type": "status",
                    "session_id": session_id,
                    "message": "Starting generation..."
                })
                
                # Generate audio
                audio_filename = f"audio_{session_id}.wav"
                audio_path = Path("static/audio") / audio_filename
                
                cmd = [
                    sys.executable,
                    "generate_cli.py",
                    "--text", text,
                    "--voice", voice,
                    "--output", str(audio_path)
                ]
                
                # Run generation
                loop = asyncio.get_event_loop()
                proc = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=120,
                        cwd=str(Path(__file__).parent)
                    )
                )
                
                if proc.returncode == 0 and audio_path.exists():
                    # Read the audio file
                    with open(audio_path, 'rb') as f:
                        audio_bytes = f.read()
                    
                    # Send as base64
                    await websocket.send_json({
                        "type": "audio_chunk",
                        "session_id": session_id,
                        "chunk_index": 1,
                        "audio": base64.b64encode(audio_bytes).decode('utf-8')
                    })
                    
                    # Send completion
                    file_size = len(audio_bytes)
                    duration = ((file_size - 44) / 2) / 24000
                    
                    await websocket.send_json({
                        "type": "complete",
                        "session_id": session_id,
                        "duration": duration,
                        "total_chunks": 1
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Failed to generate audio"
                    })
                
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    # Test if we can import the model
    can_import = False
    try:
        # Test with subprocess to avoid loading model in main process
        result = subprocess.run(
            [sys.executable, "-c", "import orpheus_tts; print('ok')"],
            capture_output=True,
            text=True,
            timeout=5
        )
        can_import = (result.returncode == 0 and 'ok' in result.stdout)
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
    
    # Check if generate_cli.py exists
    if not Path("generate_cli.py").exists():
        print("❌ Error: generate_cli.py not found!")
        print("   This file is required for audio generation.")
        sys.exit(1)
    
    # Run the server
    print("🚀 Starting Orpheus TTS Enterprise UI Server")
    print("📍 Access at: http://localhost:8000")
    print("⚠️  Note: Using subprocess isolation to avoid GPU memory conflicts")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)