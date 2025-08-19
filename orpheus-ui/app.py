#!/usr/bin/env python3
"""
Orpheus TTS Web UI Server
Enterprise-grade web interface for text-to-speech generation
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
import io
import base64

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from orpheus_tts import OrpheusModel

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

# Global model instance (initialized on first request)
model = None
model_lock = asyncio.Lock()

# Storage for generated audio sessions
audio_sessions: Dict[str, Dict] = {}

# Available voices
VOICES = ["leo", "tara", "zoe", "zac", "jess", "mia", "julia", "leah"]

# Request models
class TTSRequest(BaseModel):
    text: str
    voice: str = "leo"
    session_id: Optional[str] = None
    temperature: float = 0.6
    top_p: float = 0.8
    max_tokens: int = 1200
    include_emotions: bool = False

class SessionResponse(BaseModel):
    session_id: str
    status: str
    message: str
    audio_url: Optional[str] = None
    duration: Optional[float] = None
    file_size: Optional[int] = None

async def initialize_model():
    """Initialize the Orpheus model (singleton)"""
    global model
    async with model_lock:
        if model is None:
            print("Initializing Orpheus model...")
            model = OrpheusModel(model_name="canopylabs/orpheus-3b-0.1-ft")
            print("✅ Model loaded successfully")
    return model

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
    """Generate speech from text"""
    try:
        # Initialize model if needed
        await initialize_model()
        
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
        
        # Create unique request ID for vLLM
        request_id = f"web-{session_id}"
        
        # Generate speech
        audio_generator = model.generate_speech(
            prompt=request.text,
            voice=request.voice,
            request_id=request_id,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens
        )
        
        # Collect audio chunks
        audio_bytes = b''
        chunk_count = 0
        for chunk in audio_generator:
            chunk_count += 1
            if isinstance(chunk, bytes):
                audio_bytes += chunk
            elif isinstance(chunk, np.ndarray):
                if chunk.dtype != np.int16:
                    chunk = chunk.astype(np.int16)
                audio_bytes += chunk.tobytes()
            else:
                audio_bytes += bytes(chunk)
        
        print(f"Generated {chunk_count} chunks, {len(audio_bytes)} bytes")
        
        # Save audio to file
        audio_filename = f"audio_{session_id}.wav"
        audio_path = Path("static/audio") / audio_filename
        
        with wave.open(str(audio_path), 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(24000)  # 24kHz
            wav_file.writeframes(audio_bytes)
        
        # Calculate duration
        duration = (len(audio_bytes) / 2) / 24000
        file_size = len(audio_bytes)
        
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
        # Initialize model if needed
        await initialize_model()
        
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
                
                # Generate speech
                request_id = f"ws-{session_id}"
                audio_generator = model.generate_speech(
                    prompt=text,
                    voice=voice,
                    request_id=request_id
                )
                
                # Stream audio chunks
                audio_bytes = b''
                chunk_index = 0
                for chunk in audio_generator:
                    chunk_index += 1
                    
                    if isinstance(chunk, bytes):
                        chunk_bytes = chunk
                    elif isinstance(chunk, np.ndarray):
                        if chunk.dtype != np.int16:
                            chunk = chunk.astype(np.int16)
                        chunk_bytes = chunk.tobytes()
                    else:
                        chunk_bytes = bytes(chunk)
                    
                    audio_bytes += chunk_bytes
                    
                    # Send chunk to client
                    await websocket.send_json({
                        "type": "audio_chunk",
                        "session_id": session_id,
                        "chunk_index": chunk_index,
                        "audio": base64.b64encode(chunk_bytes).decode('utf-8')
                    })
                
                # Send completion message
                duration = (len(audio_bytes) / 2) / 24000
                await websocket.send_json({
                    "type": "complete",
                    "session_id": session_id,
                    "duration": duration,
                    "total_chunks": chunk_index
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
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "sessions_count": len(audio_sessions)
    }

if __name__ == "__main__":
    # Create audio directory if it doesn't exist
    Path("static/audio").mkdir(parents=True, exist_ok=True)
    
    # Run the server
    print("🚀 Starting Orpheus TTS Enterprise UI Server")
    print("📍 Access at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)