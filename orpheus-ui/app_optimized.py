#!/usr/bin/env python3
"""
Orpheus TTS Web UI Server - Optimized Version
Based on professional implementation with proper vLLM configuration
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
import time
import numpy as np

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Force stable V0 engine by default for better performance
os.environ.setdefault("VLLM_USE_V1", "0")

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Initialize FastAPI app
app = FastAPI(title="Orpheus TTS Enterprise UI", version="2.0.0")

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

# Global model instance - initialized once and reused
ORPHEUS_MODEL = None
MODEL_LOCK = asyncio.Lock()

# Storage for generated audio sessions
audio_sessions: Dict[str, Dict] = {}

# Available voices
VOICES = ["leo", "tara", "zoe", "zac", "jess", "mia", "julia", "leah"]

# Stats tracking
LAST_STATS = {}

# Request models
class TTSRequest(BaseModel):
    text: str
    voice: str = "leo"
    session_id: Optional[str] = None
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.8
    repetition_penalty: Optional[float] = 1.0
    max_tokens: Optional[int] = 1200

class SessionResponse(BaseModel):
    session_id: str
    status: str
    message: str
    audio_url: Optional[str] = None
    duration: Optional[float] = None
    file_size: Optional[int] = None
    tokens: Optional[int] = None
    tps: Optional[float] = None

def init_global_model():
    """Initialize the Orpheus model with optimized vLLM settings"""
    global ORPHEUS_MODEL
    if ORPHEUS_MODEL is not None:
        return ORPHEUS_MODEL
    
    from orpheus_tts import OrpheusModel
    
    # Model configuration
    model_id = os.getenv("ORPHEUS_MODEL_ID", "canopylabs/orpheus-3b-0.1-ft")
    tokenizer_id = os.getenv("ORPHEUS_TOKENIZER_ID", "canopylabs/orpheus-3b-0.1-pretrained")
    
    # vLLM engine tuning for RTX 5090 (32GB VRAM)
    # These settings prevent OOM and optimize performance
    gpu_mem_util = float(os.getenv("VLLM_GPU_MEM_UTIL", "0.85"))  # Leave some room for SNAC decoder
    max_num_seqs = int(os.getenv("VLLM_MAX_NUM_SEQS", "1"))  # Single sequence for TTS
    enforce_eager = os.getenv("VLLM_ENFORCE_EAGER", "0") == "1"  # Use CUDA graphs for speed
    enable_chunked_prefill = os.getenv("VLLM_ENABLE_CHUNKED_PREFILL", "1") == "1"
    enable_prefix_caching = os.getenv("VLLM_ENABLE_PREFIX_CACHING", "0") == "1"
    kv_cache_dtype = os.getenv("VLLM_KV_CACHE_DTYPE", "auto")  # Can use "fp8" to save memory
    max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "8192"))  # Limit context for TTS
    
    engine_kwargs = {
        "gpu_memory_utilization": gpu_mem_util,
        "max_num_seqs": max_num_seqs,
        "enforce_eager": enforce_eager,
        "enable_chunked_prefill": enable_chunked_prefill,
        "enable_prefix_caching": enable_prefix_caching,
        "kv_cache_dtype": kv_cache_dtype,
        "max_model_len": max_model_len,
        "dtype": "auto",  # Will use bfloat16 on RTX 5090
    }
    
    print(f"Initializing Orpheus model with optimized settings:")
    print(f"  GPU Memory Utilization: {gpu_mem_util}")
    print(f"  Max Model Length: {max_model_len}")
    print(f"  KV Cache Type: {kv_cache_dtype}")
    
    ORPHEUS_MODEL = OrpheusModel(
        model_name=model_id,
        tokenizer=tokenizer_id,
        **engine_kwargs
    )
    
    print("✅ Model loaded successfully")
    
    # Prewarm the model to load weights and capture CUDA graphs
    if os.getenv("ORPHEUS_PREWARM", "1") == "1":
        print("Prewarming model...")
        try:
            for _chunk in ORPHEUS_MODEL.generate_speech(prompt="Warm up.", voice="leo"):
                break
            print("✅ Model prewarmed")
        except Exception as e:
            print(f"⚠️ Prewarm failed: {e}")
    
    return ORPHEUS_MODEL

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    try:
        async with MODEL_LOCK:
            init_global_model()
    except Exception as e:
        print(f"⚠️ Model initialization failed on startup: {e}")
        print("Model will be initialized on first request")

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

async def generate_audio_async(text: str, voice: str, session_id: str, 
                              temperature: float = 0.6, top_p: float = 0.8,
                              repetition_penalty: float = 1.0, max_tokens: int = 1200):
    """Generate audio asynchronously"""
    global ORPHEUS_MODEL, LAST_STATS
    
    # Initialize model if needed
    async with MODEL_LOCK:
        if ORPHEUS_MODEL is None:
            init_global_model()
    
    # Generate unique request ID
    request_id = f"web-{session_id}"
    start_time = time.time()
    
    # Prepare generation kwargs
    gen_kwargs = {
        "prompt": text,
        "voice": voice,
        "request_id": request_id,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "max_tokens": max_tokens
    }
    
    # Generate audio chunks
    chunks = []
    token_count = 0
    
    # Run generation in executor to avoid blocking
    loop = asyncio.get_event_loop()
    
    def generate_sync():
        nonlocal chunks, token_count
        for chunk in ORPHEUS_MODEL.generate_speech(**gen_kwargs):
            if chunk:
                chunks.append(chunk)
        # Try to get token stats from model
        try:
            token_count = getattr(ORPHEUS_MODEL, 'last_llm_tokens', 0)
        except:
            token_count = 0
    
    await loop.run_in_executor(None, generate_sync)
    
    # Combine chunks into audio
    pcm_bytes = b"".join(chunks)
    
    # Add short trailing pad to avoid truncating sentence tails
    if pcm_bytes:
        pad_samples = int(0.08 * 24000)  # 80ms pad
        pad_bytes = np.zeros(pad_samples, dtype=np.int16).tobytes()
        pcm_bytes = pcm_bytes + pad_bytes
    
    # Calculate stats
    elapsed_ms = int((time.time() - start_time) * 1000)
    audio_seconds = len(pcm_bytes) / (2 * 24000)
    tokens_per_sec = (token_count / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0
    
    # Store stats
    LAST_STATS = {
        "elapsed_ms": elapsed_ms,
        "audio_seconds": audio_seconds,
        "tokens": token_count,
        "tps": tokens_per_sec,
        "voice": voice,
        "request_id": request_id
    }
    
    return pcm_bytes, elapsed_ms, audio_seconds, token_count, tokens_per_sec

@app.post("/api/generate")
async def generate_speech(request: TTSRequest):
    """Generate speech from text with optimized model"""
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
        
        print(f"Generating speech for session {session_id}")
        print(f"Text: {request.text[:100]}...")
        print(f"Voice: {request.voice}")
        
        # Generate audio
        pcm_bytes, elapsed_ms, audio_seconds, tokens, tps = await generate_audio_async(
            request.text,
            request.voice,
            session_id,
            request.temperature,
            request.top_p,
            request.repetition_penalty,
            request.max_tokens
        )
        
        if not pcm_bytes:
            raise Exception("No audio generated")
        
        print(f"Generated {len(pcm_bytes)} bytes in {elapsed_ms}ms")
        print(f"Tokens: {tokens}, TPS: {tps:.2f}")
        
        # Save audio to file
        audio_filename = f"audio_{session_id}.wav"
        audio_path = Path("static/audio") / audio_filename
        
        # Write WAV file
        with wave.open(str(audio_path), 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(24000)  # 24kHz
            wav_file.writeframes(pcm_bytes)
        
        file_size = audio_path.stat().st_size
        
        # Update session info
        audio_sessions[session_id].update({
            "status": "completed",
            "audio_file": audio_filename,
            "duration": audio_seconds,
            "file_size": file_size,
            "tokens": tokens,
            "tps": tps
        })
        
        return SessionResponse(
            session_id=session_id,
            status="completed",
            message=f"Generated {audio_seconds:.2f}s of audio in {elapsed_ms/1000:.2f}s",
            audio_url=f"/static/audio/{audio_filename}",
            duration=audio_seconds,
            file_size=file_size,
            tokens=tokens,
            tps=round(tps, 2)
        )
        
    except Exception as e:
        print(f"Error generating speech: {e}")
        import traceback
        traceback.print_exc()
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
        "tokens": session_data.get("tokens", 0),
        "tps": session_data.get("tps", 0),
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
                
                try:
                    # Generate audio
                    pcm_bytes, elapsed_ms, audio_seconds, tokens, tps = await generate_audio_async(
                        text, voice, session_id
                    )
                    
                    if pcm_bytes:
                        # Create WAV in memory
                        buffer = io.BytesIO()
                        with wave.open(buffer, 'wb') as wav_file:
                            wav_file.setnchannels(1)
                            wav_file.setsampwidth(2)
                            wav_file.setframerate(24000)
                            wav_file.writeframes(pcm_bytes)
                        wav_bytes = buffer.getvalue()
                        
                        # Send as base64
                        await websocket.send_json({
                            "type": "audio_chunk",
                            "session_id": session_id,
                            "chunk_index": 1,
                            "audio": base64.b64encode(wav_bytes).decode('utf-8')
                        })
                        
                        # Send completion with stats
                        await websocket.send_json({
                            "type": "complete",
                            "session_id": session_id,
                            "duration": audio_seconds,
                            "elapsed_ms": elapsed_ms,
                            "tokens": tokens,
                            "tps": round(tps, 2),
                            "total_chunks": 1
                        })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Failed to generate audio"
                        })
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
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
    global ORPHEUS_MODEL
    
    # Check GPU status
    gpu_info = {}
    try:
        import torch
        gpu_info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            gpu_info["cuda_device_name"] = torch.cuda.get_device_name(0)
            gpu_info["cuda_memory_allocated"] = f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB"
            gpu_info["cuda_memory_reserved"] = f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB"
    except:
        pass
    
    return {
        "status": "healthy",
        "model_loaded": ORPHEUS_MODEL is not None,
        "sessions_count": len(audio_sessions),
        "last_stats": LAST_STATS,
        "gpu_info": gpu_info
    }

if __name__ == "__main__":
    # Create audio directory if it doesn't exist
    Path("static/audio").mkdir(parents=True, exist_ok=True)
    
    # Run the server
    print("🚀 Starting Orpheus TTS Enterprise UI Server (Optimized)")
    print("📍 Access at: http://localhost:8000")
    print("⚙️  Optimized for RTX 5090 with vLLM tuning")
    print("💡 Model persists in memory for fast generation")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)