from __future__ import annotations

import io
import json
from typing import Optional
import os
import traceback
import time
import uuid
import numpy as np

from fastapi import FastAPI, HTTPException, Response, Query
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi import Body, Header
import asyncio
import audioop
import threading
import queue

app = FastAPI(title="Orpheus TTS Server (Self-Hosted)")

# CORS for local testing and LAN use. Tighten in prod as needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Gen-Ms", "X-Audio-Seconds", "X-Req-Id", "X-Voice", "X-Download-URL", "X-Tokens", "X-TPS"],
)

# Serve static UI and media
UI_DIR = os.path.join(os.path.dirname(__file__), "ui")
if os.path.isdir(UI_DIR):
    app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)
app.mount("/media", StaticFiles(directory=AUDIO_DIR, html=False), name="media")

# Serve playback test assets if present
PLAYBACK_DIR = os.path.join(os.path.dirname(__file__), "playback")
if os.path.isdir(PLAYBACK_DIR):
    app.mount("/playback", StaticFiles(directory=PLAYBACK_DIR, html=True), name="playback")

# Mount ASR WebSocket endpoints
try:
    # Prefer absolute import when running as module
    import asr_ws  # type: ignore
    app.include_router(asr_ws.router)
except Exception:
    try:
        from . import asr_ws  # type: ignore
        app.include_router(asr_ws.router)
    except Exception:
        pass

# --------------- Config helpers --------------- #

def _is_enabled(key: str, default: str = "1") -> bool:
    val = os.getenv(key, default).strip().lower()
    return val in ("1", "true", "yes", "on")

def _get_default_tts_format() -> str:
    # wav24 | pcm24 | mulaw8k
    fmt = os.getenv("ORPHEUS_TTS_DEFAULT_FORMAT", "pcm24").strip().lower()
    if fmt not in ("wav24", "pcm24", "mulaw8k"):
        fmt = "pcm24"
    return fmt


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    sample_rate: int = 24000
    emotion: Optional[str] = None
    max_duration_sec: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    # removed: speed, autocorrect (not effective)
    client_request_id: Optional[str] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def root_redirect():
    # Redirect to UI if available
    if os.path.isdir(UI_DIR):
        return RedirectResponse(url="/ui/")
    return {"status": "ok", "ui": False}


@app.get("/favicon.ico")
def favicon():
    path = os.path.join(UI_DIR, "favicon.ico")
    if os.path.isfile(path):
        return FileResponse(path)
    return Response(status_code=204)


# --------------- Model lifecycle --------------- #

ORPHEUS_MODEL = None  # type: ignore[var-annotated]


def _init_global_model():
    global ORPHEUS_MODEL
    if ORPHEUS_MODEL is not None:
        return ORPHEUS_MODEL
    # Force stable V0 engine by default unless explicitly overridden
    os.environ.setdefault("VLLM_USE_V1", "0")
    from orpheus_tts import OrpheusModel  # type: ignore
    model_dir = os.getenv("ORPHEUS_MODEL_DIR") or os.getenv("ORPHEUS_MODEL_ID", "canopylabs/orpheus-3b-0.1-ft")
    tokenizer_id = os.getenv("ORPHEUS_TOKENIZER_ID", "canopylabs/orpheus-3b-0.1-pretrained")
    # vLLM engine tuning to avoid OOM
    gpu_mem_util = float(os.getenv("VLLM_GPU_MEM_UTIL", "0.9"))
    max_num_seqs = int(os.getenv("VLLM_MAX_NUM_SEQS", "1"))
    enforce_eager = os.getenv("VLLM_ENFORCE_EAGER", "0") == "1"
    enable_chunked_prefill = os.getenv("VLLM_ENABLE_CHUNKED_PREFILL", "1") == "1"
    enable_prefix_caching = os.getenv("VLLM_ENABLE_PREFIX_CACHING", "0") == "1"
    kv_cache_dtype = os.getenv("VLLM_KV_CACHE_DTYPE")  # e.g., "fp8" or "auto"
    max_model_len = os.getenv("VLLM_MAX_MODEL_LEN")

    engine_kwargs = {
        "gpu_memory_utilization": gpu_mem_util,
        "max_num_seqs": max_num_seqs,
        "enforce_eager": enforce_eager,
        "enable_chunked_prefill": enable_chunked_prefill,
        "enable_prefix_caching": enable_prefix_caching,
    }
    if kv_cache_dtype:
        engine_kwargs["kv_cache_dtype"] = kv_cache_dtype
    if max_model_len:
        try:
            engine_kwargs["max_model_len"] = int(max_model_len)
        except Exception:
            pass

    ORPHEUS_MODEL = OrpheusModel(
        model_name=model_dir,
        dtype="auto",
        tokenizer=tokenizer_id,
        **engine_kwargs,
    )
    return ORPHEUS_MODEL


@app.on_event("startup")
def _on_startup():
    try:
        # Allow disabling TTS model initialization on startup (useful when running ASR-only)
        if _is_enabled("INIT_TTS_ON_STARTUP", "1"):
            model = _init_global_model()
            if os.getenv("ORPHEUS_PREWARM", "1") == "1":
                # Trigger minimal generation to load weights and capture CUDA graphs
                try:
                    voice = os.getenv("ORPHEUS_VOICE") or "zoe"
                    for _chunk in model.generate_speech(prompt="Warm up.", voice=voice):
                        break
                except Exception as e:
                    print("[prewarm] error:", e)
    except Exception as e:
        print("[startup] model init failed:", e)


def _synthesize_with_orpheus(
    text: str,
    sample_rate: int,
    voice: Optional[str] = None,
    emotion: Optional[str] = None,
    max_duration_sec: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> bytes:
    """Try to synthesize using Orpheus if available; else return silence."""
    try:
        # Prefer the orpheus_tts package from the cloned repo (orpheus-speech)
        from orpheus_tts import OrpheusModel  # type: ignore

        # Prefer local directory if provided, otherwise use HF repo ID
        model_dir = os.getenv("ORPHEUS_MODEL_DIR", "").strip() or None
        model_id = os.getenv("ORPHEUS_MODEL_ID", "canopylabs/orpheus-tts-0.1-finetune-prod")

        # Initialize or reuse global model
        model = _init_global_model()

        # Generate streaming audio chunks (int16 PCM bytes at 24 kHz)
        chunks: list[bytes] = []
        max_bytes_limit = (
            int(max_duration_sec) * sample_rate * 2
            if (max_duration_sec and max_duration_sec > 0)
            else None
        )
        # Try to use a voice if provided via param/env; fallback to 'zoe'
        req_voice = voice or os.getenv("ORPHEUS_VOICE") or "zoe"
        prompt_text = text
        # Rudimentary emotion steering: prepend hint to the text
        if emotion:
            prompt_text = f"emotion: {emotion}. " + text
        # Provide a unique request id and sampling params
        request_id = f"req-{uuid.uuid4().hex}"
        start_ts = time.time()
        gen_kwargs = {
            "prompt": prompt_text,
            "voice": req_voice,
            "request_id": request_id,
        }
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = repetition_penalty
        if max_tokens is not None:
            gen_kwargs["max_tokens"] = max_tokens
        # ensure updated prompt
        gen_kwargs["prompt"] = prompt_text

        for chunk in model.generate_speech(**gen_kwargs):
            if chunk:
                chunks.append(chunk)
            if max_bytes_limit is not None and sum(len(c) for c in chunks) > max_bytes_limit:
                break

        # Add a short trailing pad (80 ms) to avoid truncating sentence tails on playback
        pcm_bytes = b"".join(chunks)
        if pcm_bytes:
            try:
                tail_ms = int(os.getenv("ORPHEUS_TAIL_PAD_MS", "80"))
            except Exception:
                tail_ms = 80
            pad_samples = int((tail_ms / 1000.0) * 24000)
            if pad_samples > 0:
                pad_bytes = (np.zeros(pad_samples, dtype=np.int16)).tobytes()
                pcm_bytes = pcm_bytes + pad_bytes
        if not pcm_bytes:
            raise RuntimeError("Empty audio chunks from Orpheus")

        # Wrap PCM int16 into WAV
        import wave
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(pcm_bytes)
        wav_bytes = buffer.getvalue()
        # Compute stats
        elapsed_ms = int((time.time() - start_ts) * 1000)
        audio_seconds = len(pcm_bytes) / (2 * 24000)
        # Pull LLM token stats if available
        try:
            llm_tokens = int(getattr(model, 'last_llm_tokens', 0))
            llm_elapsed_ms = int(getattr(model, 'last_llm_elapsed_ms', elapsed_ms)) or elapsed_ms
            tokens_per_sec = (llm_tokens / (llm_elapsed_ms / 1000.0)) if llm_elapsed_ms > 0 else 0.0
        except Exception:
            llm_tokens = 0
            tokens_per_sec = 0.0
        # Persist to disk for download
        ts_label = time.strftime("%Y%m%d_%H%M%S", time.localtime(start_ts))
        safe_voice = (req_voice or "voice").replace("/", "_").replace(" ", "_")
        filename = f"{ts_label}_{request_id[:8]}_{safe_voice}.wav"
        file_path = os.path.join(AUDIO_DIR, filename)
        try:
            with open(file_path, "wb") as wf:
                wf.write(wav_bytes)
        except Exception:
            pass
        download_url = f"/media/{filename}"
        # Store recent for UI
        try:
            RECENT_SYNTH.append({
                "request_id": request_id,
                "voice": req_voice,
                "elapsed_ms": elapsed_ms,
                "audio_seconds": round(audio_seconds, 2),
                "bytes": len(wav_bytes),
                "tokens": llm_tokens,
                "tps": round(tokens_per_sec, 2),
                "ts": int(time.time()),
                "started_at": int(start_ts),
                "finished_at": int(time.time()),
                "filename": filename,
                "url": download_url,
            })
            # cap to last 25
            if len(RECENT_SYNTH) > 25:
                del RECENT_SYNTH[:-25]
        except Exception:
            pass
        # Attach stats to thread-local stash for headers
        _LAST_STATS["elapsed_ms"] = elapsed_ms
        _LAST_STATS["audio_seconds"] = audio_seconds
        _LAST_STATS["voice"] = req_voice or ""
        _LAST_STATS["request_id"] = request_id
        _LAST_STATS["download_url"] = download_url
        _LAST_STATS["tokens"] = llm_tokens
        _LAST_STATS["tps"] = tokens_per_sec
        return wav_bytes
    except Exception as e:
        print("[orpheus] synthesis error:", e)
        traceback.print_exc()

    # Audible sine-wave WAV (placeholder for testing audio path)
    import wave
    buffer = io.BytesIO()
    duration_sec = 1.5
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    freq = 440.0  # A4
    waveform = (0.2 * np.sin(2 * np.pi * freq * t))
    pcm_int16 = (waveform * 32767).astype(np.int16).tobytes()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16)
    return buffer.getvalue()


@app.post("/tts")
def tts(req: TTSRequest):
    # Ultravox external voice expects audio/wav bytes in response
    audio_wav = _synthesize_with_orpheus(
        req.text,
        req.sample_rate,
        req.voice,
        req.emotion,
        req.max_duration_sec,
        req.temperature,
        req.top_p,
        req.repetition_penalty,
        req.max_tokens,
    )
    headers = {
        "X-Gen-Ms": str(_LAST_STATS.get("elapsed_ms", 0)),
        "X-Audio-Seconds": f"{_LAST_STATS.get('audio_seconds', 0):.2f}",
        "X-Req-Id": req.client_request_id or _LAST_STATS.get("request_id", ""),
        "X-Voice": _LAST_STATS.get("voice", ""),
        "Access-Control-Expose-Headers": "X-Gen-Ms, X-Audio-Seconds, X-Req-Id, X-Voice, X-Download-URL",
        "Cache-Control": "no-store",
        "X-Download-URL": _LAST_STATS.get("download_url", ""),
        "X-Tokens": str(_LAST_STATS.get("tokens", 0)),
        "X-TPS": f"{_LAST_STATS.get('tps', 0.0):.2f}",
    }
    return Response(content=audio_wav, media_type="audio/wav", headers=headers)


@app.post("/tts-ultravox")
def tts_ultravox(body: dict):
    # Compatible with Ultravox generic externalVoice body: {"text": "..."}
    text = body.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Missing text")
    sample_rate = int(body.get("sample_rate", 24000))
    voice = body.get("voice") or os.getenv("ORPHEUS_VOICE") or "zoe"
    emotion = body.get("emotion")
    max_duration_sec = body.get("max_duration_sec")
    audio_wav = _synthesize_with_orpheus(
        text,
        sample_rate,
        voice,
        emotion,
        max_duration_sec,
        body.get("temperature"),
        body.get("top_p"),
        body.get("repetition_penalty"),
        body.get("max_tokens"),
    )
    headers = {
        "X-Gen-Ms": str(_LAST_STATS.get("elapsed_ms", 0)),
        "X-Audio-Seconds": f"{_LAST_STATS.get('audio_seconds', 0):.2f}",
        "X-Req-Id": _LAST_STATS.get("request_id", ""),
        "X-Voice": _LAST_STATS.get("voice", ""),
        "Access-Control-Expose-Headers": "X-Gen-Ms, X-Audio-Seconds, X-Req-Id, X-Voice, X-Download-URL",
        "Cache-Control": "no-store",
        "X-Download-URL": _LAST_STATS.get("download_url", ""),
        "X-Tokens": str(_LAST_STATS.get("tokens", 0)),
        "X-TPS": f"{_LAST_STATS.get('tps', 0.0):.2f}",
    }
    return Response(content=audio_wav, media_type="audio/wav", headers=headers)


# --------------- Convenience API for the Web UI --------------- #

DEFAULT_VOICES = ["zoe", "zac", "jess", "leo", "mia", "julia", "leah", "tara"]


@app.get("/api/voices")
def api_voices():
    env_voices = os.getenv("ORPHEUS_AVAILABLE_VOICES")
    if env_voices:
        try:
            voices = [v.strip() for v in env_voices.split(",") if v.strip()]
            if voices:
                return {"voices": voices}
        except Exception:
            pass
    return {"voices": DEFAULT_VOICES}


@app.get("/api/emotions")
def api_emotions():
    emotions_path = os.path.join(os.path.dirname(__file__), "Orpheus-TTS", "emotions.txt")
    emotions: list[str] = []
    try:
        if os.path.isfile(emotions_path):
            with open(emotions_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        emotions.append(line)
    except Exception:
        pass
    return {"emotions": emotions}


@app.get("/api/health-ext")
def api_health_ext():
    info: dict[str, object] = {"status": "ok"}
    try:
        import torch  # type: ignore
        info.update({
            "torch_version": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        })
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            info["cuda_device_name_0"] = torch.cuda.get_device_name(0)
            try:
                cap = torch.cuda.get_device_capability(0)
                info["cuda_device_capability_0"] = f"{cap[0]}.{cap[1]}"
            except Exception:
                pass
    except Exception:
        pass
    return info


@app.get("/api/capabilities")
def api_capabilities():
    import requests  # type: ignore

    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    def fetch_card(repo_id: str) -> str:
        base = f"https://huggingface.co/{repo_id}/raw/main/README.md"
        try:
            r = requests.get(base, headers=headers, timeout=15)
            if r.status_code == 200:
                # return trimmed content to avoid huge payloads
                text = r.text
                if len(text) > 20000:
                    text = text[:20000] + "\n\n... (truncated)"
                return text
            return f"Failed to fetch ({r.status_code})"
        except Exception as e:
            return f"Error: {e}"

    return {
        "finetune_model": "canopylabs/orpheus-3b-0.1-ft",
        "pretrained_tokenizer": "canopylabs/orpheus-3b-0.1-pretrained",
        "model_card_ft": fetch_card("canopylabs/orpheus-3b-0.1-ft"),
        "model_card_pretrained": fetch_card("canopylabs/orpheus-3b-0.1-pretrained"),
    }


@app.get("/api/tts-get")
def api_tts_get(
    text: str = Query(..., min_length=1),
    voice: Optional[str] = Query(None),
    sample_rate: int = Query(24000, ge=8000, le=48000),
    max_duration_sec: Optional[int] = Query(60, ge=1, le=600),
):
    audio_wav = _synthesize_with_orpheus(text, sample_rate, voice, None, max_duration_sec)
    headers = {
        "X-Gen-Ms": str(_LAST_STATS.get("elapsed_ms", 0)),
        "X-Audio-Seconds": f"{_LAST_STATS.get('audio_seconds', 0):.2f}",
        "X-Req-Id": _LAST_STATS.get("request_id", ""),
        "X-Voice": _LAST_STATS.get("voice", ""),
        "Access-Control-Expose-Headers": "X-Gen-Ms, X-Audio-Seconds, X-Req-Id, X-Voice, X-Download-URL, X-Tokens, X-TPS",
        "Cache-Control": "no-store",
        "X-Download-URL": _LAST_STATS.get("download_url", ""),
        "X-Tokens": str(_LAST_STATS.get("tokens", 0)),
        "X-TPS": f"{_LAST_STATS.get('tps', 0.0):.2f}",
    }
    return Response(content=audio_wav, media_type="audio/wav", headers=headers)


@app.post("/api/tts")
def api_tts(req: TTSRequest):
    if not _is_enabled("ENABLE_TTS_WAV24", "1"):
        raise HTTPException(status_code=404, detail="Endpoint disabled by config")
    audio_wav = _synthesize_with_orpheus(
        req.text,
        req.sample_rate,
        req.voice,
        req.emotion,
        req.max_duration_sec,
        req.temperature,
        req.top_p,
        req.repetition_penalty,
        req.max_tokens,
    )
    headers = {
        "X-Gen-Ms": str(_LAST_STATS.get("elapsed_ms", 0)),
        "X-Audio-Seconds": f"{_LAST_STATS.get('audio_seconds', 0):.2f}",
        "X-Req-Id": req.client_request_id or _LAST_STATS.get("request_id", ""),
        "X-Voice": _LAST_STATS.get("voice", ""),
        "X-Download-URL": _LAST_STATS.get("download_url", ""),
        "X-Tokens": str(_LAST_STATS.get("tokens", 0)),
        "X-TPS": f"{_LAST_STATS.get('tps', 0.0):.2f}",
        "Access-Control-Expose-Headers": "X-Gen-Ms, X-Audio-Seconds, X-Req-Id, X-Voice, X-Download-URL, X-Tokens, X-TPS",
        "Cache-Control": "no-store",
    }
    return Response(content=audio_wav, media_type="audio/wav", headers=headers)


@app.post("/api/tts-pcm")
def api_tts_pcm(
    text: str = Body(..., embed=True),
    voice: Optional[str] = Body(None),
    max_tokens: Optional[int] = Body(None),
    temperature: Optional[float] = Body(None),
    top_p: Optional[float] = Body(None),
    repetition_penalty: Optional[float] = Body(None),
    client_request_id: Optional[str] = Body(None),
    authorization: Optional[str] = Header(default=None, convert_underscores=False),
):
    # Log incoming request for debugging
    print(f"\n📞 [/api/tts-pcm] Received request - Text: {text[:100]}..." if len(text) > 100 else f"\n📞 [/api/tts-pcm] Received request - Text: {text}", flush=True)
    if not _is_enabled("ENABLE_TTS_PCM24", "1"):
        raise HTTPException(status_code=404, detail="Endpoint disabled by config")
    # Optional API key protection (disabled unless REQUIRE_LOCAL_API_KEY=1)
    if os.getenv("REQUIRE_LOCAL_API_KEY", "0") == "1":
        required_key = os.getenv("LOCAL_ORPHEUS_API_KEY") or os.getenv("ORPHEUS_EXTERNAL_API_KEY")
        if required_key:
            expected = f"Api-Key {required_key}"
            if authorization != expected:
                raise HTTPException(status_code=401, detail="Unauthorized")

    # Generate PCM int16 @ 24kHz, return as audio/l16 (no WAV header)
    wav_bytes = _synthesize_with_orpheus(
        text,
        24000,
        voice,
        None,
        None,
        temperature,
        top_p,
        repetition_penalty,
        max_tokens,
    )
    # Strip WAV header to yield raw PCM (16-bit, mono, 24kHz)
    # WAV header is 44 bytes for PCM mono 16-bit
    raw_pcm = b""
    try:
        if len(wav_bytes) > 44 and wav_bytes[:4] == b"RIFF" and wav_bytes[8:12] == b"WAVE":
            raw_pcm = wav_bytes[44:]
        else:
            # If it wasn't a WAV (fallback), assume already raw
            raw_pcm = wav_bytes
    except Exception:
        raw_pcm = wav_bytes

    headers = {
        "X-Gen-Ms": str(_LAST_STATS.get("elapsed_ms", 0)),
        "X-Audio-Seconds": f"{_LAST_STATS.get('audio_seconds', 0):.2f}",
        "X-Req-Id": client_request_id or _LAST_STATS.get("request_id", ""),
        "X-Voice": _LAST_STATS.get("voice", ""),
        "X-Download-URL": _LAST_STATS.get("download_url", ""),
        "X-Tokens": str(_LAST_STATS.get("tokens", 0)),
        "X-TPS": f"{_LAST_STATS.get('tps', 0.0):.2f}",
        "Access-Control-Expose-Headers": "X-Gen-Ms, X-Audio-Seconds, X-Req-Id, X-Voice, X-Download-URL, X-Tokens, X-TPS",
        "Cache-Control": "no-store",
    }
    return Response(content=raw_pcm, media_type="audio/l16", headers=headers)


# --- μ-law 8 kHz variants --- #

def _wav24k_to_mulaw8k(wav_bytes: bytes) -> bytes:
    """Convert a mono 16-bit 24k WAV payload to raw μ-law (PCMU) at 8 kHz."""
    try:
        # Strip WAV header if present (PCM mono 16-bit)
        if len(wav_bytes) > 44 and wav_bytes[:4] == b"RIFF" and wav_bytes[8:12] == b"WAVE":
            pcm24 = wav_bytes[44:]
        else:
            pcm24 = wav_bytes
        # Resample 24k -> 8k using audioop with stateful filter
        pcm8, _state = audioop.ratecv(pcm24, 2, 1, 24000, 8000, None)
        # Convert to μ-law (8-bit)
        mulaw = audioop.lin2ulaw(pcm8, 2)
        return mulaw
    except Exception:
        # Fallback: return original bytes if conversion fails
        return wav_bytes


@app.post("/api/tts-pcm8k-ulaw")
def api_tts_pcm8k_ulaw(
    text: str = Body(..., embed=True),
    voice: Optional[str] = Body(None),
    max_tokens: Optional[int] = Body(None),
    temperature: Optional[float] = Body(None),
    top_p: Optional[float] = Body(None),
    repetition_penalty: Optional[float] = Body(None),
    client_request_id: Optional[str] = Body(None),
    authorization: Optional[str] = Header(default=None, convert_underscores=False),
):
    if not _is_enabled("ENABLE_TTS_MULAW8K", "1"):
        raise HTTPException(status_code=404, detail="Endpoint disabled by config")
    if os.getenv("REQUIRE_LOCAL_API_KEY", "0") == "1":
        required_key = os.getenv("LOCAL_ORPHEUS_API_KEY") or os.getenv("ORPHEUS_EXTERNAL_API_KEY")
        if required_key:
            expected = f"Api-Key {required_key}"
            if authorization != expected:
                raise HTTPException(status_code=401, detail="Unauthorized")

    wav_bytes = _synthesize_with_orpheus(
        text,
        24000,
        voice,
        None,
        None,
        temperature,
        top_p,
        repetition_penalty,
        max_tokens,
    )
    mulaw_bytes = _wav24k_to_mulaw8k(wav_bytes)

    headers = {
        "X-Gen-Ms": str(_LAST_STATS.get("elapsed_ms", 0)),
        "X-Audio-Seconds": f"{_LAST_STATS.get('audio_seconds', 0):.2f}",
        "X-Req-Id": client_request_id or _LAST_STATS.get("request_id", ""),
        "X-Voice": _LAST_STATS.get("voice", ""),
        "X-Download-URL": _LAST_STATS.get("download_url", ""),
        "X-Tokens": str(_LAST_STATS.get("tokens", 0)),
        "X-TPS": f"{_LAST_STATS.get('tps', 0.0):.2f}",
        "X-Sample-Rate": "8000",
        "Access-Control-Expose-Headers": "X-Gen-Ms, X-Audio-Seconds, X-Req-Id, X-Voice, X-Download-URL, X-Tokens, X-TPS, X-Sample-Rate",
        "Cache-Control": "no-store",
    }
    # Many clients use audio/pcmu for μ-law; audio/basic is also common
    return Response(content=mulaw_bytes, media_type="audio/pcmu", headers=headers)


class ExternalTTSRequest(BaseModel):
    text: Optional[str] = None  # Made optional to support prompt field
    prompt: Optional[str] = None  # Baseten/Ultravox compatibility
    voice: Optional[str] = None
    # choose: wav24 | pcm24 | mulaw8k
    format: Optional[str] = None
    # advanced:
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    client_request_id: Optional[str] = None
    # optional API key header passed via normal header


@app.post("/api/tts-external")
def api_tts_external(
    req: ExternalTTSRequest,
    authorization: Optional[str] = Header(default=None, convert_underscores=False),
):
    """Single entrypoint controlled by ORPHEUS_TTS_DEFAULT_FORMAT or request.format."""
    required_key = os.getenv("LOCAL_ORPHEUS_API_KEY") or os.getenv("ORPHEUS_EXTERNAL_API_KEY")
    if required_key:
        expected = f"Api-Key {required_key}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Handle "prompt" field for Baseten/Ultravox compatibility
    if req.prompt and not req.text:
        req.text = req.prompt
    
    # Ensure we have text to synthesize
    if not req.text:
        raise HTTPException(status_code=400, detail="Either 'text' or 'prompt' field is required")

    fmt = (req.format or _get_default_tts_format()).lower()
    if fmt == "wav24":
        if not _is_enabled("ENABLE_TTS_WAV24", "1"):
            raise HTTPException(status_code=404, detail="Endpoint disabled by config")
        wav_bytes = _synthesize_with_orpheus(
            req.text, 24000, req.voice, None, None, req.temperature, req.top_p, req.repetition_penalty, req.max_tokens
        )
        headers = {
            "X-Gen-Ms": str(_LAST_STATS.get("elapsed_ms", 0)),
            "X-Audio-Seconds": f"{_LAST_STATS.get('audio_seconds', 0):.2f}",
            "X-Req-Id": req.client_request_id or _LAST_STATS.get("request_id", ""),
            "X-Voice": _LAST_STATS.get("voice", ""),
            "X-Download-URL": _LAST_STATS.get("download_url", ""),
            "X-Tokens": str(_LAST_STATS.get("tokens", 0)),
            "X-TPS": f"{_LAST_STATS.get('tps', 0.0):.2f}",
            "Access-Control-Expose-Headers": "X-Gen-Ms, X-Audio-Seconds, X-Req-Id, X-Voice, X-Download-URL, X-Tokens, X-TPS",
            "Cache-Control": "no-store",
        }
        return Response(content=wav_bytes, media_type="audio/wav", headers=headers)
    elif fmt == "mulaw8k":
        if not _is_enabled("ENABLE_TTS_MULAW8K", "1"):
            raise HTTPException(status_code=404, detail="Endpoint disabled by config")
        wav_bytes = _synthesize_with_orpheus(
            req.text, 24000, req.voice, None, None, req.temperature, req.top_p, req.repetition_penalty, req.max_tokens
        )
        mulaw_bytes = _wav24k_to_mulaw8k(wav_bytes)
        headers = {
            "X-Gen-Ms": str(_LAST_STATS.get("elapsed_ms", 0)),
            "X-Audio-Seconds": f"{_LAST_STATS.get('audio_seconds', 0):.2f}",
            "X-Req-Id": req.client_request_id or _LAST_STATS.get("request_id", ""),
            "X-Voice": _LAST_STATS.get("voice", ""),
            "X-Download-URL": _LAST_STATS.get("download_url", ""),
            "X-Tokens": str(_LAST_STATS.get("tokens", 0)),
            "X-TPS": f"{_LAST_STATS.get('tps', 0.0):.2f}",
            "X-Sample-Rate": "8000",
            "Access-Control-Expose-Headers": "X-Gen-Ms, X-Audio-Seconds, X-Req-Id, X-Voice, X-Download-URL, X-Tokens, X-TPS, X-Sample-Rate",
            "Cache-Control": "no-store",
        }
        return Response(content=mulaw_bytes, media_type="audio/pcmu", headers=headers)
    else:  # pcm24
        if not _is_enabled("ENABLE_TTS_PCM24", "1"):
            raise HTTPException(status_code=404, detail="Endpoint disabled by config")
        wav_bytes = _synthesize_with_orpheus(
            req.text, 24000, req.voice, None, None, req.temperature, req.top_p, req.repetition_penalty, req.max_tokens
        )
        # strip WAV header to raw PCM
        if len(wav_bytes) > 44 and wav_bytes[:4] == b"RIFF" and wav_bytes[8:12] == b"WAVE":
            raw_pcm = wav_bytes[44:]
        else:
            raw_pcm = wav_bytes
        headers = {
            "X-Gen-Ms": str(_LAST_STATS.get("elapsed_ms", 0)),
            "X-Audio-Seconds": f"{_LAST_STATS.get('audio_seconds', 0):.2f}",
            "X-Req-Id": req.client_request_id or _LAST_STATS.get("request_id", ""),
            "X-Voice": _LAST_STATS.get("voice", ""),
            "X-Download-URL": _LAST_STATS.get("download_url", ""),
            "X-Tokens": str(_LAST_STATS.get("tokens", 0)),
            "X-TPS": f"{_LAST_STATS.get('tps', 0.0):.2f}",
            "Access-Control-Expose-Headers": "X-Gen-Ms, X-Audio-Seconds, X-Req-Id, X-Voice, X-Download-URL, X-Tokens, X-TPS",
            "Cache-Control": "no-store",
        }
        return Response(content=raw_pcm, media_type="audio/l16", headers=headers)


@app.websocket("/ws/tts-mulaw")
async def ws_tts_mulaw(websocket: WebSocket):
    """Experimental: stream 8 kHz μ-law frames (20 ms) over WebSocket.

    Client protocol:
      1) Connect and send a JSON text message: {"text": "...", "voice": "zoe", ...}
      2) Server responds with binary PCMU frames of size 160 bytes every ~20 ms.
      3) Close on completion.
    """
    await websocket.accept()
    try:
        # Receive JSON payload
        init_text = await websocket.receive_text()
        try:
            payload = json.loads(init_text)
        except Exception:
            await websocket.close(code=1003)
            return

        text = (payload.get("text") or "").strip()
        if not text:
            await websocket.close(code=1003)
            return
        voice = payload.get("voice") or os.getenv("ORPHEUS_VOICE") or "zoe"
        temperature = payload.get("temperature")
        top_p = payload.get("top_p")
        repetition_penalty = payload.get("repetition_penalty")
        max_tokens = payload.get("max_tokens")

        model = _init_global_model()
        request_id = f"req-{uuid.uuid4().hex}"
        gen_kwargs = {"prompt": text, "voice": voice, "request_id": request_id}
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = repetition_penalty
        if max_tokens is not None:
            gen_kwargs["max_tokens"] = max_tokens

        frame_ms = int(os.getenv("MULAW_FRAME_MS", "20"))
        sr_out = 8000
        samples_per_frame = sr_out * frame_ms // 1000  # 160 for 20 ms
        bytes_per_frame = samples_per_frame  # μ-law: 1 byte per sample

        # Background producer: push 24k PCM chunks into a queue
        q: "queue.Queue[Optional[bytes]]" = queue.Queue()
        def _producer():
            try:
                for pcm_chunk in model.generate_speech(**gen_kwargs):
                    if pcm_chunk:
                        q.put(pcm_chunk)
            except Exception:
                pass
            finally:
                q.put(None)

        threading.Thread(target=_producer, daemon=True).start()

        # Stateful resampler
        state = None
        mulaw_buf = bytearray()
        next_send_time = time.monotonic()

        loop = asyncio.get_running_loop()
        while True:
            chunk = await loop.run_in_executor(None, q.get)
            if chunk is None:
                break
            try:
                pcm8, state = audioop.ratecv(chunk, 2, 1, 24000, sr_out, state)
                mulaw = audioop.lin2ulaw(pcm8, 2)
                mulaw_buf.extend(mulaw)
                while len(mulaw_buf) >= bytes_per_frame:
                    frame = bytes(mulaw_buf[:bytes_per_frame])
                    del mulaw_buf[:bytes_per_frame]
                    # Pace in real time
                    now = time.monotonic()
                    if now < next_send_time:
                        await asyncio.sleep(next_send_time - now)
                    await websocket.send_bytes(frame)
                    next_send_time += frame_ms / 1000.0
            except Exception:
                break

        # Flush remainder (pad to full frame if needed)
        if mulaw_buf:
            pad_last = os.getenv("MULAW_PAD_LAST", "1") == "1"
            if pad_last and len(mulaw_buf) % bytes_per_frame != 0:
                pad_len = bytes_per_frame - (len(mulaw_buf) % bytes_per_frame)
                mulaw_buf.extend(b"\xFF" * pad_len)  # μ-law silence is 0xFF
            while len(mulaw_buf) >= bytes_per_frame:
                frame = bytes(mulaw_buf[:bytes_per_frame])
                del mulaw_buf[:bytes_per_frame]
                now = time.monotonic()
                if now < next_send_time:
                    await asyncio.sleep(next_send_time - now)
                await websocket.send_bytes(frame)
                next_send_time += frame_ms / 1000.0
    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


# Recent synth history for UI
RECENT_SYNTH: list[dict] = []
_LAST_STATS: dict[str, object] = {}


@app.get("/api/recent")
def api_recent():
    return {"items": RECENT_SYNTH[-50:]}


