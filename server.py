import asyncio
import io
import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft as _stft, istft as _istft, resample as _resample
import torch
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from safetensors.torch import load_file, save_file

BASE_DIR = Path(__file__).resolve().parent
VOICES_DIR = BASE_DIR / "voice_safetensors"
TEMP_DIR = BASE_DIR / "temp_voices"
TEMP_DIR.mkdir(exist_ok=True)
VOICES_DIR.mkdir(exist_ok=True)
MAX_TEXT_LEN = 800
API_KEY = os.getenv("VOICELAB_API_KEY", "")

executor = ThreadPoolExecutor(max_workers=1)

_model = None


def get_model():
    global _model
    if _model is None:
        from pocket_tts import TTSModel
        print("[VoiceLab] Loading TTS model...")
        _model = TTSModel.load_model()
        print("[VoiceLab] Model ready.")
    return _model


@asynccontextmanager
async def lifespan(_app: FastAPI):
    loop = asyncio.get_running_loop()
    loop.run_in_executor(executor, get_model)
    cutoff = time.time() - 86400
    for f in TEMP_DIR.glob("*.safetensors"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass
    yield


app = FastAPI(title="VoiceLab", lifespan=lifespan)


# ── Request models ──────────────────────────────────────────────────────────

class FuseRequest(BaseModel):
    voices: list[str]
    weights: list[float]

class TestRequest(BaseModel):
    voice_id: str
    text: str
    is_temp: bool = True

class SaveRequest(BaseModel):
    voice_id: str
    name: str
    is_temp: bool = True

class SynthRequest(BaseModel):
    voice_id: str
    text: str
    is_temp: bool = False
    temperature: float = 0.7
    eos_threshold: float = -4.0
    lsd_steps: int = 1
    speed: float = 1.0
    pitch: float = 0.0
    voice_scale: float = 1.0
    voice_trim: float = 1.0


# ── Auth dependency ──────────────────────────────────────────────────────────

def _check_api_key(x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Invalid or missing X-API-Key header")


# ── DSP helpers ───────────────────────────────────────────────────────────────

@torch.no_grad()
def _prompt_to_state(model, prompt: torch.Tensor) -> dict:
    from pocket_tts.modules.stateful_module import init_states
    if model.flow_lm.insert_bos_before_voice:
        prompt = torch.cat([model.flow_lm.bos_before_voice, prompt], dim=1)
    state = init_states(model.flow_lm, batch_size=1, sequence_length=prompt.shape[1])
    model._run_flow_lm_and_increment_step(model_state=state, audio_conditioning=prompt)
    return state


def _clone_sync(audio_path: Path) -> str:
    model = get_model()
    if not model.has_voice_cloning:
        raise RuntimeError(
            "Voice cloning weights are not available. "
            "Accept terms at https://huggingface.co/kyutai/pocket-tts "
            "then run: python -m huggingface_hub.commands.huggingface_cli login"
        )
    from pocket_tts.data.audio import audio_read
    from pocket_tts.data.audio_utils import convert_audio

    audio, sr = audio_read(audio_path)
    audio = convert_audio(audio, sr, model.config.mimi.sample_rate, 1)
    max_samples = int(30 * model.config.mimi.sample_rate)
    if audio.shape[-1] > max_samples:
        audio = audio[..., :max_samples]
    with torch.no_grad():
        prompt = model._encode_audio(audio.unsqueeze(0).to(model.device))

    temp_id = str(uuid.uuid4())
    temp_path = TEMP_DIR / f"{temp_id}.safetensors"
    save_file({"audio_prompt": prompt.cpu().contiguous()}, str(temp_path))
    return temp_id


def _generate_sync(voice_path: Path, text: str) -> bytes:
    model = get_model()
    tensors = load_file(str(voice_path))
    prompt = tensors["audio_prompt"].to(model.device)
    state = _prompt_to_state(model, prompt)
    audio = model.generate_audio(state, text)
    buf = io.BytesIO()
    sf.write(buf, audio.cpu().numpy(), model.sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()


def _phase_vocoder_stretch(audio: np.ndarray, rate: float, n_fft: int = 2048) -> np.ndarray:
    """Time-stretch audio by rate using phase vocoder. Preserves pitch, changes duration."""
    if abs(rate - 1.0) < 0.001:
        return audio
    hop = n_fft // 4
    _, _, D = _stft(audio, nperseg=n_fft, noverlap=n_fft - hop, window='hann')
    n_freq, n_frames = D.shape
    n_out = max(1, int(round(n_frames * rate)))
    D_out = np.zeros((n_freq, n_out), dtype=np.complex64)
    exp_adv = 2.0 * np.pi * np.arange(n_freq) * hop / n_fft
    phase_acc = np.angle(D[:, 0])
    for i in range(n_out):
        in_pos = i / rate
        i0 = int(in_pos)
        i1 = min(i0 + 1, n_frames - 1)
        alpha = in_pos - i0
        col = (1.0 - alpha) * D[:, i0] + alpha * D[:, i1]
        if i > 0:
            pp = (i - 1) / rate
            p0, p1 = int(pp), min(int(pp) + 1, n_frames - 1)
            prev = (1.0 - (pp - p0)) * D[:, p0] + (pp - p0) * D[:, p1]
            dp = np.angle(col) - np.angle(prev) - exp_adv
            dp -= 2.0 * np.pi * np.round(dp / (2.0 * np.pi))
            phase_acc += exp_adv + dp
        D_out[:, i] = np.abs(col) * np.exp(1j * phase_acc)
    _, y = _istft(D_out, nperseg=n_fft, noverlap=n_fft - hop, window='hann')
    return y.astype(np.float32)


def _pitch_shift(audio: np.ndarray, semitones: float) -> np.ndarray:
    """Shift pitch by semitones without changing duration."""
    if abs(semitones) < 0.05:
        return audio
    factor = 2.0 ** (semitones / 12.0)
    orig_len = len(audio)
    new_len = max(4, int(round(orig_len / factor)))
    audio_r = _resample(audio, new_len)
    stretched = _phase_vocoder_stretch(audio_r, orig_len / new_len)
    if len(stretched) > orig_len:
        return stretched[:orig_len]
    if len(stretched) < orig_len:
        return np.pad(stretched, (0, orig_len - len(stretched)))
    return stretched


def _synthesize_sync(
    voice_path: Path,
    text: str,
    temperature: float,
    eos_threshold: float,
    lsd_steps: int,
    speed: float,
    pitch: float,
    voice_scale: float,
    voice_trim: float,
) -> tuple[bytes, str]:
    model = get_model()

    # Load and manipulate the voice tensor
    tensors = load_file(str(voice_path))
    prompt = tensors["audio_prompt"].float()

    # Trim: use only the first N% of temporal frames
    if voice_trim < 1.0:
        n_frames = max(1, int(prompt.shape[1] * voice_trim))
        prompt = prompt[:, :n_frames, :]

    # Scale: multiply embedding values
    if voice_scale != 1.0:
        prompt = prompt * voice_scale

    # Save modified tensor to temp (enables "Save Voice" after generation)
    temp_id = str(uuid.uuid4())
    temp_path = TEMP_DIR / f"{temp_id}.safetensors"
    save_file({"audio_prompt": prompt.contiguous()}, str(temp_path))

    # Temporarily override model generation params (safe: single-threaded executor)
    orig_temp = model.temp
    orig_eos = model.eos_threshold
    orig_lsd = model.lsd_decode_steps
    try:
        model.temp = temperature
        model.eos_threshold = eos_threshold
        model.lsd_decode_steps = lsd_steps

        tmp_tensors = load_file(str(temp_path))
        state = _prompt_to_state(model, tmp_tensors["audio_prompt"].to(model.device))
        audio = model.generate_audio(state, text)
        audio_np = audio.cpu().numpy()
    finally:
        model.temp = orig_temp
        model.eos_threshold = orig_eos
        model.lsd_decode_steps = orig_lsd

    # Speed: resample output audio
    if abs(speed - 1.0) > 0.01:
        orig_len = len(audio_np)
        target_len = max(1, int(orig_len / speed))
        audio_np = _resample(audio_np, target_len)

    # Pitch: phase vocoder shift (preserves duration)
    if abs(pitch) > 0.05:
        audio_np = _pitch_shift(audio_np, pitch)

    buf = io.BytesIO()
    sf.write(buf, audio_np, model.sample_rate, format="WAV")
    buf.seek(0)
    return buf.read(), temp_id


def weighted_fuse(voice_paths: list, weights: list) -> dict:
    total = sum(weights)
    weights = [w / total for w in weights]
    prompts = [load_file(str(p))["audio_prompt"].float() for p in voice_paths]
    min_len = min(p.shape[1] for p in prompts)
    result = sum(w * p[:, :min_len, :] for w, p in zip(weights, prompts))
    return {"audio_prompt": result.contiguous()}


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=(BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8"))


@app.get("/api/status", dependencies=[Depends(_check_api_key)])
async def status():
    return {
        "model_loaded": _model is not None,
        "voice_cloning": _model.has_voice_cloning if _model else None,
    }


@app.get("/api/voices", dependencies=[Depends(_check_api_key)])
async def list_voices():
    voices = sorted(f.stem for f in VOICES_DIR.glob("*.safetensors"))
    return {"voices": voices}


@app.post("/api/clone", dependencies=[Depends(_check_api_key)])
async def clone_voice(audio: UploadFile = File(...)):
    suffix = Path(audio.filename).suffix or ".wav"
    tmp_audio = TEMP_DIR / f"{uuid.uuid4()}{suffix}"
    try:
        tmp_audio.write_bytes(await audio.read())
        loop = asyncio.get_running_loop()
        temp_id = await loop.run_in_executor(executor, _clone_sync, tmp_audio)
        return {"temp_id": temp_id}
    finally:
        if tmp_audio.exists():
            tmp_audio.unlink()


@app.post("/api/fuse", dependencies=[Depends(_check_api_key)])
async def fuse_voices(body: FuseRequest):
    if len(body.voices) < 2:
        raise HTTPException(400, "Need at least 2 voices to fuse")
    if len(body.voices) != len(body.weights):
        raise HTTPException(400, "voices and weights length mismatch")
    if all(w == 0 for w in body.weights):
        raise HTTPException(400, "All weights are zero")

    paths = []
    for name in body.voices:
        p = VOICES_DIR / f"{name}.safetensors"
        if not p.exists():
            raise HTTPException(404, f"Voice '{name}' not found")
        paths.append(p)

    fused = weighted_fuse(paths, body.weights)
    temp_id = str(uuid.uuid4())
    save_file(fused, str(TEMP_DIR / f"{temp_id}.safetensors"))
    return {"temp_id": temp_id}


@app.post("/api/test", dependencies=[Depends(_check_api_key)])
async def test_voice(body: TestRequest):
    path = TEMP_DIR / f"{body.voice_id}.safetensors" if body.is_temp else VOICES_DIR / f"{body.voice_id}.safetensors"

    if not path.exists():
        raise HTTPException(404, "Voice not found")
    if not body.text.strip():
        raise HTTPException(400, "Text cannot be empty")
    if len(body.text.strip()) > MAX_TEXT_LEN:
        raise HTTPException(400, f"Text too long (max {MAX_TEXT_LEN} characters)")

    loop = asyncio.get_running_loop()
    wav_bytes = await loop.run_in_executor(executor, _generate_sync, path, body.text.strip())

    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=output.wav"},
    )


@app.post("/api/save", dependencies=[Depends(_check_api_key)])
async def save_voice(body: SaveRequest):
    safe_name = "".join(c for c in body.name.strip() if c.isalnum() or c in " _-()").strip()
    if not safe_name:
        raise HTTPException(400, "Invalid voice name")

    src = TEMP_DIR / f"{body.voice_id}.safetensors" if body.is_temp else VOICES_DIR / f"{body.voice_id}.safetensors"

    if not src.exists():
        raise HTTPException(404, "Voice not found")

    dest = VOICES_DIR / f"{safe_name}.safetensors"
    shutil.copy2(str(src), str(dest))
    return {"name": safe_name}


@app.delete("/api/voices/{name}", dependencies=[Depends(_check_api_key)])
async def delete_voice(name: str):
    path = VOICES_DIR / f"{name}.safetensors"
    if path.resolve().parent != VOICES_DIR.resolve():
        raise HTTPException(400, "Invalid voice name")
    if not path.exists():
        raise HTTPException(404, "Voice not found")
    path.unlink()
    return {"deleted": name}


@app.post("/api/synthesize", dependencies=[Depends(_check_api_key)])
async def synthesize_voice(body: SynthRequest):
    src = TEMP_DIR / f"{body.voice_id}.safetensors" if body.is_temp else VOICES_DIR / f"{body.voice_id}.safetensors"

    if not src.exists():
        raise HTTPException(404, "Voice not found")
    if not body.text.strip():
        raise HTTPException(400, "Text cannot be empty")
    if len(body.text.strip()) > MAX_TEXT_LEN:
        raise HTTPException(400, f"Text too long (max {MAX_TEXT_LEN} characters)")

    temperature = max(0.05, min(body.temperature, 3.0))
    eos_threshold = max(-12.0, min(body.eos_threshold, 2.0))
    lsd_steps = max(1, min(body.lsd_steps, 8))
    speed = max(0.25, min(body.speed, 3.0))
    pitch = max(-12.0, min(body.pitch, 12.0))
    voice_scale = max(0.1, min(body.voice_scale, 4.0))
    voice_trim = max(0.05, min(body.voice_trim, 1.0))

    loop = asyncio.get_running_loop()
    wav_bytes, temp_id = await loop.run_in_executor(
        executor,
        _synthesize_sync,
        src, body.text.strip(), temperature, eos_threshold, lsd_steps, speed, pitch, voice_scale, voice_trim,
    )

    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "inline; filename=synth_output.wav",
            "X-Temp-Id": temp_id,
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9007, reload=False)
