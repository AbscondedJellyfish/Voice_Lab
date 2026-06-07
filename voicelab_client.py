"""
VoiceLabClient — Python client for the VoiceLab server.

Quick start:
    from voicelab_client import VoiceLabClient

    client = VoiceLabClient("http://localhost:9007", api_key="your-key")

    # Clone a voice from audio
    temp_id = client.clone("recording.wav")
    client.save(temp_id, "my_voice")

    # Generate speech
    wav, _ = client.synthesize("my_voice", "Hello world", speed=1.1, pitch=2.0)
    with open("output.wav", "wb") as f:
        f.write(wav)

    # Fuse two saved voices 60/40
    temp_id = client.fuse(["ciri", "pana"], [60, 40])
    wav = client.test(temp_id, "This is a fused voice.", is_temp=True)
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import requests


class VoiceLabError(Exception):
    """Raised when the server returns an error response."""


class VoiceLabClient:
    def __init__(self, base_url: str = "http://localhost:9007", api_key: str = ""):
        """
        Args:
            base_url: Root URL of the VoiceLab server.
            api_key:  Value for the X-API-Key header. Leave empty if the server
                      has no VOICELAB_API_KEY env var set.
        """
        self._base = base_url.rstrip("/")
        self._session = requests.Session()
        if api_key:
            self._session.headers["X-API-Key"] = api_key

    # ── Internal helpers ────────────────────────────────────────────────────

    def _get(self, path: str) -> dict:
        r = self._session.get(f"{self._base}{path}")
        self._raise(r)
        return r.json()

    def _post(self, path: str, body: dict) -> requests.Response:
        r = self._session.post(
            f"{self._base}{path}",
            json=body,
            headers={"Content-Type": "application/json"},
        )
        self._raise(r)
        return r

    def _raise(self, r: requests.Response) -> None:
        if not r.ok:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise VoiceLabError(f"HTTP {r.status_code}: {detail}")

    # ── API methods ─────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Return model load status.

        Returns:
            {"model_loaded": bool, "voice_cloning": bool | None}
        """
        return self._get("/api/status")

    def list_voices(self) -> list[str]:
        """Return sorted list of saved voice names in the arsenal."""
        return self._get("/api/voices")["voices"]

    def clone(self, audio_path: str | Path) -> str:
        """Encode an audio file into a temporary voice embedding.

        Args:
            audio_path: Path to any audio file (wav, mp3, flac, ogg).

        Returns:
            temp_id string — pass to test(), synthesize(), or save().
        """
        audio_path = Path(audio_path)
        with audio_path.open("rb") as f:
            r = self._session.post(
                f"{self._base}/api/clone",
                files={"audio": (audio_path.name, f)},
            )
        self._raise(r)
        return r.json()["temp_id"]

    def fuse(self, voices: list[str], weights: list[float]) -> str:
        """Blend saved voices by weight into a new temporary embedding.

        Args:
            voices:  List of saved voice names (must exist in arsenal).
            weights: Corresponding blend weights (any positive numbers; auto-normalised).

        Returns:
            temp_id string.
        """
        r = self._post("/api/fuse", {"voices": voices, "weights": weights})
        return r.json()["temp_id"]

    def test(self, voice_id: str, text: str, is_temp: bool = True) -> bytes:
        """Generate speech with default model parameters.

        Args:
            voice_id: Voice name (arsenal) or temp_id.
            text:     Text to synthesise (max 800 chars).
            is_temp:  True if voice_id is a temp_id, False for an arsenal name.

        Returns:
            WAV audio as bytes.
        """
        r = self._post("/api/test", {"voice_id": voice_id, "text": text, "is_temp": is_temp})
        return r.content

    def synthesize(
        self,
        voice_id: str,
        text: str,
        *,
        is_temp: bool = False,
        temperature: float = 0.7,
        eos_threshold: float = -4.0,
        lsd_steps: int = 1,
        speed: float = 1.0,
        pitch: float = 0.0,
        voice_scale: float = 1.0,
        voice_trim: float = 1.0,
    ) -> tuple[bytes, str]:
        """Generate speech with full parameter control.

        Args:
            voice_id:      Voice name (arsenal) or temp_id.
            text:          Text to synthesise (max 800 chars).
            is_temp:       True if voice_id is a temp_id.
            temperature:   Expressiveness / randomness (0.05–3.0, default 0.7).
            eos_threshold: Trailing speech length (−12–2, default −4.0).
            lsd_steps:     Decode quality steps (1–8, default 1).
            speed:         Playback rate (0.25–3.0, default 1.0).
            pitch:         Semitone shift (−12–12, default 0.0).
            voice_scale:   Embedding intensity (0.1–4.0, default 1.0).
            voice_trim:    Fraction of reference sample used (0.05–1.0, default 1.0).

        Returns:
            (wav_bytes, temp_id) — temp_id can be passed to save().
        """
        r = self._post("/api/synthesize", {
            "voice_id": voice_id,
            "text": text,
            "is_temp": is_temp,
            "temperature": temperature,
            "eos_threshold": eos_threshold,
            "lsd_steps": lsd_steps,
            "speed": speed,
            "pitch": pitch,
            "voice_scale": voice_scale,
            "voice_trim": voice_trim,
        })
        return r.content, r.headers.get("X-Temp-Id", "")

    def save(self, voice_id: str, name: str, is_temp: bool = True) -> str:
        """Persist a voice (temp or saved) to the arsenal under a new name.

        Args:
            voice_id: temp_id or existing arsenal voice name.
            name:     Desired name (alphanumeric + spaces, underscores, hyphens, parens).
            is_temp:  True if voice_id is a temp_id.

        Returns:
            Sanitised name actually saved.
        """
        r = self._post("/api/save", {"voice_id": voice_id, "name": name, "is_temp": is_temp})
        return r.json()["name"]

    def delete(self, name: str) -> None:
        """Remove a voice from the arsenal permanently.

        Args:
            name: Arsenal voice name to delete.
        """
        r = self._session.delete(f"{self._base}/api/voices/{quote(name, safe='')}")
        self._raise(r)
