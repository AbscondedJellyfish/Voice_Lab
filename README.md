
# VoiceLab

A local server for cloning, blending, and synthesising voices. Run it, open the browser, and use the UI — or call it from Python scripts using the included client.

---

## Getting started

Install the dependencies:

```bash
pip install -r requirements.txt
```

Log into HuggingFace so the model weights can download (one-time setup):

```bash
huggingface-cli login
```

> You also need to accept the terms at https://huggingface.co/kyutai/pocket-tts before logging in.

Then start the server:

```bash
python server.py
```

The first launch downloads the model — this takes a few minutes. After that it starts instantly from cache.

- **Browser UI** → http://localhost:9007
- **API docs** → http://localhost:9007/docs

---

## Locking it down with an API key

By default the server is open — fine for local use. If you want to protect it, set an environment variable before starting:

```bash
# Windows
set VOICELAB_API_KEY=your-secret-key
python server.py

# macOS / Linux
VOICELAB_API_KEY=your-secret-key python server.py
```

Any API call will then need the header `X-API-Key: your-secret-key`. The browser UI works as normal without it.

---

## What the API does

Every endpoint accepts and returns JSON (except voice cloning, which needs a file upload). Here's a plain-English rundown:

### Check if the model is ready
`GET /api/status`

Tells you whether the model has finished loading and whether voice cloning is available.

---

### List saved voices
`GET /api/voices`

Returns the names of all voices you've saved to your arsenal.

---

### Clone a voice from audio
`POST /api/clone` — send as `multipart/form-data`, field name `audio`

Upload a WAV, MP3, FLAC, or OGG file (up to 30 seconds is used). The server encodes it and hands back a `temp_id` you can use to generate speech or save it.

---

### Blend two or more voices together
`POST /api/fuse`

```json
{
  "voices": ["ciri", "pana"],
  "weights": [70, 30]
}
```

Mixes saved voices by weight and gives back a `temp_id`. The weights just need to be positive numbers — they're normalised automatically.

---

### Generate speech (quick)
`POST /api/test`

```json
{
  "voice_id": "ciri",
  "text": "Hello, this is a test.",
  "is_temp": false
}
```

Set `is_temp: true` if you're passing a `temp_id` instead of a saved voice name. Returns a WAV file.

---

### Generate speech (with controls)
`POST /api/synthesize`

Same as above but with extra knobs. Returns a WAV file plus a `temp_id` header (`X-Temp-Id`) so you can save the modified voice afterwards.

```json
{
  "voice_id": "ciri",
  "text": "Hello, this is a test.",
  "is_temp": false,
  "temperature": 0.7,
  "eos_threshold": -4.0,
  "lsd_steps": 1,
  "speed": 1.0,
  "pitch": 0.0,
  "voice_scale": 1.0,
  "voice_trim": 1.0
}
```

| Parameter | What it does | Range |
|-----------|-------------|-------|
| `temperature` | How expressive/varied the delivery is | 0.05 – 3.0 |
| `eos_threshold` | How much trailing speech there is | -12 – 2 |
| `lsd_steps` | Quality vs. speed (higher = slower but better) | 1 – 8 |
| `speed` | How fast or slow the speech is | 0.25 – 3.0 |
| `pitch` | Pitch shift in semitones | -12 – 12 |
| `voice_scale` | How strongly the voice character comes through | 0.1 – 4.0 |
| `voice_trim` | How much of the reference clip is used | 0.05 – 1.0 |

---

### Save a voice
`POST /api/save`

```json
{
  "voice_id": "3f2a1b...",
  "name": "my_voice",
  "is_temp": true
}
```

Saves a temp voice (or an existing one under a new name) to the arsenal. Names can contain letters, numbers, spaces, underscores, hyphens, and parentheses.

---

### Delete a voice
`DELETE /api/voices/{name}`

```
DELETE /api/voices/my_voice
```

Permanently removes a voice from the arsenal.

---

## Using the Python client

If you want to drive the server from a script, `voicelab_client.py` has you covered:

```python
from voicelab_client import VoiceLabClient

client = VoiceLabClient("http://localhost:9007", api_key="your-secret-key")
# Leave out api_key if you haven't set one
```

**Clone a voice and save it**

```python
temp_id = client.clone("recording.wav")
client.save(temp_id, "my_voice")
```

**Generate speech**

```python
wav = client.test("my_voice", "Hello world.", is_temp=False)
with open("output.wav", "wb") as f:
    f.write(wav)
```

**Generate with custom settings and save the result**

```python
wav, temp_id = client.synthesize(
    "my_voice",
    "Hello world.",
    speed=1.2,
    pitch=2.0,
)
with open("output.wav", "wb") as f:
    f.write(wav)

client.save(temp_id, "my_voice_fast")  # save the modified version
```

**Mix voices together**

```python
temp_id = client.fuse(["ciri", "pana"], [60, 40])
wav = client.test(temp_id, "This is a blended voice.")
```

Any error from the server comes back as a `VoiceLabError` with a plain-English message.
