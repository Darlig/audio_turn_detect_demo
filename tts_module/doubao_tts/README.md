# Doubao TTS README

This directory contains a lightweight Volcengine/Doubao bidirectional TTS
adapter for LiveKit Agents.

## Files

- `volcengine_tts.py`: LiveKit `tts.TTS` adapter for Volcengine/Doubao TTS.
- `turn_metrics.py`: latency metric helpers used by the adapter.
- `.env`: local TTS credentials, voice, and audio settings.
- `requirements.txt`: Python dependencies needed by this adapter.

## API

- Provider: Volcengine OpenSpeech / Doubao TTS
- API style: bidirectional WebSocket TTS
- Default WebSocket URL: `wss://openspeech.bytedance.com/api/v3/tts/bidirection`
- Default resource ID: `seed-tts-2.0`
- Default audio encoding: `pcm`
- Default sample rate: `24000`
- Default UID: `livekit-user`

## Environment

Put credentials, voice, and audio settings in `doubao_tts/.env`.

Required voice setting:

```bash
VOLC_TTS_VOICE_TYPE=...
```

Authentication supports either API key auth:

```bash
VOLC_TTS_API_KEY=...
```

or app/access key auth:

```bash
VOLC_TTS_APP_ID=...
VOLC_TTS_ACCESS_KEY=...
```

Optional:

```bash
VOLC_TTS_WS_URL=wss://openspeech.bytedance.com/api/v3/tts/bidirection
VOLC_TTS_RESOURCE_ID=seed-tts-2.0
VOLC_TTS_SAMPLE_RATE=24000
VOLC_TTS_ENCODING=pcm
VOLC_TTS_SPEED_RATIO=1.0
VOLC_TTS_UID=livekit-user
VOLC_TTS_STREAM_SEND_MODE=chunk
VOLC_TTS_STREAM_MIN_CHARS=6
VOLC_TTS_STREAM_MAX_CHARS=48
VOLC_TTS_LOG_INPUT_CHUNKS=0
```

## LiveKit Integration

Install dependencies:

```bash
python -m pip install -r doubao_tts/requirements.txt
```

Load environment variables before starting the LiveKit worker, then pass
`VolcengineStreamingTTS` to `AgentSession`:

```python
from livekit.agents import AgentSession
from doubao_tts import VolcengineStreamingTTS

session = AgentSession(
    tts=VolcengineStreamingTTS(),
)
```

`VolcengineStreamingTTS.stream()` opens the bidirectional TTS WebSocket, starts a
connection and session, receives text chunks from LiveKit, sends them as
`EVENT_TASK_REQUEST` frames, then forwards returned PCM audio payloads to
LiveKit through `output_emitter.push(...)`.

`VOLC_TTS_STREAM_SEND_MODE=chunk` sends each LLM text chunk directly. Setting it
to `segment` buffers text until punctuation or `VOLC_TTS_STREAM_MAX_CHARS`.
