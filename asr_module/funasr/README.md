# FunASR README

This directory contains the FunASR WebSocket ASR service, client protocol helper,
and optional server-side LiveKit semantic end-of-utterance extension used by the
LiveKit turn detector demo.

## Files

- `funasr_2pass_server.py`: FunASR Paraformer 2-pass WebSocket service.
- `funasr_ws_client.py`: asynchronous WebSocket client for the service.
- `funasr_protocol.py`: shared start/stop message helpers and ASR result parser.
- `semantic_turn.py`: optional LiveKit multilingual semantic turn detector used
  when `--semantic-turn-detector` is enabled.
- `requirements.txt`: Python dependencies for this directory.

## Environment

Install dependencies:

```bash
python -m pip install -r funasr/requirements.txt
```

The original demo environment was:

```bash
/home/weiy/environment/mambaforge/envs/turn_detector
```

Activate it:

```bash
/home/weiy/environment/mambaforge/condabin/mamba activate turn_detector
```

## Start FunASR 2-Pass Service

Because this delivery directory is named `funasr`, run the server from inside
the directory so Python can still import the third-party `funasr` package:

```bash
cd funasr
```

CPU:

```bash
PYTHONUNBUFFERED=1 python funasr_2pass_server.py --host 127.0.0.1 --port 10095 --device cpu --ngpu 0 2>&1 | tee -a ../logs/funasr.log
```

CUDA:

```bash
PYTHONUNBUFFERED=1 python funasr_2pass_server.py --host 127.0.0.1 --port 10095 --device cuda --ngpu 1 2>&1 | tee -a ../logs/funasr.log
```

Optional server-side LiveKit semantic turn detection on online partial ASR:

```bash
PYTHONUNBUFFERED=1 python funasr_2pass_server.py --host 0.0.0.0 --port 10095 --semantic-turn-detector 2>&1 | tee -a ../logs/funasr.log
```

## Models

The first run downloads FunASR models through the normal FunASR/ModelScope cache
mechanism unless local model paths are provided.

Defaults:

- online: `paraformer-zh-streaming`
- offline: `paraformer-zh`
- VAD: `fsmn-vad`
- punctuation: `ct-punc`

Override with environment variables or matching CLI flags:

```bash
FUNASR_ONLINE_MODEL=/path/to/paraformer-zh-streaming
FUNASR_OFFLINE_MODEL=/path/to/paraformer-zh
FUNASR_VAD_MODEL=/path/to/fsmn-vad
FUNASR_PUNC_MODEL=/path/to/ct-punc
```

The service passes `disable_update=True` to FunASR `AutoModel`, so startup does
not perform the FunASR package update check.

## Client Integration

The service defaults to `FUNASR_STREAMING_VAD=true`. In that mode it keeps one
continuous websocket audio stream open, uses FunASR `fsmn-vad` streaming
endpointing to cut speech segments, emits `2pass-online` partial results during
speech, runs the offline ASR model for the detected segment at speech end, and
then clears the segment cache. The stop message remains a manual flush/end-input
fallback.

`FunASRWebSocketClient` connects to the service, sends a start message, streams
PCM16LE audio bytes, optionally sends a stop message to flush, and yields parsed
ASR events:

```python
from funasr_ws_client import FunASRWebSocketClient

async with FunASRWebSocketClient("ws://127.0.0.1:10095") as client:
    await client.send_audio(pcm16le_bytes)
    await client.finish_utterance()
    async for event in client.events():
        print(event.mode, event.is_final, event.text)
```

For LiveKit STT integration, wrap this client in a LiveKit `stt.STT` adapter that
converts `rtc.AudioFrame` data to PCM bytes and maps online/offline ASR events to
LiveKit interim/final transcript events.
