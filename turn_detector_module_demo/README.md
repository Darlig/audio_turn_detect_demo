# LiveKit Cascading Voice Agent Demo

This subproject runs the browser turn detector demo and a local cascading
LiveKit voice agent. The browser publishes microphone audio and still renders
the local EOT curve. The agent subscribes to the room, uses local FunASR for
ASR, Doubao Ark for LLM, Doubao/Volcengine for TTS, and
`livekit.agents.inference.TurnDetector(version="v1-mini")` for local audio turn
detection.

## Layout

```text
turn_detector_module_demo/
  src/turn_detector_module_demo/   Web server, detector worker, voice agent, STT adapter
  web/                             Browser UI and timeline renderer
  scripts/                         Environment and launch helpers
  tests/                           Unit and local inference smoke tests
```

## Environment

Create the requested conda environment:

```bash
cd /home/weiy/project/spoken_dialogue/livekit/audio_turn_detect_demo/turn_detector_module_demo
bash scripts/create_conda_env.sh
```

The script creates `.conda/audio_turn_detector` with Python 3.10 and installs
this subproject in editable mode with detector, voice-agent, FunASR, and test
dependencies.

## Configuration

```bash
cp .env.example .env
```

Defaults are for a local LiveKit server:

```text
LIVEKIT_URL=ws://127.0.0.1:8890
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
LIVEKIT_NODE_IP=
LIVEKIT_USE_EXTERNAL_IP=false
DEMO_ROOM=audio-turn-demo
DEMO_LANGUAGE=zh
DEMO_SSL_CERT_FILE=
DEMO_SSL_KEY_FILE=
FUNASR_URL=ws://127.0.0.1:10095
FUNASR_MODE=2pass
FUNASR_LANGUAGE=zh
AGENT_ENDPOINT_MAX_DELAY=2.5
AGENT_DEBUG_ENDPOINT_MAX_DELAY=
AGENT_INSTRUCTIONS=
AGENT_GREETING=
```

The LiveKit helper binds to `0.0.0.0` by default. Keep `LIVEKIT_URL` pointed at
`127.0.0.1` for the backend detector process; the token API rewrites the browser
URL to the host used to open the web page.

For LAN browsers, LiveKit must advertise the server's LAN IP in WebRTC ICE
candidates. `scripts/start_livekit_server.sh` tries to auto-detect it, but you
can pin it explicitly:

```bash
LIVEKIT_NODE_IP=192.168.0.179 bash scripts/start_livekit_server.sh
```

## Run

The shortest local path is the stack helper:

```bash
cd /home/weiy/project/spoken_dialogue/livekit/audio_turn_detect_demo/turn_detector_module_demo
bash scripts/start_stack.sh
```

It starts LiveKit, FunASR, the detector web app, and `voice-agent dev`. Logs are
written to `logs/`, and child processes are cleaned up when the script exits.

For separate terminals, use the commands below.

Terminal 1, start LiveKit:

```bash
cd /home/weiy/project/spoken_dialogue/livekit/audio_turn_detect_demo/turn_detector_module_demo
bash scripts/start_livekit_server.sh
```

The helper script defaults to LiveKit signaling `8890`, RTC TCP `8891`, and RTC UDP `8892`.
To override them:

```bash
LIVEKIT_PORT=8890 LIVEKIT_RTC_TCP_PORT=8891 LIVEKIT_UDP_PORT=8892 \
  bash scripts/start_livekit_server.sh
```

For LAN access, prefer:

```bash
LIVEKIT_NODE_IP=192.168.0.179 \
LIVEKIT_PORT=8890 LIVEKIT_RTC_TCP_PORT=8891 LIVEKIT_UDP_PORT=8892 \
  bash scripts/start_livekit_server.sh
```

Terminal 2, start the detector web app:

```bash
cd /home/weiy/project/spoken_dialogue/livekit/audio_turn_detect_demo/turn_detector_module_demo
bash scripts/start_demo.sh --host 127.0.0.1 --port 8090
```

Terminal 3, start FunASR:

```bash
cd /home/weiy/project/spoken_dialogue/livekit/audio_turn_detect_demo/turn_detector_module_demo
bash scripts/start_funasr_server.sh
```

For CUDA:

```bash
FUNASR_DEVICE=cuda FUNASR_NGPU=1 bash scripts/start_funasr_server.sh
```

Terminal 4, start the LiveKit voice agent:

```bash
cd /home/weiy/project/spoken_dialogue/livekit/audio_turn_detect_demo/turn_detector_module_demo
bash scripts/start_voice_agent.sh dev
```

For endpointing/max-delay debugging, keep `AGENT_DEBUG_ENDPOINT_MAX_DELAY`
unset to use the normal agent max delay. Set it only when you want a longer
debug wait window, for example:

```bash
AGENT_DEBUG_ENDPOINT_MAX_DELAY=10 bash scripts/start_voice_agent.sh dev
```

The agent entrypoint also supports the standard LiveKit Agents CLI mode:

```bash
voice-agent dev
voice-agent start
```

To expose the demo web server on the LAN:

```bash
bash scripts/start_demo.sh --host 0.0.0.0 --port 8090
```

Browser microphone capture requires a secure context. `http://127.0.0.1` works
locally, but `http://SERVER_IP:8090` on another LAN machine will not expose
`navigator.mediaDevices.getUserMedia`. For LAN microphone testing, create a dev
certificate and serve the demo over HTTPS:

```bash
bash scripts/create_dev_https_cert.sh 192.168.0.179
bash scripts/start_demo.sh --host 0.0.0.0 --port 8090 \
  --ssl-cert-file certs/dev-cert.pem \
  --ssl-key-file certs/dev-key.pem
```

Then open:

```text
https://192.168.0.179:8090
```

With a self-signed certificate, the browser may require accepting or trusting
the certificate before microphone permission is available.

Open:

```text
http://127.0.0.1:8090
```

For another machine on the same LAN, open `http://SERVER_IP:8090`.

Use **Connect**, then either **Start mic** or **Replay file**. The chart shows
audio progress, EOT score, EOT threshold, and EOT decision on one shared time
axis. When the agent publishes audio, the page subscribes to that remote audio
track and plays it through an autogenerated `<audio autoplay>` element.

For debug TTS simulation, use **Start simulation** to publish one long-lived
silent input track. After **Synthesize**, **Start replay** injects the prepared
TTS clip into that same track and automatically returns to silence when the clip
ends. You can synthesize and replay multiple clips without ending the simulation.
Use **End simulation** only when you want to unpublish the simulated input track.

The page also shows a **Turn Latency** debug panel after each agent reply:

- `STT`: aligned audio tail sent to FunASR -> FunASR final transcript.
- `EOU Wait`: FunASR final transcript -> LiveKit user turn committed.
- `LLM TTFT`: Doubao LLM request start -> first streamed text delta.
- `TTS First`: first text sent to Volcengine TTS -> first returned audio payload.
- `Total`: aligned audio tail sent to FunASR -> first returned TTS audio payload.

The audio tail anchor prefers FunASR timestamps when available, then the local
STT adapter VAD speech end, then the last audio frame forwarded to FunASR. With
preemptive generation enabled, `LLM TTFT` can begin before `EOU Wait` completes;
the panel marks those turns as `preemptive`.

## Agent Providers

The voice agent loads environment variables from:

```text
turn_detector_module_demo/.env
llm_module/doubao_llm/.env
tts_module/doubao_tts/.env
```

Expected provider variables include:

```text
ARK_API_KEY
ARK_BASE_URL
DOUBAO_MODEL
DOUBAO_STREAM
VOLC_TTS_API_KEY
VOLC_TTS_APP_ID
VOLC_TTS_ACCESS_KEY
VOLC_TTS_VOICE_TYPE
```

The default voice prompt is a short natural Chinese assistant. Set
`AGENT_INSTRUCTIONS` and `AGENT_GREETING` to override it.

## Notes

- The detector worker and voice agent both use local LiveKit turn detector
  inference. The voice agent configures it through `TurnHandlingOptions`.
- The local mini model accepts recent PCM audio only; no ASR transcript is used.
- Default language is `zh`, so the default local threshold is the v1-mini Chinese
  threshold from LiveKit.
- The browser UI imports `livekit-client` from jsDelivr. If offline browser use
  is required, vendor that ESM file into `web/vendor/` and update `web/app.js`.

## Tests

```bash
cd /home/weiy/project/spoken_dialogue/livekit/audio_turn_detect_demo/turn_detector_module_demo
conda run -p .conda/audio_turn_detector pytest
```

The `test_local_eot.py` smoke test verifies that `livekit-local-inference` can
run `EOT.predict()` locally.
