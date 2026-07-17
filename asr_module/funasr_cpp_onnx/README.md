# FunASR C++/ONNX 2pass Runtime

This directory contains the local wrapper for the official FunASR C++ websocket
runtime. It is intentionally parallel to `asr_module/funasr` and does not
replace the existing Python FunASR service.

## Build

```bash
cd /home/weiy/project/spoken_dialogue/livekit/audio_turn_detect_demo
bash turn_detector_module_demo/scripts/prepare_funasr_cpp_onnx.sh
```

The helper clones `modelscope/FunASR`, downloads the runtime dependencies, and
builds `runtime/websocket` into:

```text
asr_module/funasr_cpp_onnx/build/websocket/bin/funasr-wss-server-2pass
```

The local smoke build was verified with FunASR commit
`f9937385517cccaa8cd780b61c8b404c701c1d44` (`f993738 Show active integration
operator queue (#3099)`). Set `FUNASR_CPP_REPO_REF` to that commit if `main`
changes in a way that breaks local builds.

Local checkout, dependency, model, and build directories are git-ignored:

```text
upstream/
deps/
models/
build/
```

## Run

```bash
bash turn_detector_module_demo/scripts/start_funasr_cpp_onnx_server.sh
```

Defaults:

```text
FUNASR_CPP_URL=ws://127.0.0.1:10095
FUNASR_CPP_HOST=127.0.0.1
FUNASR_CPP_PORT=10095
FUNASR_CPP_CHUNK_SIZE=5,10,5
FUNASR_CPP_SAMPLE_RATE=16000
```

Thread defaults follow the official runtime guidance:

```text
decoder_thread_num = nproc
model_thread_num = 1
io_thread_num = ceil(decoder_thread_num / 16)
```

The LiveKit voice agent uses this backend by default through:

```text
AGENT_STT_PROVIDER=funasr_cpp_onnx
```

Use `AGENT_STT_PROVIDER=funasr_python` to return to the existing Python
FunASR backend.
