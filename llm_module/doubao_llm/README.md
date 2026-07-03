# Doubao LLM README

This directory contains a lightweight Doubao Ark Responses API adapter for
LiveKit Agents.

## Files

- `doubao_llm.py`: LiveKit `llm.LLM` adapter for Doubao Ark Responses API.
- `turn_metrics.py`: latency metric helpers used by the adapter.
- `.env`: local Doubao LLM credentials and model settings.
- `requirements.txt`: Python dependencies needed by this adapter.

## API

- Provider: Volcengine Ark
- API style: Responses API
- Default base URL: `https://ark.cn-beijing.volces.com/api/v3`
- Request URL: `https://ark.cn-beijing.volces.com/api/v3/responses`
- Default model: `doubao-seed-2-0-mini-260428`
- Streaming: enabled by default through `DOUBAO_STREAM=1`

## Environment

Put credentials and model settings in `doubao_llm/.env`.

Required:

```bash
ARK_API_KEY=...
```

Optional:

```bash
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
DOUBAO_MODEL=doubao-seed-2-0-mini-260428
DOUBAO_TEMPERATURE=
DOUBAO_MAX_OUTPUT_TOKENS=
DOUBAO_STREAM=1
```

## LiveKit Integration

Install dependencies:

```bash
python -m pip install -r doubao_llm/requirements.txt
```

Load environment variables before starting the LiveKit worker, then pass
`DoubaoResponsesLLM` to `AgentSession`:

```python
from livekit.agents import AgentSession
from doubao_llm import DoubaoResponsesLLM

session = AgentSession(
    llm=DoubaoResponsesLLM(),
)
```

`DoubaoResponsesLLM.chat()` converts LiveKit chat context messages into Ark
Responses API input messages, sends a request to `/responses`, and yields
LiveKit `llm.ChatChunk` objects. In streaming mode, SSE text deltas are forwarded
as assistant `ChoiceDelta` chunks.
