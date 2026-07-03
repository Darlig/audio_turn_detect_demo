from __future__ import annotations

from dataclasses import dataclass, field
import asyncio
import json
import os
import time
import uuid
from typing import Any

import aiohttp
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
    llm,
)

from .turn_metrics import (
    claim_llm_turn_sequence,
    elapsed_ms,
    format_metric,
    record_llm_first_token,
    record_llm_turn,
)


DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_DOUBAO_MODEL = "doubao-seed-2-0-mini-260428"


@dataclass
class DoubaoResponsesOptions:
    api_key: str
    model: str = DEFAULT_DOUBAO_MODEL
    base_url: str = DEFAULT_ARK_BASE_URL
    stream: bool = True
    temperature: float | None = None
    max_output_tokens: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def responses_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/responses"


class DoubaoResponsesLLM(llm.LLM):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        http_session: aiohttp.ClientSession | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        resolved_api_key = api_key or os.getenv("ARK_API_KEY")
        if not resolved_api_key:
            raise ValueError("ARK_API_KEY is required for DoubaoResponsesLLM")

        self._opts = DoubaoResponsesOptions(
            api_key=resolved_api_key,
            model=model or os.getenv("DOUBAO_MODEL", DEFAULT_DOUBAO_MODEL),
            base_url=base_url or os.getenv("ARK_BASE_URL", DEFAULT_ARK_BASE_URL),
            stream=_env_bool("DOUBAO_STREAM", True),
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            extra=extra or {},
        )
        self._session = http_session
        self._owns_session = http_session is None
        self._request_seq = 0

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "volcengine-ark"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> llm.LLMStream:
        if tools:
            raise NotImplementedError("DoubaoResponsesLLM does not implement LiveKit tools yet")

        request_extra: dict[str, Any] = dict(self._opts.extra)
        if extra_kwargs is not NOT_GIVEN:
            request_extra.update(extra_kwargs)

        self._request_seq += 1
        turn_sequence = claim_llm_turn_sequence(self._request_seq)
        return DoubaoResponsesStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            opts=self._opts,
            request_extra=request_extra,
            session=self._ensure_session(),
            turn_sequence=turn_sequence,
        )

    async def aclose(self) -> None:
        if self._owns_session and self._session:
            await self._session.close()
        self._session = None

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=None, connect=15)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session


class DoubaoResponsesStream(llm.LLMStream):
    def __init__(
        self,
        llm_instance: DoubaoResponsesLLM,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        conn_options: APIConnectOptions,
        opts: DoubaoResponsesOptions,
        request_extra: dict[str, Any],
        session: aiohttp.ClientSession,
        turn_sequence: int,
    ) -> None:
        super().__init__(
            llm_instance,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
        )
        self._opts = opts
        self._request_extra = request_extra
        self._session = session
        self._turn_sequence = turn_sequence

    async def _run(self) -> None:
        request_id = str(uuid.uuid4())
        payload = build_responses_payload(
            chat_ctx=self._chat_ctx,
            model=self._opts.model,
            stream=self._opts.stream,
            temperature=self._opts.temperature,
            max_output_tokens=self._opts.max_output_tokens,
            extra=self._request_extra,
        )
        headers = {
            "Authorization": f"Bearer {self._opts.api_key}",
            "Content-Type": "application/json",
        }
        retryable = True
        started_at = time.perf_counter()
        try:
            async with self._session.post(
                self._opts.responses_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=None if self._opts.stream else self._conn_options.timeout,
                    connect=self._conn_options.timeout,
                    sock_read=self._conn_options.timeout,
                ),
            ) as resp:
                if resp.status >= 400:
                    body = await _read_response_error_body(resp)
                    finished_at = time.perf_counter()
                    print(
                        "[turn_metrics] "
                        f"stage=llm seq={self._turn_sequence} "
                        f"latency_ms={format_metric(elapsed_ms(started_at, finished_at))} "
                        f"status={resp.status} output_chars=0",
                        flush=True,
                    )
                    raise APIStatusError(
                        "Doubao Responses API request failed",
                        status_code=resp.status,
                        request_id=body.get("id") if isinstance(body, dict) else None,
                        body=body,
                        retryable=resp.status >= 500,
                    )

                if self._opts.stream:
                    await self._run_streaming_response(resp, request_id, started_at)
                    retryable = False
                    return

                body = await resp.json(content_type=None)

            text = extract_response_text(body)
            finished_at = time.perf_counter()
            record_llm_turn(
                self._turn_sequence,
                llm_started_at=started_at,
                llm_finished_at=finished_at,
            )
            print(
                "[turn_metrics] "
                f"stage=llm seq={self._turn_sequence} "
                f"latency_ms={format_metric(elapsed_ms(started_at, finished_at))} "
                f"status=200 output_chars={len(text)}",
                flush=True,
            )
            retryable = False
            if text:
                self._event_ch.send_nowait(
                    llm.ChatChunk(
                        id=str(body.get("id") or request_id),
                        delta=llm.ChoiceDelta(role="assistant", content=text),
                    )
                )

            usage = extract_usage(body)
            if usage:
                self._event_ch.send_nowait(
                    llm.ChatChunk(
                        id=str(body.get("id") or request_id),
                        usage=usage,
                    )
                )
        except asyncio.TimeoutError as exc:
            raise APITimeoutError(retryable=retryable) from exc
        except APIStatusError:
            raise
        except aiohttp.ClientError as exc:
            raise APIConnectionError("failed to connect to Doubao Responses API") from exc

    async def _run_streaming_response(
        self,
        resp: aiohttp.ClientResponse,
        request_id: str,
        started_at: float,
    ) -> None:
        response_id = request_id
        output_parts: list[str] = []
        usage: llm.CompletionUsage | None = None
        first_delta_at: float | None = None

        async for event in iter_sse_json(resp.content):
            if not isinstance(event, dict):
                continue
            response_id = extract_stream_request_id(event) or response_id
            error = extract_stream_error(event)
            if error:
                finished_at = time.perf_counter()
                print(
                    "[turn_metrics] "
                    f"stage=llm seq={self._turn_sequence} "
                    f"latency_ms={format_metric(elapsed_ms(started_at, finished_at))} "
                    f"first_delta_ms={format_metric(elapsed_ms(started_at, first_delta_at))} "
                    "status=stream_error output_chars="
                    f"{sum(len(part) for part in output_parts)}",
                    flush=True,
                )
                raise APIStatusError(
                    "Doubao Responses API stream failed",
                    status_code=-1,
                    request_id=response_id,
                    body=error,
                    retryable=False,
                )

            delta = extract_response_stream_delta(event)
            if delta:
                if first_delta_at is None:
                    first_delta_at = time.perf_counter()
                    record_llm_first_token(
                        self._turn_sequence,
                        llm_started_at=started_at,
                        llm_first_token_at=first_delta_at,
                    )
                output_parts.append(delta)
                self._event_ch.send_nowait(
                    llm.ChatChunk(
                        id=response_id,
                        delta=llm.ChoiceDelta(role="assistant", content=delta),
                    )
                )

            event_usage = extract_response_stream_usage(event)
            if event_usage is not None:
                usage = event_usage

        finished_at = time.perf_counter()
        record_llm_turn(
            self._turn_sequence,
            llm_started_at=started_at,
            llm_finished_at=finished_at,
        )
        output_chars = sum(len(part) for part in output_parts)
        print(
            "[turn_metrics] "
            f"stage=llm seq={self._turn_sequence} "
            f"latency_ms={format_metric(elapsed_ms(started_at, finished_at))} "
            f"first_delta_ms={format_metric(elapsed_ms(started_at, first_delta_at))} "
            f"status=200 stream=true output_chars={output_chars}",
            flush=True,
        )
        if usage:
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id=response_id,
                    usage=usage,
                )
            )


def build_responses_payload(
    *,
    chat_ctx: llm.ChatContext,
    model: str,
    stream: bool = False,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "input": [_message_to_responses_input(message) for message in chat_ctx.messages()],
    }
    if stream:
        payload["stream"] = True
    if temperature is not None:
        payload["temperature"] = temperature
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens
    if extra:
        payload.update(extra)
    return payload


async def _read_response_error_body(resp: aiohttp.ClientResponse) -> Any:
    text = await resp.text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


async def iter_sse_json(content: aiohttp.StreamReader) -> Any:
    buffer = ""
    data_lines: list[str] = []

    async for raw_chunk in content.iter_any():
        buffer += raw_chunk.decode("utf-8", errors="replace")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")
            if not line:
                payload = "\n".join(data_lines).strip()
                data_lines = []
                if not payload or payload == "[DONE]":
                    continue
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    continue
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())

    payload = "\n".join(data_lines).strip()
    if payload and payload != "[DONE]":
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            return


def extract_response_stream_delta(data: Any) -> str:
    if not isinstance(data, dict):
        return ""

    event_type = str(data.get("type") or data.get("event") or "")
    if "reasoning" in event_type:
        return ""

    if event_type in {
        "response.output_text.delta",
        "response.text.delta",
        "response.content.delta",
    }:
        return _string_value(data.get("delta") or data.get("text"))

    delta = data.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return content
        text = delta.get("text")
        if isinstance(text, str):
            return text
    elif isinstance(delta, str) and ("text" in event_type or "delta" in event_type):
        return delta

    for choice in data.get("choices", []) or []:
        if not isinstance(choice, dict):
            continue
        choice_delta = choice.get("delta") or {}
        if isinstance(choice_delta, dict):
            content = choice_delta.get("content")
            if isinstance(content, str):
                return content
        text = choice.get("text")
        if isinstance(text, str):
            return text

    return ""


def extract_response_stream_usage(data: Any) -> llm.CompletionUsage | None:
    if not isinstance(data, dict):
        return None
    usage = data.get("usage")
    if not isinstance(usage, dict):
        response = data.get("response")
        if isinstance(response, dict):
            usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    return extract_usage({"usage": usage})


def extract_stream_request_id(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    for key in ("id", "response_id", "request_id"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    response = data.get("response")
    if isinstance(response, dict):
        value = response.get("id")
        if isinstance(value, str):
            return value
    return ""


def extract_stream_error(data: Any) -> Any:
    if not isinstance(data, dict):
        return None
    event_type = str(data.get("type") or data.get("event") or "")
    if event_type in {"response.failed", "error"}:
        return data.get("error") or data
    error = data.get("error")
    return error if error else None


def _string_value(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _message_to_responses_input(message: llm.ChatMessage) -> dict[str, Any]:
    role = "assistant" if message.role == "assistant" else "user"
    if message.role in {"system", "developer"}:
        role = "system"

    text = message.text_content or ""
    content_type = "output_text" if role == "assistant" else "input_text"
    item = {
        "type": "message",
        "role": role,
        "content": [
            {
                "type": content_type,
                "text": text,
            }
        ],
    }
    if role == "assistant":
        item["status"] = "completed"
    return item


def extract_response_text(data: Any) -> str:
    if not isinstance(data, dict):
        return ""

    if isinstance(data.get("output_text"), str):
        return data["output_text"].strip()

    texts: list[str] = []
    for item in data.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                texts.append(text)

    if texts:
        return "".join(texts).strip()

    # Some OpenAI-compatible providers return chat-completion shaped data.
    for choice in data.get("choices", []) or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)

    return "".join(texts).strip()


def extract_usage(data: Any) -> llm.CompletionUsage | None:
    if not isinstance(data, dict) or not isinstance(data.get("usage"), dict):
        return None

    usage = data["usage"]
    input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    cached_tokens = int(usage.get("input_cached_tokens") or usage.get("prompt_cached_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
    return llm.CompletionUsage(
        completion_tokens=output_tokens,
        prompt_tokens=input_tokens,
        prompt_cached_tokens=cached_tokens,
        total_tokens=total_tokens,
    )


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
