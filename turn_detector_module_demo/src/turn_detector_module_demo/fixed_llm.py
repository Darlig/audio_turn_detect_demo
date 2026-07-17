from __future__ import annotations

import time
from typing import Any

from livekit.agents import (
    APIConnectOptions,
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
    llm,
)

from .turn_metrics import (
    claim_llm_turn_sequence,
    format_metric,
    record_llm_first_token,
    record_llm_turn,
)


class FixedTextLLM(llm.LLM):
    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text
        self._request_seq = 0

    @property
    def text(self) -> str:
        return self._text

    @property
    def model(self) -> str:
        return "fixed-text"

    @property
    def provider(self) -> str:
        return "debug"

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
        self._request_seq += 1
        return FixedTextLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            text=self._text,
            turn_sequence=claim_llm_turn_sequence(self._request_seq),
        )


class FixedTextLLMStream(llm.LLMStream):
    def __init__(
        self,
        llm_instance: FixedTextLLM,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        conn_options: APIConnectOptions,
        text: str,
        turn_sequence: int,
    ) -> None:
        super().__init__(
            llm_instance,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
        )
        self._text = text
        self._turn_sequence = turn_sequence

    async def _run(self) -> None:
        started_at = time.perf_counter()
        first_token_at = time.perf_counter()
        record_llm_first_token(
            self._turn_sequence,
            llm_started_at=started_at,
            llm_first_token_at=first_token_at,
        )
        self._event_ch.send_nowait(
            llm.ChatChunk(
                id=f"fixed-{self._turn_sequence}",
                delta=llm.ChoiceDelta(role="assistant", content=self._text),
            )
        )
        finished_at = time.perf_counter()
        record_llm_turn(
            self._turn_sequence,
            llm_started_at=started_at,
            llm_finished_at=finished_at,
        )
        print(
            "[turn_metrics] "
            f"stage=llm seq={self._turn_sequence} "
            f"latency_ms={format_metric((finished_at - started_at) * 1000.0)} "
            f"first_delta_ms={format_metric((first_token_at - started_at) * 1000.0)} "
            f"status=fixed output_chars={len(self._text)}",
            flush=True,
        )
