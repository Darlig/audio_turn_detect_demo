from __future__ import annotations

import logging
import os

from doubao_llm import DoubaoResponsesLLM
from livekit.agents import llm

from .fixed_llm import FixedTextLLM


DEFAULT_LLM_PROVIDER = "doubao"
DEFAULT_FIXED_TEXT = "你好，我是豆包"
logger = logging.getLogger("cascade-voice-agent")


def create_llm(
    provider: str | None = None,
    *,
    fixed_text: str | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> llm.LLM:
    selected = (provider or os.getenv("LLM_PROVIDER") or DEFAULT_LLM_PROVIDER).strip().lower()
    normalized = selected.replace("-", "_")
    if normalized in {"fixed", "fixed_text", "debug"}:
        text = fixed_text if fixed_text is not None else os.getenv("LLM_FIXED_TEXT", DEFAULT_FIXED_TEXT)
        text = text.strip()
        if not text:
            raise ValueError("LLM_FIXED_TEXT must not be empty when LLM_PROVIDER=fixed")
        logger.info("using LLM provider: fixed text=%r", text)
        return FixedTextLLM(text)
    if normalized in {"doubao", "volcengine", "volcengine_ark"}:
        logger.info("using LLM provider: doubao")
        return DoubaoResponsesLLM(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    raise ValueError(
        f"unsupported LLM_PROVIDER={selected!r}; expected doubao or fixed"
    )
