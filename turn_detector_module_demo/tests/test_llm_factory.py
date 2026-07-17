from __future__ import annotations

import pytest
from livekit.agents import llm

from turn_detector_module_demo.fixed_llm import FixedTextLLM
from turn_detector_module_demo.llm_factory import create_llm


def test_llm_factory_creates_fixed_provider() -> None:
    fixed = create_llm("fixed", fixed_text="你好，我是豆包")

    assert isinstance(fixed, FixedTextLLM)
    assert fixed.text == "你好，我是豆包"
    assert fixed.model == "fixed-text"
    assert fixed.provider == "debug"


@pytest.mark.asyncio
async def test_fixed_provider_streams_configured_text() -> None:
    fixed = create_llm("fixed", fixed_text="你好，我是豆包")
    chunks = []

    async for chunk in fixed.chat(chat_ctx=llm.ChatContext.empty()):
        chunks.append(chunk.delta.content or "")

    assert "".join(chunks) == "你好，我是豆包"


def test_llm_factory_rejects_empty_fixed_text() -> None:
    with pytest.raises(ValueError, match="LLM_FIXED_TEXT"):
        create_llm("fixed", fixed_text="  ")


def test_llm_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unsupported LLM_PROVIDER"):
        create_llm("unknown")
