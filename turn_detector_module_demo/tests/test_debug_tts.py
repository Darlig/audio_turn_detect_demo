from __future__ import annotations

import io
from types import SimpleNamespace
import wave

from aiohttp import web
import pytest

from turn_detector_module_demo import app as app_module


class DummyRequest:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    async def json(self) -> dict[str, object]:
        return self._payload


@pytest.mark.asyncio
async def test_debug_tts_returns_wav(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_synthesize(text: str) -> SimpleNamespace:
        assert text == "你好"
        return SimpleNamespace(pcm=b"\x00\x00" * 240, sample_rate=24000, num_channels=1)

    monkeypatch.setattr(app_module, "load_debug_tts_synthesizer", lambda: fake_synthesize)

    response = await app_module.debug_tts(DummyRequest({"text": "  你好  "}))  # type: ignore[arg-type]

    assert response.status == 200
    assert response.content_type == "audio/wav"
    assert response.body is not None
    with wave.open(io.BytesIO(response.body), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 24000
        assert wav.getnframes() == 240


@pytest.mark.asyncio
async def test_debug_tts_rejects_empty_text() -> None:
    with pytest.raises(web.HTTPBadRequest):
        await app_module.debug_tts(DummyRequest({"text": "   "}))  # type: ignore[arg-type]

