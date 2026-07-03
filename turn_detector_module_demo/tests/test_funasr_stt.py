from __future__ import annotations

import json
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for import_path in (
    REPO_ROOT,
    REPO_ROOT / "agents" / "livekit-agents",
    REPO_ROOT / "turn_detector_module_demo" / "src",
):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)

import pytest
from livekit import rtc
from livekit.agents import stt

from asr_module.funasr.funasr_protocol import ASRResult
from asr_module.funasr.funasr_protocol import parse_server_message
from turn_detector_module_demo import funasr_stt
from turn_detector_module_demo.funasr_stt import (
    FunASRSTT,
    _UtteranceTiming,
    _audio_frame_bytes,
    _resolve_final_audio_tail,
)


def _audio_frame() -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=(b"\x01\x00" * 320),
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=320,
    )


@pytest.mark.asyncio
async def test_funasr_stream_maps_online_and_offline_results(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeFunASRClient:
        instances: list["FakeFunASRClient"] = []

        def __init__(self, *_args, **_kwargs) -> None:
            self.messages: list[str | bytes] = []
            self.results: asyncio.Queue[ASRResult | None] = asyncio.Queue()
            FakeFunASRClient.instances.append(self)

        async def connect(self) -> None:
            self.messages.append("start")

        async def send_audio(self, pcm16le: bytes) -> None:
            self.messages.append(pcm16le)
            await self.results.put(
                ASRResult(
                    text="你好",
                    mode="2pass-online",
                    is_final=False,
                    raw={"mode": "2pass-online", "text": "你好"},
                )
            )

        async def finish_utterance(self) -> None:
            self.messages.append("stop")
            await self.results.put(
                ASRResult(
                    text="你好，世界。",
                    mode="2pass-offline",
                    is_final=True,
                    raw={"mode": "2pass-offline", "text": "你好，世界。"},
                )
            )

        async def events(self):
            while True:
                item = await self.results.get()
                if item is None:
                    break
                yield item

        async def close(self) -> None:
            self.messages.append("close")
            await self.results.put(None)

    monkeypatch.setattr(funasr_stt, "FunASRWebSocketClient", FakeFunASRClient)
    funasr = FunASRSTT(
        url="ws://fake-funasr",
        internal_vad=False,
        final_wait_timeout=1.0,
    )
    stream = funasr.stream()
    events: list[stt.SpeechEvent] = []

    async with stream:
        stream.push_frame(_audio_frame())
        stream.flush()
        async for event in stream:
            events.append(event)
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                break

    messages = FakeFunASRClient.instances[0].messages
    assert messages[0] == "start"
    assert any(isinstance(item, bytes) for item in messages)
    assert "stop" in messages

    assert [event.type for event in events] == [
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
    ]
    assert events[0].alternatives[0].text == "你好"
    assert events[1].alternatives[0].text == "你好，世界。"


@pytest.mark.asyncio
async def test_funasr_stream_ignores_empty_text_before_close(monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptyResultFunASRClient:
        def __init__(self, *_args, **_kwargs) -> None:
            self.results: asyncio.Queue[ASRResult | None] = asyncio.Queue()

        async def connect(self) -> None:
            return None

        async def send_audio(self, _pcm16le: bytes) -> None:
            return None

        async def finish_utterance(self) -> None:
            await self.results.put(
                ASRResult(
                    text="",
                    mode="2pass-offline",
                    is_final=True,
                    raw={"mode": "2pass-offline", "text": ""},
                )
            )
            await self.results.put(None)

        async def events(self):
            while True:
                item = await self.results.get()
                if item is None:
                    break
                yield item

        async def close(self) -> None:
            await self.results.put(None)

    monkeypatch.setattr(funasr_stt, "FunASRWebSocketClient", EmptyResultFunASRClient)
    funasr = FunASRSTT(
        url="ws://fake-funasr",
        internal_vad=False,
        final_wait_timeout=0.1,
    )
    stream = funasr.stream()
    events: list[stt.SpeechEvent] = []

    async with stream:
        stream.push_frame(_audio_frame())
        stream.end_input()
        async for event in stream:
            events.append(event)

    assert events == []


def test_funasr_protocol_ignores_invalid_or_empty_messages() -> None:
    assert parse_server_message("") is None
    assert parse_server_message("{not-json") is None


def test_audio_frame_bytes_downmixes_to_mono_pcm16le() -> None:
    frame = rtc.AudioFrame(
        data=(
            (1000).to_bytes(2, "little", signed=True)
            + (3000).to_bytes(2, "little", signed=True)
            + (-1000).to_bytes(2, "little", signed=True)
            + (1000).to_bytes(2, "little", signed=True)
        ),
        sample_rate=16000,
        num_channels=2,
        samples_per_channel=2,
    )

    assert _audio_frame_bytes(frame) == (
        (2000).to_bytes(2, "little", signed=True)
        + (0).to_bytes(2, "little", signed=True)
    )


def test_latency_anchor_ignores_zero_funasr_timestamp() -> None:
    result = ASRResult(
        text="你好",
        mode="2pass-offline",
        is_final=True,
        raw={"timestamp": [[0, 0]]},
    )
    timing = _UtteranceTiming(
        seq=1,
        audio_started_at=100.0,
        audio_tail_at=120.0,
        speech_tail_at=119.5,
    )

    tail_at, source = _resolve_final_audio_tail(result, timing)

    assert tail_at == 119.5
    assert source == "internal_vad"


def test_latency_anchor_rejects_timestamp_far_before_vad_tail() -> None:
    result = ASRResult(
        text="你好",
        mode="2pass-offline",
        is_final=True,
        raw={"timestamp": [[0, 600]]},
    )
    timing = _UtteranceTiming(
        seq=1,
        audio_started_at=100.0,
        audio_tail_at=120.0,
        speech_tail_at=119.5,
    )

    tail_at, source = _resolve_final_audio_tail(result, timing)

    assert tail_at == 119.5
    assert source == "internal_vad"
