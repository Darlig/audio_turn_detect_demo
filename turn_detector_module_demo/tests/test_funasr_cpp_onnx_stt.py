from __future__ import annotations

import asyncio
import json
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

from asr_module.funasr_cpp_onnx.funasr_cpp_protocol import (
    ASRResult,
    parse_server_message,
    start_message,
)
from asr_module.funasr_cpp_onnx.funasr_cpp_ws_client import FunASRCppOnnxWebSocketClient
from asr_module.funasr_cpp_onnx import funasr_cpp_ws_client
from turn_detector_module_demo import funasr_cpp_onnx_stt
from turn_detector_module_demo.funasr_cpp_onnx_stt import FunASRCppOnnxSTT
from turn_detector_module_demo.funasr_stt import FunASRSTT
from turn_detector_module_demo.stt_factory import create_stt


def _audio_frame() -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=(b"\x01\x00" * 320),
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=320,
    )


def test_funasr_cpp_start_message_uses_official_pcm_2pass_shape() -> None:
    payload = json.loads(
        start_message(
            mode="2pass",
            chunk_size=(5, 10, 5),
            sample_rate=16000,
            wav_name="unit-test",
            hotwords="热词 10",
            use_itn=True,
        )
    )

    assert payload == {
        "mode": "2pass",
        "wav_name": "unit-test",
        "wav_format": "pcm",
        "is_speaking": True,
        "chunk_size": [5, 10, 5],
        "audio_fs": 16000,
        "hotwords": "热词 10",
        "itn": True,
    }


@pytest.mark.asyncio
async def test_funasr_cpp_client_sends_start_audio_and_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeWebSocket:
        def __init__(self) -> None:
            self.messages: list[str | bytes] = []
            self.closed = asyncio.Event()

        async def send(self, message: str | bytes) -> None:
            self.messages.append(message)

        async def close(self) -> None:
            self.closed.set()

        def __aiter__(self):
            return self

        async def __anext__(self):
            await self.closed.wait()
            raise StopAsyncIteration

    fake_ws = FakeWebSocket()

    async def fake_connect(*_args, **_kwargs):
        return fake_ws

    monkeypatch.setattr(funasr_cpp_ws_client.websockets, "connect", fake_connect)

    client = FunASRCppOnnxWebSocketClient("ws://fake", wav_name="client-test")
    await client.connect()
    await client.send_audio(b"\x01\x00")
    await client.finish_utterance()
    await client.close()

    assert json.loads(fake_ws.messages[0])["wav_format"] == "pcm"
    assert fake_ws.messages[1] == b"\x01\x00"
    assert json.loads(fake_ws.messages[2]) == {"is_speaking": False}


@pytest.mark.asyncio
async def test_funasr_cpp_stream_maps_online_and_offline_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
                    text="你",
                    mode="2pass-online",
                    is_final=False,
                    raw={"mode": "2pass-online", "text": "你"},
                )
            )

        async def finish_utterance(self) -> None:
            self.messages.append("stop")
            await self.results.put(
                ASRResult(
                    text="你好。",
                    mode="2pass-offline",
                    is_final=True,
                    raw={"mode": "2pass-offline", "text": "你好。"},
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

    monkeypatch.setattr(
        funasr_cpp_onnx_stt,
        "FunASRCppOnnxWebSocketClient",
        FakeFunASRClient,
    )
    funasr = FunASRCppOnnxSTT(url="ws://fake", final_wait_timeout=1.0)
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
    assert [event.alternatives[0].text for event in events] == ["你", "你好。"]


@pytest.mark.asyncio
async def test_funasr_cpp_stream_accumulates_online_partial_fragments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FragmentFunASRClient:
        def __init__(self, *_args, **_kwargs) -> None:
            self.results: asyncio.Queue[ASRResult | None] = asyncio.Queue()
            self.fragments = iter(["你", "好"])

        async def connect(self) -> None:
            return None

        async def send_audio(self, _pcm16le: bytes) -> None:
            fragment = next(self.fragments)
            await self.results.put(
                ASRResult(
                    text=fragment,
                    mode="2pass-online",
                    is_final=False,
                    raw={"mode": "2pass-online", "text": fragment},
                )
            )

        async def finish_utterance(self) -> None:
            await self.results.put(
                ASRResult(
                    text="你好。",
                    mode="2pass-offline",
                    is_final=True,
                    raw={"mode": "2pass-offline", "text": "你好。"},
                )
            )

        async def events(self):
            while True:
                item = await self.results.get()
                if item is None:
                    break
                yield item

        async def close(self) -> None:
            await self.results.put(None)

    monkeypatch.setattr(
        funasr_cpp_onnx_stt,
        "FunASRCppOnnxWebSocketClient",
        FragmentFunASRClient,
    )
    funasr = FunASRCppOnnxSTT(url="ws://fake", final_wait_timeout=1.0)
    stream = funasr.stream()
    events: list[stt.SpeechEvent] = []

    async with stream:
        stream.push_frame(_audio_frame())
        stream.push_frame(_audio_frame())
        stream.flush()
        async for event in stream:
            events.append(event)
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                break

    assert [event.alternatives[0].text for event in events] == ["你", "你好", "你好。"]


def test_funasr_cpp_protocol_ignores_invalid_or_empty_messages() -> None:
    assert parse_server_message("") is None
    assert parse_server_message("{not-json") is None
    assert parse_server_message('{"mode":"2pass-online","text":""}') is None


def test_funasr_cpp_protocol_treats_online_is_final_as_interim() -> None:
    result = parse_server_message('{"mode":"2pass-online","text":"你好","is_final":true}')

    assert result is not None
    assert not result.is_final


def test_stt_factory_selects_new_default_and_old_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_STT_PROVIDER", raising=False)
    assert isinstance(create_stt(), FunASRCppOnnxSTT)
    assert isinstance(create_stt("funasr_python"), FunASRSTT)


def test_stt_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError):
        create_stt("unknown")
