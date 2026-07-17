from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from livekit.agents import (
    APIConnectOptions,
    APIConnectionError,
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    LanguageCode,
    NotGivenOr,
    stt,
)
from livekit.agents.utils import AudioBuffer


def _install_project_import_paths() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    for path in (repo_root, repo_root / "asr_module"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_install_project_import_paths()

from asr_module.funasr_cpp_onnx import FunASRCppOnnxWebSocketClient  # noqa: E402
from asr_module.funasr_cpp_onnx.funasr_cpp_protocol import ASRResult  # noqa: E402

from .funasr_stt import (  # noqa: E402
    _PartialTranscriptAccumulator,
    _UtteranceTiming,
    _audio_frame_bytes,
    _env_bool,
    _funasr_timestamp_end_seconds,
    _iter_audio_buffer,
    _parse_chunk_size,
    _resolve_language,
)
from .turn_metrics import record_asr_turn  # noqa: E402


DEFAULT_FUNASR_CPP_URL = "ws://127.0.0.1:10095"


@dataclass(frozen=True)
class FunASRCppOnnxSTTOptions:
    url: str = DEFAULT_FUNASR_CPP_URL
    mode: str = "2pass"
    language: str = "zh"
    sample_rate: int = 16000
    chunk_size: tuple[int, int, int] = (5, 10, 5)
    hotwords: str = ""
    wav_name: str = "livekit-agent"
    use_itn: bool = True
    final_wait_timeout: float = 2.0


class FunASRCppOnnxSTT(stt.STT):
    """LiveKit STT adapter for the official FunASR C++/ONNX 2pass websocket server."""

    def __init__(
        self,
        *,
        url: str | None = None,
        mode: str | None = None,
        language: str | None = None,
        sample_rate: int | None = None,
        chunk_size: tuple[int, int, int] | None = None,
        hotwords: str | None = None,
        wav_name: str | None = None,
        use_itn: bool | None = None,
        final_wait_timeout: float | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                diarization=False,
                offline_recognize=True,
            )
        )
        self._opts = FunASRCppOnnxSTTOptions(
            url=url or os.getenv("FUNASR_CPP_URL", DEFAULT_FUNASR_CPP_URL),
            mode=mode or os.getenv("FUNASR_CPP_MODE", "2pass"),
            language=language or os.getenv("FUNASR_CPP_LANGUAGE", "zh"),
            sample_rate=sample_rate
            if sample_rate is not None
            else int(os.getenv("FUNASR_CPP_SAMPLE_RATE", "16000")),
            chunk_size=chunk_size or _parse_chunk_size(os.getenv("FUNASR_CPP_CHUNK_SIZE", "5,10,5")),
            hotwords=hotwords if hotwords is not None else os.getenv("FUNASR_CPP_HOTWORDS", ""),
            wav_name=wav_name or os.getenv("FUNASR_CPP_WAV_NAME", "livekit-agent"),
            use_itn=use_itn
            if use_itn is not None
            else _env_bool("FUNASR_CPP_USE_ITN", True),
            final_wait_timeout=final_wait_timeout
            if final_wait_timeout is not None
            else float(os.getenv("FUNASR_CPP_FINAL_WAIT_TIMEOUT", "2.0")),
        )

    @property
    def model(self) -> str:
        return self._opts.mode

    @property
    def provider(self) -> str:
        return "funasr_cpp_onnx"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        stream = self.stream(language=language, conn_options=conn_options)
        async with stream:
            for frame in _iter_audio_buffer(buffer):
                stream.push_frame(frame)
            stream.end_input()
            async for event in stream:
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    return event

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    language=LanguageCode(_resolve_language(language, self._opts.language)),
                    text="",
                )
            ],
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.RecognizeStream:
        return FunASRCppOnnxRecognizeStream(
            stt=self,
            opts=self._opts,
            language=_resolve_language(language, self._opts.language),
            conn_options=conn_options,
        )


class FunASRCppOnnxRecognizeStream(stt.RecognizeStream):
    def __init__(
        self,
        *,
        stt: FunASRCppOnnxSTT,
        opts: FunASRCppOnnxSTTOptions,
        language: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._language = language
        self._request_id = f"funasr-cpp-onnx-{uuid.uuid4().hex[:12]}"

    async def _run(self) -> None:
        client = FunASRCppOnnxWebSocketClient(
            self._opts.url,
            mode=self._opts.mode,
            chunk_size=self._opts.chunk_size,
            sample_rate=self._opts.sample_rate,
            hotwords=self._opts.hotwords,
            wav_name=self._opts.wav_name,
            use_itn=self._opts.use_itn,
        )
        finish_lock = asyncio.Lock()
        final_seen = asyncio.Event()
        sent_audio_since_finish = False
        utterance_seq = 1
        active_timing = _UtteranceTiming(seq=utterance_seq)
        stream_audio_origin_at: float | None = None
        partial_transcript = _PartialTranscriptAccumulator()

        async def finish_current_utterance() -> None:
            nonlocal sent_audio_since_finish
            async with finish_lock:
                if not sent_audio_since_finish:
                    return
                active_timing.endpoint_sent_at = time.perf_counter()
                await client.finish_utterance()
                sent_audio_since_finish = False

        async def forward_audio() -> None:
            nonlocal active_timing
            nonlocal sent_audio_since_finish
            nonlocal stream_audio_origin_at
            async for item in self._input_ch:
                if isinstance(item, self._FlushSentinel):
                    await finish_current_utterance()
                    continue

                await client.send_audio(_audio_frame_bytes(item))
                frame_tail_at = time.perf_counter()
                if active_timing.audio_started_at is None:
                    active_timing.audio_started_at = frame_tail_at - item.duration
                    if stream_audio_origin_at is None:
                        stream_audio_origin_at = active_timing.audio_started_at
                active_timing.audio_tail_at = frame_tail_at
                sent_audio_since_finish = True

            await finish_current_utterance()

        async def receive_results() -> None:
            nonlocal active_timing
            nonlocal utterance_seq
            async for result in client.events():
                timing = active_timing if result.is_final else None
                event = self._speech_event_from_result(
                    result,
                    timing=timing,
                    stream_audio_origin_at=stream_audio_origin_at,
                    partial_transcript=partial_transcript,
                )
                if event is None:
                    continue
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    final_seen.set()
                    utterance_seq += 1
                    active_timing = _UtteranceTiming(
                        seq=utterance_seq,
                        audio_started_at=active_timing.audio_tail_at,
                    )
                self._event_ch.send_nowait(event)

        try:
            await client.connect()
        except Exception as exc:
            raise APIConnectionError(
                f"failed to connect to FunASR C++/ONNX websocket: {self._opts.url}"
            ) from exc

        tasks = {
            asyncio.create_task(forward_audio(), name="FunASRCppOnnxSTT.forward_audio"),
            asyncio.create_task(receive_results(), name="FunASRCppOnnxSTT.receive_results"),
        }

        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()

            forward_done = any(task.get_name() == "FunASRCppOnnxSTT.forward_audio" for task in done)
            if forward_done:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(final_seen.wait(), timeout=self._opts.final_wait_timeout)
            elif pending:
                raise APIConnectionError("FunASR C++/ONNX websocket closed before audio input ended")
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await client.close()

    def _speech_event_from_result(
        self,
        result: ASRResult,
        *,
        timing: _UtteranceTiming | None = None,
        stream_audio_origin_at: float | None = None,
        partial_transcript: _PartialTranscriptAccumulator | None = None,
    ) -> stt.SpeechEvent | None:
        if result.is_final:
            text = result.text
            if partial_transcript is not None:
                partial_transcript.reset()
        else:
            text = result.display_text
            if partial_transcript is not None:
                text = partial_transcript.update(text)
        text = text.strip()
        if not text:
            return None

        event_type = (
            stt.SpeechEventType.FINAL_TRANSCRIPT
            if result.is_final
            else stt.SpeechEventType.INTERIM_TRANSCRIPT
        )
        start_time = 0.0
        end_time = 0.0
        metadata: dict[str, Any] = {"funasr_cpp_onnx": result.raw}
        if timing is not None and result.is_final:
            audio_tail_at, audio_tail_source = _resolve_cpp_final_audio_tail(result, timing)
            asr_final_at = time.perf_counter()
            if stream_audio_origin_at is not None:
                if timing.audio_started_at is not None:
                    start_time = max(0.0, timing.audio_started_at - stream_audio_origin_at)
                if audio_tail_at is not None:
                    end_time = max(0.0, audio_tail_at - stream_audio_origin_at)
            metadata["latency"] = {
                "seq": timing.seq,
                "audio_tail_source": audio_tail_source,
                "audio_tail_at": audio_tail_at,
                "asr_final_at": asr_final_at,
            }
            record_asr_turn(
                timing.seq,
                speech_started_at=timing.audio_started_at,
                speech_ended_at=audio_tail_at,
                endpoint_sent_at=timing.endpoint_sent_at,
                asr_final_at=asr_final_at,
                audio_tail_source=audio_tail_source,
                transcript=text,
            )

        return stt.SpeechEvent(
            type=event_type,
            request_id=self._request_id,
            alternatives=[
                stt.SpeechData(
                    language=LanguageCode(self._language),
                    text=text,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=1.0 if result.is_final else 0.0,
                    metadata=metadata,
                )
            ],
        )


def _resolve_cpp_final_audio_tail(
    result: ASRResult,
    timing: _UtteranceTiming,
) -> tuple[float | None, str]:
    timestamp_end_seconds = _funasr_timestamp_end_seconds(result.raw.get("timestamp"))
    if timestamp_end_seconds is not None and timing.audio_started_at is not None:
        timestamp_tail_at = timing.audio_started_at + timestamp_end_seconds
        if timing.audio_tail_at is not None:
            timestamp_tail_at = min(timestamp_tail_at, timing.audio_tail_at)
        if timestamp_tail_at > timing.audio_started_at + 0.05:
            return timestamp_tail_at, "funasr_timestamp"
    if timing.audio_tail_at is not None:
        return timing.audio_tail_at, "last_audio_frame"
    return timing.endpoint_sent_at, "endpoint_sent"
