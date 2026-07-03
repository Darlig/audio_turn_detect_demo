from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from livekit import rtc
from livekit.agents import (
    APIConnectOptions,
    APIConnectionError,
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    LanguageCode,
    NotGivenOr,
    inference,
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

from asr_module.funasr import FunASRWebSocketClient  # noqa: E402
from asr_module.funasr.funasr_protocol import ASRResult  # noqa: E402

from .turn_metrics import record_asr_turn


DEFAULT_FUNASR_URL = "ws://127.0.0.1:10095"


@dataclass(frozen=True)
class FunASRSTTOptions:
    url: str = DEFAULT_FUNASR_URL
    mode: str = "2pass"
    language: str = "zh"
    sample_rate: int = 16000
    chunk_size: tuple[int, int, int] = (8, 8, 4)
    chunk_interval: int = 10
    hotwords: str = ""
    wav_name: str = "livekit-agent"
    internal_vad: bool = False
    final_wait_timeout: float = 2.0


@dataclass
class _UtteranceTiming:
    seq: int
    audio_started_at: float | None = None
    audio_tail_at: float | None = None
    speech_tail_at: float | None = None
    endpoint_sent_at: float | None = None


class FunASRSTT(stt.STT):
    """LiveKit streaming STT adapter for the local FunASR 2-pass websocket service."""

    def __init__(
        self,
        *,
        url: str | None = None,
        mode: str | None = None,
        language: str | None = None,
        sample_rate: int | None = None,
        chunk_size: tuple[int, int, int] | None = None,
        chunk_interval: int | None = None,
        hotwords: str | None = None,
        wav_name: str | None = None,
        internal_vad: bool | None = None,
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
        self._opts = FunASRSTTOptions(
            url=url or os.getenv("FUNASR_URL", DEFAULT_FUNASR_URL),
            mode=mode or os.getenv("FUNASR_MODE", "2pass"),
            language=language or os.getenv("FUNASR_LANGUAGE", "zh"),
            sample_rate=sample_rate or int(os.getenv("FUNASR_SAMPLE_RATE", "16000")),
            chunk_size=chunk_size or _parse_chunk_size(os.getenv("FUNASR_CHUNK_SIZE")),
            chunk_interval=chunk_interval
            if chunk_interval is not None
            else int(os.getenv("FUNASR_CHUNK_INTERVAL", "10")),
            hotwords=hotwords if hotwords is not None else os.getenv("FUNASR_HOTWORDS", ""),
            wav_name=wav_name or os.getenv("FUNASR_WAV_NAME", "livekit-agent"),
            internal_vad=internal_vad
            if internal_vad is not None
            else _env_bool("FUNASR_INTERNAL_VAD", False),
            final_wait_timeout=final_wait_timeout
            if final_wait_timeout is not None
            else float(os.getenv("FUNASR_FINAL_WAIT_TIMEOUT", "2.0")),
        )

    @property
    def model(self) -> str:
        return self._opts.mode

    @property
    def provider(self) -> str:
        return "funasr"

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
        return FunASRRecognizeStream(
            stt=self,
            opts=self._opts,
            language=_resolve_language(language, self._opts.language),
            conn_options=conn_options,
        )


class FunASRRecognizeStream(stt.RecognizeStream):
    def __init__(
        self,
        *,
        stt: FunASRSTT,
        opts: FunASRSTTOptions,
        language: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._language = language
        self._request_id = f"funasr-{uuid.uuid4().hex[:12]}"

    async def _run(self) -> None:
        client = FunASRWebSocketClient(
            self._opts.url,
            mode=self._opts.mode,
            chunk_size=self._opts.chunk_size,
            chunk_interval=self._opts.chunk_interval,
            hotwords=self._opts.hotwords,
            wav_name=self._opts.wav_name,
        )
        vad_stream = inference.VAD(
            min_silence_duration=0.25,
            max_buffered_speech=60.0,
        ).stream() if self._opts.internal_vad else None

        finish_lock = asyncio.Lock()
        final_seen = asyncio.Event()
        sent_audio_since_finish = False
        utterance_seq = 0
        active_timing = _UtteranceTiming(seq=0)
        pending_timings: deque[_UtteranceTiming] = deque()
        stream_audio_origin_at: float | None = None
        latest_audio_tail_at: float | None = None
        final_seq = 0
        partial_transcript = _PartialTranscriptAccumulator()

        async def finish_current_utterance() -> None:
            nonlocal active_timing, sent_audio_since_finish
            async with finish_lock:
                if not sent_audio_since_finish:
                    return
                active_timing.endpoint_sent_at = time.perf_counter()
                await client.finish_utterance()
                pending_timings.append(active_timing)
                active_timing = _UtteranceTiming(seq=0)
                sent_audio_since_finish = False

        async def forward_audio() -> None:
            nonlocal active_timing
            nonlocal sent_audio_since_finish
            nonlocal stream_audio_origin_at
            nonlocal latest_audio_tail_at
            nonlocal utterance_seq
            async for item in self._input_ch:
                if isinstance(item, self._FlushSentinel):
                    if vad_stream is not None:
                        vad_stream.flush()
                    await finish_current_utterance()
                    continue

                if vad_stream is not None:
                    vad_stream.push_frame(item)

                await client.send_audio(_audio_frame_bytes(item))
                frame_tail_at = time.perf_counter()
                latest_audio_tail_at = frame_tail_at
                if active_timing.seq == 0:
                    utterance_seq += 1
                    active_timing.seq = utterance_seq
                    active_timing.audio_started_at = frame_tail_at - item.duration
                    if stream_audio_origin_at is None:
                        stream_audio_origin_at = active_timing.audio_started_at
                active_timing.audio_tail_at = frame_tail_at
                sent_audio_since_finish = True

            if vad_stream is not None:
                vad_stream.end_input()
            await finish_current_utterance()

        async def receive_results() -> None:
            nonlocal final_seq
            async for result in client.events():
                timing = None
                if result.is_final and pending_timings:
                    timing = pending_timings[0]
                elif result.is_final:
                    final_seq += 1
                    timing = _server_segment_timing(
                        result,
                        seq=final_seq,
                        stream_audio_origin_at=stream_audio_origin_at,
                        latest_audio_tail_at=latest_audio_tail_at,
                    )
                event = self._speech_event_from_result(
                    result,
                    timing=timing,
                    stream_audio_origin_at=stream_audio_origin_at,
                    partial_transcript=partial_transcript,
                )
                if event is None:
                    continue
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    if pending_timings:
                        pending_timings.popleft()
                    final_seen.set()
                self._event_ch.send_nowait(event)

        async def watch_vad() -> None:
            if vad_stream is None:
                return
            nonlocal active_timing
            async for event in vad_stream:
                if event.type.name == "END_OF_SPEECH":
                    silence_duration = float(getattr(event, "silence_duration", 0.0) or 0.0)
                    inference_duration = float(getattr(event, "inference_duration", 0.0) or 0.0)
                    speech_tail_at = time.perf_counter() - silence_duration - inference_duration
                    if active_timing.audio_started_at is not None:
                        active_timing.speech_tail_at = max(
                            active_timing.audio_started_at,
                            speech_tail_at,
                        )
                    await finish_current_utterance()

        try:
            await client.connect()
        except Exception as exc:
            raise APIConnectionError(f"failed to connect to FunASR websocket: {self._opts.url}") from exc

        tasks = {
            asyncio.create_task(forward_audio(), name="FunASRSTT.forward_audio"),
            asyncio.create_task(receive_results(), name="FunASRSTT.receive_results"),
        }
        if vad_stream is not None:
            tasks.add(asyncio.create_task(watch_vad(), name="FunASRSTT.watch_vad"))

        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()

            forward_done = any(task.get_name() == "FunASRSTT.forward_audio" for task in done)
            if forward_done:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(final_seen.wait(), timeout=self._opts.final_wait_timeout)
            elif pending:
                raise APIConnectionError("FunASR websocket closed before audio input ended")
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if vad_stream is not None:
                await vad_stream.aclose()
            await client.close()

    def _speech_event_from_result(
        self,
        result: ASRResult,
        *,
        timing: _UtteranceTiming | None = None,
        stream_audio_origin_at: float | None = None,
        partial_transcript: "_PartialTranscriptAccumulator | None" = None,
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
        metadata: dict[str, Any] = {"funasr": result.raw}
        if timing is not None and result.is_final:
            audio_tail_at, audio_tail_source = _resolve_final_audio_tail(result, timing)
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


class _PartialTranscriptAccumulator:
    def __init__(self) -> None:
        self.text = ""

    def reset(self) -> None:
        self.text = ""

    def update(self, partial: str) -> str:
        partial = partial.strip()
        if not partial:
            return self.text
        if not self.text:
            self.text = partial
        elif partial == self.text:
            pass
        elif partial.startswith(self.text):
            self.text = partial
        elif self.text in partial and len(partial) > len(self.text):
            self.text = partial
        elif not self.text.endswith(partial):
            self.text += partial
        return self.text


def _iter_audio_buffer(buffer: AudioBuffer) -> list[rtc.AudioFrame]:
    if isinstance(buffer, rtc.AudioFrame):
        return [buffer]
    return list(buffer)


def _audio_frame_bytes(frame: rtc.AudioFrame) -> bytes:
    if frame.num_channels > 1:
        return _downmix_to_mono_pcm16le(frame)
    data = frame.data
    if hasattr(data, "tobytes"):
        return data.tobytes()
    return bytes(data)


def _downmix_to_mono_pcm16le(frame: rtc.AudioFrame) -> bytes:
    samples = _pcm16_samples(frame)
    channels = frame.num_channels
    if channels <= 0:
        return b""

    mono = bytearray(frame.samples_per_channel * 2)
    for sample_idx in range(frame.samples_per_channel):
        offset = sample_idx * channels
        mixed = 0
        for channel_idx in range(channels):
            mixed += int(samples[offset + channel_idx])
        mixed = round(mixed / channels)
        mixed = max(-32768, min(32767, mixed))
        mono[sample_idx * 2 : sample_idx * 2 + 2] = int(mixed).to_bytes(
            2,
            "little",
            signed=True,
        )
    return bytes(mono)


def _pcm16_samples(frame: rtc.AudioFrame) -> memoryview:
    data = frame.data if isinstance(frame.data, memoryview) else memoryview(frame.data)
    if data.format == "h":
        return data
    return data.cast("h")


def _resolve_final_audio_tail(
    result: ASRResult,
    timing: _UtteranceTiming,
) -> tuple[float | None, str]:
    timestamp_end_seconds = _funasr_timestamp_end_seconds(result.raw.get("timestamp"))
    if timestamp_end_seconds is not None and timing.audio_started_at is not None:
        timestamp_tail_at = timing.audio_started_at + timestamp_end_seconds
        if timing.audio_tail_at is not None:
            timestamp_tail_at = min(timestamp_tail_at, timing.audio_tail_at)
        if _is_plausible_timestamp_tail(timestamp_tail_at, timing):
            return timestamp_tail_at, "funasr_timestamp"
    if timing.speech_tail_at is not None:
        return timing.speech_tail_at, "internal_vad"
    if timing.audio_tail_at is not None:
        return timing.audio_tail_at, "last_audio_frame"
    return timing.endpoint_sent_at, "endpoint_sent"


def _server_segment_timing(
    result: ASRResult,
    *,
    seq: int,
    stream_audio_origin_at: float | None,
    latest_audio_tail_at: float | None,
) -> _UtteranceTiming | None:
    if stream_audio_origin_at is None:
        return None
    segment_end_ms = _numeric_raw_value(result.raw.get("segment_end_ms"))
    if segment_end_ms is None:
        return None
    segment_start_ms = _numeric_raw_value(result.raw.get("segment_start_ms")) or 0.0
    segment_seq = int(_numeric_raw_value(result.raw.get("segment_seq")) or seq)
    audio_started_at = stream_audio_origin_at + segment_start_ms / 1000.0
    audio_tail_at = stream_audio_origin_at + segment_end_ms / 1000.0
    if latest_audio_tail_at is not None:
        audio_tail_at = min(audio_tail_at, latest_audio_tail_at)
    return _UtteranceTiming(
        seq=segment_seq,
        audio_started_at=audio_started_at,
        audio_tail_at=audio_tail_at,
        speech_tail_at=audio_tail_at,
    )


def _numeric_raw_value(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _is_plausible_timestamp_tail(timestamp_tail_at: float, timing: _UtteranceTiming) -> bool:
    if timing.audio_started_at is not None and timestamp_tail_at <= timing.audio_started_at + 0.05:
        return False
    if timing.speech_tail_at is not None:
        # FunASR timestamps are useful when they agree with the local speech-end
        # estimate. If they point far earlier, they usually came from empty/old
        # timestamp data and produce a fake multi-second STT latency.
        return abs(timestamp_tail_at - timing.speech_tail_at) <= 2.0
    if timing.audio_tail_at is not None and timestamp_tail_at > timing.audio_tail_at + 0.1:
        return False
    return True


def _funasr_timestamp_end_seconds(value: Any) -> float | None:
    end_ms = _max_numeric_leaf(value, prefer_index=1)
    if end_ms is None or end_ms <= 0:
        return None
    # FunASR timestamp values are normally in milliseconds. If a custom backend
    # already returns seconds, keep small values as seconds.
    return end_ms / 1000.0 if end_ms > 60.0 else end_ms


def _max_numeric_leaf(value: Any, *, prefer_index: int | None = None) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        candidates = [_max_numeric_leaf(item, prefer_index=prefer_index) for item in value.values()]
        return _max_or_none(candidates)
    if isinstance(value, (list, tuple)):
        candidates: list[float | None] = []
        for item in value:
            if (
                prefer_index is not None
                and isinstance(item, (list, tuple))
                and len(item) > prefer_index
            ):
                candidates.append(_max_numeric_leaf(item[prefer_index], prefer_index=None))
            else:
                candidates.append(_max_numeric_leaf(item, prefer_index=prefer_index))
        return _max_or_none(candidates)
    return None


def _max_or_none(values: list[float | None]) -> float | None:
    numeric = [item for item in values if item is not None]
    return max(numeric) if numeric else None


def _resolve_language(language: NotGivenOr[str], default: str) -> str:
    return language if language is not NOT_GIVEN and language else default


def _parse_chunk_size(raw: str | None) -> tuple[int, int, int]:
    if not raw:
        return (8, 8, 4)
    parts = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if len(parts) != 3:
        raise ValueError("FUNASR_CHUNK_SIZE must have exactly three comma-separated integers")
    return (parts[0], parts[1], parts[2])


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
