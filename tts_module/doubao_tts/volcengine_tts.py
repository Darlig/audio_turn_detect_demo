from __future__ import annotations

from dataclasses import dataclass
import asyncio
import gzip
import json
import os
import struct
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
    tts,
)
from livekit.agents import utils

from .turn_metrics import claim_tts_turn_sequence, elapsed_ms, format_metric, record_tts_first_audio
from .turn_metrics import publish_turn_latency_metrics


DEFAULT_TTS_WS_URL = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
DEFAULT_TTS_RESOURCE_ID = "seed-tts-2.0"
LEGACY_TTS_RESOURCE_IDS = {"volc.service_type.10029"}

PROTOCOL_VERSION = 0x1
HEADER_SIZE_WORDS = 0x1
FLAG_WITH_EVENT = 0x4

MESSAGE_TYPE_FULL_CLIENT_REQUEST = 0x1
MESSAGE_TYPE_AUDIO_ONLY_REQUEST = 0x2
MESSAGE_TYPE_FULL_SERVER_RESPONSE = 0x9
MESSAGE_TYPE_AUDIO_ONLY_RESPONSE = 0xB
MESSAGE_TYPE_ERROR = 0xF

SERIALIZATION_RAW = 0x0
SERIALIZATION_JSON = 0x1
COMPRESSION_NONE = 0x0
COMPRESSION_GZIP = 0x1

EVENT_START_CONNECTION = 1
EVENT_FINISH_CONNECTION = 2
EVENT_CONNECTION_STARTED = 50
EVENT_CONNECTION_FAILED = 51
EVENT_CONNECTION_FINISHED = 52
EVENT_START_SESSION = 100
EVENT_CANCEL_SESSION = 101
EVENT_FINISH_SESSION = 102
EVENT_SESSION_STARTED = 150
EVENT_SESSION_CANCELED = 151
EVENT_SESSION_FINISHED = 152
EVENT_SESSION_FAILED = 153
EVENT_TASK_REQUEST = 200
EVENT_TTS_SENTENCE_START = 350
EVENT_TTS_SENTENCE_END = 351
EVENT_TTS_RESPONSE = 352

FAILED_EVENTS = {
    EVENT_CONNECTION_FAILED,
    EVENT_SESSION_CANCELED,
    EVENT_SESSION_FAILED,
}
CONNECTION_EVENTS = {
    EVENT_CONNECTION_STARTED,
    EVENT_CONNECTION_FAILED,
    EVENT_CONNECTION_FINISHED,
}
SESSION_EVENTS = {
    EVENT_SESSION_STARTED,
    EVENT_SESSION_CANCELED,
    EVENT_SESSION_FINISHED,
    EVENT_SESSION_FAILED,
    EVENT_TTS_SENTENCE_START,
    EVENT_TTS_SENTENCE_END,
    EVENT_TTS_RESPONSE,
}
OK_STATUS_CODE = 20000000
DEFAULT_STREAM_BREAK_CHARS = "，,。.!！?？;；:：\n"


@dataclass
class VolcengineTTSOptions:
    api_key: str | None
    access_key: str | None
    app_id: str | None
    voice_type: str
    ws_url: str = DEFAULT_TTS_WS_URL
    resource_id: str = DEFAULT_TTS_RESOURCE_ID
    sample_rate: int = 24000
    encoding: str = "pcm"
    speed_ratio: float = 1.0
    uid: str = "livekit-user"


@dataclass
class VolcengineFrame:
    message_type: int
    flags: int
    serialization: int
    compression: int
    event: int | None
    payload: bytes
    payload_json: dict[str, Any] | None = None
    connection_id: str | None = None
    session_id: str | None = None
    error_code: int | None = None


@dataclass(frozen=True)
class SynthesizedPCM:
    pcm: bytes
    sample_rate: int
    num_channels: int = 1


class VolcengineStreamingTTS(tts.TTS):
    """LiveKit TTS adapter for Doubao/Volcengine bidirectional TTS V3.

    The public surface is LiveKit's `tts.TTS` interface. Provider-specific V3
    websocket framing and event handling are intentionally isolated in this
    module so the rest of the voice agent can swap TTS providers later.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        access_key: str | None = None,
        app_id: str | None = None,
        voice_type: str | None = None,
        ws_url: str | None = None,
        resource_id: str | None = None,
        sample_rate: int | None = None,
        encoding: str | None = None,
        speed_ratio: float | None = None,
        uid: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        resolved_api_key = api_key or os.getenv("VOLC_TTS_API_KEY")
        resolved_access_key = access_key or os.getenv("VOLC_TTS_ACCESS_KEY")
        resolved_app_id = app_id or os.getenv("VOLC_TTS_APP_ID")
        resolved_voice_type = voice_type or os.getenv("VOLC_TTS_VOICE_TYPE")
        if not resolved_voice_type:
            raise ValueError("missing required Volcengine TTS config: VOLC_TTS_VOICE_TYPE")
        if not resolved_api_key and not (resolved_app_id and resolved_access_key):
            raise ValueError(
                "missing required Volcengine TTS auth: set VOLC_TTS_API_KEY or "
                "both VOLC_TTS_APP_ID and VOLC_TTS_ACCESS_KEY"
            )

        resolved_resource_id = normalize_resource_id(
            resource_id or os.getenv("VOLC_TTS_RESOURCE_ID", DEFAULT_TTS_RESOURCE_ID)
        )
        resolved_encoding = encoding or os.getenv("VOLC_TTS_ENCODING", "pcm")

        self._opts = VolcengineTTSOptions(
            api_key=resolved_api_key,
            access_key=resolved_access_key,
            app_id=resolved_app_id,
            voice_type=resolved_voice_type,
            ws_url=ws_url or os.getenv("VOLC_TTS_WS_URL", DEFAULT_TTS_WS_URL),
            resource_id=resolved_resource_id,
            sample_rate=sample_rate or int(os.getenv("VOLC_TTS_SAMPLE_RATE", "24000")),
            encoding=resolved_encoding,
            speed_ratio=speed_ratio
            if speed_ratio is not None
            else float(os.getenv("VOLC_TTS_SPEED_RATIO", "1.0")),
            uid=uid or os.getenv("VOLC_TTS_UID", "livekit-user"),
        )
        mime_type = "audio/pcm" if self._opts.encoding == "pcm" else f"audio/{self._opts.encoding}"
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, aligned_transcript=False),
            sample_rate=self._opts.sample_rate,
            num_channels=1,
        )
        self._mime_type = mime_type
        self._session = http_session
        self._owns_session = http_session is None
        self._request_seq = 0

    @property
    def model(self) -> str:
        return self._opts.voice_type

    @property
    def provider(self) -> str:
        return "volcengine-tts"

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.SynthesizeStream:
        self._request_seq += 1
        turn_sequence = claim_tts_turn_sequence(self._request_seq)
        return VolcengineSynthesizeStream(
            tts=self,
            conn_options=conn_options,
            turn_sequence=turn_sequence,
        )

    async def aclose(self) -> None:
        if self._owns_session and self._session:
            await self._session.close()
        self._session = None

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _connect(self, request_id: str, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        header_variants = build_volcengine_ws_header_variants(self._opts, request_id=request_id)
        last_status_error: APIStatusError | None = None

        for idx, headers in enumerate(header_variants):
            try:
                return await asyncio.wait_for(
                    session.ws_connect(
                        self._opts.ws_url,
                        headers=headers,
                        max_msg_size=0,
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError as exc:
                raise APITimeoutError("Volcengine TTS websocket connection timed out") from exc
            except aiohttp.ClientResponseError as exc:
                body = await self._fetch_handshake_error_body(session, headers)
                last_status_error = APIStatusError(
                    f"Volcengine TTS websocket rejected the connection: {body or exc.message}",
                    status_code=exc.status,
                    request_id=request_id,
                    body=body,
                    retryable=exc.status >= 500,
                )
                can_try_next = idx < len(header_variants) - 1 and exc.status in {401, 403}
                if can_try_next:
                    continue
                raise last_status_error from exc
            except aiohttp.ClientError as exc:
                raise APIConnectionError("failed to connect to Volcengine TTS websocket") from exc

        if last_status_error:
            raise last_status_error
        raise APIConnectionError("failed to connect to Volcengine TTS websocket")

    async def _fetch_handshake_error_body(
        self, session: aiohttp.ClientSession, headers: dict[str, str]
    ) -> str:
        try:
            async with session.get(
                self._opts.ws_url.replace("wss://", "https://", 1).replace(
                    "ws://", "http://", 1
                ),
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                return (await resp.text())[:500]
        except Exception:
            return ""


async def synthesize_text_to_pcm(
    text: str,
    *,
    client: VolcengineStreamingTTS | None = None,
    conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
) -> SynthesizedPCM:
    """Synthesize one text string and collect all PCM bytes.

    The LiveKit agent consumes the streaming TTS interface above. The web text-audio
    demo needs a fully materialized WAV file first, so it uses the same Volcengine
    protocol but collects audio frames instead of emitting them to LiveKit.
    """

    clean_text = text.strip()
    if not clean_text:
        return SynthesizedPCM(pcm=b"", sample_rate=24000)

    owns_client = client is None
    tts_client = client or VolcengineStreamingTTS(encoding="pcm")
    opts = tts_client._opts
    if opts.encoding != "pcm":
        raise ValueError("synthesize_text_to_pcm requires PCM TTS encoding")

    request_id = str(uuid.uuid4())
    session_id = uuid.uuid4().hex[:12]
    chunks: list[bytes] = []
    ws: aiohttp.ClientWebSocketResponse | None = None

    try:
        ws = await tts_client._connect(request_id, conn_options.timeout)
        await ws.send_bytes(build_client_event_frame(EVENT_START_CONNECTION, {}))
        frame = await receive_volcengine_frame(ws, timeout=conn_options.timeout)
        raise_for_failed_frame(frame, request_id=request_id)
        if frame.event != EVENT_CONNECTION_STARTED:
            raise unexpected_event_error(
                "start connection",
                frame,
                request_id=request_id,
                expected=EVENT_CONNECTION_STARTED,
            )

        await ws.send_bytes(
            build_client_event_frame(
                EVENT_START_SESSION,
                build_tts_session_payload(opts, "", event=EVENT_START_SESSION),
                session_id=session_id,
            )
        )

        task_sent = False
        finish_sent = False
        session_finished = False
        connection_finished = False

        while True:
            frame = await receive_volcengine_frame(ws, timeout=conn_options.timeout)
            raise_for_failed_frame(frame, request_id=request_id)

            if frame.event == EVENT_SESSION_STARTED:
                if not task_sent:
                    await ws.send_bytes(
                        build_client_event_frame(
                            EVENT_TASK_REQUEST,
                            build_tts_session_payload(
                                opts,
                                clean_text,
                                event=EVENT_TASK_REQUEST,
                            ),
                            session_id=session_id,
                        )
                    )
                    task_sent = True
                if not finish_sent:
                    await ws.send_bytes(
                        build_client_event_frame(
                            EVENT_FINISH_SESSION,
                            {"event": EVENT_FINISH_SESSION},
                            session_id=session_id,
                        )
                    )
                    finish_sent = True
                continue

            if frame.event == EVENT_TTS_RESPONSE and frame.payload:
                chunks.append(frame.payload)
                continue

            if frame.event in {EVENT_TTS_SENTENCE_START, EVENT_TTS_SENTENCE_END}:
                continue

            if frame.event == EVENT_SESSION_FINISHED:
                session_finished = True
                break

            if frame.event == EVENT_CONNECTION_FINISHED:
                connection_finished = True
                break

        if not finish_sent and not session_finished and not connection_finished:
            await ws.send_bytes(
                build_client_event_frame(
                    EVENT_FINISH_SESSION,
                    {"event": EVENT_FINISH_SESSION},
                    session_id=session_id,
                )
            )

        if not connection_finished:
            await ws.send_bytes(build_client_event_frame(EVENT_FINISH_CONNECTION, {}))
            while True:
                frame = await receive_volcengine_frame(ws, timeout=conn_options.timeout)
                raise_for_failed_frame(frame, request_id=request_id)
                if frame.event == EVENT_CONNECTION_FINISHED:
                    break
                if frame.event in {EVENT_SESSION_FINISHED, EVENT_TTS_SENTENCE_END}:
                    continue
                break
    finally:
        if ws is not None:
            await ws.close()
        if owns_client:
            await tts_client.aclose()

    return SynthesizedPCM(pcm=b"".join(chunks), sample_rate=opts.sample_rate)


class VolcengineSynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        tts: VolcengineStreamingTTS,
        conn_options: APIConnectOptions,
        turn_sequence: int,
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: VolcengineStreamingTTS = tts
        self._opts = tts._opts
        self._turn_sequence = turn_sequence

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = str(uuid.uuid4())
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type=self._tts._mime_type,
            stream=True,
            frame_size_ms=80,
        )

        ws = await self._tts._connect(request_id, self._conn_options.timeout)
        session_id = uuid.uuid4().hex[:12]
        segment_started = False
        first_audio_logged = False
        session_finished = False
        connection_finished = False
        finish_sent = False
        tts_started_at: float | None = None
        sent_text_chars = 0
        sent_segments = 0

        try:
            await ws.send_bytes(build_client_event_frame(EVENT_START_CONNECTION, {}))
            frame = await receive_volcengine_frame(ws, timeout=self._conn_options.timeout)
            raise_for_failed_frame(frame, request_id=request_id)
            if frame.event != EVENT_CONNECTION_STARTED:
                raise unexpected_event_error(
                    "start connection",
                    frame,
                    request_id=request_id,
                    expected=EVENT_CONNECTION_STARTED,
                )

            await ws.send_bytes(
                build_client_event_frame(
                    EVENT_START_SESSION,
                    build_tts_session_payload(self._opts, "", event=EVENT_START_SESSION),
                    session_id=session_id,
                )
            )

            while True:
                frame = await receive_volcengine_frame(ws, timeout=self._conn_options.timeout)
                raise_for_failed_frame(frame, request_id=request_id)
                if frame.event == EVENT_SESSION_STARTED:
                    break
                if frame.event == EVENT_SESSION_FINISHED:
                    session_finished = True
                    output_emitter.end_input()
                    return
                if frame.event == EVENT_CONNECTION_FINISHED:
                    connection_finished = True
                    output_emitter.end_input()
                    return

            first_task_sent = asyncio.Event()

            async def send_input_text() -> None:
                nonlocal finish_sent, sent_text_chars, sent_segments, tts_started_at

                send_mode = os.getenv("VOLC_TTS_STREAM_SEND_MODE", "chunk").strip().lower()
                log_input_chunks = _env_bool("VOLC_TTS_LOG_INPUT_CHUNKS", False)
                segmenter = None
                if send_mode == "segment":
                    segmenter = _StreamingTextSegmenter(
                        min_chars=int(os.getenv("VOLC_TTS_STREAM_MIN_CHARS", "6")),
                        max_chars=int(os.getenv("VOLC_TTS_STREAM_MAX_CHARS", "48")),
                        break_chars=os.getenv(
                            "VOLC_TTS_STREAM_BREAK_CHARS",
                            DEFAULT_STREAM_BREAK_CHARS,
                        ),
                    )

                async def send_text_piece(piece: str) -> None:
                    nonlocal sent_text_chars, sent_segments, tts_started_at
                    text = piece.strip() if segmenter is not None else piece
                    if not text.strip():
                        return
                    if tts_started_at is None:
                        tts_started_at = time.perf_counter()
                        self._mark_started()
                    await ws.send_bytes(
                        build_client_event_frame(
                            EVENT_TASK_REQUEST,
                            build_tts_session_payload(
                                self._opts,
                                text,
                                event=EVENT_TASK_REQUEST,
                            ),
                            session_id=session_id,
                        )
                    )
                    sent_text_chars += len(text)
                    sent_segments += 1
                    first_task_sent.set()
                    if log_input_chunks:
                        print(
                            "[turn_metrics] "
                            f"stage=tts_input seq={self._turn_sequence} "
                            f"segment={sent_segments} chars={len(text)} text={text}",
                            flush=True,
                        )

                async for data in self._input_ch:
                    if isinstance(data, self._FlushSentinel):
                        ready_segments = segmenter.flush() if segmenter is not None else []
                    else:
                        ready_segments = (
                            segmenter.push(data) if segmenter is not None else [data]
                        )
                    for ready_text in ready_segments:
                        await send_text_piece(ready_text)

                if segmenter is not None:
                    for ready_text in segmenter.close():
                        await send_text_piece(ready_text)

                await ws.send_bytes(
                    build_client_event_frame(
                        EVENT_FINISH_SESSION,
                        {"event": EVENT_FINISH_SESSION},
                        session_id=session_id,
                    )
                )
                finish_sent = True

            sender_task = asyncio.create_task(send_input_text(), name="VolcengineTTS.send_input")

            while True:
                if sender_task.done():
                    sender_task.result()

                if not first_task_sent.is_set():
                    first_task_waiter = asyncio.create_task(first_task_sent.wait())
                    await asyncio.wait(
                        {first_task_waiter, sender_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if not first_task_waiter.done():
                        first_task_waiter.cancel()
                    if sender_task.done() and not first_task_sent.is_set():
                        sender_task.result()
                        break

                frame = await receive_volcengine_frame(ws, timeout=self._conn_options.timeout)
                raise_for_failed_frame(frame, request_id=request_id)

                if frame.event == EVENT_TTS_RESPONSE and frame.payload:
                    if not segment_started:
                        output_emitter.start_segment(segment_id=session_id)
                        segment_started = True
                    output_emitter.push(frame.payload)
                    if not first_audio_logged:
                        first_audio_at = time.perf_counter()
                        timing = record_tts_first_audio(
                            self._turn_sequence,
                            tts_started_at=tts_started_at or first_audio_at,
                            tts_first_audio_at=first_audio_at,
                        )
                        print(
                            "[turn_metrics] "
                            f"stage=tts seq={self._turn_sequence} "
                            f"first_audio_ms={format_metric(elapsed_ms(tts_started_at, first_audio_at))} "
                            f"user_speech_end_to_tts_first_audio_ms="
                            f"{format_metric(elapsed_ms(timing.speech_ended_at, first_audio_at))} "
                            f"asr_final_to_tts_first_audio_ms="
                            f"{format_metric(elapsed_ms(timing.asr_final_at, first_audio_at))} "
                            f"text_chars_sent={sent_text_chars} "
                            f"segments_sent={sent_segments} "
                            f"first_audio_bytes={len(frame.payload)}",
                            flush=True,
                        )
                        publish_turn_latency_metrics(self._turn_sequence)
                        first_audio_logged = True
                    continue

                if frame.event in {EVENT_TTS_SENTENCE_START, EVENT_TTS_SENTENCE_END}:
                    continue

                if frame.event == EVENT_SESSION_FINISHED:
                    session_finished = True
                    break

                if frame.event == EVENT_CONNECTION_FINISHED:
                    connection_finished = True
                    break

            if segment_started:
                output_emitter.end_segment()

            if not sender_task.done():
                sender_task.cancel()
                try:
                    await sender_task
                except asyncio.CancelledError:
                    pass
            else:
                sender_task.result()

            if not finish_sent and not session_finished and not connection_finished:
                await ws.send_bytes(
                    build_client_event_frame(
                        EVENT_FINISH_SESSION,
                        {"event": EVENT_FINISH_SESSION},
                        session_id=session_id,
                    )
                )

            if not connection_finished:
                await ws.send_bytes(build_client_event_frame(EVENT_FINISH_CONNECTION, {}))
                while True:
                    frame = await receive_volcengine_frame(ws, timeout=self._conn_options.timeout)
                    raise_for_failed_frame(frame, request_id=request_id)
                    if frame.event == EVENT_CONNECTION_FINISHED:
                        break
                    if frame.event in {EVENT_SESSION_FINISHED, EVENT_TTS_SENTENCE_END}:
                        continue
                    break
        finally:
            await ws.close()
            output_emitter.end_input()

    async def _collect_text(self) -> str:
        parts: list[str] = []
        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                continue
            if data:
                parts.append(data)
        return "".join(parts).strip()


class _StreamingTextSegmenter:
    def __init__(self, *, min_chars: int, max_chars: int, break_chars: str) -> None:
        self._min_chars = max(1, min_chars)
        self._max_chars = max(self._min_chars, max_chars)
        self._break_chars = set(break_chars)
        self._buffer = ""

    def push(self, text: str) -> list[str]:
        self._buffer += text
        return self._pop_ready(force=False)

    def flush(self) -> list[str]:
        return self._pop_ready(force=True)

    def close(self) -> list[str]:
        return self.flush()

    def _pop_ready(self, *, force: bool) -> list[str]:
        segments: list[str] = []
        while self._buffer:
            split_at = self._find_split(force=force)
            if split_at <= 0:
                break
            segment = self._buffer[:split_at].strip()
            self._buffer = self._buffer[split_at:]
            if segment:
                segments.append(segment)
            if force:
                continue
        return segments

    def _find_split(self, *, force: bool) -> int:
        if force:
            return len(self._buffer)

        last_break = -1
        for idx, char in enumerate(self._buffer):
            if char in self._break_chars and idx + 1 >= self._min_chars:
                last_break = idx + 1

        if last_break > 0:
            return last_break

        if len(self._buffer) >= self._max_chars:
            return self._max_chars

        return -1


def normalize_resource_id(resource_id: str | None) -> str:
    if not resource_id or resource_id in LEGACY_TTS_RESOURCE_IDS:
        return DEFAULT_TTS_RESOURCE_ID
    return resource_id


def build_volcengine_ws_headers(opts: VolcengineTTSOptions, *, request_id: str) -> dict[str, str]:
    return build_volcengine_ws_header_variants(opts, request_id=request_id)[0]


def build_volcengine_ws_header_variants(
    opts: VolcengineTTSOptions, *, request_id: str
) -> list[dict[str, str]]:
    headers = {
        "X-Api-Resource-Id": normalize_resource_id(opts.resource_id),
        "X-Api-Connect-Id": request_id,
    }
    variants: list[dict[str, str]] = []
    if opts.api_key:
        variants.append({**headers, "X-Api-Key": opts.api_key})

    if opts.app_id and opts.access_key:
        variants.append(
            {
                **headers,
                "X-Api-App-Id": opts.app_id,
                "X-Api-Access-Key": opts.access_key,
            }
        )

    if variants:
        return variants

    raise ValueError("missing Volcengine TTS websocket credentials")


def build_tts_session_payload(
    opts: VolcengineTTSOptions, text: str, *, event: int
) -> dict[str, Any]:
    audio_params: dict[str, Any] = {
        "format": opts.encoding,
        "sample_rate": opts.sample_rate,
    }
    speech_rate = speed_ratio_to_speech_rate(opts.speed_ratio)
    if speech_rate:
        audio_params["speech_rate"] = speech_rate

    return {
        "user": {"uid": opts.uid},
        "event": event,
        "namespace": "BidirectionalTTS",
        "req_params": {
            "text": text,
            "speaker": opts.voice_type,
            "audio_params": audio_params,
        },
    }


def speed_ratio_to_speech_rate(speed_ratio: float) -> int:
    speech_rate = round((speed_ratio - 1.0) * 100)
    return max(-50, min(100, speech_rate))


def build_client_event_frame(
    event: int,
    payload: dict[str, Any] | bytes | None,
    *,
    session_id: str | None = None,
    message_type: int = MESSAGE_TYPE_FULL_CLIENT_REQUEST,
    serialization: int = SERIALIZATION_JSON,
) -> bytes:
    if payload is None:
        payload_bytes = b"{}" if serialization == SERIALIZATION_JSON else b""
    elif isinstance(payload, bytes):
        payload_bytes = payload
    else:
        payload_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )

    frame = bytearray(
        [
            (PROTOCOL_VERSION << 4) | HEADER_SIZE_WORDS,
            (message_type << 4) | FLAG_WITH_EVENT,
            (serialization << 4) | COMPRESSION_NONE,
            0,
        ]
    )
    frame.extend(struct.pack(">i", event))

    if session_id is not None:
        session_id_bytes = session_id.encode("utf-8")
        frame.extend(struct.pack(">I", len(session_id_bytes)))
        frame.extend(session_id_bytes)

    frame.extend(struct.pack(">I", len(payload_bytes)))
    frame.extend(payload_bytes)
    return bytes(frame)


async def receive_volcengine_frame(
    ws: aiohttp.ClientWebSocketResponse, *, timeout: float
) -> VolcengineFrame:
    try:
        msg = await ws.receive(timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise APITimeoutError("Volcengine TTS websocket receive timed out") from exc

    if msg.type in (
        aiohttp.WSMsgType.CLOSE,
        aiohttp.WSMsgType.CLOSED,
        aiohttp.WSMsgType.CLOSING,
    ):
        raise APIConnectionError("Volcengine TTS websocket closed")

    if msg.type == aiohttp.WSMsgType.ERROR:
        raise APIConnectionError("Volcengine TTS websocket error")

    return parse_tts_ws_message(msg)


def parse_tts_ws_message(msg: aiohttp.WSMessage) -> VolcengineFrame:
    if msg.type == aiohttp.WSMsgType.TEXT:
        raise APIStatusError(
            f"Volcengine TTS returned unexpected text frame: {msg.data}",
            body=msg.data,
            retryable=False,
        )
    if msg.type != aiohttp.WSMsgType.BINARY:
        raise APIConnectionError(f"unexpected Volcengine TTS websocket frame: {msg.type}")

    return parse_volcengine_binary_frame(bytes(msg.data))


def parse_volcengine_binary_frame(data: bytes) -> VolcengineFrame:
    if len(data) < 4:
        raise APIConnectionError("Volcengine TTS returned a truncated frame")

    version = data[0] >> 4
    header_words = data[0] & 0x0F
    if version != PROTOCOL_VERSION:
        raise APIConnectionError(f"unsupported Volcengine TTS protocol version: {version}")

    offset = header_words * 4
    if len(data) < offset:
        raise APIConnectionError("Volcengine TTS returned an invalid header size")

    message_type = data[1] >> 4
    flags = data[1] & 0x0F
    serialization = data[2] >> 4
    compression = data[2] & 0x0F
    event: int | None = None
    error_code: int | None = None
    connection_id: str | None = None
    session_id: str | None = None

    if message_type == MESSAGE_TYPE_ERROR:
        error_code, offset = read_i32(data, offset)
        payload = read_sized_payload_if_present(data, offset)
        payload = maybe_decompress(payload, compression=compression)
        payload_json = decode_payload_json(payload, serialization=serialization)
        return VolcengineFrame(
            message_type=message_type,
            flags=flags,
            serialization=serialization,
            compression=compression,
            event=None,
            payload=payload,
            payload_json=payload_json,
            error_code=error_code,
        )

    if flags & FLAG_WITH_EVENT:
        event, offset = read_i32(data, offset)

    if event in CONNECTION_EVENTS:
        connection_id, offset = read_optional_id(data, offset)
    elif event in SESSION_EVENTS:
        session_id, offset = read_optional_id(data, offset)

    payload, offset = read_sized_payload(data, offset)
    payload = maybe_decompress(payload, compression=compression)
    payload_json = decode_payload_json(payload, serialization=serialization)

    return VolcengineFrame(
        message_type=message_type,
        flags=flags,
        serialization=serialization,
        compression=compression,
        event=event,
        payload=payload,
        payload_json=payload_json,
        connection_id=connection_id,
        session_id=session_id,
    )


def read_i32(data: bytes, offset: int) -> tuple[int, int]:
    if offset + 4 > len(data):
        raise APIConnectionError("Volcengine TTS returned a truncated int32 field")
    return struct.unpack(">i", data[offset : offset + 4])[0], offset + 4


def read_u32(data: bytes, offset: int) -> tuple[int, int]:
    if offset + 4 > len(data):
        raise APIConnectionError("Volcengine TTS returned a truncated uint32 field")
    return struct.unpack(">I", data[offset : offset + 4])[0], offset + 4


def read_optional_id(data: bytes, offset: int) -> tuple[str | None, int]:
    if offset + 4 > len(data):
        return None, offset
    id_len, offset_after_len = read_u32(data, offset)
    if offset_after_len + id_len > len(data):
        return None, offset
    raw_id = data[offset_after_len : offset_after_len + id_len]
    return raw_id.decode("utf-8", errors="replace"), offset_after_len + id_len


def read_sized_payload(data: bytes, offset: int) -> tuple[bytes, int]:
    payload_len, offset = read_u32(data, offset)
    if offset + payload_len > len(data):
        raise APIConnectionError("Volcengine TTS returned a truncated payload")
    return data[offset : offset + payload_len], offset + payload_len


def read_sized_payload_if_present(data: bytes, offset: int) -> bytes:
    if offset + 4 > len(data):
        return data[offset:]
    payload_len, next_offset = read_u32(data, offset)
    if next_offset + payload_len <= len(data):
        return data[next_offset : next_offset + payload_len]
    return data[offset:]


def maybe_decompress(payload: bytes, *, compression: int) -> bytes:
    if compression == COMPRESSION_GZIP and payload:
        return gzip.decompress(payload)
    return payload


def decode_payload_json(payload: bytes, *, serialization: int) -> dict[str, Any] | None:
    if serialization != SERIALIZATION_JSON or not payload:
        return None
    decoded = json.loads(payload.decode("utf-8"))
    return decoded if isinstance(decoded, dict) else {"value": decoded}


def raise_for_failed_frame(frame: VolcengineFrame, *, request_id: str) -> None:
    body = frame.payload_json or (frame.payload.decode("utf-8", errors="replace") if frame.payload else "")
    status_code = extract_status_code(frame)
    if frame.message_type == MESSAGE_TYPE_ERROR:
        raise APIStatusError(
            format_frame_error("Volcengine TTS error frame", frame, body),
            status_code=frame.error_code or status_code or -1,
            request_id=request_id,
            body=body,
            retryable=is_retryable_status(frame.error_code or status_code),
        )

    if frame.event in FAILED_EVENTS or (status_code is not None and status_code != OK_STATUS_CODE):
        raise APIStatusError(
            format_frame_error("Volcengine TTS request failed", frame, body),
            status_code=status_code or -1,
            request_id=request_id,
            body=body,
            retryable=is_retryable_status(status_code),
        )


def extract_status_code(frame: VolcengineFrame) -> int | None:
    if not frame.payload_json:
        return None
    value = frame.payload_json.get("status_code") or frame.payload_json.get("code")
    return value if isinstance(value, int) else None


def format_frame_error(prefix: str, frame: VolcengineFrame, body: object) -> str:
    message = ""
    if isinstance(body, dict):
        value = body.get("message") or body.get("error")
        if isinstance(value, str):
            message = value
    elif isinstance(body, str):
        message = body
    details = f"event={frame.event}"
    if frame.error_code is not None:
        details += f", error_code={frame.error_code}"
    if message:
        details += f", message={message}"
    return f"{prefix}: {details}"


def is_retryable_status(status_code: int | None) -> bool | None:
    if status_code is None:
        return None
    return status_code >= 50000000 or status_code >= 500


def unexpected_event_error(
    stage: str, frame: VolcengineFrame, *, request_id: str, expected: int
) -> APIStatusError:
    return APIStatusError(
        f"unexpected Volcengine TTS event during {stage}: expected={expected}, got={frame.event}",
        status_code=-1,
        request_id=request_id,
        body=frame.payload_json or frame.payload,
        retryable=False,
    )


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
