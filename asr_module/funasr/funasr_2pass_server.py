from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
import json
import os
from typing import Any

import websockets
from websockets.asyncio.server import ServerConnection

try:
    from .semantic_turn import PartialTranscript, SemanticTurnDetector
except ImportError:
    from semantic_turn import PartialTranscript, SemanticTurnDetector

PCM16_BYTES_PER_SAMPLE = 2


def _json(data: dict[str, Any]) -> str:
    return json.dumps(_to_python(data), ensure_ascii=False)


def _to_python(value: Any) -> Any:
    try:
        import numpy as np
        import torch
    except Exception:
        np = None
        torch = None

    if np is not None and isinstance(value, np.generic):
        return value.item()
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(item) for item in value]
    return value


def _parse_chunk_size(value: Any) -> list[int]:
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [int(item) for item in value]
    return [8, 8, 4]


def _parse_silence_schedule(value: str | None) -> list[tuple[float, int]] | None:
    if not value:
        return None
    schedule: list[tuple[float, int]] = []
    for item in value.split(","):
        if not item.strip():
            continue
        limit, silence = item.split(":", maxsplit=1)
        limit = limit.strip().lower()
        limit_ms = float("inf") if limit in {"inf", "infinity"} else float(limit)
        schedule.append((limit_ms, int(float(silence.strip()))))
    return schedule or None


def _load_dynamic_streaming_vad_cls() -> Any | None:
    try:
        from funasr.models.fsmn_vad_streaming.dynamic_vad import DynamicStreamingVAD
    except Exception:
        return None
    return DynamicStreamingVAD


def _pcm16le_to_float_tensor(audio: bytes) -> Any:
    import numpy as np
    import torch

    samples = np.frombuffer(audio, dtype="<i2").astype("float32") / 32768.0
    return torch.from_numpy(samples)


def _audio_bytes_to_ms(audio_bytes: int, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return audio_bytes / (PCM16_BYTES_PER_SAMPLE * sample_rate) * 1000.0


def _ms_to_byte_offset(ms: float, sample_rate: int) -> int:
    samples = max(0, int(round(ms * sample_rate / 1000.0)))
    return samples * PCM16_BYTES_PER_SAMPLE


def _slice_pcm16le(audio: bytearray, start_ms: float, end_ms: float, sample_rate: int) -> bytes:
    start = min(len(audio), _ms_to_byte_offset(start_ms, sample_rate))
    end = min(len(audio), _ms_to_byte_offset(end_ms, sample_rate))
    if end <= start:
        return b""
    return bytes(audio[start:end])


def _peer_label(websocket: ServerConnection) -> str:
    peer = getattr(websocket, "remote_address", None)
    if isinstance(peer, tuple):
        return ":".join(str(part) for part in peer)
    return str(peer or "unknown")


def _log_ws_send(
    *,
    kind: str,
    websocket: ServerConnection,
    state: dict[str, Any],
    payload_json: str,
    audio_bytes: int,
    text: str,
    is_final: bool,
) -> None:
    print(
        "[funasr_ws_send] "
        f"kind={kind} "
        f"target={_peer_label(websocket)} "
        f"wav_name={state.get('wav_name') or ''} "
        f"session_mode={state.get('mode') or ''} "
        f"is_final={is_final} "
        f"audio_bytes={audio_bytes} "
        f"text_chars={len(text)} "
        f"payload={payload_json}",
        flush=True,
    )


async def _run_blocking(executor: ThreadPoolExecutor, fn: Any, *args: Any, **kwargs: Any) -> Any:
    loop = asyncio.get_running_loop()
    call = functools.partial(fn, *args, **kwargs)
    return await loop.run_in_executor(executor, call)


class FunASR2PassService:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.executor = ThreadPoolExecutor(max_workers=args.worker_threads)
        self.online_sem = asyncio.Semaphore(args.concurrent_online)
        self.offline_sem = asyncio.Semaphore(args.concurrent_offline)
        self.semantic_turn_detector = SemanticTurnDetector(args.semantic_turn_detector)

        print("Loading FunASR models...")
        from funasr import AutoModel

        self.sample_rate = args.sample_rate
        self.dynamic_streaming_vad_cls = None
        self.streaming_vad_model = None
        if args.streaming_vad and args.vad_model:
            self.dynamic_streaming_vad_cls = _load_dynamic_streaming_vad_cls()
            if self.dynamic_streaming_vad_cls is None:
                print(
                    "FunASR DynamicStreamingVAD is unavailable; falling back to client stop endpointing.",
                    flush=True,
                )
            else:
                self.streaming_vad_model = AutoModel(
                    model=args.vad_model,
                    ngpu=args.ngpu,
                    ncpu=args.ncpu,
                    device=args.device,
                    disable_pbar=True,
                    disable_log=True,
                    disable_update=True,
                )

        self.offline_model = AutoModel(
            model=args.asr_model,
            vad_model=args.vad_model or None,
            punc_model=args.punc_model or None,
            ngpu=args.ngpu,
            ncpu=args.ncpu,
            device=args.device,
            disable_pbar=True,
            disable_log=True,
            disable_update=True,
        )
        self.online_model = AutoModel(
            model=args.asr_model_online,
            ngpu=args.ngpu,
            ncpu=args.ncpu,
            device=args.device,
            disable_pbar=True,
            disable_log=True,
            disable_update=True,
        )
        print("FunASR models loaded.")

    def _new_streaming_vad(self) -> Any | None:
        if self.streaming_vad_model is None or self.dynamic_streaming_vad_cls is None:
            return None
        return self.dynamic_streaming_vad_cls(
            self.streaming_vad_model,
            chunk_size_ms=self.args.vad_chunk_ms,
            speech_noise_thres=self.args.vad_speech_noise_threshold,
            speech_to_sil_thres_ms=self.args.vad_speech_to_sil_ms,
            silence_schedule=self.args.vad_silence_schedule,
            sample_rate=self.sample_rate,
        )

    async def serve(self) -> None:
        if self.args.semantic_turn_detector:
            print("Loading LiveKit semantic turn detector...")
            await self.semantic_turn_detector.warmup()
            print("LiveKit semantic turn detector loaded.")
        async with websockets.serve(
            self._handle_client,
            self.args.host,
            self.args.port,
            max_size=None,
            ping_interval=None,
            subprotocols=["binary"],
        ) as server:
            print(f"FunASR 2pass websocket service listening on ws://{self.args.host}:{self.args.port}")
            await server.serve_forever()

    async def _handle_client(self, websocket: ServerConnection) -> None:
        state: dict[str, Any] = {
            "mode": "2pass",
            "wav_name": "microphone",
            "chunk_interval": 10,
            "chunk_size": [8, 8, 4],
            "hotword": "",
            "online_cache": {},
            "online_frames": [],
            "all_frames": [],
            "online_frame_count": 0,
            "is_speaking": True,
            "semantic_turn_detection": self.args.semantic_turn_detector,
            "partial_transcript": PartialTranscript(),
            "streaming_vad": self._new_streaming_vad(),
            "segment_audio": bytearray(),
            "segment_base_ms": 0.0,
            "stream_audio_ms": 0.0,
            "speech_active": False,
            "speech_start_ms": None,
            "segment_seq": 0,
        }
        print("Client connected.")
        try:
            async for message in websocket:
                if isinstance(message, str):
                    await self._handle_control(websocket, state, message)
                    continue

                await self._handle_audio(websocket, state, message)
        except websockets.ConnectionClosed:
            pass
        finally:
            print("Client disconnected.")

    async def _handle_control(
        self, websocket: ServerConnection, state: dict[str, Any], message: str
    ) -> None:
        data = json.loads(message)
        if "mode" in data:
            state["mode"] = data["mode"] or "2pass"
        if "wav_name" in data:
            state["wav_name"] = data["wav_name"] or "microphone"
        if "chunk_interval" in data:
            state["chunk_interval"] = int(data["chunk_interval"])
        if "chunk_size" in data:
            state["chunk_size"] = _parse_chunk_size(data["chunk_size"])
        if "hotwords" in data:
            state["hotword"] = data["hotwords"] or ""
        if "semantic_turn_detection" in data:
            state["semantic_turn_detection"] = bool(data["semantic_turn_detection"])
        if "is_speaking" in data:
            state["is_speaking"] = bool(data["is_speaking"])
            if not state["is_speaking"]:
                if state.get("streaming_vad") is not None:
                    await self._finalize_streaming_vad(websocket, state)
                else:
                    await self._send_online(websocket, state, is_final=True)
                    await self._send_offline(websocket, state)
                    self._reset_legacy_utterance(state)

    async def _handle_audio(
        self, websocket: ServerConnection, state: dict[str, Any], message: bytes
    ) -> None:
        if state.get("streaming_vad") is None:
            state["all_frames"].append(message)
            state["online_frames"].append(message)
            state["online_frame_count"] += 1
            if state["online_frame_count"] % int(state["chunk_interval"]) == 0:
                await self._send_online(websocket, state, is_final=False)
            return

        segment_started_now = False
        state["segment_audio"].extend(message)
        state["stream_audio_ms"] += _audio_bytes_to_ms(len(message), self.sample_rate)

        vad = state["streaming_vad"]
        audio_tensor = _pcm16le_to_float_tensor(message)
        segments = await _run_blocking(self.executor, vad.feed, audio_tensor, False)

        if not state["speech_active"]:
            speech_start_ms = self._current_vad_speech_start_ms(state, segments)
            if speech_start_ms is not None:
                self._start_streaming_segment(state, speech_start_ms)
                segment_started_now = True

        if state["speech_active"] and not segment_started_now:
            state["online_frames"].append(message)
            state["online_frame_count"] += 1

        if (
            state["speech_active"]
            and state["online_frame_count"] > 0
            and state["online_frame_count"] % int(state["chunk_interval"]) == 0
        ):
            await self._send_online(websocket, state, is_final=False)

        for segment in segments:
            await self._finalize_streaming_segment(websocket, state, segment)

    def _current_vad_speech_start_ms(
        self, state: dict[str, Any], segments: list[list[int]]
    ) -> float | None:
        vad = state["streaming_vad"]
        current_start = getattr(vad, "current_speech_start", None)
        if current_start is not None:
            return float(current_start)
        for segment in segments:
            if segment and segment[0] >= 0:
                return float(segment[0])
        return None

    def _start_streaming_segment(self, state: dict[str, Any], speech_start_ms: float) -> None:
        start_ms = max(0.0, speech_start_ms - self.args.segment_pre_padding_ms)
        state["speech_active"] = True
        state["speech_start_ms"] = start_ms
        primed_audio = _slice_pcm16le(
            state["segment_audio"],
            start_ms,
            _audio_bytes_to_ms(len(state["segment_audio"]), self.sample_rate),
            self.sample_rate,
        )
        if primed_audio:
            state["online_frames"].append(primed_audio)
            state["online_frame_count"] += 1

    async def _finalize_streaming_vad(
        self, websocket: ServerConnection, state: dict[str, Any]
    ) -> None:
        vad = state.get("streaming_vad")
        if vad is None:
            return

        segments = await _run_blocking(self.executor, vad.finalize)
        if not segments and state["speech_active"]:
            speech_start_ms = float(state["speech_start_ms"] or 0.0)
            segment_end_ms = _audio_bytes_to_ms(len(state["segment_audio"]), self.sample_rate)
            segments = [[int(speech_start_ms), int(segment_end_ms)]]

        if not segments:
            self._reset_streaming_segment(state)
            return

        for segment in segments:
            await self._finalize_streaming_segment(websocket, state, segment)

    async def _finalize_streaming_segment(
        self, websocket: ServerConnection, state: dict[str, Any], segment: list[int]
    ) -> None:
        if len(segment) < 2:
            return
        local_start_ms = max(0.0, float(segment[0]))
        local_end_ms = max(local_start_ms, float(segment[1]))
        if state["speech_start_ms"] is not None:
            local_start_ms = min(local_start_ms, float(state["speech_start_ms"]))

        segment_audio_end_ms = _audio_bytes_to_ms(len(state["segment_audio"]), self.sample_rate)
        local_start_ms = max(0.0, local_start_ms - self.args.segment_pre_padding_ms)
        local_end_ms = min(segment_audio_end_ms, local_end_ms + self.args.segment_post_padding_ms)
        audio = _slice_pcm16le(
            state["segment_audio"],
            local_start_ms,
            local_end_ms,
            self.sample_rate,
        )
        if not audio:
            self._reset_streaming_segment(state)
            return

        state["segment_seq"] += 1
        segment_seq = state["segment_seq"]
        absolute_start_ms = state["segment_base_ms"] + local_start_ms
        absolute_end_ms = state["segment_base_ms"] + local_end_ms

        await self._send_online(websocket, state, is_final=True)
        await self._send_offline(
            websocket,
            state,
            audio=audio,
            segment_seq=segment_seq,
            segment_start_ms=absolute_start_ms,
            segment_end_ms=absolute_end_ms,
        )
        self._reset_streaming_segment(state)

    def _reset_legacy_utterance(self, state: dict[str, Any]) -> None:
        state["online_cache"] = {}
        state["online_frames"] = []
        state["all_frames"] = []
        state["online_frame_count"] = 0
        state["partial_transcript"].reset()

    def _reset_streaming_segment(self, state: dict[str, Any]) -> None:
        state["online_cache"] = {}
        state["online_frames"] = []
        state["all_frames"] = []
        state["online_frame_count"] = 0
        state["partial_transcript"].reset()
        state["segment_audio"] = bytearray()
        state["segment_base_ms"] = state["stream_audio_ms"]
        state["speech_active"] = False
        state["speech_start_ms"] = None
        state["streaming_vad"] = self._new_streaming_vad()

    async def _send_online(
        self, websocket: ServerConnection, state: dict[str, Any], *, is_final: bool
    ) -> None:
        if state["mode"] not in {"online", "2pass"} or not state["online_frames"]:
            return
        audio = b"".join(state["online_frames"])
        state["online_frames"] = []
        kwargs = {
            "cache": state["online_cache"],
            "is_final": is_final,
            "chunk_size": state["chunk_size"],
        }
        if state["hotword"]:
            kwargs["hotword"] = state["hotword"]

        async with self.online_sem:
            result = await _run_blocking(self.executor, self.online_model.generate, input=audio, **kwargs)
        item = result[0] if result else {}
        text = str(item.get("text") or "").strip()
        if text:
            mode = "2pass-online" if state["mode"] == "2pass" else "online"
            payload: dict[str, Any] = {
                "mode": mode,
                "text": text,
                "wav_name": state["wav_name"],
                "is_final": bool(is_final and state["mode"] == "online"),
            }
            semantic_payload = await self._semantic_payload(state, text)
            if semantic_payload:
                payload.update(semantic_payload)
            payload_json = _json(payload)
            _log_ws_send(
                kind="online",
                websocket=websocket,
                state=state,
                payload_json=payload_json,
                audio_bytes=len(audio),
                text=text,
                is_final=bool(is_final and state["mode"] == "online"),
            )
            await websocket.send(payload_json)

    async def _send_offline(
        self,
        websocket: ServerConnection,
        state: dict[str, Any],
        *,
        audio: bytes | None = None,
        segment_seq: int | None = None,
        segment_start_ms: float | None = None,
        segment_end_ms: float | None = None,
    ) -> None:
        if state["mode"] not in {"offline", "2pass"}:
            return
        audio = audio if audio is not None else b"".join(state["all_frames"])
        if not audio:
            return
        kwargs: dict[str, Any] = {}
        if state["hotword"]:
            kwargs["hotword"] = state["hotword"]

        async with self.offline_sem:
            result = await _run_blocking(self.executor, self.offline_model.generate, input=audio, **kwargs)
        item = result[0] if result else {}
        text = str(item.get("text") or "").strip()
        mode = "2pass-offline" if state["mode"] == "2pass" else "offline"
        payload = {
            "mode": mode,
            "text": text,
            "wav_name": state["wav_name"],
            "is_final": True,
            "timestamp": item.get("timestamp"),
            "raw": {k: v for k, v in item.items() if k in {"key", "text"}},
        }
        if segment_seq is not None:
            payload["segment_seq"] = segment_seq
        if segment_start_ms is not None:
            payload["segment_start_ms"] = int(round(segment_start_ms))
        if segment_end_ms is not None:
            payload["segment_end_ms"] = int(round(segment_end_ms))
        payload_json = _json(payload)
        _log_ws_send(
            kind="offline",
            websocket=websocket,
            state=state,
            payload_json=payload_json,
            audio_bytes=len(audio),
            text=text,
            is_final=True,
        )
        await websocket.send(payload_json)

    async def _semantic_payload(self, state: dict[str, Any], partial: str) -> dict[str, Any]:
        current_text = state["partial_transcript"].update(partial)
        if not state.get("semantic_turn_detection"):
            return {"partial_text": current_text}

        probability = await self.semantic_turn_detector.predict(current_text)
        if probability is None:
            if self.semantic_turn_detector.unavailable_reason:
                return {
                    "partial_text": current_text,
                    "semantic_turn": "unavailable",
                    "semantic_reason": self.semantic_turn_detector.unavailable_reason,
                }
            return {"partial_text": current_text}

        return {
            "partial_text": current_text,
            "semantic_turn": "complete" if probability >= self.args.semantic_threshold else "incomplete",
            "semantic_probability": probability,
            "semantic_threshold": self.args.semantic_threshold,
        }

    async def aclose(self) -> None:
        await self.semantic_turn_detector.aclose()
        self.executor.shutdown(wait=False, cancel_futures=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FunASR Paraformer 2pass websocket service")
    parser.add_argument("--host", default=os.getenv("FUNASR_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("FUNASR_PORT", "10095")))
    parser.add_argument("--device", default=os.getenv("FUNASR_DEVICE", "cpu"))
    parser.add_argument("--ngpu", type=int, default=int(os.getenv("FUNASR_NGPU", "0")))
    parser.add_argument("--ncpu", type=int, default=int(os.getenv("FUNASR_NCPU", "4")))
    parser.add_argument("--sample-rate", type=int, default=int(os.getenv("FUNASR_SAMPLE_RATE", "16000")))
    parser.add_argument("--asr_model", default=os.getenv("FUNASR_OFFLINE_MODEL", "paraformer-zh"))
    parser.add_argument(
        "--asr_model_online",
        default=os.getenv("FUNASR_ONLINE_MODEL", "paraformer-zh-streaming"),
    )
    parser.add_argument("--vad_model", default=os.getenv("FUNASR_VAD_MODEL", "fsmn-vad"))
    parser.add_argument("--punc_model", default=os.getenv("FUNASR_PUNC_MODEL", "ct-punc"))
    parser.add_argument(
        "--streaming-vad",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("FUNASR_STREAMING_VAD", "true").strip().lower()
        not in {"0", "false", "no", "off"},
        help="Use FunASR fsmn-vad streaming endpointing to cut websocket ASR segments.",
    )
    parser.add_argument(
        "--vad-chunk-ms",
        type=int,
        default=int(os.getenv("FUNASR_VAD_CHUNK_MS", "60")),
    )
    parser.add_argument(
        "--vad-speech-noise-threshold",
        type=float,
        default=float(os.getenv("FUNASR_VAD_SPEECH_NOISE_THRESHOLD", "0.5")),
    )
    parser.add_argument(
        "--vad-speech-to-sil-ms",
        type=int,
        default=int(os.getenv("FUNASR_VAD_SPEECH_TO_SIL_MS", "150")),
    )
    parser.add_argument(
        "--vad-silence-schedule",
        type=_parse_silence_schedule,
        default=_parse_silence_schedule(os.getenv("FUNASR_VAD_SILENCE_SCHEDULE")),
        help="Optional dynamic VAD schedule like '5000:2000,10000:1500,inf:400'.",
    )
    parser.add_argument(
        "--segment-pre-padding-ms",
        type=int,
        default=int(os.getenv("FUNASR_SEGMENT_PRE_PADDING_MS", "100")),
    )
    parser.add_argument(
        "--segment-post-padding-ms",
        type=int,
        default=int(os.getenv("FUNASR_SEGMENT_POST_PADDING_MS", "120")),
    )
    parser.add_argument("--worker_threads", type=int, default=4)
    parser.add_argument("--concurrent_online", type=int, default=1)
    parser.add_argument("--concurrent_offline", type=int, default=1)
    parser.add_argument(
        "--semantic-turn-detector",
        action="store_true",
        help="Run LiveKit multilingual semantic EOU on server-side partial ASR text.",
    )
    parser.add_argument("--semantic-threshold", type=float, default=0.5)
    return parser


async def async_main() -> None:
    args = build_arg_parser().parse_args()
    service = FunASR2PassService(args)
    try:
        await service.serve()
    finally:
        await service.aclose()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
