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
        }
        print("Client connected.")
        try:
            async for message in websocket:
                if isinstance(message, str):
                    await self._handle_control(websocket, state, message)
                    continue

                state["all_frames"].append(message)
                state["online_frames"].append(message)
                state["online_frame_count"] += 1

                if state["online_frame_count"] % int(state["chunk_interval"]) == 0:
                    await self._send_online(websocket, state, is_final=False)
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
                await self._send_online(websocket, state, is_final=True)
                await self._send_offline(websocket, state)
                state["online_cache"] = {}
                state["online_frames"] = []
                state["all_frames"] = []
                state["online_frame_count"] = 0
                state["partial_transcript"].reset()

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

    async def _send_offline(self, websocket: ServerConnection, state: dict[str, Any]) -> None:
        if state["mode"] not in {"offline", "2pass"}:
            return
        audio = b"".join(state["all_frames"])
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
        if not state.get("semantic_turn_detection"):
            return {}

        current_text = state["partial_transcript"].update(partial)
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
    parser.add_argument("--asr_model", default=os.getenv("FUNASR_OFFLINE_MODEL", "paraformer-zh"))
    parser.add_argument(
        "--asr_model_online",
        default=os.getenv("FUNASR_ONLINE_MODEL", "paraformer-zh-streaming"),
    )
    parser.add_argument("--vad_model", default=os.getenv("FUNASR_VAD_MODEL", "fsmn-vad"))
    parser.add_argument("--punc_model", default=os.getenv("FUNASR_PUNC_MODEL", "ct-punc"))
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
