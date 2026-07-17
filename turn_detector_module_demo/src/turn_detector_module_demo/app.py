from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import ssl
import sys
import uuid
import wave
from pathlib import Path
from urllib.parse import SplitResult, urlsplit, urlunsplit

import aiohttp
from aiohttp import web
from livekit import api

from .config import PROJECT_DIR, WEB_DIR, DemoConfig, load_config, load_dotenv
from .detector_worker import DetectorWorker, DetectorWorkerConfig
from .events import EventHub
from .token import browser_signaling_url, build_token


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = PROJECT_DIR.parent
MAX_DEBUG_TTS_CHARS = 500


async def index(_request: web.Request) -> web.FileResponse:
    response = web.FileResponse(WEB_DIR / "index.html")
    response.headers["Cache-Control"] = "no-cache"
    return response


async def health(request: web.Request) -> web.Response:
    config: DemoConfig = request.app["config"]
    return web.json_response(
        {
            "ok": True,
            "room": config.room,
            "language": config.language,
            "model": "turn-detector-v1-mini",
            "agent": config.agent_name,
            "autoDispatchAgent": config.auto_dispatch_agent,
        }
    )


async def token(request: web.Request) -> web.Response:
    config: DemoConfig = request.app["config"]
    room = request.query.get("room", config.room).strip() or config.room
    identity = request.query.get("identity", f"web-{uuid.uuid4().hex[:8]}").strip()
    name = request.query.get("name", identity).strip()
    access_token = build_token(
        api_key=config.livekit_api_key,
        api_secret=config.livekit_api_secret,
        room=room,
        identity=identity,
        name=name,
        can_publish=True,
        can_subscribe=True,
    )
    return web.json_response(
        {
            "url": browser_signaling_url(
                livekit_url=config.livekit_url,
                request_host=request.host,
                request_scheme=request.scheme,
                public_livekit_url=config.public_livekit_url,
            ),
            "token": access_token,
            "room": room,
            "identity": identity,
        }
    )


async def dispatch_agent(request: web.Request) -> web.Response:
    config: DemoConfig = request.app["config"]
    if not config.auto_dispatch_agent:
        return web.json_response({"ok": True, "dispatched": False, "reason": "disabled"})
    if not config.agent_name:
        return web.json_response({"ok": True, "dispatched": False, "reason": "agent_name_empty"})

    room = request.query.get("room", config.room).strip() or config.room
    participant_identity = (
        request.query.get("participantIdentity")
        or request.query.get("participant_identity")
        or ""
    ).strip()
    metadata = {
        "source": "audio-turn-demo",
    }
    if participant_identity:
        metadata["participant_identity"] = participant_identity
    client = api.LiveKitAPI(
        livekit_api_url(config.livekit_url),
        config.livekit_api_key,
        config.livekit_api_secret,
    )
    try:
        dispatch = await client.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                room=room,
                agent_name=config.agent_name,
                metadata=json.dumps(metadata, ensure_ascii=False),
            )
        )
    finally:
        await client.aclose()

    logger.info(
        "agent dispatch requested room=%s agent=%s participant=%s dispatch=%s",
        room,
        config.agent_name,
        participant_identity or "auto",
        dispatch.id,
    )
    return web.json_response(
        {
            "ok": True,
            "dispatched": True,
            "id": dispatch.id,
            "agentName": dispatch.agent_name or config.agent_name,
            "room": room,
            "participantIdentity": participant_identity or None,
        }
    )


async def debug_tts(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise web.HTTPBadRequest(text="request body must be JSON") from exc

    text = str(payload.get("text", "")).strip()
    if not text:
        raise web.HTTPBadRequest(text="text is required")
    if len(text) > MAX_DEBUG_TTS_CHARS:
        raise web.HTTPBadRequest(text=f"text must be at most {MAX_DEBUG_TTS_CHARS} characters")

    try:
        synthesize_text_to_pcm = load_debug_tts_synthesizer()
        synthesized = await synthesize_text_to_pcm(text)
        if not synthesized.pcm:
            raise web.HTTPBadGateway(text="TTS returned empty audio")
        wav_bytes = pcm16le_to_wav_bytes(
            synthesized.pcm,
            sample_rate=synthesized.sample_rate,
            num_channels=synthesized.num_channels,
        )
    except web.HTTPException:
        raise
    except Exception as exc:
        logger.exception("debug TTS synthesis failed")
        raise web.HTTPBadGateway(text=f"debug TTS synthesis failed: {exc}") from exc

    return web.Response(
        body=wav_bytes,
        content_type="audio/wav",
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": 'inline; filename="debug-tts.wav"',
        },
    )


def load_debug_tts_synthesizer() -> object:
    """Load the existing Doubao/Volcengine TTS helper only for the debug endpoint."""

    load_dotenv(REPO_ROOT / "tts_module" / "doubao_tts" / ".env")
    for path in (REPO_ROOT, REPO_ROOT / "tts_module"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    from doubao_tts import synthesize_text_to_pcm

    return synthesize_text_to_pcm


def pcm16le_to_wav_bytes(pcm: bytes, *, sample_rate: int, num_channels: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(num_channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


async def events_ws(request: web.Request) -> web.WebSocketResponse:
    hub: EventHub = request.app["hub"]
    worker: DetectorWorker = request.app["worker"]
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    await hub.add(ws)
    await ws.send_str(
        json.dumps(
            {
                "type": "status",
                "level": "ok",
                "message": "event stream connected",
            }
        )
    )
    try:
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            try:
                payload = json.loads(msg.data)
                action = payload.get("action")
                if action == "start":
                    await worker.start_track(
                        ws=ws,
                        room=str(payload.get("room", "")),
                        identity=str(payload.get("participantIdentity", "")),
                        track_sid=str(payload.get("trackSid", "")),
                        run_id=str(payload.get("runId", "")),
                    )
                elif action == "stop":
                    await worker.stop_track(
                        track_sid=str(payload.get("trackSid", "")), ws=ws
                    )
                else:
                    raise ValueError(f"unknown detector action: {action}")
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                await hub.send(
                    ws, {"type": "status", "level": "error", "message": str(exc)}
                )
    finally:
        await worker.stop_client(ws)
        await hub.remove(ws)
    return ws


async def livekit_proxy(request: web.Request) -> web.StreamResponse:
    if request.headers.get("upgrade", "").lower() == "websocket":
        return await livekit_ws_proxy(request)
    return await livekit_http_proxy(request)


async def livekit_ws_proxy(request: web.Request) -> web.WebSocketResponse:
    config: DemoConfig = request.app["config"]
    upstream_url = livekit_rtc_url(
        config.livekit_url,
        request.query_string,
        suffix=request.match_info.get("tail", ""),
    )
    client_ws = web.WebSocketResponse(heartbeat=30, max_msg_size=0)
    await client_ws.prepare(request)

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(upstream_url, heartbeat=30, max_msg_size=0) as upstream_ws:
            tasks = {
                asyncio.create_task(relay_websocket(client_ws, upstream_ws)),
                asyncio.create_task(relay_websocket(upstream_ws, client_ws)),
            }
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in done:
                task.result()

    return client_ws


async def livekit_http_proxy(request: web.Request) -> web.Response:
    config: DemoConfig = request.app["config"]
    upstream_url = livekit_http_url(
        config.livekit_url,
        request.query_string,
        suffix=request.match_info.get("tail", ""),
    )
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower()
        not in {
            "connection",
            "host",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailer",
            "transfer-encoding",
            "upgrade",
        }
    }
    body = await request.read()
    async with aiohttp.ClientSession() as session:
        async with session.request(
            request.method,
            upstream_url,
            headers=headers,
            data=body if body else None,
            allow_redirects=False,
        ) as upstream:
            response_headers = {
                key: value
                for key, value in upstream.headers.items()
                if key.lower()
                not in {
                    "connection",
                    "keep-alive",
                    "proxy-authenticate",
                    "proxy-authorization",
                    "te",
                    "trailer",
                    "transfer-encoding",
                    "upgrade",
                }
            }
            return web.Response(
                status=upstream.status,
                headers=response_headers,
                body=await upstream.read(),
            )


async def relay_websocket(source: object, target: object) -> None:
    async for msg in source:  # type: ignore[attr-defined]
        if msg.type == aiohttp.WSMsgType.TEXT:
            await target.send_str(msg.data)  # type: ignore[attr-defined]
        elif msg.type == aiohttp.WSMsgType.BINARY:
            await target.send_bytes(msg.data)  # type: ignore[attr-defined]
        elif msg.type in {
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.CLOSING,
        }:
            await target.close()  # type: ignore[attr-defined]
            return
        elif msg.type == aiohttp.WSMsgType.ERROR:
            await target.close()  # type: ignore[attr-defined]
            return


def livekit_rtc_url(livekit_url: str, query_string: str, *, suffix: str = "") -> str:
    parsed = urlsplit(livekit_url)
    if parsed.scheme not in {"ws", "wss"}:
        raise ValueError(f"LiveKit URL must start with ws:// or wss://: {livekit_url}")
    path = _livekit_proxy_path(parsed.path, suffix)
    return urlunsplit(
        SplitResult(
            scheme=parsed.scheme,
            netloc=parsed.netloc,
            path=path,
            query=query_string,
            fragment="",
        )
    )


def livekit_http_url(livekit_url: str, query_string: str, *, suffix: str = "") -> str:
    parsed = urlsplit(livekit_url)
    if parsed.scheme not in {"ws", "wss", "http", "https"}:
        raise ValueError(f"LiveKit URL must start with ws://, wss://, http://, or https://: {livekit_url}")
    scheme = {"ws": "http", "wss": "https"}.get(parsed.scheme, parsed.scheme)
    return urlunsplit(
        SplitResult(
            scheme=scheme,
            netloc=parsed.netloc,
            path=_livekit_proxy_path(parsed.path, suffix),
            query=query_string,
            fragment="",
        )
    )


def livekit_api_url(livekit_url: str) -> str:
    parsed = urlsplit(livekit_url)
    if parsed.scheme == "ws":
        return urlunsplit(parsed._replace(scheme="http"))
    if parsed.scheme == "wss":
        return urlunsplit(parsed._replace(scheme="https"))
    return livekit_url


def _livekit_proxy_path(base_path: str, suffix: str) -> str:
    path = base_path.rstrip("/") + "/rtc" if base_path else "/rtc"
    suffix = suffix.strip("/")
    if suffix:
        path = f"{path}/{suffix}"
    return path


async def start_worker(app: web.Application) -> None:
    config: DemoConfig = app["config"]
    hub: EventHub = app["hub"]
    worker = DetectorWorker(
        DetectorWorkerConfig(
            livekit_url=config.livekit_url,
            api_key=config.livekit_api_key,
            api_secret=config.livekit_api_secret,
            room=config.room,
            language=config.language,
        ),
        hub,
    )
    app["worker"] = worker
    await worker.start()


async def stop_worker(app: web.Application) -> None:
    worker: DetectorWorker | None = app.get("worker")
    if worker is not None:
        await worker.stop()


def make_app(config: DemoConfig | None = None, *, start_detector: bool = True) -> web.Application:
    app = web.Application()
    app["config"] = config or load_config()
    app["hub"] = EventHub()
    app.router.add_get("/", index)
    app.router.add_get("/health", health)
    app.router.add_get("/token", token)
    app.router.add_post("/dispatch", dispatch_agent)
    app.router.add_post("/debug/tts", debug_tts)
    app.router.add_get("/events", events_ws)
    app.router.add_route("*", "/rtc", livekit_proxy)
    app.router.add_route("*", "/rtc/{tail:.*}", livekit_proxy)
    app.router.add_static("/static", WEB_DIR, show_index=False)
    if start_detector:
        app.on_startup.append(start_worker)
        app.on_cleanup.append(stop_worker)
    return app


def build_arg_parser() -> argparse.ArgumentParser:
    config = load_config()
    parser = argparse.ArgumentParser(description="LiveKit audio turn detector module demo")
    parser.add_argument("--host", default=config.host)
    parser.add_argument("--port", type=int, default=config.port)
    parser.add_argument("--ssl-cert-file", default=config.ssl_cert_file)
    parser.add_argument("--ssl-key-file", default=config.ssl_key_file)
    parser.add_argument("--no-detector", action="store_true", help="Serve UI/API without joining LiveKit.")
    return parser


def build_ssl_context(cert_file: str, key_file: str) -> ssl.SSLContext | None:
    if not cert_file and not key_file:
        return None
    if not cert_file or not key_file:
        raise SystemExit("Both --ssl-cert-file and --ssl-key-file are required for HTTPS.")

    cert_path = Path(cert_file).expanduser()
    key_path = Path(key_file).expanduser()
    if not cert_path.exists():
        raise SystemExit(f"SSL certificate file not found: {cert_path}")
    if not key_path.exists():
        raise SystemExit(f"SSL key file not found: {key_path}")

    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(cert_path, key_path)
    return context


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config()
    ssl_context = build_ssl_context(args.ssl_cert_file, args.ssl_key_file)
    scheme = "https" if ssl_context else "http"
    logger.info("serving demo on %s://%s:%s room=%s", scheme, args.host, args.port, config.room)
    web.run_app(
        make_app(config, start_detector=not args.no_detector),
        host=args.host,
        port=args.port,
        ssl_context=ssl_context,
    )


if __name__ == "__main__":
    main()
