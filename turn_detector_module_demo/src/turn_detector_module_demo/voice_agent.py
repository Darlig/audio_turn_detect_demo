from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_DIR = REPO_ROOT / "turn_detector_module_demo"


def _install_project_import_paths() -> None:
    for path in (
        REPO_ROOT,
        REPO_ROOT / "asr_module",
        REPO_ROOT / "llm_module",
        REPO_ROOT / "tts_module",
        REPO_ROOT / "agents" / "livekit-agents",
    ):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_runtime_env() -> None:
    _load_env_file(PROJECT_DIR / ".env")
    _load_env_file(REPO_ROOT / "llm_module" / "doubao_llm" / ".env")
    _load_env_file(REPO_ROOT / "tts_module" / "doubao_tts" / ".env")
    os.environ.setdefault("LIVEKIT_URL", "ws://127.0.0.1:8890")
    os.environ.setdefault("LIVEKIT_API_KEY", "devkey")
    os.environ.setdefault("LIVEKIT_API_SECRET", "secret")
    os.environ.setdefault("LIVEKIT_AGENT_NAME", "cascade-voice-agent")
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,::1")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost,::1")
    if os.environ["LIVEKIT_URL"].startswith(("ws://127.0.0.1", "ws://localhost")):
        for key in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "http_proxy",
            "https_proxy",
            "ALL_PROXY",
            "all_proxy",
        ):
            os.environ.pop(key, None)


_install_project_import_paths()
load_runtime_env()

from doubao_tts import VolcengineStreamingTTS  # noqa: E402
from livekit.agents import (  # noqa: E402
    Agent,
    AgentServer,
    AgentSession,
    AutoSubscribe,
    JobContext,
    MetricsCollectedEvent,
    TurnHandlingOptions,
    cli,
    inference,
    metrics,
)
from livekit.agents.voice.room_io import RoomOptions  # noqa: E402

from .stt_factory import create_stt  # noqa: E402
from .llm_factory import create_llm  # noqa: E402
from .turn_metrics import (  # noqa: E402
    record_eou_confirmed,
    record_playback_started,
    record_vad_speech_tail,
    reset_turn_metrics,
    set_metrics_callback,
)


logger = logging.getLogger("cascade-voice-agent")


DEFAULT_INSTRUCTIONS = (
    "你是一个实时语音助手。请用中文自然、简短地回答用户。"
    "不要使用 Markdown、项目符号、表情符号或不适合语音播报的特殊格式。"
    "如果信息不确定，请直接说明不确定，并给出下一步建议。"
)
DEFAULT_GREETING = "用一句自然简短的中文问候用户，并说明可以开始对话。"
EOU_DEBUG_TOPIC = "audio-turn-demo.agent-eou-debug"
LATENCY_METRICS_TOPIC = "audio-turn-demo.latency-metrics"


class CascadeVoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=os.getenv("AGENT_INSTRUCTIONS") or DEFAULT_INSTRUCTIONS)

    async def on_enter(self) -> None:
        greeting = (os.getenv("AGENT_GREETING") or DEFAULT_GREETING).strip()
        if greeting:
            self.session.generate_reply(instructions=greeting)


server = AgentServer(load_fnc=lambda: 0.0)


@server.rtc_session(agent_name=os.getenv("LIVEKIT_AGENT_NAME", "cascade-voice-agent"))
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}
    participant_identity = _metadata_participant_identity(getattr(ctx.job, "metadata", ""))
    logger.info(
        "starting cascade voice agent session room=%s participant=%s",
        ctx.room.name,
        participant_identity or "auto",
    )
    reset_turn_metrics()
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    endpoint_min_delay = _env_float("AGENT_ENDPOINT_MIN_DELAY", 0.3)
    endpoint_system_max_delay = _env_float("AGENT_ENDPOINT_MAX_DELAY", 2.5)
    endpoint_debug_max_delay = _env_float("AGENT_DEBUG_ENDPOINT_MAX_DELAY")
    endpoint_max_delay = (
        endpoint_debug_max_delay
        if endpoint_debug_max_delay is not None
        else endpoint_system_max_delay
    )
    endpoint_max_source = "debug" if endpoint_debug_max_delay is not None else "system"
    require_eou_positive = _env_bool("AGENT_REQUIRE_EOU_POSITIVE", False)
    logger.info(
        "endpointing config min_delay=%s max_delay=%s max_source=%s system_max_delay=%s require_eou_positive=%s",
        endpoint_min_delay,
        endpoint_max_delay,
        endpoint_max_source,
        endpoint_system_max_delay,
        require_eou_positive,
    )

    tts_client = VolcengineStreamingTTS()
    session = AgentSession(
        stt=create_stt(),
        llm=create_llm(
            temperature=_env_float("DOUBAO_TEMPERATURE"),
            max_output_tokens=_env_int("DOUBAO_MAX_OUTPUT_TOKENS"),
        ),
        tts=tts_client,
        vad=inference.VAD(
            model="silero",
            min_speech_duration=_env_float("AGENT_VAD_MIN_SPEECH_DURATION", 0.05),
            min_silence_duration=_env_float("AGENT_VAD_MIN_SILENCE_DURATION", 0.25),
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(version="v1-mini"),
            endpointing={
                "min_delay": endpoint_min_delay,
                "max_delay": endpoint_max_delay,
                "require_eou_positive": require_eou_positive,
            },
            interruption={
                "resume_false_interruption": True,
                "false_interruption_timeout": _env_float(
                    "AGENT_FALSE_INTERRUPTION_TIMEOUT", 1.0
                ),
            },
            preemptive_generation={
                "enabled": _env_bool("AGENT_PREEMPTIVE_GENERATION", True),
                "max_retries": _env_int("AGENT_PREEMPTIVE_MAX_RETRIES", 3),
            },
        ),
        aec_warmup_duration=_env_float("AGENT_AEC_WARMUP_DURATION", 3.0),
    )

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)

    loop = asyncio.get_running_loop()

    def _on_latency_metrics(payload: dict[str, object]) -> None:
        loop.call_soon_threadsafe(
            lambda: asyncio.create_task(
                _publish_latency_metrics(
                    ctx=ctx,
                    payload=payload,
                    participant_identity=participant_identity,
                )
            )
        )

    set_metrics_callback(_on_latency_metrics)

    @session.on("eot_prediction")
    def _on_eot_prediction(ev: object) -> None:
        asyncio.create_task(
            _publish_eou_debug(
                ctx=ctx,
                ev=ev,
                participant_identity=participant_identity,
                endpoint_min_delay=endpoint_min_delay,
                endpoint_max_delay=endpoint_max_delay,
                endpoint_system_max_delay=endpoint_system_max_delay,
                endpoint_debug_max_delay=endpoint_debug_max_delay,
                endpoint_max_source=endpoint_max_source,
            )
        )

    @session.on("user_turn_committed")
    def _on_user_turn_committed(ev: object) -> None:
        record_eou_confirmed(
            eou_confirmed_at=time.perf_counter(),
            transcript=str(getattr(ev, "transcript", "") or ""),
        )

    @session.on("user_state_changed")
    def _on_user_state_changed(ev: object) -> None:
        new_state = getattr(ev, "new_state", None)
        if new_state == "speaking":
            tts_client.prewarm()
        elif getattr(ev, "old_state", None) == "speaking" and new_state == "listening":
            event_wall_time = float(getattr(ev, "created_at", time.time()))
            speech_ended_at = time.perf_counter() - max(0.0, time.time() - event_wall_time)
            record_vad_speech_tail(speech_ended_at=speech_ended_at)

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: object) -> None:
        if getattr(ev, "new_state", None) == "speaking":
            record_playback_started(playback_started_at=time.perf_counter())

    async def log_usage() -> None:
        logger.info("session usage: %s", session.usage)

    ctx.add_shutdown_callback(log_usage)
    ctx.add_shutdown_callback(_clear_latency_metrics_callback)
    room_options = (
        RoomOptions(participant_identity=participant_identity)
        if participant_identity
        else RoomOptions()
    )
    await session.start(agent=CascadeVoiceAgent(), room=ctx.room, room_options=room_options)


async def _publish_eou_debug(
    *,
    ctx: JobContext,
    ev: object,
    participant_identity: str | None,
    endpoint_min_delay: float,
    endpoint_max_delay: float,
    endpoint_system_max_delay: float,
    endpoint_debug_max_delay: float | None,
    endpoint_max_source: str,
) -> None:
    probability = float(getattr(ev, "probability", 0.0))
    threshold = float(getattr(ev, "threshold", 0.0))
    result = "max_delay" if probability < threshold else "min_delay"
    chosen_delay = getattr(ev, "endpointing_delay", None)
    if chosen_delay is None:
        chosen_delay = endpoint_max_delay if result == "max_delay" else endpoint_min_delay
    payload = {
        "type": "agent_eou_debug",
        "room": ctx.room.name,
        "participantIdentity": participant_identity,
        "probability": probability,
        "threshold": threshold,
        "endpointingDelay": float(chosen_delay),
        "result": result,
        "trigger": getattr(ev, "trigger", None) or "unknown",
        "inferenceDuration": float(getattr(ev, "inference_duration", 0.0)),
        "predictionDelay": float(getattr(ev, "delay", 0.0)),
        "endpointMinDelay": endpoint_min_delay,
        "endpointMaxDelay": endpoint_max_delay,
        "endpointSystemMaxDelay": endpoint_system_max_delay,
        "endpointDebugMaxDelay": endpoint_debug_max_delay,
        "endpointMaxSource": endpoint_max_source,
        "createdAt": float(getattr(ev, "created_at", 0.0)),
    }
    destination_identities = [participant_identity] if participant_identity else []
    try:
        await ctx.room.local_participant.publish_data(
            json.dumps(payload, ensure_ascii=False),
            reliable=True,
            destination_identities=destination_identities,
            topic=EOU_DEBUG_TOPIC,
        )
    except Exception:
        logger.exception("failed to publish EOU debug event")
    else:
        logger.info(
            "agent eou debug result=%s probability=%.4f threshold=%.4f endpointing_delay=%.3f trigger=%s source=%s",
            result,
            probability,
            threshold,
            float(chosen_delay),
            payload["trigger"],
            endpoint_max_source,
        )


async def _clear_latency_metrics_callback() -> None:
    set_metrics_callback(None)


async def _publish_latency_metrics(
    *,
    ctx: JobContext,
    payload: dict[str, object],
    participant_identity: str | None,
) -> None:
    enriched_payload = {
        **payload,
        "room": ctx.room.name,
        "participantIdentity": participant_identity,
        "createdAt": time.time(),
    }
    destination_identities = [participant_identity] if participant_identity else []
    try:
        await ctx.room.local_participant.publish_data(
            json.dumps(enriched_payload, ensure_ascii=False),
            reliable=True,
            destination_identities=destination_identities,
            topic=LATENCY_METRICS_TOPIC,
        )
    except Exception:
        logger.exception("failed to publish latency metrics")
    else:
        metrics_payload = enriched_payload.get("metrics") or {}
        if isinstance(metrics_payload, dict):
            logger.info(
                "latency metrics seq=%s asr=%s eou_wait=%s llm=%s tts=%s total=%s source=%s",
                enriched_payload.get("seq"),
                metrics_payload.get("audio_tail_to_asr_final_ms"),
                metrics_payload.get("asr_final_to_eou_confirmed_ms"),
                metrics_payload.get("llm_input_to_first_token_ms"),
                metrics_payload.get("tts_text_to_first_audio_ms"),
                metrics_payload.get("audio_tail_to_tts_first_audio_ms"),
                (enriched_payload.get("debug") or {}).get("audioTailSource")
                if isinstance(enriched_payload.get("debug"), dict)
                else None,
            )


def _metadata_participant_identity(metadata: str | bytes | None) -> str | None:
    if not metadata:
        return None
    if isinstance(metadata, bytes):
        metadata = metadata.decode("utf-8", errors="replace")
    try:
        payload = json.loads(metadata)
    except json.JSONDecodeError:
        logger.warning("agent dispatch metadata is not valid JSON: %s", metadata)
        return None
    if not isinstance(payload, dict):
        return None
    identity = payload.get("participant_identity") or payload.get("participantIdentity")
    if identity is None:
        return None
    identity = str(identity).strip()
    return identity or None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int | None = None) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float | None = None) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def main() -> None:
    load_runtime_env()
    cli.run_app(server)


if __name__ == "__main__":
    main()
