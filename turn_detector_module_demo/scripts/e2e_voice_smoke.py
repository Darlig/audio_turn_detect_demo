from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO_ROOT / "turn_detector_module_demo"
for path in (
    REPO_ROOT,
    REPO_ROOT / "llm_module",
    REPO_ROOT / "tts_module",
    REPO_ROOT / "agents" / "livekit-agents",
):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from livekit import api, rtc  # noqa: E402
from livekit.protocol.room import RoomConfiguration  # noqa: E402
from doubao_tts import synthesize_text_to_pcm  # noqa: E402


DEFAULT_ROOM = "audio-turn-e2e"
DEFAULT_TEXT = "你好，请用一句话介绍一下你自己。"


@dataclass
class SmokeResult:
    room: str
    input_wav: Path
    participants: list[str] = field(default_factory=list)
    subscribed_tracks: list[str] = field(default_factory=list)
    transcripts: list[str] = field(default_factory=list)
    agent_transcripts_after_user_final: list[str] = field(default_factory=list)
    agent_audio_frames: int = 0
    agent_audio_samples: int = 0
    agent_audio_rms_peak: float = 0.0
    agent_audio_frames_after_user_final: int = 0
    agent_audio_rms_peak_after_user_final: float = 0.0
    audio_errors: list[str] = field(default_factory=list)


def load_env_file(path: Path) -> None:
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
    load_env_file(PROJECT_DIR / ".env")
    load_env_file(REPO_ROOT / "llm_module" / "doubao_llm" / ".env")
    load_env_file(REPO_ROOT / "tts_module" / "doubao_tts" / ".env")
    os.environ.setdefault("LIVEKIT_AGENT_NAME", "cascade-voice-agent")
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,::1")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost,::1")


def bypass_proxy_for_local_url(url: str) -> None:
    if url.startswith("ws://127.0.0.1") or url.startswith("ws://localhost"):
        for key in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "http_proxy",
            "https_proxy",
            "ALL_PROXY",
            "all_proxy",
        ):
            os.environ.pop(key, None)


def livekit_api_url(livekit_url: str) -> str:
    if livekit_url.startswith("ws://"):
        return "http://" + livekit_url.removeprefix("ws://")
    if livekit_url.startswith("wss://"):
        return "https://" + livekit_url.removeprefix("wss://")
    return livekit_url


async def create_agent_dispatch(
    livekit_url: str,
    room_name: str,
    agent_name: str,
    participant_identity: str,
) -> api.AgentDispatch:
    client = api.LiveKitAPI(
        livekit_api_url(livekit_url),
        os.getenv("LIVEKIT_API_KEY", "devkey"),
        os.getenv("LIVEKIT_API_SECRET", "secret"),
    )
    try:
        dispatch = await client.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                room=room_name,
                agent_name=agent_name,
                metadata=json.dumps(
                    {
                        "source": "e2e_voice_smoke",
                        "participant_identity": participant_identity,
                    }
                ),
            )
        )
        print(
            "[smoke] agent dispatch created: "
            f"id={dispatch.id or '(empty)'} agent={dispatch.agent_name or '(default)'}"
        )
        return dispatch
    finally:
        await client.aclose()


def make_token(identity: str, room: str, *, agent_name: str = "") -> str:
    api_key = os.getenv("LIVEKIT_API_KEY", "devkey")
    api_secret = os.getenv("LIVEKIT_API_SECRET", "secret")
    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
            )
        )
    )
    if agent_name:
        room_config = RoomConfiguration()
        room_config.agents.add(
            agent_name=agent_name,
            metadata=json.dumps(
                {
                    "source": "e2e_voice_smoke",
                    "participant_identity": identity,
                }
            ),
        )
        token.with_room_config(room_config)
    return token.to_jwt()


async def synthesize_input_wav(text: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pcm = await synthesize_text_to_pcm(text)
    if not pcm.pcm:
        raise RuntimeError("TTS returned empty input audio")

    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(pcm.num_channels)
        wav.setsampwidth(2)
        wav.setframerate(pcm.sample_rate)
        wav.writeframes(pcm.pcm)
    return output_path


def wav_frames(path: Path, *, frame_ms: int = 20) -> list[rtc.AudioFrame]:
    frames: list[rtc.AudioFrame] = []
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        chunk_samples = max(1, sample_rate * frame_ms // 1000)
        while True:
            raw = wav.readframes(chunk_samples)
            if not raw:
                break
            samples = len(raw) // (channels * 2)
            if samples <= 0:
                continue
            frames.append(
                rtc.AudioFrame(
                    data=raw,
                    sample_rate=sample_rate,
                    num_channels=channels,
                    samples_per_channel=samples,
                )
            )
    return frames


def silence_frame(sample_rate: int, channels: int, duration_ms: int) -> rtc.AudioFrame:
    samples = max(1, sample_rate * duration_ms // 1000)
    return rtc.AudioFrame(
        data=bytes(samples * channels * 2),
        sample_rate=sample_rate,
        num_channels=channels,
        samples_per_channel=samples,
    )


def frame_rms(frame: rtc.AudioFrame) -> float:
    data = frame.data
    if isinstance(data, memoryview):
        samples = data if data.format == "h" else data.cast("B").cast("h")
    else:
        samples = memoryview(data).cast("B").cast("h")
    if len(samples) == 0:
        return 0.0
    acc = sum(int(sample) * int(sample) for sample in samples)
    return math.sqrt(acc / len(samples)) / 32768.0


async def collect_agent_audio(
    track: rtc.RemoteAudioTrack,
    result: SmokeResult,
    stop_event: asyncio.Event,
    user_final_seen: asyncio.Event,
) -> None:
    stream = rtc.AudioStream(track)
    try:
        async for event in stream:
            rms = frame_rms(event.frame)
            result.agent_audio_frames += 1
            result.agent_audio_samples += event.frame.samples_per_channel
            result.agent_audio_rms_peak = max(result.agent_audio_rms_peak, rms)
            if user_final_seen.is_set():
                result.agent_audio_frames_after_user_final += 1
                result.agent_audio_rms_peak_after_user_final = max(
                    result.agent_audio_rms_peak_after_user_final,
                    rms,
                )
            if stop_event.is_set():
                break
    except Exception as exc:
        result.audio_errors.append(f"{type(exc).__name__}: {exc}")
        print(f"[smoke] audio stream error: {type(exc).__name__}: {exc}")
    finally:
        await stream.aclose()


async def run_smoke(args: argparse.Namespace) -> SmokeResult:
    load_runtime_env()
    room_name = args.room or f"{DEFAULT_ROOM}-{uuid.uuid4().hex[:8]}"
    input_wav = Path(args.input_wav).resolve() if args.input_wav else (
        PROJECT_DIR / "logs" / f"{room_name}-input.wav"
    )
    result = SmokeResult(room=room_name, input_wav=input_wav)

    if not input_wav.exists():
        print(f"[smoke] synthesizing input wav: {input_wav}")
        await synthesize_input_wav(args.text, input_wav)
    else:
        print(f"[smoke] using input wav: {input_wav}")

    frames = wav_frames(input_wav)
    if not frames:
        raise RuntimeError(f"input wav has no audio frames: {input_wav}")

    livekit_url = os.getenv("LIVEKIT_URL", "ws://127.0.0.1:8890")
    bypass_proxy_for_local_url(livekit_url)
    user_identity = f"e2e-user-{uuid.uuid4().hex[:6]}"
    token = make_token(
        identity=user_identity,
        room=room_name,
        agent_name=args.agent_name if not args.api_dispatch else "",
    )
    room = rtc.Room()
    stop_audio = asyncio.Event()
    audio_tasks: list[asyncio.Task[None]] = []

    connected = asyncio.Event()
    agent_audio_seen = asyncio.Event()
    user_final_seen = asyncio.Event()

    @room.on("connected")
    def _connected() -> None:
        print("[smoke] room connected")
        connected.set()

    @room.on("participant_connected")
    def _participant_connected(participant: rtc.RemoteParticipant) -> None:
        print(f"[smoke] participant joined: {participant.identity}")
        result.participants.append(participant.identity)

    @room.on("track_subscribed")
    def _track_subscribed(
        track: rtc.Track,
        _publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        print(f"[smoke] track subscribed: {participant.identity} {track.kind}")
        result.subscribed_tracks.append(f"{participant.identity}:{track.kind}")
        if isinstance(track, rtc.RemoteAudioTrack):
            agent_audio_seen.set()
            audio_tasks.append(
                asyncio.create_task(collect_agent_audio(track, result, stop_audio, user_final_seen))
            )

    @room.on("transcription_received")
    def _transcription_received(
        segments: list[rtc.TranscriptionSegment],
        participant: rtc.Participant | None,
        _publication: rtc.TrackPublication | None,
    ) -> None:
        speaker = getattr(participant, "identity", "unknown")
        for segment in segments:
            text = (segment.text or "").strip()
            if text:
                status = "final" if segment.final else "interim"
                line = f"{status} {speaker}: {text}"
                print(f"[smoke] transcript {line}")
                result.transcripts.append(line)
                if user_final_seen.is_set() and speaker.startswith("agent-"):
                    result.agent_transcripts_after_user_final.append(line)
                if segment.final and speaker.startswith("e2e-user-"):
                    user_final_seen.set()

    print(f"[smoke] connecting to {livekit_url}, room={room_name}")
    await room.connect(livekit_url, token)
    connected.set()
    await asyncio.wait_for(connected.wait(), timeout=args.connect_timeout)
    if args.api_dispatch:
        await create_agent_dispatch(livekit_url, room_name, args.agent_name, user_identity)

    source = rtc.AudioSource(frames[0].sample_rate, frames[0].num_channels)
    track = rtc.LocalAudioTrack.create_audio_track("e2e-user-audio", source)
    await room.local_participant.publish_track(
        track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )
    print("[smoke] published local audio track")

    agent_join_deadline = time.monotonic() + args.agent_wait_timeout
    while time.monotonic() < agent_join_deadline and not result.participants:
        await source.capture_frame(
            silence_frame(frames[0].sample_rate, frames[0].num_channels, 100)
        )
        await asyncio.sleep(0.1)
    if not result.participants:
        raise RuntimeError("agent did not join after dispatch")

    if args.pre_speech_delay_ms > 0:
        print(f"[smoke] waiting {args.pre_speech_delay_ms}ms before user speech")
        for _ in range(args.pre_speech_delay_ms // 100):
            await source.capture_frame(
                silence_frame(frames[0].sample_rate, frames[0].num_channels, 100)
            )
            await asyncio.sleep(0.1)

    print(f"[smoke] published {len(frames)} input frames")

    for frame in frames:
        await source.capture_frame(frame)
        await asyncio.sleep(frame.samples_per_channel / frame.sample_rate)

    for _ in range(args.trailing_silence_ms // 100):
        await source.capture_frame(silence_frame(frames[0].sample_rate, frames[0].num_channels, 100))
        await asyncio.sleep(0.1)

    deadline = time.monotonic() + args.reply_timeout
    while time.monotonic() < deadline:
        post_user_audio_ready = (
            result.agent_audio_frames_after_user_final >= args.min_agent_audio_frames
            and result.agent_audio_rms_peak_after_user_final >= args.min_agent_audio_rms_peak
        )
        if (
            user_final_seen.is_set()
            and post_user_audio_ready
        ):
            break
        if (
            user_final_seen.is_set()
            and not args.require_decoded_audio
            and bool(result.agent_transcripts_after_user_final)
        ):
            break
        await asyncio.sleep(0.1)

    stop_audio.set()
    await asyncio.sleep(0.2)
    for task in audio_tasks:
        task.cancel()
    await asyncio.gather(*audio_tasks, return_exceptions=True)
    await room.disconnect()

    if not result.participants:
        raise RuntimeError("no remote participant joined; voice agent may not be running")
    if not result.subscribed_tracks:
        raise RuntimeError("no agent track subscribed")
    user_finals = [
        line for line in result.transcripts
        if line.startswith("final e2e-user-")
    ]
    if not user_finals:
        raise RuntimeError("no final user transcript received")
    if (
        not result.agent_transcripts_after_user_final
        and result.agent_audio_frames_after_user_final < args.min_agent_audio_frames
    ):
        raise RuntimeError("no post-user agent transcript or post-user agent audio received")
    if args.require_decoded_audio and result.agent_audio_frames_after_user_final < args.min_agent_audio_frames:
        raise RuntimeError(
            "agent audio was not received: "
            f"post_user_frames={result.agent_audio_frames_after_user_final}, "
            f"frames={result.agent_audio_frames}, peak_rms={result.agent_audio_rms_peak:.4f}"
        )
    if (
        args.require_decoded_audio
        and result.agent_audio_rms_peak_after_user_final < args.min_agent_audio_rms_peak
    ):
        raise RuntimeError(
            "agent audio after user final was silent: "
            f"post_user_peak_rms={result.agent_audio_rms_peak_after_user_final:.6f}, "
            f"min={args.min_agent_audio_rms_peak:.6f}"
        )
    if result.agent_audio_frames_after_user_final < args.min_agent_audio_frames:
        print(
            "[smoke] warning: decoded post-user agent PCM below threshold; "
            f"post_user_frames={result.agent_audio_frames_after_user_final}, "
            f"frames={result.agent_audio_frames}, peak_rms={result.agent_audio_rms_peak:.4f}. "
            "The synchronized agent transcript was received."
        )
    return result


def parse_args() -> argparse.Namespace:
    load_runtime_env()
    parser = argparse.ArgumentParser(description="Run real LiveKit voice agent E2E smoke test")
    parser.add_argument("--room", default="")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--input-wav", default="")
    parser.add_argument("--connect-timeout", type=float, default=15.0)
    parser.add_argument("--agent-wait-timeout", type=float, default=20.0)
    parser.add_argument("--reply-timeout", type=float, default=45.0)
    parser.add_argument("--trailing-silence-ms", type=int, default=1800)
    parser.add_argument("--pre-speech-delay-ms", type=int, default=7000)
    parser.add_argument("--min-agent-audio-frames", type=int, default=5)
    parser.add_argument("--min-agent-audio-rms-peak", type=float, default=0.001)
    parser.add_argument("--require-decoded-audio", action="store_true")
    parser.add_argument("--api-dispatch", action="store_true")
    parser.add_argument("--agent-name", default=os.getenv("LIVEKIT_AGENT_NAME", "cascade-voice-agent"))
    return parser.parse_args()


async def amain() -> None:
    result = await run_smoke(parse_args())
    print("[smoke] PASS")
    print(f"[smoke] room={result.room}")
    print(f"[smoke] input_wav={result.input_wav}")
    print(f"[smoke] participants={result.participants}")
    print(f"[smoke] tracks={result.subscribed_tracks}")
    print(f"[smoke] transcripts={result.transcripts}")
    print(f"[smoke] post_user_agent_transcripts={result.agent_transcripts_after_user_final}")
    print(
        "[smoke] agent_audio="
        f"frames={result.agent_audio_frames} "
        f"post_user_frames={result.agent_audio_frames_after_user_final} "
        f"samples={result.agent_audio_samples} "
        f"peak_rms={result.agent_audio_rms_peak:.6f} "
        f"post_user_peak_rms={result.agent_audio_rms_peak_after_user_final:.6f}"
    )
    if result.audio_errors:
        print(f"[smoke] audio_errors={result.audio_errors}")


if __name__ == "__main__":
    asyncio.run(amain())
