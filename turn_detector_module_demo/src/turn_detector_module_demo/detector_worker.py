from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass

from livekit import rtc
from livekit.agents import LanguageCode
from livekit.agents.inference import TurnDetector

from .audio_math import decision_from_score, pcm16_rms01
from .events import EventHub
from .token import build_token


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectorWorkerConfig:
    livekit_url: str
    api_key: str
    api_secret: str
    room: str
    language: str = "zh"
    identity: str = "audio-eot-detector"
    sample_rate: int = 16000
    frame_size_ms: int = 40
    predict_interval_ms: int = 120


class DetectorWorker:
    def __init__(self, config: DetectorWorkerConfig, hub: EventHub) -> None:
        self._config = config
        self._hub = hub
        self._room: rtc.Room | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        token = build_token(
            api_key=self._config.api_key,
            api_secret=self._config.api_secret,
            room=self._config.room,
            identity=self._config.identity,
            name="Audio EOT Detector",
            can_publish=False,
            can_subscribe=True,
        )
        room = rtc.Room()
        self._room = room

        @room.on("track_subscribed")
        def _on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            if track.kind != rtc.TrackKind.KIND_AUDIO:
                return
            task = asyncio.create_task(self._process_track(track, publication, participant))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        @room.on("track_unsubscribed")
        def _on_track_unsubscribed(
            track: rtc.Track,
            _publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            asyncio.create_task(
                self._hub.broadcast(
                    {
                        "type": "status",
                        "level": "info",
                        "message": f"audio track unsubscribed: {participant.identity}",
                        "trackSid": getattr(track, "sid", ""),
                    }
                )
            )

        await room.connect(self._config.livekit_url, token)
        await self._hub.broadcast(
            {
                "type": "status",
                "level": "ok",
                "message": f"detector connected to room {self._config.room}",
                "room": self._config.room,
                "model": "turn-detector-v1-mini",
            }
        )

    async def stop(self) -> None:
        self._stopping.set()
        for task in list(self._tasks):
            task.cancel()
        for task in list(self._tasks):
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self._room is not None:
            await self._room.disconnect()
            self._room = None

    async def _process_track(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        detector = TurnDetector(version="v1-mini", sample_rate=self._config.sample_rate)
        stream = detector.stream()
        audio_stream = rtc.AudioStream.from_track(
            track=track,
            sample_rate=self._config.sample_rate,
            num_channels=1,
            frame_size_ms=self._config.frame_size_ms,
        )
        started_at = time.monotonic()
        elapsed_sec = 0.0
        next_predict_sec = 0.0
        language = LanguageCode(self._config.language)
        threshold = await detector.unlikely_threshold(language)
        track_sid = publication.sid

        await self._hub.broadcast(
            {
                "type": "track_started",
                "participant": participant.identity,
                "trackSid": track_sid,
                "threshold": threshold,
                "language": self._config.language,
                "model": detector.model,
            }
        )

        try:
            async for event in audio_stream:
                if self._stopping.is_set():
                    break

                frame = event.frame
                stream.push_audio(frame)
                elapsed_sec += frame.samples_per_channel / frame.sample_rate
                rms = pcm16_rms01(frame.data)

                if elapsed_sec >= next_predict_sec:
                    next_predict_sec = elapsed_sec + self._config.predict_interval_ms / 1000.0
                    probability = await self._predict(stream)
                    decision = decision_from_score(probability, threshold)
                    await self._hub.broadcast(
                        {
                            "type": "point",
                            "participant": participant.identity,
                            "trackSid": track_sid,
                            "time": round(elapsed_sec, 3),
                            "wallTime": round(time.monotonic() - started_at, 3),
                            "rms": round(rms, 4),
                            "score": round(probability, 4),
                            "threshold": threshold,
                            "decision": decision,
                        }
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("audio track processing failed")
            await self._hub.broadcast(
                {
                    "type": "status",
                    "level": "error",
                    "message": f"track processing failed: {exc}",
                    "trackSid": track_sid,
                }
            )
        finally:
            stream.end_input()
            await stream.aclose()
            await self._hub.broadcast(
                {
                    "type": "track_ended",
                    "participant": participant.identity,
                    "trackSid": track_sid,
                    "time": round(elapsed_sec, 3),
                }
            )

    async def _predict(self, stream: object) -> float:
        try:
            future = stream.predict()
            event = await asyncio.wait_for(future, timeout=2.0)
            return float(event.end_of_turn_probability)
        except Exception:
            logger.exception("turn detector prediction failed")
            return 0.0
