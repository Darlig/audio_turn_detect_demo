from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass

from aiohttp import web
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
        self._track_tasks: dict[str, asyncio.Task[None]] = {}
        self._registrations: dict[str, tuple[web.WebSocketResponse, str, str]] = {}
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
            registration = self._registrations.get(publication.sid)
            if track.kind != rtc.TrackKind.KIND_AUDIO or registration is None:
                return
            ws, identity, run_id = registration
            if participant.identity != identity or publication.sid in self._track_tasks:
                return
            task = asyncio.create_task(
                self._process_track(track, publication, participant, ws=ws, run_id=run_id)
            )
            self._tasks.add(task)
            self._track_tasks[publication.sid] = task
            task.add_done_callback(self._tasks.discard)
            task.add_done_callback(
                lambda _task, sid=publication.sid: self._track_tasks.pop(sid, None)
            )

        @room.on("track_unsubscribed")
        def _on_track_unsubscribed(
            track: rtc.Track,
            _publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            track_sid = _publication.sid
            registration = self._registrations.get(track_sid)
            if registration is not None:
                ws, _identity, run_id = registration
                asyncio.create_task(
                    self._hub.send(
                        ws,
                        {
                            "type": "status",
                            "level": "info",
                            "message": f"audio track unsubscribed: {participant.identity}",
                            "trackSid": track_sid,
                            "room": self._config.room,
                            "runId": run_id,
                        },
                    )
                )

        await room.connect(
            self._config.livekit_url,
            token,
            options=rtc.RoomOptions(auto_subscribe=False),
        )
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

    async def start_track(
        self,
        *,
        ws: web.WebSocketResponse,
        room: str,
        identity: str,
        track_sid: str,
        run_id: str,
    ) -> None:
        if room != self._config.room:
            raise ValueError(f"detector is connected to room {self._config.room}, not {room}")
        if not identity or not track_sid or not run_id:
            raise ValueError("identity, trackSid, and runId are required")
        existing = self._registrations.get(track_sid)
        if existing is not None and existing[0] is not ws:
            raise ValueError(f"track is already registered by another client: {track_sid}")
        await self.stop_track(track_sid=track_sid, ws=ws)
        publication = self._find_publication(identity, track_sid)
        for _ in range(20):
            if publication is not None:
                break
            await asyncio.sleep(0.05)
            publication = self._find_publication(identity, track_sid)
        if publication is None or publication.kind != rtc.TrackKind.KIND_AUDIO:
            raise ValueError(f"audio track not found for {identity}: {track_sid}")
        self._registrations[track_sid] = (ws, identity, run_id)
        if publication.track is not None:
            participant = self._room.remote_participants[identity]  # type: ignore[union-attr]
            task = asyncio.create_task(
                self._process_track(
                    publication.track, publication, participant, ws=ws, run_id=run_id
                )
            )
            self._tasks.add(task)
            self._track_tasks[track_sid] = task
            task.add_done_callback(self._tasks.discard)
            task.add_done_callback(
                lambda _task, sid=track_sid: self._track_tasks.pop(sid, None)
            )
        else:
            publication.set_subscribed(True)

    async def stop_track(
        self, *, track_sid: str, ws: web.WebSocketResponse | None = None
    ) -> None:
        registration = self._registrations.get(track_sid)
        if ws is not None and registration is not None and registration[0] is not ws:
            raise ValueError(f"track is registered by another client: {track_sid}")
        self._registrations.pop(track_sid, None)
        task = self._track_tasks.pop(track_sid, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        publication = self._find_publication(None, track_sid)
        if publication is not None and publication.subscribed:
            publication.set_subscribed(False)

    async def stop_client(self, ws: web.WebSocketResponse) -> None:
        track_sids = [
            sid
            for sid, registration in self._registrations.items()
            if registration[0] is ws
        ]
        for track_sid in track_sids:
            await self.stop_track(track_sid=track_sid, ws=ws)

    def _find_publication(
        self, identity: str | None, track_sid: str
    ) -> rtc.RemoteTrackPublication | None:
        if self._room is None:
            return None
        participants = (
            [self._room.remote_participants.get(identity)]
            if identity is not None
            else list(self._room.remote_participants.values())
        )
        for participant in participants:
            if participant is None:
                continue
            publication = participant.track_publications.get(track_sid)
            if isinstance(publication, rtc.RemoteTrackPublication):
                return publication
        return None

    async def _process_track(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
        *,
        ws: web.WebSocketResponse,
        run_id: str,
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

        await self._hub.send(
            ws,
            {
                "type": "track_started",
                "participant": participant.identity,
                "trackSid": track_sid,
                "threshold": threshold,
                "language": self._config.language,
                "model": detector.model,
                "room": self._config.room,
                "runId": run_id,
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
                    await self._hub.send(
                        ws,
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
                            "room": self._config.room,
                            "runId": run_id,
                        }
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("audio track processing failed")
            await self._hub.send(
                ws,
                {
                    "type": "status",
                    "level": "error",
                    "message": f"track processing failed: {exc}",
                    "trackSid": track_sid,
                    "room": self._config.room,
                    "runId": run_id,
                }
            )
        finally:
            stream.end_input()
            await stream.aclose()
            await self._hub.send(
                ws,
                {
                    "type": "track_ended",
                    "participant": participant.identity,
                    "trackSid": track_sid,
                    "time": round(elapsed_sec, 3),
                    "room": self._config.room,
                    "runId": run_id,
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
