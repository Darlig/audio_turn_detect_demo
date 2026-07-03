from __future__ import annotations

from collections.abc import AsyncIterator
import asyncio
import contextlib
import ssl

import websockets
from websockets.asyncio.client import ClientConnection

try:
    from .funasr_protocol import ASRResult, parse_server_message, start_message, stop_message
except ImportError:
    from funasr_protocol import ASRResult, parse_server_message, start_message, stop_message


class FunASRWebSocketClient:
    def __init__(
        self,
        uri: str,
        *,
        mode: str = "2pass",
        chunk_size: tuple[int, int, int] = (8, 8, 4),
        chunk_interval: int = 10,
        hotwords: str = "",
        wav_name: str = "microphone",
        semantic_turn_detection: bool = False,
    ) -> None:
        self.uri = uri
        self.mode = mode
        self.chunk_size = chunk_size
        self.chunk_interval = chunk_interval
        self.hotwords = hotwords
        self.wav_name = wav_name
        self.semantic_turn_detection = semantic_turn_detection
        self._ws: ClientConnection | None = None
        self._rx_task: asyncio.Task[None] | None = None
        self._events: asyncio.Queue[ASRResult | None] = asyncio.Queue()
        self._utterance_open = False

    async def __aenter__(self) -> "FunASRWebSocketClient":
        await self.connect()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def connect(self) -> None:
        ssl_context = ssl.create_default_context() if self.uri.startswith("wss://") else None
        self._ws = await websockets.connect(
            self.uri,
            ssl=ssl_context,
            max_size=None,
            subprotocols=["binary"],
        )
        await self._start_utterance()
        self._rx_task = asyncio.create_task(self._receive_loop(), name="funasr-ws-receive")

    async def send_audio(self, pcm16le: bytes) -> None:
        if not self._ws:
            raise RuntimeError("FunASR websocket is not connected")
        if not self._utterance_open:
            await self._start_utterance()
        await self._ws.send(pcm16le)

    async def finish_utterance(self) -> None:
        if self._ws and self._utterance_open:
            await self._ws.send(stop_message())
            self._utterance_open = False

    async def events(self) -> AsyncIterator[ASRResult]:
        while True:
            item = await self._events.get()
            if item is None:
                break
            yield item

    async def close(self) -> None:
        if self._ws:
            if self._utterance_open:
                with contextlib.suppress(Exception):
                    await self._ws.send(stop_message())
                self._utterance_open = False
            await self._ws.close()
            self._ws = None
        if self._rx_task:
            self._rx_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._rx_task
            self._rx_task = None
        await self._events.put(None)

    async def _start_utterance(self) -> None:
        if not self._ws:
            raise RuntimeError("FunASR websocket is not connected")
        await self._ws.send(
            start_message(
                mode=self.mode,
                chunk_size=self.chunk_size,
                chunk_interval=self.chunk_interval,
                wav_name=self.wav_name,
                hotwords=self.hotwords,
                semantic_turn_detection=self.semantic_turn_detection,
            )
        )
        self._utterance_open = True

    async def _receive_loop(self) -> None:
        assert self._ws is not None
        try:
            async for message in self._ws:
                result = parse_server_message(message)
                if result:
                    await self._events.put(result)
        finally:
            await self._events.put(None)
