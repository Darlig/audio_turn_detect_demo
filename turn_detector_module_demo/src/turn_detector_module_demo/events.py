from __future__ import annotations

import asyncio
import json
from typing import Any

from aiohttp import web


class EventHub:
    def __init__(self) -> None:
        self._clients: set[web.WebSocketResponse] = set()
        self._lock = asyncio.Lock()

    async def add(self, ws: web.WebSocketResponse) -> None:
        async with self._lock:
            self._clients.add(ws)

    async def remove(self, ws: web.WebSocketResponse) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        message = json.dumps(payload, separators=(",", ":"))
        async with self._lock:
            clients = list(self._clients)

        stale: list[web.WebSocketResponse] = []
        for ws in clients:
            if ws.closed:
                stale.append(ws)
                continue
            try:
                await ws.send_str(message)
            except ConnectionResetError:
                stale.append(ws)

        if stale:
            async with self._lock:
                for ws in stale:
                    self._clients.discard(ws)

    async def send(self, ws: web.WebSocketResponse, payload: dict[str, Any]) -> bool:
        """Send a detector event only to the client that registered its track."""
        if ws.closed:
            await self.remove(ws)
            return False
        try:
            await ws.send_str(json.dumps(payload, separators=(",", ":")))
            return True
        except ConnectionResetError:
            await self.remove(ws)
            return False
