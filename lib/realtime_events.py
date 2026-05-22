import asyncio
from typing import Optional


class EvaluationEventHub:
    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._clients = set()
        self._lock = asyncio.Lock()

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    async def add_client(self, websocket, initial_payload=None):
        await websocket.accept()
        async with self._lock:
            self._clients.add(websocket)
        if initial_payload is not None:
            try:
                await websocket.send_json(initial_payload)
            except Exception:
                pass

    async def remove_client(self, websocket):
        async with self._lock:
            self._clients.discard(websocket)

    async def _broadcast(self, payload: dict):
        async with self._lock:
            clients = list(self._clients)
        if not clients:
            return
        stale = []
        for client in clients:
            try:
                await client.send_json(payload)
            except Exception:
                stale.append(client)
        if stale:
            async with self._lock:
                for client in stale:
                    self._clients.discard(client)

    def notify(self, payload: dict):
        if not self._loop or self._loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)


evaluation_event_hub = EvaluationEventHub()

