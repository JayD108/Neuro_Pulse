"""
NeuroPulse WebSocket Connection Manager
========================================
Tracks all active WebSocket connections and provides
safe broadcast / disconnect helpers.
"""

from typing import Set
from fastapi import WebSocket


class WebSocketManager:
    """Lightweight manager for active WebSocket connections."""

    def __init__(self):
        self._active: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._active.add(websocket)
        print(f"[WS] Client connected | Active={len(self._active)}")

    def disconnect(self, websocket: WebSocket) -> None:
        self._active.discard(websocket)
        print(f"[WS] Client disconnected | Active={len(self._active)}")

    def is_connected(self, websocket: WebSocket) -> bool:
        return websocket in self._active

    async def broadcast(self, message: dict) -> None:
        """Send a JSON message to ALL connected clients."""
        dead = set()
        for ws in self._active:
            try:
                await ws.send_json(message)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self._active.discard(ws)

    @property
    def connection_count(self) -> int:
        return len(self._active)
