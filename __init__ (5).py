"""
WebSocket Connection Manager.
Maintains per-session connection pools and broadcasts operational transforms.
Designed for sub-100ms message relay under concurrent load.
"""

import asyncio
import json
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


@dataclass
class ConnectedClient:
    websocket: WebSocket
    user_id: str
    connected_at: datetime = field(default_factory=datetime.utcnow)


class ConnectionManager:
    def __init__(self):
        # session_id -> list of connected clients
        self._sessions: dict[str, list[ConnectedClient]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, session_id: str, user_id: str) -> None:
        await websocket.accept()
        client = ConnectedClient(websocket=websocket, user_id=user_id)
        async with self._lock:
            self._sessions[session_id].append(client)
        logger.info(f"User {user_id} connected to session {session_id} "
                    f"(total: {len(self._sessions[session_id])})")

    async def disconnect(self, websocket: WebSocket, session_id: str) -> None:
        async with self._lock:
            self._sessions[session_id] = [
                c for c in self._sessions[session_id] if c.websocket is not websocket
            ]
            if not self._sessions[session_id]:
                del self._sessions[session_id]
        logger.info(f"Client disconnected from session {session_id}")

    async def broadcast(
        self,
        session_id: str,
        message: dict[str, Any],
        exclude_ws: WebSocket | None = None,
    ) -> None:
        """Broadcast a message to all clients in a session (optionally excluding sender)."""
        payload = json.dumps(message)
        clients = self._sessions.get(session_id, [])
        dead: list[ConnectedClient] = []

        await asyncio.gather(
            *[
                self._safe_send(c, payload, dead)
                for c in clients
                if c.websocket is not exclude_ws
            ],
            return_exceptions=True,
        )

        # Prune dead connections
        if dead:
            async with self._lock:
                for c in dead:
                    self._sessions[session_id] = [
                        x for x in self._sessions[session_id] if x is not c
                    ]

    @staticmethod
    async def _safe_send(client: ConnectedClient, payload: str, dead: list) -> None:
        try:
            await client.websocket.send_text(payload)
        except Exception:
            dead.append(client)

    def active_users(self, session_id: str) -> list[str]:
        return [c.user_id for c in self._sessions.get(session_id, [])]

    def session_count(self) -> int:
        return len(self._sessions)


# Singleton shared across the app
manager = ConnectionManager()
