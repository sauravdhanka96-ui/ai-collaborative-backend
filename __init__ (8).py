"""Session management routes + WebSocket endpoint."""

import json
import logging
import uuid

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres import get_db
from app.models.models import CollabSession
from app.services.collaboration.ws_manager import manager

logger = logging.getLogger(__name__)
router = APIRouter()


class CreateSessionRequest(BaseModel):
    name: str
    owner_id: str


class SessionResponse(BaseModel):
    id: uuid.UUID
    name: str
    owner_id: str


@router.post("/", response_model=SessionResponse, status_code=201)
async def create_session(payload: CreateSessionRequest, db: AsyncSession = Depends(get_db)):
    session = CollabSession(name=payload.name, owner_id=payload.owner_id)
    db.add(session)
    await db.flush()
    return session


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(CollabSession).where(CollabSession.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("/{session_id}/users")
async def active_users(session_id: uuid.UUID):
    return {"users": manager.active_users(str(session_id))}


@router.websocket("/{session_id}/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: uuid.UUID,
    user_id: str,
):
    """
    WebSocket endpoint for real-time collaboration.
    Clients send operational transform (OT) events; the server
    broadcasts them to all other participants in the session.
    """
    sid = str(session_id)
    await manager.connect(websocket, sid, user_id)

    # Notify others that a new user joined
    await manager.broadcast(
        sid,
        {"type": "user_joined", "user_id": user_id, "active_users": manager.active_users(sid)},
        exclude_ws=websocket,
    )

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "detail": "Invalid JSON"}))
                continue

            event["sender"] = user_id
            await manager.broadcast(sid, event, exclude_ws=websocket)

    except WebSocketDisconnect:
        await manager.disconnect(websocket, sid)
        await manager.broadcast(
            sid,
            {"type": "user_left", "user_id": user_id, "active_users": manager.active_users(sid)},
        )
