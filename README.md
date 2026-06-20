"""Health and readiness probe endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres import get_db
from app.services.collaboration.ws_manager import manager

router = APIRouter()


@router.get("/")
async def health():
    return {"status": "ok"}


@router.get("/ready")
async def readiness(db: AsyncSession = Depends(get_db)):
    """Checks DB connectivity and reports active WebSocket sessions."""
    await db.execute(text("SELECT 1"))
    return {
        "status": "ready",
        "active_sessions": manager.session_count(),
    }
