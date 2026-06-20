"""AI suggestion endpoints — standard and streaming."""

import uuid
from openai import AsyncOpenAI
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.postgres import get_db
from app.services.ai.rag_pipeline import generate_suggestion, stream_suggestion

router = APIRouter()

_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


class SuggestionRequest(BaseModel):
    session_id: uuid.UUID
    query: str


@router.post("/suggest")
async def suggest(payload: SuggestionRequest, db: AsyncSession = Depends(get_db)):
    """Returns a complete AI suggestion with latency metadata."""
    result = await generate_suggestion(db, _client, payload.query, payload.session_id)
    return result


@router.post("/suggest/stream")
async def suggest_stream(payload: SuggestionRequest, db: AsyncSession = Depends(get_db)):
    """Streams the AI suggestion token-by-token for low-latency UX."""
    async def token_generator():
        async for token in stream_suggestion(db, _client, payload.query, payload.session_id):
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain")
