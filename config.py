"""Document CRUD routes with automatic chunk embedding on save."""

import uuid
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres import get_db
from app.models.models import Document
from app.services.vector.pgvector import upsert_chunks

logger = logging.getLogger(__name__)
router = APIRouter()

CHUNK_SIZE = 512  # characters per chunk


def chunk_text(text: str, size: int = CHUNK_SIZE) -> List[str]:
    """Simple fixed-size chunking with overlap."""
    overlap = size // 4
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks or [""]


class CreateDocumentRequest(BaseModel):
    session_id: uuid.UUID
    title: str
    content: str = ""


class UpdateDocumentRequest(BaseModel):
    content: str


class DocumentResponse(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    title: str
    content: str
    version: int


@router.post("/", response_model=DocumentResponse, status_code=201)
async def create_document(payload: CreateDocumentRequest, db: AsyncSession = Depends(get_db)):
    doc = Document(session_id=payload.session_id, title=payload.title, content=payload.content)
    db.add(doc)
    await db.flush()

    if payload.content:
        await upsert_chunks(db, doc.id, chunk_text(payload.content))

    return doc


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.patch("/{doc_id}", response_model=DocumentResponse)
async def update_document(
    doc_id: uuid.UUID,
    payload: UpdateDocumentRequest,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    doc.content = payload.content
    doc.version += 1

    # Re-embed on every save (debounce this client-side for hot paths)
    await upsert_chunks(db, doc.id, chunk_text(payload.content))
    return doc
