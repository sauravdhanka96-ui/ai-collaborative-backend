"""
pgvector-backed vector store service.
Handles embedding generation, upsert, and similarity search.
"""

import logging
import uuid
from typing import Optional
from openai import AsyncOpenAI
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.models import DocumentChunk

logger = logging.getLogger(__name__)
_openai_client: Optional[AsyncOpenAI] = None


async def init_vector_store() -> None:
    global _openai_client
    _openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    logger.info("Vector store (pgvector) initialised")


async def embed_text(text_input: str) -> list[float]:
    """Generate an embedding vector for the given text."""
    response = await _openai_client.embeddings.create(
        input=text_input,
        model=settings.EMBEDDING_MODEL,
    )
    return response.data[0].embedding


async def upsert_chunks(
    db: AsyncSession,
    document_id: uuid.UUID,
    chunks: list[str],
) -> list[DocumentChunk]:
    """Embed and store document chunks, replacing any existing ones."""
    # Delete old chunks
    await db.execute(
        text("DELETE FROM document_chunks WHERE document_id = :doc_id"),
        {"doc_id": str(document_id)},
    )

    stored = []
    for idx, chunk_text in enumerate(chunks):
        embedding = await embed_text(chunk_text)
        chunk = DocumentChunk(
            document_id=document_id,
            chunk_index=idx,
            content=chunk_text,
            embedding=embedding,
        )
        db.add(chunk)
        stored.append(chunk)

    await db.flush()
    logger.info(f"Upserted {len(stored)} chunks for document {document_id}")
    return stored


async def similarity_search(
    db: AsyncSession,
    query: str,
    session_id: uuid.UUID,
    top_k: int = None,
) -> list[dict]:
    """
    Find the most relevant document chunks for a query within a session.
    Uses pgvector cosine distance operator <=> for fast ANN search.
    """
    top_k = top_k or settings.TOP_K_RESULTS
    query_embedding = await embed_text(query)
    embedding_str = f"[{','.join(map(str, query_embedding))}]"

    result = await db.execute(
        text("""
            SELECT
                dc.id,
                dc.content,
                dc.document_id,
                d.title AS document_title,
                1 - (dc.embedding <=> :embedding::vector) AS similarity
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            JOIN collab_sessions cs ON cs.id = d.session_id
            WHERE cs.id = :session_id
              AND 1 - (dc.embedding <=> :embedding::vector) >= :threshold
            ORDER BY dc.embedding <=> :embedding::vector
            LIMIT :top_k
        """),
        {
            "embedding": embedding_str,
            "session_id": str(session_id),
            "threshold": settings.SIMILARITY_THRESHOLD,
            "top_k": top_k,
        },
    )

    rows = result.mappings().all()
    return [dict(row) for row in rows]
