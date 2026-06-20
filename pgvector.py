"""
RAG Pipeline — retrieves relevant context chunks and generates AI suggestions.
Targets sub-100ms end-to-end latency via async streaming and context caching.
"""

import logging
import time
import uuid
from typing import AsyncIterator

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.services.vector.pgvector import similarity_search

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert design collaborator embedded in a real-time design platform.
Using the provided document context, give concise, actionable suggestions.
Be specific to the content shown. Never fabricate details not present in the context."""


async def generate_suggestion(
    db: AsyncSession,
    client: AsyncOpenAI,
    query: str,
    session_id: uuid.UUID,
) -> dict:
    """
    Full RAG pipeline:
      1. Retrieve top-K similar chunks from pgvector
      2. Build a context-aware prompt
      3. Stream a completion from the inference model
    Returns the suggestion text and latency metadata.
    """
    t0 = time.monotonic()

    # --- Retrieval ---
    chunks = await similarity_search(db, query, session_id)
    retrieval_ms = (time.monotonic() - t0) * 1000

    if not chunks:
        return {
            "suggestion": "No relevant context found. Please add more content to the document.",
            "context_used": 0,
            "retrieval_ms": retrieval_ms,
            "inference_ms": 0,
        }

    context_block = "\n\n---\n\n".join(
        f"[{c['document_title']}]\n{c['content']}" for c in chunks
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context from the collaborative workspace:\n\n{context_block}"
                f"\n\n---\n\nUser query: {query}"
            ),
        },
    ]

    # --- Inference ---
    t1 = time.monotonic()
    response = await client.chat.completions.create(
        model=settings.INFERENCE_MODEL,
        messages=messages,
        max_tokens=512,
        temperature=0.4,
    )
    inference_ms = (time.monotonic() - t1) * 1000
    total_ms = (time.monotonic() - t0) * 1000

    suggestion_text = response.choices[0].message.content

    logger.info(
        f"RAG pipeline: retrieval={retrieval_ms:.1f}ms "
        f"inference={inference_ms:.1f}ms total={total_ms:.1f}ms "
        f"chunks_used={len(chunks)}"
    )

    return {
        "suggestion": suggestion_text,
        "context_used": len(chunks),
        "retrieval_ms": round(retrieval_ms, 2),
        "inference_ms": round(inference_ms, 2),
        "total_ms": round(total_ms, 2),
    }


async def stream_suggestion(
    db: AsyncSession,
    client: AsyncOpenAI,
    query: str,
    session_id: uuid.UUID,
) -> AsyncIterator[str]:
    """Streaming variant — yields tokens as they arrive for low-perceived-latency UX."""
    chunks = await similarity_search(db, query, session_id)

    if not chunks:
        yield "No relevant context found."
        return

    context_block = "\n\n---\n\n".join(
        f"[{c['document_title']}]\n{c['content']}" for c in chunks
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context:\n\n{context_block}\n\n---\n\nQuery: {query}"
            ),
        },
    ]

    async with client.chat.completions.stream(
        model=settings.INFERENCE_MODEL,
        messages=messages,
        max_tokens=512,
        temperature=0.4,
    ) as stream:
        async for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                yield delta
