"""
NexusCollaborate AI - Main Application Entry Point
High-performance backend for real-time collaborative design with RAG-based AI pipeline.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.api.routes import sessions, documents, ai_suggestions, health
from app.core.config import settings
from app.core.logging import setup_logging
from app.db.postgres import init_db, close_db
from app.services.vector.pgvector import init_vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle."""
    setup_logging()
    await init_db()
    await init_vector_store()
    yield
    await close_db()


app = FastAPI(
    title="NexusCollaborate AI",
    description="Real-time collaborative design backend with RAG-based AI suggestions",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(sessions.router, prefix="/api/v1/sessions", tags=["sessions"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(ai_suggestions.router, prefix="/api/v1/ai", tags=["ai"])
