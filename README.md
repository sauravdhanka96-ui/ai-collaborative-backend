# AI-Collaborative Backend

AI-native backend for real-time collaborative applications — featuring WebSocket-based multi-user synchronization and a RAG (Retrieval-Augmented Generation) pipeline powered by PostgreSQL and pgvector.

---

## What It Does

- **Real-time collaboration** — multiple users connect simultaneously via WebSockets; shared session state stays consistent across all clients
- **AI assistance** — a RAG pipeline retrieves context from stored embeddings (pgvector) and delivers context-aware AI responses
- **Health & readiness probes** — `/health` and `/ready` endpoints report system status and active WebSocket session count

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Framework | FastAPI |
| Real-time | WebSockets |
| Database | PostgreSQL + pgvector |
| ORM | SQLAlchemy (async) |
| AI Pipeline | RAG (Retrieval-Augmented Generation) |
| Containerization | Docker |

---

## Project Structure
