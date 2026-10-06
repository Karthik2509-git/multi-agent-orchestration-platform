# Deployment & Infrastructure Operations

This document describes the container architecture, infrastructure composition, and deployment verification procedures.

---

## 1. Container Architecture

The production environment is packaged as a lean, secure Docker container:

- **Base Image**: `python:3.12-slim` (minimal surface area, Debian Bookworm).
- **Security Posture**: Executes under an unprivileged non-root user (`appuser` with UID `10001`).
- **File System Permissions**: Pre-provisions `/app/data` with ownership assigned to `appuser:appuser` before dropping root privileges, ensuring persistent vector storage writes succeed.
- **Port**: Listens on TCP port `8000`.
- **Healthcheck**: Evaluates `curl -f http://localhost:8000/health || exit 1`.

---

## 2. Infrastructure Services (`docker-compose.yml`)

The platform defines three coordinated services:

```yaml
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-postgres}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-postgres}
      POSTGRES_DB: ${POSTGRES_DB:-multi_agent_db}
    ports:
      - "127.0.0.1:5432:5432"   # Bound to loopback interface only
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-postgres} -d ${POSTGRES_DB:-multi_agent_db}"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "127.0.0.1:6379:6379"   # Bound to loopback interface only
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  backend:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - chroma_data:/app/data/chroma
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
```

### Key Infrastructure Hardening
1. **Loopback Port Binding**: Postgres (`5432`) and Redis (`6379`) are explicitly bound to `127.0.0.1`, preventing exposure to external networks.
2. **Persistent Named Volumes**:
   - `postgres_data`: Preserves LangGraph checkpoints across container restarts.
   - `chroma_data`: Preserves indexed RAG documents and long-term semantic memory vectors.
   - `redis_data`: Preserves operational cache state.
3. **Conditioned Dependency Startup**: `backend` does not launch until PostgreSQL and Redis pass their container healthchecks.

---

## 3. Deployment Verification Script (`scripts/verify_deployment.py`)

The repository includes a standalone deployment smoke script (`scripts/verify_deployment.py`) to validate a live deployment.

### Usage
```bash
# Validate against local instance
python scripts/verify_deployment.py --base-url http://127.0.0.1:8000

# Validate with custom timeout
python scripts/verify_deployment.py --base-url http://127.0.0.1:8000 --timeout 60
```

### Verification Pipeline
The script executes 5 strict sequential validation gates:

1. **Step A — Liveness**: Calls `GET /health` and validates `{"status": "ok"}` with status code `200`.
2. **Step B — Readiness**: Calls `GET /api/v1/health` and verifies that database and vector store subsystems report ready.
3. **Step C — RAG Document Ingestion & Query**:
   - Ingests a synthetic document into `/api/v1/knowledge/documents`.
   - Executes `/api/v1/knowledge/query` and confirms retrieved citations.
4. **Step D — Semantic Memory Storage & Search**:
   - Stores a memory record via `/api/v1/memory`.
   - Searches via `/api/v1/memory/search` and verifies ranking retrieval.
5. **Step E — Multi-Agent Orchestration**:
   - Submits a task to `/api/v1/orchestration/run`.
   - Requires status code `200`, `status ∈ {"completed", "interrupted"}`, valid matching `thread_id`, and a populated answer string. Exits with code `1` on any failure.

---

## 4. Current Verification Status & Honest Boundaries

To maintain engineering integrity, the platform distinguishes between what has been tested versus what remains bounded by environment constraints:

| Verification Area | Method | Status |
|---|---|---|
| **FastAPI REST API Surface** | Automated Pytest / TestClient | ✅ Fully verified (320 tests pass) |
| **LangGraph Multi-Agent Flow** | In-process & Mock Provider Tests | ✅ Fully verified |
| **PostgreSQL Pool & Checkpointer** | Async unit & self-healing tests | ✅ Fully verified |
| **Chroma Vector Ingestion & Search** | Local filesystem integration tests | ✅ Fully verified |
| **Docker Configuration Syntax** | `docker compose config` linting | ✅ Fully verified |
| **Live Container Startup on Host** | Docker Desktop Engine | ⚠️ **Deferred**: Docker Desktop daemon was offline on the host machine during Phase 8. |
| **Live Container Restart Persistence** | Live Docker named volume restart | ⚠️ **Deferred**: Requires running Docker daemon to validate physical volume remounts. |
