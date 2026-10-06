# Local Development Setup

This guide walks through configuring a local development environment for the Multi-Agent AI Orchestration Platform.

---

## 1. System Requirements

- **Python**: Version 3.11, 3.12, or 3.13 (Python 3.12 is recommended and used in the container image).
- **Git**: Modern version for version control.
- **Operating System**: Linux, macOS, or Windows (tested on both POSIX and Windows PowerShell environments).
- **Docker & Docker Compose**: Optional for local execution; required for containerized deployment.

---

## 2. Step-by-Step Installation

### Step 2.1 — Clone the Repository
```bash
git clone https://github.com/Karthik2509-git/multi-agent-orchestration-platform.git
cd multi-agent-orchestration-platform
```

### Step 2.2 — Create and Activate a Virtual Environment
```bash
# On Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 2.3 — Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements-dev.txt
```

> **Note**: `requirements-dev.txt` installs all runtime production dependencies plus testing (`pytest`, `pytest-asyncio`, `anyio`) and code formatting tools (`ruff`).

---

## 3. Environment Configuration

Copy the template configuration file:
```bash
cp .env.example .env
```

### Zero-Cost Offline Development Mode (Recommended)
To run the platform completely offline without external API keys or network requests:
```ini
LLM_PROVIDER=mock
MOCK_LLM_RESPONSE="Deterministic mock response for local testing"
DATABASE_URL=
CHROMA_PERSIST_DIR=./data/chroma
```
When `LLM_PROVIDER=mock`, the system routes all agent completions through `MockLLMProvider`, allowing complete testing of the supervisor, specialists, tool registry, memory, and RAG pipelines without billing or external network dependencies.

### Real Provider Configuration
If connecting to an external LLM provider, set `LLM_PROVIDER` and provide the corresponding API key:

- **OpenRouter** (access to free and commercial models):
  ```ini
  LLM_PROVIDER=openrouter
  OPENROUTER_API_KEY=your_openrouter_api_key
  OPENROUTER_MODEL=openrouter/free
  ```
- **Google Gemini**:
  ```ini
  LLM_PROVIDER=gemini
  GEMINI_API_KEY=your_gemini_api_key
  GEMINI_MODEL=gemini-2.5-flash
  ```
- **Groq** (ultra-low-latency LPU inference):
  ```ini
  LLM_PROVIDER=groq
  GROQ_API_KEY=your_groq_api_key
  GROQ_MODEL=llama-3.3-70b-versatile
  ```

---

## 4. Storage & Persistence Services

### PostgreSQL (Working Memory Checkpointer)
- In production, the system connects to PostgreSQL via `DATABASE_URL` (e.g. `postgresql+asyncpg://postgres:postgres@localhost:5432/multi_agent_db`).
- In local development, if `DATABASE_URL` is omitted or empty, the checkpointer falls back automatically to an in-memory `MemorySaver`. This allows instant zero-dependency local runs.

### Chroma DB (Vector & Semantic Memory)
- Chroma vectors are stored on disk at `CHROMA_PERSIST_DIR` (default `./data/chroma`).
- The directory is automatically created on startup if it does not exist.

### Redis Status
- Redis is configured in `docker-compose.yml` and settings for distributed caching/queuing.
- **Runtime Dependency Status**: Redis is currently **non-runtime-critical**; the backend operates fully without an active Redis instance.

---

## 5. Running the Application

Launch the ASGI application using Uvicorn:
```bash
python -m uvicorn src.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Once started:
- **Interactive OpenAPI Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Liveness Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- **Readiness Check**: [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)
