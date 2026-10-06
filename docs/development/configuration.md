# Configuration Reference

The platform uses **Pydantic Settings** (`src/app/core/config.py`) to manage typed environment variables with runtime validation, defaults, and security constraints.

---

## 1. Application & Environment Settings

| Variable | Type | Default | Description |
|---|---|---|---|
| `ENVIRONMENT` | string | `development` | Deployment environment: `development`, `testing`, `production`. |
| `DEBUG` | boolean | `false` | Enable verbose debugging and trace logging. |
| `API_V1_STR` | string | `/api/v1` | Root prefix for version 1 REST endpoints. |
| `PROJECT_NAME` | string | `Multi-Agent Orchestration Platform` | Application display title in OpenAPI documentation. |
| `ALLOWED_HOSTS` | list | `["*"]` | Allowed HTTP Host headers for CORS middleware. |

---

## 2. LLM Provider Settings

| Variable | Type | Default | Description |
|---|---|---|---|
| `LLM_PROVIDER` | string | `openrouter` | Active LLM provider: `openrouter`, `gemini`, `groq`, `mock`, `openai`. |
| `OPENROUTER_API_KEY` | secret | `""` | API key for OpenRouter gateway. |
| `OPENROUTER_MODEL` | string | `openrouter/free` | Target model ID on OpenRouter. |
| `GEMINI_API_KEY` | secret | `""` | API key for Google Gemini GenAI SDK. |
| `GEMINI_MODEL` | string | `gemini-2.5-flash` | Gemini model variant. |
| `GROQ_API_KEY` | secret | `""` | API key for Groq Cloud inference. |
| `GROQ_MODEL` | string | `llama-3.3-70b-versatile` | Groq LPU model variant. |
| `OPENAI_API_KEY` | secret | `""` | Optional direct OpenAI API key. |
| `OPENAI_MODEL` | string | `gpt-4o-mini` | Direct OpenAI model name. |
| `MOCK_LLM_RESPONSE` | string | `"Mock response"` | Default completion string when using `mock` provider. |

---

## 3. Database & Working Memory Checkpointer

| Variable | Type | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | string | `""` | PostgreSQL async connection string (`postgresql+asyncpg://...`). If empty, checkpointer uses in-memory `MemorySaver`. |
| `POSTGRES_USER` | string | `postgres` | PostgreSQL username for database service. |
| `POSTGRES_PASSWORD` | secret | `postgres` | PostgreSQL password. **Production Validator**: `config.py` enforces that in `ENVIRONMENT=production`, weak passwords like `postgres` or `admin` raise a startup validation error. |
| `POSTGRES_DB` | string | `multi_agent_db` | Primary database name. |
| `POSTGRES_PORT` | integer | `5432` | PostgreSQL listener port. |

---

## 4. Vector Store & RAG Configuration

| Variable | Type | Default | Description |
|---|---|---|---|
| `CHROMA_PERSIST_DIR` | path | `./data/chroma` | Persistent on-disk directory for Chroma DB collections. |
| `RAG_EMBEDDING_PROVIDER` | string | `onnx` | Embedding model: `onnx` (local `all-MiniLM-L6-v2`) or `mock`. |
| `RAG_CHUNK_SIZE` | integer | `500` | Target character size per text chunk. |
| `RAG_CHUNK_OVERLAP` | integer | `50` | Overlap characters between adjacent chunks. |
| `RAG_HYBRID_ALPHA` | float | `0.6` | Weight assigned to dense vector search in Reciprocal Rank Fusion ($0.0 \dots 1.0$). |
| `RAG_RRF_K` | integer | `60` | Rank constant $k$ used in the RRF denominator. |

---

## 5. Semantic Memory Configuration

| Variable | Type | Default | Description |
|---|---|---|---|
| `MEMORY_COLLECTION_NAME` | string | `agent_memory` | Name of the Chroma collection reserved for semantic memories. |
| `MEMORY_SIMILARITY_THRESHOLD` | float | `0.85` | Cosine similarity threshold above which duplicate memories are merged. |
| `MEMORY_DEFAULT_TTL_DAYS` | integer | `30` | Default time-to-live before non-permanent memories expire. |
| `MEMORY_RETRIEVAL_TOP_K` | integer | `5` | Maximum number of memories retrieved for supervisor injection. |

---

## 6. Tool Guardrails & Security Ceilings

| Variable | Type | Default | Description |
|---|---|---|---|
| `TOOL_MAX_CALLS_PER_RUN` | integer | `10` | Maximum allowable tool invocations per single orchestration execution. |
| `TOOL_MAX_FAILURES_BEFORE_DISABLE` | integer | `3` | Consecutive failure threshold before an individual tool is disabled. |
| `TOOL_MAX_PAYLOAD_SIZE_BYTES` | integer | `65536` | Maximum allowed input payload (64 KB). |
| `TOOL_MAX_OUTPUT_SIZE_BYTES` | integer | `1048576` | Maximum allowed output payload before truncation (1 MB). |
| `ALLOWED_HTTP_DOMAINS` | list | `["httpbin.org", "api.github.com"]` | Strict allowlist of domains permitted for external HTTP GET requests. |
| `TOOL_HTTP_TIMEOUT_SECONDS` | integer | `10` | Timeout ceiling for external HTTP tool requests. |
| `TOOL_HTTP_MAX_SIZE_BYTES` | integer | `102400` | Maximum response byte limit for HTTP downloads (100 KB). |

---

## 7. Observability & Telemetry

| Variable | Type | Default | Description |
|---|---|---|---|
| `ENABLE_TELEMETRY` | boolean | `true` | Enables OpenTelemetry span generation and metric collection. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | string | `""` | Optional remote OpenTelemetry collector endpoint. If empty, uses local tracer. |
| `METRICS_ENABLED` | boolean | `true` | Enables in-memory system performance counters and histograms. |
| `LOG_LEVEL` | string | `INFO` | Application log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
