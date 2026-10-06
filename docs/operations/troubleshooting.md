# Operational Troubleshooting & Diagnostics

This guide provides remediation steps for common operational issues encountered during local development and deployment.

---

## 1. Readiness Check Returns 503 (`database: disconnected`)

### Symptoms
- `GET /health` returns `200 OK`, but `GET /api/v1/health` returns `503 Service Unavailable`.
- Log shows `Failed to initialize PostgreSQL checkpointer pool`.

### Root Cause
- PostgreSQL container has not finished initializing its authentication or database schema.
- Or `DATABASE_URL` credentials / host configuration are incorrect.

### Diagnostic & Remediation
1. Verify PostgreSQL container status:
   ```bash
   docker compose ps postgres
   ```
2. Inspect PostgreSQL startup logs:
   ```bash
   docker compose logs postgres
   ```
3. Test connectivity directly using `pg_isready`:
   ```bash
   docker compose exec postgres pg_isready -U postgres -d multi_agent_db
   ```
4. **Self-Healing Verification**: Wait 5–10 seconds and re-query `GET /api/v1/health`. The platform will automatically re-attempt pool initialization without requiring an application restart.

---

## 2. Chroma Vector Store Permission Denied in Container

### Symptoms
- Container startup log shows `PermissionError: [Errno 13] Permission denied: '/app/data/chroma'`.

### Root Cause
- Volume directory mounted to `/app/data/chroma` was created on the host by a root process, while the container runs as unprivileged `appuser` (UID 10001).

### Diagnostic & Remediation
1. The `Dockerfile` includes `RUN mkdir -p /app/data && chown -R appuser:appuser /app/data`.
2. When mounting named volumes with Docker Compose, Docker initializes permissions to match the image mount point.
3. If using host bind mounts on Linux hosts, adjust permissions:
   ```bash
   sudo chown -R 10001:10001 ./data/chroma
   ```

---

## 3. Tool Execution Budget Exhausted

### Symptoms
- Orchestration response contains incomplete answers with warning:
  `Tool execution budget exhausted (10 calls). Please synthesize answer with existing data.`

### Root Cause
- The multi-agent workflow encountered a task requiring more tool interactions than permitted by `TOOL_MAX_CALLS_PER_RUN` (default 10).

### Diagnostic & Remediation
1. Inspect the agent execution trajectory to verify whether agents were trapped in an inefficient querying loop.
2. If the task legitimately requires additional tool calls, adjust the ceiling in `.env`:
   ```ini
   TOOL_MAX_CALLS_PER_RUN=20
   ```

---

## 4. HTTP Tool Rejection (`Domain or IP forbidden`)

### Symptoms
- `http_get` returns `ToolResult(success=False, error="URL domain 'example.com' is not in allowed domains")` or `Resolved IP is in a restricted private/loopback range`.

### Root Cause
- Platform SSRF protections enforce:
  1. A strict domain allowlist (`ALLOWED_HTTP_DOMAINS`).
  2. Blocking of private IPs (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), loopback (`127.0.0.0/8`), and cloud metadata (`169.254.169.254`).

### Diagnostic & Remediation
1. Verify target domain:
   Only explicitly permitted public domains are reachable. To add a legitimate external API, update `.env`:
   ```ini
   ALLOWED_HTTP_DOMAINS=["httpbin.org", "api.github.com", "api.weather.gov"]
   ```
2. **Never** attempt to whitelist internal hostnames (`localhost`, `10.0.0.1`, `postgres`) as this violates platform security invariants.

---

## 5. Thread Paused at `interrupted` Status

### Symptoms
- `POST /api/v1/orchestration/run` returns `status: "interrupted"`.

### Root Cause
- An agent proposed an action flagged as sensitive or supervisor routing confidence fell below $0.65$, triggering a Human-in-the-Loop approval gate.

### Diagnostic & Remediation
1. Inspect pending action details:
   ```bash
   curl http://127.0.0.1:8000/api/v1/hitl/pending/{thread_id}
   ```
2. Approve or reject to resume execution:
   ```bash
   curl -X POST http://127.0.0.1:8000/api/v1/hitl/resume/{thread_id} \
     -H "Content-Type: application/json" \
     -d '{"approved": true, "user_feedback": "Approved by administrator"}'
   ```
