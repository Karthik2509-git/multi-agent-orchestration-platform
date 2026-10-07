# Platform Security Model & Defense-in-Depth

The platform is designed with a defense-in-depth posture centered on **defensive execution boundaries, structural guardrails, and deterministic blast radius containment**.

> **Important Boundary Clarification**: The structural guardrails, payload ceilings, and tool filters implemented in this platform provide structural and operational isolation. They **do not** constitute a semantic prompt-injection classifier or an AI firewall. The platform does not claim formal security certification.

---

## 1. Network & Tool Security Controls

### 1.1 SSRF & DNS-Rebinding Protection (`SafeHTTPTool`)
External HTTP requests initiated by agents are guarded against Server-Side Request Forgery (SSRF) and DNS rebinding attacks:
- **IP Range Validation**: Before opening a socket, the tool resolves target hostnames and validates all resolved IP addresses using Python's `ipaddress` library. The request is aborted if any IP falls within:
  - Private networks (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`)
  - Loopback addresses (`127.0.0.0/8`, `::1`)
  - Link-local and cloud metadata addresses (`169.254.0.0/16`, `fe80::/10`)
  - Multicast and reserved ranges (`224.0.0.0/4`, `240.0.0.0/4`)
- **DNS Rebinding Defense**: Hostname resolution and IP validation occur immediately prior to connection, rejecting dynamic resolution to unrouted internal subnets.
- **Domain Allowlisting**: Only hosts explicitly listed in `ALLOWED_HTTP_DOMAINS` (`httpbin.org`, `api.github.com`) are allowed.
- **Redirect Disablement**: Automatic redirects (`follow_redirects=False`) are disabled to prevent open-redirect SSRF bypasses.
- **Response Size Cap**: Responses are read incrementally up to `TOOL_HTTP_MAX_SIZE_BYTES` (100 KB) to prevent memory exhaustion denial-of-service.

### 1.2 AST Arithmetic Parser (`CalculatorTool`)
Arithmetic evaluation uses Python's Abstract Syntax Tree (`ast`) module:
- Completely avoids `eval()` and `exec()`.
- Whitelists only safe mathematical operators (`+`, `-`, `*`, `/`, `//`, `%`, `**`).
- Strictly rejects function calls, identifiers, imports, attribute access, and variable assignments.
- Limits power operations (`x ** y`) to exponent ceilings (`y <= 10000`) to prevent CPU hang exploits.

### 1.3 Code Agent Non-Execution Guarantee
The `CodeAgent` is strictly an authoring, reviewing, and linting persona:
- Generated code is passed to the state scratchpad as formatted strings.
- **Zero-Execution Policy**: The platform contains no Python execution sandbox, subprocess runner, or bash execution node for the `CodeAgent`. Generated code is never executed on the host runtime.

---

## 2. Execution Guardrails & Blast Radius Containment

### 2.1 Tool Budget & Run-Scoped Tool Failure Isolation
- **Call Budget Ceiling**: A maximum of `TOOL_MAX_CALLS_PER_RUN` (10 calls) is allowed per orchestration run to prevent infinite agent tool loops.
- **Consecutive-Failure Tool Disablement**: Failures are tracked per tool; a successful execution resets the count. If an individual tool encounters 3 consecutive failures, it is disabled for the remainder of the current run to prevent repeated failing invocations.
- **Payload Limits**: Tool inputs are rejected if they exceed `TOOL_MAX_PAYLOAD_SIZE_BYTES` (64 KB); tool outputs exceeding `TOOL_MAX_OUTPUT_SIZE_BYTES` (1 MB) are truncated with warnings.
- **Failure Isolation**: Uncaught exceptions during tool execution are trapped, preventing crashed tools from aborting the overarching LangGraph state loop.

### 2.2 Replay & Mock Injection Safety
When using `POST /api/v1/orchestration/replay`:
- Mock tool results supplied by callers must conform to the target tool's Pydantic schema.
- Payloads are subject to the same payload size limitations as live executions.
- Unsanitized arbitrary code injection into graph state is rejected.

---

## 3. Data Protection, Memory Isolation & MCP Security

### 3.1 Model Context Protocol (MCP) Controls
- MCP tools are discoverable only from registered, allowlisted server definitions.
- Local MCP tool calls are mapped through typed schemas and executed in bounded subprocesses or isolated functions.

### 3.2 Semantic Memory Scope Isolation
- Memories stored in Chroma are partitioned by `scope`: `user`, `agent`, `session`, and `global`.
- Queries specify scope filters and verify zero cross-scope memory leakage across the tested isolation scenarios.

### 3.3 Credential Redaction & Telemetry Restrictions
- API keys (`OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `POSTGRES_PASSWORD`) are marked as secret fields in Pydantic Settings.
- OpenTelemetry spans record high-level tool names and execution duration, avoiding full-body logging of sensitive user prompts in production telemetry.
- Docker containers run as unprivileged `appuser` (UID 10001), preventing host privilege escalation.
