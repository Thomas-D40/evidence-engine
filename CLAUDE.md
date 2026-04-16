# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Evidence Engine is a standalone FastAPI service that exposes a `POST /analyze` endpoint for fact-checking
individual arguments. It extracts analysis agents from `video-analyzer-workflow` into an independently
deployable REST service, hardened with enterprise-grade security.

## Development Commands

### Local Development (Docker)
```bash
docker compose up -d --build
curl http://localhost:8001/health
docker compose logs -f evidence-engine
docker compose down
```

### Local Development (No Docker)
```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
export ALLOWED_API_KEYS="your-key-here"
uvicorn app.api:app --reload --port 8001
```

### Testing
```bash
pytest tests/ -v --tb=short
pytest tests/unit/ -v
pytest tests/mocked/ -v
```

## Architecture

### Package Structure

```
app/
├── api.py              # FastAPI app, routes, middleware registration
├── config.py           # Settings via pydantic-settings
├── pipeline.py         # analyze_argument() orchestrator
├── core/               # Security layer (auth, rate limiting, sanitization)
├── agents/
│   ├── orchestration/  # topic_classifier, query_generator, adversarial_query
│   ├── enrichment/     # screening, fulltext, common
│   └── analysis/       # pros_cons, aggregate, consensus
├── services/research/  # External API clients (no LLM)
├── models/             # Pydantic request/response models
├── constants/          # All constants (security, analysis, LLM, scoring...)
├── logger/             # Structured logger
├── prompts/            # Reusable prompt components
└── utils/              # api_helpers, relevance_filter
```

### Key Design Decisions

- **Dual queries**: Every analysis generates both support and refutation queries in parallel
- **Framing bias prevention**: `retrieved_for` tag stripped from sources before LLM call
- **Deterministic consensus**: `compute_consensus()` is pure Python, zero LLM calls
- **Empty = reject all**: Unlike video-analyzer, `ALLOWED_API_KEYS` empty means reject all (misconfiguration)
- **Agents vs Services**: Agents use LLM; Services are pure API clients

### Hybrid Prompt Pattern

All LLM agents declare prompts as module-level constants at the top of the file:

```python
# ============================================================================
# PROMPTS
# ============================================================================
SYSTEM_PROMPT = """..."""
USER_PROMPT_TEMPLATE = """..."""

# ============================================================================
# LOGIC
# ============================================================================
def agent_function(...): ...
```

### Security Layer

- API key auth via `X-API-Key` header (HTTP 401 if missing, 403 if invalid)
- Brute-force protection: 10 failed attempts in 10 min → IP blocked 15 min
- Rate limiting: 60 req/min per IP, 100 req/hour per API key
- Payload limit: 10 KB max request body
- Prompt injection sanitization on `argument` and `context` fields
- Security headers middleware (CSP, HSTS, X-Frame-Options, etc.)

## Configuration

Required `.env`:
```env
OPENAI_API_KEY=sk-...
ALLOWED_API_KEYS=key1,key2   # Required — empty = reject all
```

Optional:
```env
OPENAI_MODEL=gpt-4o-mini
OPENAI_SMART_MODEL=gpt-4o
NEWSAPI_KEY=
GNEWS_API_KEY=
GOOGLE_FACTCHECK_API_KEY=
CLAIMBUSTER_API_KEY=
FULLTEXT_SCREENING_ENABLED=true
FULLTEXT_TOP_N=3
FULLTEXT_MIN_SCORE=0.6
ADVERSARIAL_QUERIES_ENABLED=true
```

## Code Style

- All comments in English
- Prompts declared as module-level constants (top of file)
- No magic strings — use constants from `app/constants/`
- Type hints throughout
- Async/await for all I/O
