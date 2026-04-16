# Evidence Engine

A standalone FastAPI service for fact-checking individual arguments via a multi-source adversarial research pipeline. Extracted from [video-analyzer-workflow](https://github.com/Thomas-D40/video-analyzer-workflow) as an independently deployable REST API.

## Quick Start

```bash
# Copy and fill in your keys
cp .env.example .env

# Docker
docker compose up -d
curl http://localhost:8001/health

# Or local
pip install -r requirements.txt
uvicorn app.api:app --reload --port 8001
```

**Analyze an argument:**
```bash
curl -X POST http://localhost:8001/analyze \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"argument": "Coffee reduces liver cancer risk", "mode": "medium"}'
```

---

## Architecture

```
evidence-engine/
├── app/
│   ├── api.py                  # FastAPI app, routes, middleware
│   ├── config.py               # Settings via pydantic-settings
│   ├── pipeline.py             # analyze_argument() — 11-step orchestrator
│   ├── core/
│   │   ├── auth.py             # API key validation + brute-force IP blocking
│   │   ├── rate_limiter.py     # slowapi — per-IP and per-key limits
│   │   └── security.py        # Prompt injection sanitization + security headers
│   ├── agents/
│   │   ├── orchestration/     # Topic classification, query generation (support + adversarial)
│   │   ├── enrichment/        # Relevance screening, full-text fetching
│   │   └── analysis/          # Pros/cons extraction, reliability scoring, consensus
│   ├── services/research/     # External API clients (no LLM)
│   ├── models/                # AnalyzeRequest, AnalysisResult (Pydantic)
│   ├── constants/             # All constants organized by domain
│   ├── prompts/               # Reusable LLM prompt components
│   └── utils/                 # api_helpers, relevance_filter
└── tests/
    ├── unit/                  # Pure Python — no mocking needed
    └── mocked/                # API endpoints + pipeline (all external calls mocked)
```

**Key distinction:** `agents/` = LLM-powered logic. `services/research/` = pure API clients, no LLM.

---

## Pipeline

`POST /analyze` runs 11 steps for each argument:

```
1. Topic classification      → select research services (e.g. pubmed, arxiv, oecd)
       ↓
2. Query generation          → support queries  +  refutation queries  (parallel, LLM)
       ↓
3. Dual research             → both query sets executed against all services (concurrent)
       ↓
4. Tag sources               → each source stamped retrieved_for = "support" | "refutation"
       ↓
5. Relevance filtering       → keyword-based pre-filter (no LLM)
       ↓
6. Screening                 → LLM batch-scores abstracts, selects top-N for full-text
       ↓
7. Full-text fetch           → async concurrent HTTP fetch (medium / hard modes only)
       ↓
8. Strip retrieved_for tag   → prevents framing bias before LLM sees sources
       ↓
9. Pros / cons extraction    → LLM identifies supporting and contradicting evidence
       ↓
10. Reliability aggregation  → LLM scores 0.0–1.0 based on source quality and consensus
       ↓
11. Consensus computation    → pure Python, deterministic, zero LLM
```

### Analysis Modes

| Mode | Full-text fetch | Sources screened |
|------|----------------|-----------------|
| `simple` | No | Abstracts only |
| `medium` | Top 3 (score ≥ 0.6) | Yes |
| `hard` | Top 6 (score ≥ 0.5) | Yes |

### Adversarial Design

Two independent query sets are generated for every argument:
- **Support queries** — find evidence that corroborates the claim
- **Refutation queries** — find evidence that contradicts or challenges it

The `retrieved_for` tag is attached to each source for post-processing bookkeeping, then **stripped before passing sources to the LLM** — preventing the model from being anchored by retrieval intent when extracting pros/cons.

---

## Research Services

All services are pure async HTTP clients — no LLM involved.

| Service | Domain | Notes |
|---------|--------|-------|
| `pubmed` | Medical / biomedical | NCBI E-utilities, 39M+ citations |
| `europepmc` | Biomedical | Europe PMC open access |
| `arxiv` | Physics, CS, math | Pre-prints via `arxiv` library |
| `semantic_scholar` | All academic | 200M+ papers, AI-powered |
| `crossref` | All academic | DOI metadata |
| `core` | Open access | 350M+ sources |
| `doaj` | Open access journals | Peer-reviewed only |
| `oecd` | Economics / statistics | SDMX API |
| `world_bank` | Development data | `wbgapi` library |
| `newsapi` | Current events | Requires `NEWSAPI_KEY` |
| `gnews` | Current events | Requires `GNEWS_API_KEY` |
| `google_factcheck` | Fact-checking | Requires `GOOGLE_FACTCHECK_API_KEY` |
| `claimbuster` | Claim scoring | Requires `CLAIMBUSTER_API_KEY` |

Services are selected automatically based on topic classification. Missing API keys cause the service to be silently skipped — the pipeline continues with available sources.

---

## API Reference

### `POST /analyze`

Requires `X-API-Key` header.

**Request**
```json
{
  "argument": "Coffee reduces liver cancer risk",
  "mode": "medium",
  "context": "Optional additional context",
  "language": "en"
}
```

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| `argument` | string | Yes | 10–2000 chars |
| `mode` | `simple` \| `medium` \| `hard` | No | Default: `medium` |
| `context` | string | No | Max 1000 chars |
| `language` | string | No | Default: `en` |

**Response**
```json
{
  "argument": "Coffee reduces liver cancer risk",
  "argument_en": "Coffee reduces liver cancer risk",
  "reliability_score": 0.74,
  "consensus_ratio": 0.667,
  "consensus_label": "Moderate consensus",
  "pros": [
    { "claim": "Meta-analysis of 9 studies shows 40% risk reduction.", "source": "https://pubmed.ncbi.nlm.nih.gov/..." }
  ],
  "cons": [
    { "claim": "Cohort study found no statistically significant association.", "source": "https://pubmed.ncbi.nlm.nih.gov/..." }
  ],
  "sources_count": 8,
  "support_sources": 5,
  "refutation_sources": 3
}
```

**Consensus labels**

| `consensus_ratio` | `consensus_label` |
|-------------------|-------------------|
| ≥ 0.75 | Strong consensus |
| ≥ 0.55 | Moderate consensus |
| ≥ 0.35 | Contested |
| < 0.35 | Minority position |
| No evidence | Insufficient data |

### `GET /health`

No authentication required. Returns `{"status": "ok"}`.

---

## Security

| Concern | Implementation |
|---------|---------------|
| Authentication | `X-API-Key` header — HTTP 401 if missing, 403 if invalid |
| Empty `ALLOWED_API_KEYS` | Rejects all requests (misconfiguration, unlike open-access default) |
| Brute-force protection | IP blocked for 15 min after 10 failed attempts in 10 min |
| Rate limiting | 60 req/min per IP · 100 req/hour per API key (HTTP 429) |
| Payload limit | 10 KB max body (HTTP 413) |
| Prompt injection | Patterns stripped from `argument` and `context` before any LLM call |
| Security headers | `X-Frame-Options`, `CSP`, `HSTS`, `X-Content-Type-Options`, etc. |

---

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | Fails at startup if unset |
| `ALLOWED_API_KEYS` | Yes | — | Comma-separated keys; empty = reject all |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Model for screening, pros/cons, aggregation |
| `OPENAI_SMART_MODEL` | No | `gpt-4o` | Reserved for future complex reasoning tasks |
| `FULLTEXT_SCREENING_ENABLED` | No | `true` | Toggle LLM screening step |
| `FULLTEXT_TOP_N` | No | `3` | Sources fetched for full text (medium mode) |
| `FULLTEXT_MIN_SCORE` | No | `0.6` | Minimum relevance score to fetch full text |
| `ADVERSARIAL_QUERIES_ENABLED` | No | `true` | Toggle dual-query pipeline |
| `NEWSAPI_KEY` | No | — | Required for news/politics topics |
| `GNEWS_API_KEY` | No | — | Required for news/politics topics |
| `GOOGLE_FACTCHECK_API_KEY` | No | — | Required for fact-check topics |
| `CLAIMBUSTER_API_KEY` | No | — | Required for claim scoring |

---

## Testing

```bash
# All tests (46) — no live API calls
pytest tests/ -v

# Unit tests only (pure Python, instant)
pytest tests/unit/

# Mocked integration tests
pytest tests/mocked/
```
