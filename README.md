# Evidence Engine

A standalone FastAPI service for fact-checking individual arguments via a multi-source adversarial research pipeline. Extracted from [video-analyzer-workflow](https://github.com/Thomas-D40/video-analyzer-workflow) as an independently deployable REST API.

## Design Philosophy — Honesty First

Evidence Engine is built around one core principle: **never imply more certainty than the evidence supports**.

This shapes every layer of the system:

- **Field names signal intent** — `estimated_reliability` (not `reliability_score`) and `evidence_balance_label` (not `consensus_label`) make it explicit that outputs are AI estimates over retrieved sources, not scientific measurements or peer-reviewed verdicts.
- **`reliability_basis` is always visible** — every response includes a plain-text string stating how many sources were used, how many were full text vs. abstract, and the disclaimer *"Not a verified fact-check."*
- **Source transparency** — each evidence item carries `source_type` (`academic`, `news`, `fact_check`, `statistical`) and `content_depth` (`full_text`, `abstract`, `snippet`) so consumers know the quality of the underlying material.
- **Source breakdown** — `sources` is a typed object breaking down counts by category and content depth, replacing an opaque integer count.
- **Adversarial quality gate** — refutation queries that are too similar to support queries (surface negations) are discarded before research. A poor adversarial query is worse than none — it pollutes the refutation pool with sources that support the claim rather than challenge it.
- **Framing bias prevention** — retrieval intent (`retrieved_for`) is stripped from sources before any LLM call, so the model cannot be anchored by whether a source was fetched to support or refute the argument.
- **No symmetric capping** — sources are never artificially balanced between pro and con. If evidence clearly leans one way, that is what the response reflects.

---

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
│   ├── pipeline.py             # analyze_argument() — 13-step orchestrator
│   ├── core/
│   │   ├── auth.py             # API key validation + brute-force IP blocking
│   │   ├── rate_limiter.py     # slowapi — per-IP and per-key limits
│   │   └── security.py        # Prompt injection sanitization + security headers
│   ├── agents/
│   │   ├── orchestration/     # Topic classification, query generation (support + adversarial)
│   │   ├── enrichment/        # Relevance screening, full-text fetching
│   │   └── analysis/          # Pros/cons extraction, reliability scoring, evidence balance
│   ├── services/research/     # External API clients (no LLM)
│   ├── models/                # AnalyzeRequest, AnalysisResult, SourceBreakdown (Pydantic)
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

`POST /analyze` runs 13 steps for each argument:

```
1. Topic classification        → select research services (e.g. pubmed, arxiv, oecd)
       ↓
2. Query generation            → support queries  +  adversarial queries  (parallel, LLM)
       ↓
3. Adversarial quality gate    → discard refutation queries too similar to support queries
       ↓
4. Dual research               → both query sets executed against all services (concurrent)
       ↓
5. Tag sources                 → each source stamped retrieved_for = "support" | "refutation"
       ↓
6. Relevance filtering         → keyword-based pre-filter (no LLM)
       ↓
7. Screening                   → LLM batch-scores abstracts, selects top-N for full-text
       ↓
8. Full-text fetch             → async concurrent HTTP fetch (medium / hard modes only)
       ↓
9. Strip retrieved_for tag     → prevents framing bias before LLM sees sources
       ↓
10. Pros / cons extraction     → LLM identifies supporting and contradicting evidence
       ↓
11. Enrich evidence items      → source_type and content_depth added per EvidenceItem
       ↓
12. Reliability aggregation    → LLM estimates 0.0–1.0 based on source quality and balance
       ↓
13. Evidence balance           → pure Python, deterministic, zero LLM
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
- **Adversarial queries** — find evidence that genuinely challenges it

Adversarial queries are generated via structured two-step reasoning across five angles (confounders, subgroup exceptions, methodological weaknesses, opposing causal mechanisms, null findings), then filtered by a quality gate: any query with more than 60% keyword overlap with the corresponding support query is discarded as a surface negation.

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
  "estimated_reliability": 0.74,
  "reliability_basis": "AI estimate based on 8 sources (2 full text, 6 abstract only). Not a verified fact-check.",
  "evidence_balance_ratio": 0.625,
  "evidence_balance_label": "More supporting than contradicting evidence found",
  "pros": [
    {
      "claim": "Meta-analysis of 9 studies shows 40% risk reduction.",
      "source": "https://pubmed.ncbi.nlm.nih.gov/...",
      "source_type": "academic",
      "content_depth": "abstract"
    }
  ],
  "cons": [
    {
      "claim": "Cohort study found no statistically significant association.",
      "source": "https://pubmed.ncbi.nlm.nih.gov/...",
      "source_type": "academic",
      "content_depth": "full_text"
    }
  ],
  "sources": {
    "total": 8,
    "academic": 6,
    "statistical": 0,
    "news": 2,
    "fact_check": 0,
    "full_text": 2,
    "abstract_only": 6
  },
  "support_sources": 5,
  "refutation_sources": 3,
  "used_adversarial_queries": true
}
```

**Field reference**

| Field | Type | Description |
|-------|------|-------------|
| `estimated_reliability` | float 0–1 | LLM estimate — not a verified score |
| `reliability_basis` | string | How the estimate was formed (source count, depth, disclaimer) |
| `evidence_balance_ratio` | float 0–1 \| null | pros / (pros + cons); null if no evidence found |
| `evidence_balance_label` | string | Human-readable summary of the ratio (see table below) |
| `pros[].source_type` | string | `academic` \| `news` \| `fact_check` \| `statistical` |
| `pros[].content_depth` | string | `full_text` \| `abstract` \| `snippet` |
| `sources` | object | Typed breakdown of all sources used |
| `used_adversarial_queries` | bool | False if all refutation queries were filtered by the quality gate |

**Evidence balance labels**

| `evidence_balance_ratio` | `evidence_balance_label` |
|--------------------------|--------------------------|
| ≥ 0.75 | Mostly supporting evidence found |
| ≥ 0.55 | More supporting than contradicting evidence found |
| ≥ 0.35 | Mixed evidence found |
| < 0.35 | Mostly contradicting evidence found |
| No evidence | Insufficient sources found |

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
# All tests (148) — no live API calls
pytest tests/ -v

# Unit tests only (pure Python, instant)
pytest tests/unit/

# Mocked integration tests
pytest tests/mocked/
```
