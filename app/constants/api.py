"""
API and Network Configuration Constants.

Timeouts and rate limits for external API clients.
"""

# ============================================================================
# TIMEOUTS (seconds)
# ============================================================================

DEFAULT_TIMEOUT_HTTPX = 60
"""Default timeout for HTTP requests using httpx client."""

PUBMED_REQUEST_TIMEOUT = 10
"""Timeout for PubMed search requests."""

PUBMED_FETCH_TIMEOUT = 15
"""Timeout for fetching individual PubMed articles."""

MCP_WEB_FETCH_DEFAULT_TIMEOUT = 30
"""Default timeout for web fetch operations."""

# ============================================================================
# RATE LIMITS (calls per second)
# ============================================================================

PUBMED_RATE_LIMIT_WITHOUT_KEY = 0.34
"""PubMed rate limit without API key (~3 requests per second)."""

PUBMED_RATE_LIMIT_WITH_KEY = 0.11
"""PubMed rate limit with API key (~10 requests per second)."""

RATE_LIMIT_OECD_CALLS_PER_SEC = 1.0
RATE_LIMIT_WORLD_BANK_CALLS_PER_SEC = 2.0
RATE_LIMIT_ARXIV_CALLS_PER_SEC = 1.0
RATE_LIMIT_PUBMED_CALLS_PER_SEC = 3.0
RATE_LIMIT_SEMANTIC_SCHOLAR_CALLS_PER_SEC = 1.0

# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

DEFAULT_CIRCUIT_BREAKER_THRESHOLD = 5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT_OECD = 300
CIRCUIT_BREAKER_RECOVERY_TIMEOUT_ACADEMIC = 180
