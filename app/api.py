"""
FastAPI application — routes, middleware, and lifespan.
"""
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, Request
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.core.auth import verify_api_key
from app.core.rate_limiter import limiter, LIMIT_PER_IP, LIMIT_PER_KEY, key_func_by_api_key
from app.core.security import SecurityHeadersMiddleware, MaxBodySizeMiddleware, sanitize_argument
from app.models.request import AnalyzeRequest
from app.models.response import AnalysisResult
from app.pipeline import analyze_argument

logger = logging.getLogger(__name__)

# ============================================================================
# LIFESPAN
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate required settings at startup; fail fast if misconfigured."""
    try:
        settings = get_settings()
        if not settings.openai_api_key:
            logger.error("OPENAI_API_KEY is not set — refusing to start")
            sys.exit(1)
        if not settings.api_keys_set:
            logger.warning(
                "ALLOWED_API_KEYS is empty — all requests will be rejected with HTTP 403"
            )
        logger.info(f"Evidence Engine starting on port {settings.port} [env={settings.env}]")
    except Exception as e:
        logger.error(f"Startup configuration error: {e}")
        sys.exit(1)

    yield


# ============================================================================
# APPLICATION
# ============================================================================

app = FastAPI(
    title="Evidence Engine",
    description="Fact-checking API with adversarial dual-query research pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

# Rate limiter state
app.state.limiter = limiter

# Middleware (order matters: outermost applied last)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(MaxBodySizeMiddleware)

# Rate limit error handler
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ============================================================================
# ROUTES
# ============================================================================


@app.get("/health")
async def health():
    """Health check endpoint. No auth required."""
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalysisResult)
@limiter.limit(LIMIT_PER_IP)
@limiter.limit(LIMIT_PER_KEY, key_func=key_func_by_api_key)
async def analyze(
    request: Request,
    body: AnalyzeRequest,
    api_key: str = Depends(verify_api_key),
) -> AnalysisResult:
    """
    Analyze an argument and return fact-checking results.

    - Requires a valid X-API-Key header
    - Rate limited: 60/min per IP, 100/hour per API key
    - Body limit: 10 KB
    """
    # Sanitize against prompt injection before passing to LLM agents
    body.argument = sanitize_argument(body.argument)
    if body.context:
        body.context = sanitize_argument(body.context)

    return await analyze_argument(body)
