"""
Shared retry strategy for all research HTTP services.

Uses tenacity for exponential backoff on transient errors:
- Timeouts and network errors
- HTTP 429 (rate limit) and 5xx (server errors)

Non-retryable errors (400, 401, 403, 404) propagate immediately.
"""
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
import httpx

# ============================================================================
# CONSTANTS
# ============================================================================

RETRY_MAX_ATTEMPTS = 3
RETRY_MIN_WAIT_SECONDS = 1
RETRY_MAX_WAIT_SECONDS = 8


# ============================================================================
# RETRY LOGIC
# ============================================================================

def _is_retryable(exc: BaseException) -> bool:
    """Return True for transient errors worth retrying."""
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


RETRY_STRATEGY = retry(
    stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT_SECONDS, max=RETRY_MAX_WAIT_SECONDS),
    retry=retry_if_exception(_is_retryable),
    reraise=True
)
