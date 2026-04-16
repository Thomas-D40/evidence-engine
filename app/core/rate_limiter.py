"""
Rate limiting setup using slowapi.

Two limits applied on the /analyze endpoint:
- 60 requests per minute per IP address
- 100 requests per hour per API key
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.constants.security import (
    RATE_LIMIT_PER_IP_PER_MINUTE,
    RATE_LIMIT_PER_KEY_PER_HOUR,
)

# ============================================================================
# CONSTANTS
# ============================================================================

LIMIT_PER_IP = f"{RATE_LIMIT_PER_IP_PER_MINUTE}/minute"
LIMIT_PER_KEY = f"{RATE_LIMIT_PER_KEY_PER_HOUR}/hour"

# ============================================================================
# LIMITER
# ============================================================================

limiter = Limiter(key_func=get_remote_address)


def key_func_by_api_key(request) -> str:
    """Extract API key from header for per-key rate limiting."""
    return request.headers.get("X-API-Key", "anonymous")
