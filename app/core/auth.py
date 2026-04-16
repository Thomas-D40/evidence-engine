"""
API key authentication with brute-force protection.

In-memory store tracks auth failures per IP.
After MAX_AUTH_FAILURES_BEFORE_BLOCK failures in AUTH_FAILURE_WINDOW_SECONDS,
the IP is blocked for IP_BLOCK_DURATION_SECONDS.
Counter resets on successful authentication.
"""
import time
from dataclasses import dataclass, field
from typing import Dict

from fastapi import HTTPException, Request
from fastapi.security import APIKeyHeader

from app.config import get_settings
from app.constants.security import (
    MAX_AUTH_FAILURES_BEFORE_BLOCK,
    AUTH_FAILURE_WINDOW_SECONDS,
    IP_BLOCK_DURATION_SECONDS,
)

# ============================================================================
# HEADER
# ============================================================================

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# ============================================================================
# BRUTE-FORCE STATE
# ============================================================================


@dataclass
class IPState:
    failure_count: int = 0
    first_failure_time: float = 0.0
    blocked_until: float = 0.0


# In-memory store: IP → IPState
_ip_states: Dict[str, IPState] = {}


def _get_state(ip: str) -> IPState:
    if ip not in _ip_states:
        _ip_states[ip] = IPState()
    return _ip_states[ip]


def _check_ip_not_blocked(ip: str) -> None:
    """Raise HTTP 429 if the IP is currently blocked."""
    state = _get_state(ip)
    now = time.time()
    if state.blocked_until and now < state.blocked_until:
        remaining = int(state.blocked_until - now)
        raise HTTPException(
            status_code=429,
            detail=f"IP blocked due to repeated auth failures. Retry in {remaining}s."
        )


def _record_failure(ip: str) -> None:
    """Record a failed authentication attempt, blocking the IP if threshold reached."""
    state = _get_state(ip)
    now = time.time()

    # Reset window if too old
    if state.first_failure_time and (now - state.first_failure_time) > AUTH_FAILURE_WINDOW_SECONDS:
        state.failure_count = 0
        state.first_failure_time = 0.0

    if state.failure_count == 0:
        state.first_failure_time = now

    state.failure_count += 1

    if state.failure_count >= MAX_AUTH_FAILURES_BEFORE_BLOCK:
        state.blocked_until = now + IP_BLOCK_DURATION_SECONDS
        state.failure_count = 0
        state.first_failure_time = 0.0


def _reset_failures(ip: str) -> None:
    """Reset failure counter after a successful authentication."""
    if ip in _ip_states:
        _ip_states[ip] = IPState()


def _is_valid_key(key: str) -> bool:
    """Check whether the key exists in the configured set."""
    settings = get_settings()
    valid_keys = settings.api_keys_set
    # Empty = misconfiguration → reject all
    return bool(valid_keys) and key in valid_keys


# ============================================================================
# DEPENDENCY
# ============================================================================


async def verify_api_key(
    request: Request,
    x_api_key: str = None,
) -> str:
    """
    FastAPI dependency that validates the X-API-Key header.

    Raises:
        HTTP 429 — IP is blocked
        HTTP 401 — key missing
        HTTP 403 — key invalid
    """
    # Extract header manually to avoid auto_error behavior
    x_api_key = request.headers.get("X-API-Key")

    ip = request.client.host if request.client else "unknown"

    _check_ip_not_blocked(ip)

    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    if not _is_valid_key(x_api_key):
        _record_failure(ip)
        raise HTTPException(status_code=403, detail="Invalid API key")

    _reset_failures(ip)
    return x_api_key
