"""
Input sanitization and security headers middleware.
"""
import re
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.constants.security import (
    PROMPT_INJECTION_PATTERNS,
    SECURITY_HEADERS,
    MAX_REQUEST_BODY_BYTES,
)

# ============================================================================
# INPUT SANITIZATION
# ============================================================================


def sanitize_argument(text: str) -> str:
    """Remove prompt injection patterns from user-supplied text."""
    for pattern in PROMPT_INJECTION_PATTERNS:
        text = re.sub(pattern, "[removed]", text, flags=re.IGNORECASE)
    return text


# ============================================================================
# SECURITY HEADERS MIDDLEWARE
# ============================================================================


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to every response."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        for header, value in SECURITY_HEADERS.items():
            response.headers[header] = value
        return response


# ============================================================================
# PAYLOAD SIZE MIDDLEWARE
# ============================================================================


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """Reject requests whose body exceeds MAX_REQUEST_BODY_BYTES."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BODY_BYTES:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request body too large (max {MAX_REQUEST_BODY_BYTES} bytes)"}
            )
        return await call_next(request)
