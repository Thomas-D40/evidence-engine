"""
Security Constants.

Rate limiting, brute-force protection, payload limits,
and prompt injection patterns.
"""
import re

# ============================================================================
# RATE LIMITING
# ============================================================================

RATE_LIMIT_PER_IP_PER_MINUTE = 60
"""Maximum requests per IP per minute."""

RATE_LIMIT_PER_KEY_PER_HOUR = 100
"""Maximum requests per API key per hour."""

# ============================================================================
# BRUTE-FORCE PROTECTION
# ============================================================================

MAX_AUTH_FAILURES_BEFORE_BLOCK = 10
"""Number of failed auth attempts before blocking an IP."""

AUTH_FAILURE_WINDOW_SECONDS = 600
"""Time window (seconds) in which failures are counted (10 minutes)."""

IP_BLOCK_DURATION_SECONDS = 900
"""Duration (seconds) to block an IP after too many failures (15 minutes)."""

# ============================================================================
# PAYLOAD LIMITS
# ============================================================================

MAX_REQUEST_BODY_BYTES = 10_240
"""Maximum allowed request body size in bytes (10 KB)."""

# ============================================================================
# PROMPT INJECTION PATTERNS
# ============================================================================

PROMPT_INJECTION_PATTERNS = [
    r"ignore (previous|prior|all) instructions",
    r"you are now",
    r"forget (everything|all|your instructions)",
    r"system\s*:",
    r"<\|.*?\|>",
]
"""Regex patterns used to detect and strip prompt injection attempts."""

# ============================================================================
# SECURITY HEADERS
# ============================================================================

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'none'",
    "Referrer-Policy": "no-referrer",
}
"""HTTP security headers added to every response."""
