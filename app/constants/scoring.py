"""
Scoring and Reliability Constants.
"""

# ============================================================================
# RELIABILITY CALCULATION
# ============================================================================

RELIABILITY_NO_SOURCES = 0.0
"""Reliability score when no sources are available."""

RELIABILITY_MAX_FALLBACK = 0.9
"""Maximum reliability score in fallback calculation."""

RELIABILITY_BASE_SCORE = 0.3
"""Base reliability score for fallback calculation."""

RELIABILITY_PER_SOURCE_INCREMENT = 0.1
"""Score increment per additional source in fallback."""

# ============================================================================
# RELEVANCE THRESHOLDS
# ============================================================================

RELEVANCE_THRESHOLD_HIGH = 0.7
"""Threshold for high relevance classification."""

RELEVANCE_THRESHOLD_MEDIUM_MIN = 0.4
"""Minimum threshold for medium relevance classification."""

RELEVANCE_DEFAULT_MIN_SCORE = 0.2
"""Default minimum score for relevance filtering."""

RELEVANCE_DEFAULT_MAX_RESULTS = 2
"""Default maximum results after relevance filtering."""
