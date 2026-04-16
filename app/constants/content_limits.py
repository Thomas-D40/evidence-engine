"""
Content and Token Limit Constants.

Maximum lengths, token counts, and other content-related limits.
"""

# ============================================================================
# ANALYSIS CONTENT LIMITS
# ============================================================================

PROS_CONS_MAX_CONTENT_LENGTH = 40000
"""Maximum content length for pros/cons analysis."""

PROS_CONS_MIN_PARTIAL_CONTENT = 500
"""Minimum space to include partial content."""

AGGREGATE_MAX_PROS_PER_ARG = 5
"""Maximum number of pros items per argument."""

AGGREGATE_MAX_CONS_PER_ARG = 5
"""Maximum number of cons items per argument."""

AGGREGATE_MAX_CLAIM_LENGTH = 200
"""Maximum characters per claim."""

AGGREGATE_MAX_ARGUMENT_LENGTH = 300
"""Maximum characters for argument text."""

AGGREGATE_MAX_ITEMS_TEXT_LENGTH = 10000
"""Maximum total length for items text in aggregation."""

# ============================================================================
# RESEARCH RESULT LIMITS
# ============================================================================

PUBMED_SNIPPET_MAX_LENGTH = 500
"""Maximum snippet length for PubMed results."""

PUBMED_MAX_AUTHORS_DISPLAYED = 3
"""Maximum number of authors to display in PubMed results."""

FULLTEXT_MAX_CONTENT_LENGTH = 50000
"""Maximum full-text content length."""

SCREENING_TITLE_MAX_LENGTH = 150
"""Maximum title length for screening prompts."""

SCREENING_SNIPPET_MAX_LENGTH = 300
"""Maximum snippet length for screening prompts."""

SCREENING_MAX_TOKENS = 800
"""Maximum tokens for LLM screening response."""
