"""
Analysis Constants.

Mode configurations and score thresholds for the analysis pipeline.
"""

# ============================================================================
# MODE CONFIGURATION
# ============================================================================

MODE_CONFIG = {
    "simple": {"enabled": False, "top_n": 0, "min_score": 0.0},
    "medium": {"enabled": True,  "top_n": 3, "min_score": 0.6},
    "hard":   {"enabled": True,  "top_n": 6, "min_score": 0.5},
}
"""Full-text enrichment config per analysis mode."""

# ============================================================================
# CONSENSUS THRESHOLDS
# ============================================================================

CONSENSUS_STRONG_THRESHOLD = 0.75
"""Minimum pros ratio for 'Strong consensus' label."""

CONSENSUS_MODERATE_THRESHOLD = 0.55
"""Minimum pros ratio for 'Moderate consensus' label."""

CONSENSUS_CONTESTED_THRESHOLD = 0.35
"""Minimum pros ratio for 'Contested' label. Below this: 'Minority position'."""

# ============================================================================
# SOURCE TYPE CLASSIFICATION
# ============================================================================

SOURCE_TYPE_MAP: dict[str, set[str]] = {
    "academic":    {"pubmed", "semantic_scholar", "arxiv", "crossref", "europepmc", "core", "doaj"},
    "statistical": {"oecd", "world_bank"},
    "news":        {"newsapi", "gnews"},
    "fact_check":  {"google_factcheck", "claimbuster"},
}
"""Maps source category to the set of service names belonging to that category."""
