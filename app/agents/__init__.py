"""
Agents package — LLM-powered processing.

Organized into:
- orchestration: topic classification, query generation (support + adversarial)
- enrichment: relevance screening, full-text fetching
- analysis: pros/cons extraction, reliability aggregation, consensus scoring
"""
from .orchestration import (
    generate_search_queries,
    generate_adversarial_queries,
    classify_argument_topic,
    get_agents_for_argument,
    get_research_strategy,
)
from .enrichment import (
    screen_sources_by_relevance,
    get_screening_stats,
    fetch_fulltext_for_sources,
)
from .analysis import extract_pros_cons, aggregate_results, compute_consensus

__all__ = [
    "generate_search_queries",
    "generate_adversarial_queries",
    "classify_argument_topic",
    "get_agents_for_argument",
    "get_research_strategy",
    "screen_sources_by_relevance",
    "get_screening_stats",
    "fetch_fulltext_for_sources",
    "extract_pros_cons",
    "aggregate_results",
    "compute_consensus",
]
