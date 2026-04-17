"""
Main analysis pipeline.

Orchestrates the full adversarial dual-query fact-checking pipeline:
1. Topic classification → select research services
2. Generate support + refutation queries in parallel
3. Apply adversarial quality gate — discard queries too similar to support queries
4. Execute both query sets against research services concurrently
5. Tag sources with retrieval intent
6. Relevance filtering
7. Screening (LLM batch)
8. Full-text fetch (medium/hard only)
9. Strip retrieved_for tag to prevent framing bias
10. Pros/cons extraction (LLM)
11. Enrich EvidenceItems with source_type and content_depth
12. Reliability aggregation (LLM)
13. Consensus computation (pure Python, no LLM)
"""
import asyncio
import logging
from typing import Dict, List, Any

from app.config import get_settings
from app.models.request import AnalyzeRequest, AnalysisMode
from app.models.response import AnalysisResult, EvidenceItem, SourceBreakdown
from app.constants.analysis import MODE_CONFIG, SOURCE_TYPE_MAP
from app.constants import RELIABILITY_NO_SOURCES

from app.agents.orchestration import (
    get_research_strategy,
    generate_search_queries,
    generate_adversarial_queries,
)
from app.agents.enrichment import (
    screen_sources_by_relevance,
    fetch_fulltext_for_sources,
    get_screening_stats,
)
from app.agents.enrichment.common import detect_source_type
from app.agents.analysis import (
    extract_pros_cons,
    aggregate_results,
    compute_consensus,
)
from app.utils.relevance_filter import filter_relevant_results

from app.services.research import (
    search_arxiv,
    search_world_bank_data,
    search_pubmed,
    search_semantic_scholar,
    search_crossref,
    search_oecd_data,
    search_core,
    search_doaj,
    search_europepmc,
)

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

AGENT_FUNCS = {
    "pubmed":           lambda q: search_pubmed(q, 5),
    "europepmc":        lambda q: search_europepmc(q, 5),
    "arxiv":            lambda q: search_arxiv(q, 5),
    "semantic_scholar": lambda q: search_semantic_scholar(q, 5),
    "crossref":         lambda q: search_crossref(q, max_results=3),
    "core":             lambda q: search_core(q, 5),
    "doaj":             lambda q: search_doaj(q, 5),
    "oecd":             lambda q: search_oecd_data(q, max_results=3),
    "world_bank":       lambda q: search_world_bank_data(q),
}

# ============================================================================
# HELPERS
# ============================================================================


def _is_genuinely_adversarial(
    support_query: str,
    adversarial_query: str,
    threshold: float = 0.6
) -> bool:
    """
    Return False if adversarial query shares too many keywords with the support query.
    A high overlap means the adversarial query is a surface negation, not a genuine challenge.
    """
    stop_words = {"the", "a", "an", "of", "in", "for", "and", "or", "to", "with"}
    support_tokens = {w for w in support_query.lower().split() if len(w) > 3 and w not in stop_words}
    adv_tokens = {w for w in adversarial_query.lower().split() if len(w) > 3 and w not in stop_words}

    if not support_tokens or not adv_tokens:
        return False

    overlap = len(support_tokens & adv_tokens) / len(support_tokens | adv_tokens)
    return overlap < threshold


def _classify_source_type(service_name: str) -> str:
    """Map a service name to its source category using SOURCE_TYPE_MAP."""
    for category, services in SOURCE_TYPE_MAP.items():
        if service_name in services:
            return category
    return "unknown"


def _get_content_depth(source: Dict) -> str:
    """Determine content depth available for a source."""
    if source.get("has_full_text") or "fulltext" in source:
        return "full_text"
    if "abstract" in source:
        return "abstract"
    return "snippet"


def _build_source_breakdown(sources: List[Dict]) -> SourceBreakdown:
    """Build a SourceBreakdown from a list of source dicts."""
    counts: Dict[str, int] = {"academic": 0, "statistical": 0, "news": 0, "fact_check": 0}
    full_text_count = 0
    abstract_only_count = 0

    for source in sources:
        service = detect_source_type(source)
        category = _classify_source_type(service)
        if category in counts:
            counts[category] += 1

        depth = _get_content_depth(source)
        if depth == "full_text":
            full_text_count += 1
        else:
            abstract_only_count += 1

    return SourceBreakdown(
        total=len(sources),
        academic=counts["academic"],
        statistical=counts["statistical"],
        news=counts["news"],
        fact_check=counts["fact_check"],
        full_text=full_text_count,
        abstract_only=abstract_only_count,
    )


def _enrich_evidence_items(
    items: List[Dict],
    url_to_source: Dict[str, Dict]
) -> List[EvidenceItem]:
    """
    Convert raw pros/cons dicts to EvidenceItem, adding source_type and content_depth
    by looking up each item's source URL in the original source map.
    """
    evidence = []
    for item in items:
        url = item.get("source", "")
        source = url_to_source.get(url, {})
        service = detect_source_type(source)
        evidence.append(EvidenceItem(
            claim=item.get("claim", ""),
            source=url,
            source_type=_classify_source_type(service),
            content_depth=_get_content_depth(source),
        ))
    return evidence


# ============================================================================
# RESEARCH EXECUTION
# ============================================================================


async def _run_single_agent(
    agent_name: str,
    query: str,
    tag: str
) -> tuple[str, List[Dict], Exception | None]:
    """Execute one research agent and tag results with retrieval intent."""
    if not query:
        return (agent_name, [], None)

    func = AGENT_FUNCS.get(agent_name)
    if not func:
        return (agent_name, [], None)

    try:
        results = await func(query)
        for r in results:
            r["retrieved_for"] = tag
        return (agent_name, results, None)
    except Exception as e:
        logger.warning(f"agent_search_failed agent={agent_name} error={e}")
        return (agent_name, [], e)


async def _run_research(queries: Dict[str, str], agents: List[str], tag: str) -> List[Dict]:
    """Execute all agents in parallel for one query set."""
    tasks = [
        _run_single_agent(agent, queries.get(agent, ""), tag)
        for agent in agents
        if queries.get(agent, "")
    ]
    results = await asyncio.gather(*tasks)

    sources = []
    for _, agent_results, error in results:
        if not error:
            sources.extend(agent_results)
    return sources


# ============================================================================
# MAIN PIPELINE
# ============================================================================


async def analyze_argument(request: AnalyzeRequest) -> AnalysisResult:
    """
    Full adversarial fact-checking pipeline for a single argument.

    Args:
        request: Validated and sanitized AnalyzeRequest

    Returns:
        AnalysisResult with estimated_reliability, evidence balance, pros, cons, and counts
    """
    settings = get_settings()
    argument_en = request.argument  # Translation not in scope for v1

    # 1. Classify topic → select research services
    try:
        strategy = await asyncio.to_thread(get_research_strategy, argument_en)
        selected_agents = strategy["agents"]
    except Exception as e:
        logger.error(f"research_strategy_failed error={e}")
        selected_agents = ["semantic_scholar", "crossref"]

    # 2. Generate support and refutation queries in parallel
    support_queries: Dict[str, str] = {}
    refutation_queries: Dict[str, str] = {}

    try:
        support_queries, refutation_queries = await asyncio.gather(
            asyncio.to_thread(generate_search_queries, argument_en, selected_agents),
            asyncio.to_thread(generate_adversarial_queries, argument_en, selected_agents)
            if settings.adversarial_queries_enabled
            else asyncio.coroutine(lambda: {})()
        )
    except Exception as e:
        logger.error(f"query_generation_failed error={e}")
        # Partial recovery: support queries only
        if not support_queries:
            try:
                support_queries = await asyncio.to_thread(
                    generate_search_queries, argument_en, selected_agents
                )
            except Exception:
                support_queries = {}

    # 3. Quality gate — discard adversarial queries too similar to support queries
    validated_refutation_queries = {
        agent: query
        for agent, query in refutation_queries.items()
        if _is_genuinely_adversarial(support_queries.get(agent, ""), query)
    }
    used_adversarial_queries = bool(validated_refutation_queries)

    # 4. Execute both query sets concurrently
    support_sources, refutation_sources = await asyncio.gather(
        _run_research(support_queries, selected_agents, tag="support"),
        _run_research(validated_refutation_queries, selected_agents, tag="refutation"),
    )

    all_sources = support_sources + refutation_sources

    # 5. Relevance filtering (keyword-based, no LLM)
    filtered = filter_relevant_results(argument_en, all_sources, min_score=0.0, max_results=len(all_sources))

    # 6. Determine enrichment config from mode
    config = MODE_CONFIG.get(request.mode.value, MODE_CONFIG["medium"])
    enrichment_enabled = config["enabled"]
    top_n = config["top_n"]
    min_score = config["min_score"]

    # 7. Screening (LLM batch evaluation)
    if enrichment_enabled and filtered:
        try:
            selected_sources, rejected_sources = await asyncio.to_thread(
                screen_sources_by_relevance,
                argument_en,
                filtered,
                "en",
                top_n,
                min_score
            )
        except Exception as e:
            logger.error(f"screening_failed error={e}")
            selected_sources = filtered[:top_n]
            rejected_sources = filtered[top_n:]
    else:
        selected_sources = []
        rejected_sources = filtered

    # 8. Full-text fetch (medium/hard only)
    if enrichment_enabled and selected_sources:
        try:
            enhanced = await fetch_fulltext_for_sources(selected_sources)
            final_sources = enhanced + rejected_sources
        except Exception as e:
            logger.error(f"fulltext_fetch_failed error={e}")
            final_sources = all_sources
    else:
        final_sources = all_sources

    # Count retrieval intent before stripping the tag
    support_count = sum(1 for s in final_sources if s.get("retrieved_for") == "support")
    refutation_count = sum(1 for s in final_sources if s.get("retrieved_for") == "refutation")

    # Build URL → source map for EvidenceItem enrichment (before stripping tags)
    url_to_source = {s.get("url", ""): s for s in final_sources if s.get("url")}

    # 9. Strip retrieved_for tag before LLM call — prevents framing bias
    sources_for_llm = [
        {k: v for k, v in s.items() if k != "retrieved_for"}
        for s in final_sources
    ]

    # 10. Pros/cons extraction (LLM)
    try:
        analysis = await asyncio.to_thread(extract_pros_cons, argument_en, sources_for_llm)
    except Exception as e:
        logger.error(f"pros_cons_failed error={e}")
        analysis = {"pros": [], "cons": []}

    # 11. Reliability aggregation (LLM)
    try:
        agg = await asyncio.to_thread(
            aggregate_results,
            [{
                "argument": argument_en,
                "pros": analysis["pros"],
                "cons": analysis["cons"],
                "stance": "affirmatif",
            }]
        )
        estimated_reliability = agg["arguments"][0]["reliability"] if agg["arguments"] else RELIABILITY_NO_SOURCES
    except Exception as e:
        logger.error(f"aggregation_failed error={e}")
        estimated_reliability = RELIABILITY_NO_SOURCES

    # 12. Consensus — deterministic, no LLM
    consensus = compute_consensus(analysis.get("pros", []), analysis.get("cons", []))

    # Build structured source breakdown
    source_breakdown = _build_source_breakdown(final_sources)

    # Construct reliability basis string (pipeline-generated, not LLM)
    full_text_count = source_breakdown.full_text
    abstract_only_count = source_breakdown.abstract_only
    reliability_basis = (
        f"AI estimate based on {len(final_sources)} sources "
        f"({full_text_count} full text, {abstract_only_count} abstract only). "
        f"Not a verified fact-check."
    )

    # Enrich EvidenceItems with source_type and content_depth
    pros = _enrich_evidence_items(analysis.get("pros", []), url_to_source)
    cons = _enrich_evidence_items(analysis.get("cons", []), url_to_source)

    return AnalysisResult(
        argument=request.argument,
        argument_en=argument_en,
        estimated_reliability=estimated_reliability,
        reliability_basis=reliability_basis,
        evidence_balance_ratio=consensus["evidence_balance_ratio"],
        evidence_balance_label=consensus["evidence_balance_label"],
        pros=pros,
        cons=cons,
        sources=source_breakdown,
        support_sources=support_count,
        refutation_sources=refutation_count,
        used_adversarial_queries=used_adversarial_queries,
    )
