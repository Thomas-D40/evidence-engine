"""
Main analysis pipeline.

Orchestrates the full adversarial dual-query fact-checking pipeline:
1. Topic classification → select research services
2. Generate support + refutation queries in parallel
3. Execute both query sets against research services concurrently
4. Tag sources with retrieval intent
5. Relevance filtering
6. Screening (LLM batch)
7. Full-text fetch (medium/hard only)
8. Strip retrieved_for tag to prevent framing bias
9. Pros/cons extraction (LLM)
10. Reliability aggregation (LLM)
11. Consensus computation (pure Python, no LLM)
"""
import asyncio
import logging
from typing import Dict, List, Any

from app.config import get_settings
from app.models.request import AnalyzeRequest, AnalysisMode
from app.models.response import AnalysisResult, EvidenceItem
from app.constants.analysis import MODE_CONFIG
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
        AnalysisResult with reliability_score, consensus, pros, cons, and counts
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

    # 3. Execute both query sets concurrently
    support_sources, refutation_sources = await asyncio.gather(
        _run_research(support_queries, selected_agents, tag="support"),
        _run_research(refutation_queries, selected_agents, tag="refutation"),
    )

    all_sources = support_sources + refutation_sources

    # 4. Relevance filtering (keyword-based, no LLM)
    filtered = filter_relevant_results(argument_en, all_sources, min_score=0.0, max_results=len(all_sources))

    # 5. Determine enrichment config from mode
    config = MODE_CONFIG.get(request.mode.value, MODE_CONFIG["medium"])
    enrichment_enabled = config["enabled"]
    top_n = config["top_n"]
    min_score = config["min_score"]

    # 6. Screening (LLM batch evaluation)
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

    # 7. Full-text fetch (medium/hard only)
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

    # 8. Strip retrieved_for tag before LLM call — prevents framing bias
    sources_for_llm = [
        {k: v for k, v in s.items() if k != "retrieved_for"}
        for s in final_sources
    ]

    # 9. Pros/cons extraction (LLM)
    try:
        analysis = await asyncio.to_thread(extract_pros_cons, argument_en, sources_for_llm)
    except Exception as e:
        logger.error(f"pros_cons_failed error={e}")
        analysis = {"pros": [], "cons": []}

    # 10. Reliability aggregation (LLM)
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
        reliability_score = agg["arguments"][0]["reliability"] if agg["arguments"] else RELIABILITY_NO_SOURCES
    except Exception as e:
        logger.error(f"aggregation_failed error={e}")
        reliability_score = RELIABILITY_NO_SOURCES

    # 11. Consensus — deterministic, no LLM
    consensus = compute_consensus(analysis.get("pros", []), analysis.get("cons", []))

    return AnalysisResult(
        argument=request.argument,
        argument_en=argument_en,
        reliability_score=reliability_score,
        consensus_ratio=consensus["ratio"],
        consensus_label=consensus["label"],
        pros=[EvidenceItem(**p) for p in analysis.get("pros", [])],
        cons=[EvidenceItem(**c) for c in analysis.get("cons", [])],
        sources_count=len(final_sources),
        support_sources=support_count,
        refutation_sources=refutation_count,
    )
