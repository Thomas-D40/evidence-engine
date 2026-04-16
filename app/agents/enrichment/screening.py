"""
Relevance screening agent for intelligent source selection.

Evaluates source abstracts to determine which deserve full-text retrieval,
optimizing token usage by fetching complete content only for the most relevant sources.
"""
import json
import logging
from typing import List, Dict, Tuple

from openai import OpenAI

from app.config import get_settings
from .common import extract_source_content, truncate_content
from app.constants import (
    SCREENING_TITLE_MAX_LENGTH,
    SCREENING_SNIPPET_MAX_LENGTH,
    SCREENING_MAX_TOKENS,
    LLM_TEMP_RELEVANCE_SCREENING,
    RELEVANCE_THRESHOLD_HIGH,
    RELEVANCE_THRESHOLD_MEDIUM_MIN,
)
from app.prompts import JSON_OUTPUT_STRICT

logger = logging.getLogger(__name__)

# ============================================================================
# PROMPTS
# ============================================================================

SCORE_GUIDE = """Score Guide:
- 0.9-1.0: Directly addresses the argument with specific evidence or data
- 0.7-0.8: Highly relevant, discusses the main topic in detail
- 0.5-0.6: Somewhat relevant, related to the topic
- 0.3-0.4: Tangentially related, minor relevance
- 0.0-0.2: Not relevant to the argument"""

SCREENING_INSTRUCTION = "Be strict: Only sources that DIRECTLY help fact-check this argument should score above 0.6."

SCREENING_PROMPT_TEMPLATE = """You are a research relevance evaluator for fact-checking.

Argument to fact-check: "{argument}"

Evaluate {num_sources} sources below for relevance to this specific argument.

{sources_text}

For EACH source, assign a relevance score:

{score_guide}

{screening_instruction}

{json_instruction}

**RESPONSE FORMAT:**
{{{{
  "scores": [
    {{{{"source_id": 1, "score": 0.85, "reason": "One brief sentence"}}}},
    {{{{"source_id": 2, "score": 0.65, "reason": "One brief sentence"}}}},
    ...
  ]
}}}}"""

# ============================================================================
# LOGIC
# ============================================================================


def _build_screening_prompt(argument: str, sources: List[Dict]) -> str:
    sources_text = ""
    for i, source in enumerate(sources):
        title = truncate_content(source.get("title", "N/A"), SCREENING_TITLE_MAX_LENGTH)
        content = extract_source_content(source, prefer_fulltext=False)
        snippet = truncate_content(content, SCREENING_SNIPPET_MAX_LENGTH)
        sources_text += f"\nSource {i + 1}:\nTitle: {title}\nAbstract: {snippet}\n---\n"

    return SCREENING_PROMPT_TEMPLATE.format(
        argument=argument,
        num_sources=len(sources),
        sources_text=sources_text,
        score_guide=SCORE_GUIDE,
        screening_instruction=SCREENING_INSTRUCTION,
        json_instruction=JSON_OUTPUT_STRICT
    )


def _parse_screening_response(content: str, num_sources: int) -> Dict[int, Dict]:
    try:
        result = json.loads(content)
        scores_data = result.get("scores", [])
        if not scores_data:
            return {}

        scores = {}
        for item in scores_data:
            source_id = item.get("source_id", 0)
            score = float(item.get("score", 0.5))
            reason = item.get("reason", "")
            idx = source_id - 1
            if 0 <= idx < num_sources:
                scores[idx] = {
                    "score": max(0.0, min(1.0, score)),
                    "reason": reason
                }
        return scores

    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"[Screening] Parse error: {e}")
        return {}


def _attach_scores_to_sources(sources: List[Dict], scores: Dict[int, Dict]) -> List[Dict]:
    scored = []
    for i, source in enumerate(sources):
        score_data = scores.get(i, {"score": 0.5, "reason": "Not evaluated"})
        entry = source.copy()
        entry["relevance_score"] = score_data["score"]
        entry["relevance_reason"] = score_data["reason"]
        scored.append(entry)
    return scored


def _select_top_sources(
    scored_sources: List[Dict],
    top_n: int,
    min_score: float
) -> Tuple[List[Dict], List[Dict]]:
    sorted_sources = sorted(
        scored_sources,
        key=lambda x: x.get("relevance_score", 0),
        reverse=True
    )

    selected = []
    rejected = []

    for source in sorted_sources:
        score = source.get("relevance_score", 0)
        if len(selected) < top_n and score >= min_score:
            selected.append(source)
        else:
            rejected.append(source)

    return (selected, rejected)


def screen_sources_by_relevance(
    argument: str,
    sources: List[Dict],
    language: str = "en",
    top_n: int = 3,
    min_score: float = 0.6
) -> Tuple[List[Dict], List[Dict]]:
    """
    Screen sources by relevance and select top candidates for full-text retrieval.

    Uses a single batch LLM call to evaluate all sources efficiently.

    Args:
        argument: The argument to fact-check
        sources: List of source dictionaries from research agents
        language: Argument language (for logging)
        top_n: Maximum number of sources to select for full-text
        min_score: Minimum relevance score threshold

    Returns:
        Tuple of (selected_sources, rejected_sources)
    """
    settings = get_settings()

    if not getattr(settings, "fulltext_screening_enabled", True):
        logger.info("[Screening] Disabled in config, using simple top-N selection")
        return (sources[:top_n], sources[top_n:])

    if not sources:
        return ([], [])

    if len(sources) <= top_n:
        logger.info(f"[Screening] Only {len(sources)} sources, selecting all")
        return (sources, [])

    logger.info(f"[Screening] Evaluating {len(sources)} sources...")

    try:
        client = OpenAI(api_key=settings.openai_api_key)
        prompt = _build_screening_prompt(argument, sources)

        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=LLM_TEMP_RELEVANCE_SCREENING,
            max_tokens=SCREENING_MAX_TOKENS,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        scores = _parse_screening_response(content, len(sources))

        if not scores:
            logger.warning("[Screening] No valid scores, using default selection")
            return (sources[:top_n], sources[top_n:])

        scored_sources = _attach_scores_to_sources(sources, scores)
        selected, rejected = _select_top_sources(scored_sources, top_n, min_score)

        logger.info(f"[Screening] Selected {len(selected)} for full-text, {len(rejected)} abstract-only")
        return (selected, rejected)

    except Exception as e:
        logger.error(f"[Screening] Error: {e}")
        return (sources[:top_n], sources[top_n:])


def get_screening_stats(sources: List[Dict]) -> Dict:
    """Calculate statistics about screening results."""
    if not sources:
        return {"total": 0, "avg_score": 0.0, "high_relevance": 0, "medium_relevance": 0, "low_relevance": 0}

    scores = [s.get("relevance_score", 0.5) for s in sources if "relevance_score" in s]

    if not scores:
        return {"total": len(sources), "avg_score": 0.0, "high_relevance": 0, "medium_relevance": 0, "low_relevance": 0}

    return {
        "total": len(sources),
        "avg_score": sum(scores) / len(scores),
        "min_score": min(scores),
        "max_score": max(scores),
        "high_relevance": sum(1 for s in scores if s >= RELEVANCE_THRESHOLD_HIGH),
        "medium_relevance": sum(1 for s in scores if RELEVANCE_THRESHOLD_MEDIUM_MIN <= s < RELEVANCE_THRESHOLD_HIGH),
        "low_relevance": sum(1 for s in scores if s < RELEVANCE_THRESHOLD_MEDIUM_MIN),
    }
