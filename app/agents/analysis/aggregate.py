"""
Final aggregation agent for analysis results.

Aggregates pros/cons results and calculates a reliability score
using an LLM to weigh source quality, consensus, and argument tone.
"""
import json
from typing import List, Dict

from openai import OpenAI

from app.config import get_settings
from app.logger import get_logger
from app.constants import (
    AGGREGATE_MAX_PROS_PER_ARG,
    AGGREGATE_MAX_CONS_PER_ARG,
    AGGREGATE_MAX_CLAIM_LENGTH,
    AGGREGATE_MAX_ARGUMENT_LENGTH,
    AGGREGATE_MAX_ITEMS_TEXT_LENGTH,
    LLM_TEMP_RELIABILITY_AGGREGATION,
    RELIABILITY_BASE_SCORE,
    RELIABILITY_PER_SOURCE_INCREMENT,
    RELIABILITY_MAX_FALLBACK,
    RELIABILITY_NO_SOURCES,
)
from app.prompts import JSON_OUTPUT_STRICT

logger = get_logger(__name__)

# ============================================================================
# PROMPTS
# ============================================================================

SYSTEM_PROMPT = f"""You are an expert in evaluating the reliability of scientific arguments.
Aggregate analysis results and calculate a reliability score (0.0-1.0) for each argument.

**SCORING CRITERIA:**
- 0.0-0.3: Very low (few sources, major contradictions)
- 0.4-0.6: Average (some sources, partial consensus)
- 0.7-0.8: Good (several reliable sources, relative consensus)
- 0.9-1.0: Very high (numerous scientific sources, strong consensus)

**FACTORS TO CONSIDER:**
- Number of sources
- Consensus among sources
- Quality (scientific > general)
- Argument tone (affirmative vs conditional)
- Balance between pros and cons

**IMPORTANT:** Abstract-only sources are still VALUABLE and RELIABLE for fact-checking.
Do NOT penalize sources for being abstract-only or requiring subscription.

{JSON_OUTPUT_STRICT}

**RESPONSE FORMAT:**
{{
  "arguments": [
    {{
      "argument": "...",
      "pros": [{{"claim": "...", "source": "..."}}],
      "cons": [{{"claim": "...", "source": "..."}}],
      "reliability": 0.75,
      "stance": "affirmatif"
    }}
  ]
}}"""

USER_PROMPT_TEMPLATE = """Aggregate the following results and calculate reliability scores:

{items_text}

Return only JSON, no additional text."""

# ============================================================================
# LOGIC
# ============================================================================


def aggregate_results(items: List[Dict], video_id: str = "") -> Dict:
    """
    Aggregate analysis results and calculate reliability scores.

    Args:
        items: List of {"argument", "pros", "cons", "stance"} dicts
        video_id: Optional identifier for logging

    Returns:
        {"arguments": [{argument, pros, cons, reliability, stance}]}
    """
    settings = get_settings()

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY not configured")

    if not items:
        return {"arguments": []}

    client = OpenAI(api_key=settings.openai_api_key)

    # Truncate items to reduce token usage
    items_context = []
    for item in items:
        pros = item.get("pros", [])[:AGGREGATE_MAX_PROS_PER_ARG]
        cons = item.get("cons", [])[:AGGREGATE_MAX_CONS_PER_ARG]

        items_context.append({
            "argument": item.get("argument", "")[:AGGREGATE_MAX_ARGUMENT_LENGTH],
            "pros": [{"claim": p.get("claim", "")[:AGGREGATE_MAX_CLAIM_LENGTH], "source": p.get("source", "")} for p in pros],
            "cons": [{"claim": c.get("claim", "")[:AGGREGATE_MAX_CLAIM_LENGTH], "source": c.get("source", "")} for c in cons],
            "stance": item.get("stance", "affirmatif"),
        })

    items_text = json.dumps(items_context, ensure_ascii=False, separators=(",", ":"))
    truncated_items = items_text[:AGGREGATE_MAX_ITEMS_TEXT_LENGTH]

    user_prompt = USER_PROMPT_TEMPLATE.format(items_text=truncated_items)

    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=LLM_TEMP_RELIABILITY_AGGREGATION,
            response_format={"type": "json_object"}
        )

        parsed = json.loads(response.choices[0].message.content)
        validated = []

        if isinstance(parsed, dict) and "arguments" in parsed:
            for arg in parsed["arguments"]:
                if isinstance(arg, dict) and "argument" in arg:
                    reliability = arg.get("reliability", 0.5)
                    if not isinstance(reliability, (int, float)):
                        reliability = 0.5
                    reliability = max(0.0, min(1.0, float(reliability)))

                    pros = arg.get("pros", [])
                    cons = arg.get("cons", [])
                    stance = arg.get("stance", "affirmatif")
                    if stance not in ["affirmatif", "conditionnel"]:
                        stance = "affirmatif"

                    validated.append({
                        "argument": arg["argument"].strip(),
                        "pros": pros if isinstance(pros, list) else [],
                        "cons": cons if isinstance(cons, list) else [],
                        "reliability": reliability,
                        "stance": stance,
                    })

        if not validated and items:
            return _fallback_aggregation(items)

        return {"arguments": validated}

    except (json.JSONDecodeError, Exception) as e:
        logger.error("aggregation_failed", detail=str(e))
        return _fallback_aggregation(items)


def _fallback_aggregation(items: List[Dict]) -> Dict:
    """Fallback aggregation without LLM when API call fails."""
    arguments = []
    for item in items:
        num_sources = len(item.get("pros", [])) + len(item.get("cons", []))

        if num_sources == 0:
            reliability = RELIABILITY_NO_SOURCES
        else:
            reliability = min(
                RELIABILITY_MAX_FALLBACK,
                RELIABILITY_BASE_SCORE + (num_sources * RELIABILITY_PER_SOURCE_INCREMENT)
            )

        arguments.append({
            "argument": item.get("argument", ""),
            "pros": item.get("pros", []),
            "cons": item.get("cons", []),
            "reliability": reliability,
            "stance": item.get("stance", "affirmatif"),
        })

    return {"arguments": arguments}
