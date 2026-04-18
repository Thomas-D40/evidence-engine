"""
Adversarial query generation agent.

Generates refutation-oriented search queries for each research service,
enabling dual retrieval (support + refutation) to reduce structural bias.
Uses structured reasoning before generation to avoid surface-level negations.
"""
import json
import logging
from typing import Dict, List, Optional

from openai import OpenAI

from app.config import get_settings
from app.utils.api_helpers import retry_with_backoff, TransientAPIError
from app.constants import (
    LLM_TEMP_QUERY_GENERATION,
    QUERY_GENERATOR_MAX_RETRY_ATTEMPTS,
    QUERY_GENERATOR_BASE_DELAY,
)
from app.prompts import JSON_OUTPUT_STRICT

logger = logging.getLogger(__name__)

# ============================================================================
# PROMPTS
# ============================================================================

SYSTEM_PROMPT = "You are a scientific devil's advocate specialized in finding genuine weaknesses in claims."

USER_PROMPT_TEMPLATE = """Argument: "{argument}"

STEP 1 — Identify genuine weaknesses (reason before generating):
For each angle below, write one sentence explaining how it applies to this specific argument.
If an angle does not apply, write "N/A".

  a) Confounders: what alternative factor could explain the same observation?
  b) Exceptions: in what subgroup, context, or condition might the effect reverse or disappear?
  c) Methodological weakness: what flaw commonly affects studies on this topic?
  d) Opposing mechanism: what mechanism contradicts the claim at a causal level?
  e) Null findings: what would a failed replication or heterogeneous meta-analysis look like?

STEP 2 — Generate one adversarial query per research service.
Each query MUST emerge from one of the weaknesses identified above (state which: a/b/c/d/e).
Do NOT simply negate the argument. The query must surface evidence that makes the claim
less convincing, not evidence that merely mentions it.

Services: {agents}

{json_instruction}

Return JSON with this exact format:
{{
  "reasoning": {{
    "a": "...",
    "b": "...",
    "c": "...",
    "d": "...",
    "e": "..."
  }},
  "queries": {{
    "service_name": {{"adversarial_query": "...", "angle": "c", "confidence": 0.8}}
  }}
}}

Only include services from the provided list.
Use empty string "" for adversarial_query if the service is irrelevant to this argument."""

# ============================================================================
# LOGIC
# ============================================================================

_adversarial_generator: Optional["AdversarialQueryGenerator"] = None


class AdversarialQueryGenerator:
    """
    LLM-based generator producing one adversarial query per research service.
    Uses structured two-step reasoning to surface genuine weaknesses rather than
    surface-level negations.
    """

    def __init__(self):
        self.settings = get_settings()
        if self.settings.openai_api_key:
            self.client = OpenAI(api_key=self.settings.openai_api_key)
            self.available = True
            logger.info("Adversarial query generator initialized")
        else:
            self.client = None
            self.available = False
            logger.warning("Adversarial query generator not available (no OpenAI key)")

    @retry_with_backoff(
        max_attempts=QUERY_GENERATOR_MAX_RETRY_ATTEMPTS,
        base_delay=QUERY_GENERATOR_BASE_DELAY
    )
    def _call_llm(self, argument: str, agents: List[str]) -> Dict:
        if not self.available:
            raise TransientAPIError("OpenAI client not available")

        prompt = USER_PROMPT_TEMPLATE.format(
            argument=argument,
            agents=", ".join(agents),
            json_instruction=JSON_OUTPUT_STRICT
        )

        try:
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=LLM_TEMP_QUERY_GENERATION
            )
            return json.loads(response.choices[0].message.content)

        except json.JSONDecodeError as e:
            raise TransientAPIError(f"JSON parse error: {e}")
        except Exception as e:
            raise TransientAPIError(f"LLM error: {e}")

    def generate(self, argument: str, agents: List[str]) -> Dict[str, str]:
        """
        Generate adversarial queries for each research service.

        Args:
            argument: English argument text
            agents: List of research service names

        Returns:
            {service_name: adversarial_query_string}
            Returns empty dict on any failure so the pipeline can continue.
        """
        if not argument or len(argument.strip()) < 3:
            logger.warning("Argument too short for adversarial query generation")
            return {}

        try:
            raw = self._call_llm(argument, agents)

            # Extract from nested "queries" key — angle/confidence retained in raw for logging
            raw_queries = raw.get("queries", {})
            queries: Dict[str, str] = {
                agent: raw_queries.get(agent, {}).get("adversarial_query", "")
                if isinstance(raw_queries.get(agent), dict)
                else ""
                for agent in agents
            }

            logger.info(
                "adversarial_queries_generated",
                agents=len(queries),
                non_empty=sum(1 for q in queries.values() if q)
            )
            return queries

        except TransientAPIError as e:
            logger.warning(f"Adversarial query generation failed: {e}, returning empty")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error in adversarial query generation: {e}")
            return {}


def generate_adversarial_queries(argument: str, agents: List[str]) -> Dict[str, str]:
    """
    Generate adversarial (refutation-oriented) search queries for each research service.

    Module-level convenience function. Returns empty dict on any failure so the
    pipeline can continue with support-only queries.

    Args:
        argument: English argument text
        agents: List of research service names

    Returns:
        {service_name: adversarial_query_string}
    """
    global _adversarial_generator
    if _adversarial_generator is None:
        _adversarial_generator = AdversarialQueryGenerator()

    return _adversarial_generator.generate(argument, agents)
