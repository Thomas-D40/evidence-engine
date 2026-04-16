"""
Query generation agent with API-specific optimization and fallback strategies.
"""
import json
import logging
from typing import Dict, List, Any, Optional

from openai import OpenAI

from app.config import get_settings
from app.utils.api_helpers import retry_with_backoff, TransientAPIError
from app.constants import (
    LLM_TEMP_QUERY_GENERATION,
    QUERY_GENERATOR_MAX_RETRY_ATTEMPTS,
    QUERY_GENERATOR_BASE_DELAY,
    QUERY_GEN_MIN_WORD_LENGTH,
    QUERY_GEN_MAX_KEYWORDS,
)
from app.prompts import JSON_OUTPUT_STRICT

logger = logging.getLogger(__name__)

# ============================================================================
# PROMPTS
# ============================================================================

SYSTEM_PROMPT = "You are a precise research query optimizer that responds in JSON format."

USER_PROMPT_TEMPLATE = """You are an expert in academic and statistical information retrieval.
Generate HIGHLY OPTIMIZED search queries for the following research sources.

Argument to research: "{argument}"
Detected language: {language}

Generate search queries for these sources:
{agent_instructions}

**CRITICAL REQUIREMENTS:**

1. OECD/World Bank queries: Use standard indicator names (2-4 words max). E.g. "GDP growth", "unemployment rate"
2. PubMed queries: Medical terminology only. Empty string "" if not medical.
3. ArXiv queries: Scientific/technical only (physics, CS, math). Empty string "" if not scientific.
4. Semantic Scholar / CrossRef: Can handle any academic topic.
5. NewsAPI / GNews: Current events only. Empty string "" if not newsworthy.
6. Google Fact Check: Specific verifiable claims.
7. ClaimBuster: Full sentence factual claims.

Also provide fallback queries for each agent.

{json_instruction}

**EXAMPLE OUTPUT FORMAT:**
{{{{
    "pubmed": {{{{
        "query": "coffee cancer risk epidemiology",
        "fallback": ["coffee health effects", "caffeine cancer"],
        "confidence": 0.85
    }}}},
    "oecd": {{{{
        "query": "GDP growth",
        "fallback": ["economic growth"],
        "confidence": 0.90
    }}}}
}}}}

Set confidence 0.0–1.0 based on query quality.
If a source is not relevant, use empty string "" for query and empty array for fallback."""

# ============================================================================
# CLASS
# ============================================================================


class QueryGenerator:
    """LLM-based query generator with per-service optimization."""

    AGENT_REQUIREMENTS = {
        "pubmed":           {"language": "English", "style": "Medical terminology, MeSH terms", "length": "3-5 keywords", "example": "coffee consumption cancer risk epidemiology"},
        "arxiv":            {"language": "English", "style": "Academic, technical terms", "length": "4-6 keywords", "example": "machine learning neural networks optimization"},
        "semantic_scholar": {"language": "English", "style": "Broad academic query with synonyms", "length": "5-8 keywords", "example": "artificial intelligence applications healthcare"},
        "crossref":         {"language": "English", "style": "Formal academic terms", "length": "3-5 keywords", "example": "climate change economic impact"},
        "oecd":             {"language": "English", "style": "Standard indicator names (GDP, unemployment, etc.)", "length": "2-4 keywords", "example": "GDP growth rate"},
        "world_bank":       {"language": "English", "style": "Economic/development indicators", "length": "2-4 keywords", "example": "poverty rate income inequality"},
        "newsapi":          {"language": "English", "style": "News keywords, current events", "length": "3-6 keywords", "example": "climate change policy announcement"},
        "gnews":            {"language": "English", "style": "News keywords, recent events", "length": "3-6 keywords", "example": "vaccine mandate government decision"},
        "google_factcheck": {"language": "English", "style": "Specific claim or statement", "length": "5-10 words", "example": "coffee causes cancer health claim"},
        "claimbuster":      {"language": "English", "style": "Factual claim to verify", "length": "Full sentence", "example": "The unemployment rate decreased by 2% last year"},
    }

    def __init__(self):
        self.settings = get_settings()
        if self.settings.openai_api_key:
            self.client = OpenAI(api_key=self.settings.openai_api_key)
            self.available = True
        else:
            self.client = None
            self.available = False
            logger.warning("Query generator not available (no OpenAI key)")

    def _build_prompt(self, argument: str, agents: List[str], language: str = "en") -> str:
        agent_instructions = []
        for agent in agents:
            if agent in self.AGENT_REQUIREMENTS:
                req = self.AGENT_REQUIREMENTS[agent]
                agent_instructions.append(
                    f'\n{agent}:\n'
                    f'  - Language: {req["language"]}\n'
                    f'  - Style: {req["style"]}\n'
                    f'  - Length: {req["length"]}\n'
                    f'  - Example: "{req["example"]}"'
                )

        return USER_PROMPT_TEMPLATE.format(
            argument=argument,
            language=language,
            agent_instructions="".join(agent_instructions),
            json_instruction=JSON_OUTPUT_STRICT
        )

    @retry_with_backoff(
        max_attempts=QUERY_GENERATOR_MAX_RETRY_ATTEMPTS,
        base_delay=QUERY_GENERATOR_BASE_DELAY
    )
    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        if not self.available:
            raise TransientAPIError("OpenAI client not available")

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

    def _fallback_queries(self, argument: str, agents: List[str]) -> Dict[str, Any]:
        words = argument.lower().split()
        keywords = [w for w in words if len(w) > QUERY_GEN_MIN_WORD_LENGTH][:QUERY_GEN_MAX_KEYWORDS]
        simple_query = " ".join(keywords)

        queries = {}
        for agent in agents:
            if agent in ["oecd", "world_bank"]:
                economic_terms = ["gdp", "unemployment", "inflation", "poverty", "growth", "trade"]
                found = [t for t in economic_terms if t in argument.lower()]
                query = " ".join(found) if found else "economic indicators"
            else:
                query = simple_query
            queries[agent] = {"query": query, "fallback": [simple_query], "confidence": 0.3}

        logger.info("Using fallback query generation")
        return queries

    def generate_queries(
        self,
        argument: str,
        agents: Optional[List[str]] = None,
        language: str = "en",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate optimized search queries for multiple agents.

        Returns:
            {agent_name: {"query": str, "fallback": list, "confidence": float}}
        """
        if not argument or len(argument.strip()) < 3:
            logger.warning("Argument too short for query generation")
            return {}

        if agents is None:
            agents = ["pubmed", "arxiv", "semantic_scholar", "crossref", "oecd", "world_bank"]

        try:
            prompt = self._build_prompt(argument, agents, language)
            queries = self._call_llm(prompt)

            for agent in agents:
                if agent not in queries:
                    queries[agent] = {"query": "", "fallback": [], "confidence": 0.0}
                if "fallback" not in queries[agent]:
                    queries[agent]["fallback"] = []
                if "confidence" not in queries[agent]:
                    queries[agent]["confidence"] = 0.5

            logger.info(f"Generated queries for {len(queries)} agents")
            return queries

        except TransientAPIError as e:
            logger.warning(f"Query generation failed: {e}, using fallback")
            return self._fallback_queries(argument, agents)
        except Exception as e:
            logger.error(f"Unexpected error in query generation: {e}")
            return self._fallback_queries(argument, agents)


# Module-level singleton for backward compatibility
_query_generator: Optional[QueryGenerator] = None


def generate_search_queries(
    argument: str,
    agents: Optional[List[str]] = None,
    language: str = "en"
) -> Dict[str, str]:
    """
    Generate optimized search queries (simplified format).

    Returns:
        {agent_name: query_string}
    """
    global _query_generator
    if _query_generator is None:
        _query_generator = QueryGenerator()

    enhanced = _query_generator.generate_queries(argument, agents, language)
    return {agent: data.get("query", "") for agent, data in enhanced.items()}
