"""
Thematic classification agent for arguments.

Uses an LLM to determine the scientific domain of an argument
to select appropriate research sources.
"""
import json
from typing import List

from openai import OpenAI

from app.config import get_settings
from app.constants import LLM_TEMP_TOPIC_CLASSIFICATION
from app.prompts import JSON_OUTPUT_STRICT
from app.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

CATEGORY_AGENTS_MAP = {
    "medicine":         ["pubmed", "semantic_scholar", "crossref", "google_factcheck"],
    "biology":          ["pubmed", "semantic_scholar", "crossref", "arxiv"],
    "economics":        ["oecd", "world_bank", "semantic_scholar", "crossref"],
    "physics":          ["arxiv", "semantic_scholar", "crossref"],
    "computer_science": ["arxiv", "semantic_scholar", "crossref"],
    "mathematics":      ["arxiv", "semantic_scholar", "crossref"],
    "environment":      ["arxiv", "semantic_scholar", "crossref", "oecd"],
    "social_sciences":  ["semantic_scholar", "crossref", "oecd"],
    "psychology":       ["pubmed", "semantic_scholar", "crossref"],
    "education":        ["semantic_scholar", "crossref", "oecd"],
    "politics":         ["semantic_scholar", "crossref", "newsapi", "gnews"],
    "current_events":   ["newsapi", "gnews", "google_factcheck", "semantic_scholar"],
    "fact_check":       ["google_factcheck", "claimbuster", "semantic_scholar"],
    "general":          ["semantic_scholar", "crossref"],
}

AVAILABLE_CATEGORIES = list(CATEGORY_AGENTS_MAP.keys())

PRIORITY_AGENT_MAP = {
    "medicine":         "pubmed",
    "biology":          "pubmed",
    "psychology":       "pubmed",
    "economics":        "oecd",
    "physics":          "arxiv",
    "computer_science": "arxiv",
    "mathematics":      "arxiv",
    "environment":      "semantic_scholar",
    "social_sciences":  "semantic_scholar",
    "education":        "semantic_scholar",
    "politics":         "semantic_scholar",
    "current_events":   "newsapi",
    "fact_check":       "google_factcheck",
    "general":          "semantic_scholar",
}

# ============================================================================
# PROMPTS
# ============================================================================

SYSTEM_PROMPT = "You are a precise scientific classifier that responds in JSON format."

USER_PROMPT_TEMPLATE = """You are an expert in scientific classification.
Analyze the following argument and identify the relevant scientific domains.

Argument: "{argument}"

Available categories:
{categories}

Choose 1 to 3 categories that best match this argument.
If the argument touches multiple domains, list them in order of relevance.
If no specific category matches, use "general".

Examples:
- "Coffee increases cancer risk" → ["medicine", "fact_check"]
- "French GDP is rising" → ["economics"]
- "Black holes emit radiation" → ["physics"]
- "Climate change threatens biodiversity" → ["environment", "biology"]

{json_instruction}

**RESPONSE FORMAT:**
{{
    "categories": ["category1", "category2"]
}}"""

# ============================================================================
# LOGIC
# ============================================================================


def classify_argument_topic(argument: str) -> List[str]:
    """
    Classify an argument by scientific domain.

    Args:
        argument: The argument to classify

    Returns:
        List of categories (e.g., ["medicine", "biology"])
    """
    settings = get_settings()
    if not settings.openai_api_key:
        logger.warning("no_openai_key_fallback_general")
        return ["general"]

    client = OpenAI(api_key=settings.openai_api_key)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        argument=argument,
        categories=", ".join(AVAILABLE_CATEGORIES),
        json_instruction=JSON_OUTPUT_STRICT
    )

    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=LLM_TEMP_TOPIC_CLASSIFICATION
        )

        content = response.choices[0].message.content
        data = json.loads(content)
        categories_list = data.get("categories", ["general"])

        valid = [c for c in categories_list if c in AVAILABLE_CATEGORIES]
        result = valid if valid else ["general"]

        logger.debug("argument_classified", categories=result)
        return result

    except Exception as e:
        logger.error("classification_failed", detail=str(e))
        return ["general"]


def get_agents_for_argument(argument: str) -> List[str]:
    """
    Determine which research agents to use for a given argument.

    Args:
        argument: The argument to analyze

    Returns:
        Deduplicated list of agent names
    """
    categories = classify_argument_topic(argument)

    agents = []
    seen = set()

    for category in categories:
        for agent in CATEGORY_AGENTS_MAP.get(category, CATEGORY_AGENTS_MAP["general"]):
            if agent not in seen:
                agents.append(agent)
                seen.add(agent)

    logger.debug("agents_selected", agents=agents)
    return agents


def get_research_strategy(argument: str) -> dict:
    """
    Return a complete research strategy for an argument.

    Returns:
        {categories, agents, priority}
    """
    categories = classify_argument_topic(argument)
    agents = get_agents_for_argument(argument)
    primary_category = categories[0] if categories else "general"
    priority_agent = PRIORITY_AGENT_MAP.get(primary_category, "semantic_scholar")

    return {
        "categories": categories,
        "agents": agents,
        "priority": priority_agent,
    }
