"""
Pros/cons extraction agent.

Analyzes scientific articles to identify points that support (pros)
or contradict (cons) the argument under study.
"""
import json
import hashlib
from typing import List, Dict

from openai import OpenAI

from app.config import get_settings
from app.logger import get_logger
from app.constants import (
    PROS_CONS_MAX_CONTENT_LENGTH,
    PROS_CONS_MIN_PARTIAL_CONTENT,
    LLM_TEMP_PROS_CONS_ANALYSIS,
)
from app.prompts import JSON_OUTPUT_STRICT, CITATION_INSTRUCTION

logger = get_logger(__name__)

# ============================================================================
# PROMPTS
# ============================================================================

SYSTEM_PROMPT = f"""You are an expert in scientific analysis and argument critique.
Analyze scientific articles to identify points that support (pros) or contradict (cons) an argument.

**STRICT VERIFICATION RULES:**
1. **Explicit Evidence Required**: Each point ("claim") MUST be explicitly supported by the text of a provided article.
2. **No Invention**: If no article mentions a point, DO NOT INVENT IT.
3. **Citation Required**: Each claim must be associated with the exact URL of the article containing it.
4. **Relevance**: Only retain points directly related to the analyzed argument.
5. **Access Level**: Abstracts and summaries are VALID sources — do not dismiss sources for lacking full text.

For each article, identify:
- Claims that SUPPORT the argument (pros)
- Claims that CONTRADICT or QUESTION the argument (cons)

{JSON_OUTPUT_STRICT}

**RESPONSE FORMAT:**
{{
    "pros": [{{"claim": "point description", "source": "article URL"}}],
    "cons": [{{"claim": "point description", "source": "article URL"}}]
}}

If no article contains relevant information, return empty lists."""

USER_PROMPT_TEMPLATE = """Argument to analyze: {argument}

Scientific articles:
{articles_context}

Analyze these articles and extract supporting (pros) and contradicting (cons) points for this argument."""

# ============================================================================
# LOGIC
# ============================================================================


def extract_pros_cons(
    argument: str,
    articles: List[Dict],
    argument_id: str = ""
) -> Dict[str, List[Dict]]:
    """
    Extract supporting and contradicting arguments from scientific articles.

    Args:
        argument: Text of the argument to analyze
        articles: List of articles with fields title, url, snippet/abstract/fulltext
        argument_id: Unique identifier for the argument (optional)

    Returns:
        {"pros": [{claim, source}], "cons": [{claim, source}]}
    """
    settings = get_settings()

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY not configured")

    if not argument or not articles:
        return {"pros": [], "cons": []}

    if not argument_id:
        argument_id = hashlib.md5(argument.encode()).hexdigest()[:8]

    # Build articles context — prefer fulltext over snippet
    articles_context = ""
    current_length = 0

    for article in articles:
        if "fulltext" in article and article["fulltext"]:
            content = article["fulltext"]
            content_type = "Full Text"
        else:
            content = article.get("snippet") or article.get("abstract") or article.get("summary", "")
            content_type = "Summary"

        article_text = (
            f"Article: {article.get('title', '')}\n"
            f"URL: {article.get('url', '')}\n"
            f"{content_type}: {content}\n\n"
        )

        if current_length + len(article_text) > PROS_CONS_MAX_CONTENT_LENGTH:
            remaining = PROS_CONS_MAX_CONTENT_LENGTH - current_length
            if remaining > PROS_CONS_MIN_PARTIAL_CONTENT:
                articles_context += (
                    f"Article: {article.get('title', '')}\n"
                    f"URL: {article.get('url', '')}\n"
                    f"{content_type}: {content[:remaining]}\n\n"
                )
            break

        articles_context += article_text
        current_length += len(article_text)

    # Fallback if first article is already too long
    if articles and not articles_context:
        first = articles[0]
        first_content = first.get("fulltext") or first.get("snippet", "")
        articles_context = (
            f"Article: {first.get('title', '')}\n"
            f"URL: {first.get('url', '')}\n"
            f"Content: {first_content[:PROS_CONS_MAX_CONTENT_LENGTH]}\n\n"
        )

    client = OpenAI(api_key=settings.openai_api_key)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        argument=argument,
        articles_context=articles_context
    )

    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=LLM_TEMP_PROS_CONS_ANALYSIS,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        return {
            "pros": result.get("pros", []),
            "cons": result.get("cons", [])
        }

    except json.JSONDecodeError as e:
        logger.error("pros_cons_json_error", detail=str(e))
        return {"pros": [], "cons": []}
    except Exception as e:
        logger.error("pros_cons_failed", detail=str(e))
        return {"pros": [], "cons": []}
