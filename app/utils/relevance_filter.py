"""
Utility for filtering web search results by relevance.
"""
from typing import List, Dict
import re


def extract_keywords(text: str, min_length: int = 3) -> set:
    """
    Extract significant keywords from text.

    Args:
        text: Source text
        min_length: Minimum word length to keep

    Returns:
        Set of lowercase keywords
    """
    # Common French stop words
    stop_words = {
        'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 'ou', 'mais',
        'est', 'sont', 'a', 'ont', 'pour', 'dans', 'sur', 'avec', 'par',
        'ce', 'cette', 'ces', 'que', 'qui', 'quoi', 'dont', 'où',
        'il', 'elle', 'ils', 'elles', 'nous', 'vous', 'leur', 'leurs',
        'son', 'sa', 'ses', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes',
        'plus', 'moins', 'très', 'aussi', 'bien', 'pas', 'ne', 'si'
    }

    words = re.findall(r'\b[a-zàâäéèêëïîôùûüÿæœç]+\b', text.lower())

    return {word for word in words if len(word) >= min_length and word not in stop_words}


def calculate_relevance_score(argument: str, result_snippet: str) -> float:
    """
    Calculate a relevance score between an argument and a result snippet.

    Args:
        argument: Argument text
        result_snippet: Search result snippet

    Returns:
        Score between 0.0 and 1.0
    """
    if not argument or not result_snippet:
        return 0.0

    arg_keywords = extract_keywords(argument)
    snippet_keywords = extract_keywords(result_snippet)

    if not arg_keywords:
        return 0.0

    common = arg_keywords.intersection(snippet_keywords)
    return min(len(common) / len(arg_keywords), 1.0)


def filter_relevant_results(
    argument: str,
    results: List[Dict],
    min_score: float = 0.2,
    max_results: int = 2
) -> List[Dict]:
    """
    Filter search results by relevance to the argument.

    Args:
        argument: Argument text
        results: List of search results
        min_score: Minimum relevance score (0.0-1.0)
        max_results: Maximum number of results to return

    Returns:
        Filtered and sorted list by relevance
    """
    if not results:
        return []

    scored = []
    for result in results:
        snippet = result.get("snippet", "") or result.get("abstract", "") or result.get("summary", "")
        score = calculate_relevance_score(argument, snippet)
        if score >= min_score:
            entry = result.copy()
            entry["relevance_score"] = score
            scored.append(entry)

    scored.sort(key=lambda x: x["relevance_score"], reverse=True)
    return scored[:max_results]
