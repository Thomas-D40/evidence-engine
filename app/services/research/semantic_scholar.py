"""
Research agent for Semantic Scholar.

Semantic Scholar is an AI-powered academic database
that covers ~200M scientific articles across all disciplines.

API Documentation: https://www.semanticscholar.org/product/api
Rate Limits: 100 requests per 5 minutes (no API key required)
"""
from typing import List, Dict

import httpx

from ...logger import get_logger
from ..retry import RETRY_STRATEGY

logger = get_logger(__name__)

_HEADERS = {"User-Agent": "VideoAnalyzerWorkflow/1.0 (Research Tool)"}


@RETRY_STRATEGY
async def search_semantic_scholar(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search for academic articles on Semantic Scholar.

    Semantic Scholar uses AI to identify the most relevant articles
    across all academic disciplines (sciences, medicine, social sciences, etc.)

    Args:
        query: Search query (ideally in English)
        max_results: Maximum number of results (default: 5)

    Returns:
        List of dictionaries containing:
        - title: Article title
        - url: URL to the article on Semantic Scholar
        - snippet: Article abstract/summary
        - source: "Semantic Scholar"
        - year: Publication year
        - citations: Number of citations
        - authors: List of authors

    Raises:
        httpx.HTTPStatusError: On HTTP 5xx / 429 after retries
        httpx.TimeoutException: On timeout after retries
    """
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,abstract,year,citationCount,authors,url,openAccessPdf"
    }

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(base_url, params=params, headers=_HEADERS)
        response.raise_for_status()
        data = response.json()

    if "data" not in data or not data["data"]:
        logger.debug("semantic_scholar_no_results", query=query)
        return []

    articles = []
    for paper in data["data"]:
        authors = []
        if paper.get("authors"):
            authors = [a.get("name", "") for a in paper["authors"][:3]]

        paper_id = paper.get("paperId", "")
        paper_url = (
            f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id
            else paper.get("url", "")
        )

        abstract = paper.get("abstract", "")
        if not abstract:
            abstract = f"Article from {paper.get('year', 'N/A')} with {paper.get('citationCount', 0)} citations"

        is_open_access = bool(paper.get("openAccessPdf"))
        if is_open_access:
            access_type = "open_access"
            has_full_text = True
            access_note = "Full text available (open access)"
        else:
            access_type = "abstract_only"
            has_full_text = False
            access_note = "Abstract only - full text may require subscription"

        articles.append({
            "title": paper.get("title", "Untitled"),
            "url": paper_url,
            "snippet": abstract[:500],
            "source": "Semantic Scholar",
            "year": paper.get("year", "N/A"),
            "citations": paper.get("citationCount", 0),
            "authors": ", ".join(authors) if authors else "N/A",
            "access_type": access_type,
            "has_full_text": has_full_text,
            "access_note": access_note
        })

    logger.info("semantic_scholar_search_end", articles_count=len(articles), query=query)
    return articles


@RETRY_STRATEGY
async def get_paper_details(paper_id: str) -> Dict:
    """
    Retrieve full details of an article by its Semantic Scholar ID.

    Args:
        paper_id: Semantic Scholar ID of the article

    Returns:
        Dictionary with full article details
    """
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
    params = {
        "fields": "title,abstract,year,citationCount,authors,url,openAccessPdf,references,citations"
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error("semantic_scholar_details_error", detail=str(e))
        return {}
