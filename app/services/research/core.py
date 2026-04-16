"""
Research agent for CORE (COnnecting REpositories).

CORE aggregates open access research papers from repositories
and journals worldwide. It provides access to 350M+ open access papers.

API Documentation: https://core.ac.uk/services/api
Rate Limits: Free tier allows reasonable usage without API key
"""
import re
from typing import List, Dict

import httpx

from ...logger import get_logger
from ..retry import RETRY_STRATEGY

logger = get_logger(__name__)

_HEADERS = {"User-Agent": "VideoAnalyzerWorkflow/1.0 (Research Tool)"}


@RETRY_STRATEGY
async def search_core(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search for open access research papers on CORE.

    CORE aggregates millions of open access papers from repositories
    worldwide, making it an excellent source for freely accessible research.

    Args:
        query: Search query (ideally in English)
        max_results: Maximum number of results (default: 5)

    Returns:
        List of dictionaries containing:
        - title: Article title
        - url: URL to the article
        - snippet: Article abstract/summary
        - source: "CORE"
        - year: Publication year
        - authors: List of authors
        - repository: Source repository
        - access_type: Always "open_access"
        - has_full_text: Always True
        - access_note: Full text availability note

    Raises:
        httpx.HTTPStatusError: On HTTP 5xx / 429 after retries
        httpx.TimeoutException: On timeout after retries
    """
    base_url = "https://api.core.ac.uk/v3/search/works"
    params = {"q": query, "limit": max_results, "offset": 0}

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(base_url, params=params, headers=_HEADERS)
        response.raise_for_status()
        data = response.json()

    if "results" not in data or not data["results"]:
        logger.debug("core_no_results", query=query)
        return []

    articles = []
    for paper in data["results"]:
        title = paper.get("title", "Untitled")
        abstract = paper.get("abstract", "")

        year = "N/A"
        pub_date = paper.get("publishedDate", "") or paper.get("yearPublished", "")
        if pub_date:
            year_match = re.search(r'(\d{4})', str(pub_date))
            if year_match:
                year = year_match.group(1)

        authors_list = paper.get("authors", [])
        authors = []
        for author in authors_list[:3]:
            name = author.get("name", "") if isinstance(author, dict) else str(author)
            if name:
                authors.append(name)

        doi = paper.get("doi", "")
        if doi:
            url = f"https://doi.org/{doi}"
        else:
            url = paper.get("links", [{}])[0].get("url", "") if paper.get("links") else ""
            if not url:
                url = paper.get("downloadUrl", "")

        repository = "CORE"
        if paper.get("publisher"):
            repository = paper.get("publisher")
        elif paper.get("journals"):
            journals = paper.get("journals", [])
            if journals and isinstance(journals, list) and journals[0]:
                repository = journals[0].get("title", "CORE") if isinstance(journals[0], dict) else str(journals[0])

        has_pdf = bool(paper.get("downloadUrl"))
        access_note = "Full text PDF available (open access)" if has_pdf else "Open access article (full text may be available)"

        if not abstract or len(abstract.strip()) < 20:
            abstract = f"Open access article from {repository} ({year})"

        articles.append({
            "title": title,
            "url": url if url else f"https://core.ac.uk/search?q={query}",
            "snippet": abstract[:500],
            "source": "CORE",
            "year": year,
            "authors": ", ".join(authors) if authors else "N/A",
            "repository": repository,
            "access_type": "open_access",
            "has_full_text": True,
            "access_note": access_note
        })

    logger.info("core_search_end", articles_count=len(articles), query=query)
    return articles
