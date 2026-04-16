"""
Research agent for DOAJ (Directory of Open Access Journals).

DOAJ indexes ~2 million articles from quality, peer-reviewed
open access journals across all disciplines.

API Documentation: https://doaj.org/api/docs
Rate Limits: Reasonable usage without API key
"""
from typing import List, Dict

import httpx

from ...logger import get_logger
from ..retry import RETRY_STRATEGY

logger = get_logger(__name__)

_HEADERS = {"User-Agent": "VideoAnalyzerWorkflow/1.0 (Research Tool)"}


@RETRY_STRATEGY
async def search_doaj(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search for articles in open access journals via DOAJ.

    DOAJ provides access to high-quality, peer-reviewed open access journals.
    All content is guaranteed to be open access with full text available.

    Args:
        query: Search query (ideally in English)
        max_results: Maximum number of results (default: 5)

    Returns:
        List of dictionaries containing:
        - title: Article title
        - url: URL to the article
        - snippet: Article abstract/summary
        - source: "DOAJ"
        - year: Publication year
        - authors: List of authors
        - journal: Journal name
        - doi: DOI if available
        - access_type: Always "open_access"
        - has_full_text: Always True
        - access_note: Full text availability note

    Raises:
        httpx.HTTPStatusError: On HTTP 5xx / 429 after retries
        httpx.TimeoutException: On timeout after retries
    """
    base_url = "https://doaj.org/api/search/articles"
    params = {"q": query, "pageSize": max_results, "page": 1}

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(base_url, params=params, headers=_HEADERS)
        response.raise_for_status()
        data = response.json()

    results = data.get("results", [])
    if not results:
        logger.debug("doaj_no_results", query=query)
        return []

    articles = []
    for item in results:
        bibjson = item.get("bibjson", {})

        title = bibjson.get("title", "Untitled")
        abstract = bibjson.get("abstract", "")

        year = "N/A"
        if bibjson.get("year"):
            year = str(bibjson["year"])

        authors_list = bibjson.get("author", [])
        authors = [a.get("name", "") for a in authors_list[:3] if a.get("name")]

        journal_info = bibjson.get("journal", {})
        journal = journal_info.get("title", "N/A")

        doi = ""
        url = ""
        for identifier in bibjson.get("identifier", []):
            if identifier.get("type") == "doi":
                doi = identifier.get("id", "")
                if doi:
                    url = f"https://doi.org/{doi}"
                    break

        if not url:
            for link in bibjson.get("link", []):
                if link.get("type") == "fulltext":
                    url = link.get("url", "")
                    break

        if not url:
            url = item.get("id", "")

        if not abstract or len(abstract.strip()) < 20:
            abstract = f"Open access article from {journal} ({year})"

        article = {
            "title": title,
            "url": url if url else f"https://doaj.org/search/articles?q={query}",
            "snippet": abstract[:500],
            "source": "DOAJ",
            "year": year,
            "authors": ", ".join(authors) if authors else "N/A",
            "journal": journal,
            "doi": doi if doi else "N/A",
            "access_type": "open_access",
            "has_full_text": True,
            "access_note": "Full text available (peer-reviewed open access journal)"
        }

        articles.append(article)

    logger.info("doaj_search_end", articles_count=len(articles), query=query)
    return articles


@RETRY_STRATEGY
async def search_doaj_journals(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search for open access journals (not articles) in DOAJ.

    Args:
        query: Search query (journal name or subject)
        max_results: Maximum number of results (default: 5)

    Returns:
        List of dictionaries containing journal information
    """
    base_url = "https://doaj.org/api/search/journals"
    params = {"q": query, "pageSize": max_results, "page": 1}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(base_url, params=params, headers=_HEADERS)
            response.raise_for_status()
            data = response.json()

        journals = []
        for item in data.get("results", []):
            bibjson = item.get("bibjson", {})
            journals.append({
                "title": bibjson.get("title", "Untitled"),
                "publisher": bibjson.get("publisher", {}).get("name", "N/A"),
                "subjects": ", ".join([s.get("term", "") for s in bibjson.get("subject", [])[:3]]),
                "url": bibjson.get("ref", {}).get("journal", ""),
                "source": "DOAJ"
            })

        logger.info("doaj_journals_search_end", journals_count=len(journals), query=query)
        return journals

    except Exception as e:
        logger.error("doaj_journals_search_error", detail=str(e))
        return []
