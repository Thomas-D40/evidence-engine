"""
Research agent for CrossRef.

CrossRef provides metadata for academic publications via DOI,
including citations, funding, licenses and bibliographic data.

API Documentation: https://www.crossref.org/documentation/retrieve-metadata/rest-api/
Rate Limits: Variable based on "polite" usage (with contact email)
"""
import re
from typing import List, Dict, Optional

import httpx

from ...logger import get_logger
from ..retry import RETRY_STRATEGY

logger = get_logger(__name__)

# "Polite" headers for better rate limits
_HEADERS = {"User-Agent": "VideoAnalyzerWorkflow/1.0 (mailto:research@example.com)"}


@RETRY_STRATEGY
async def search_crossref(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search for academic publications via CrossRef.

    CrossRef is useful for obtaining complete metadata and citation
    information to validate source credibility.

    Args:
        query: Search query (ideally in English)
        max_results: Maximum number of results (default: 5)

    Returns:
        List of dictionaries containing:
        - title: Publication title
        - url: URL to the publication (via DOI)
        - snippet: Abstract/summary
        - source: "CrossRef"
        - doi: Publication DOI
        - type: Publication type (journal-article, book-chapter, etc.)
        - year: Publication year
        - citations: Number of citations
        - publisher: Publisher
        - authors: List of authors

    Raises:
        httpx.HTTPStatusError: On HTTP 5xx / 429 after retries
        httpx.TimeoutException: On timeout after retries
    """
    base_url = "https://api.crossref.org/works"
    params = {
        "query": query,
        "rows": max_results,
        "select": "DOI,title,author,published,abstract,type,publisher,is-referenced-by-count,URL"
    }

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(base_url, params=params, headers=_HEADERS)
        response.raise_for_status()
        data = response.json()

    if "message" not in data or "items" not in data["message"]:
        logger.debug("crossref_no_results", query=query)
        return []

    articles = []
    for item in data["message"]["items"]:
        title_list = item.get("title", [])
        title = title_list[0] if title_list else "Untitled"

        doi = item.get("DOI", "")
        url = item.get("URL", f"https://doi.org/{doi}" if doi else "")

        abstract = item.get("abstract", "")
        if not abstract:
            subtitle = item.get("subtitle", [])
            if subtitle:
                abstract = subtitle[0]
            else:
                abstract = f"{item.get('type', 'Publication')} from {item.get('publisher', 'N/A')}"

        # Clean HTML tags from abstract (CrossRef sometimes includes them)
        if abstract:
            abstract = re.sub(r'<[^>]+>', '', abstract)

        published = item.get("published", {})
        if "date-parts" in published and published["date-parts"]:
            year = published["date-parts"][0][0] if published["date-parts"][0] else "N/A"
        else:
            year = "N/A"

        authors_list = item.get("author", [])
        authors = []
        for author in authors_list[:3]:
            given = author.get("given", "")
            family = author.get("family", "")
            if family:
                name = f"{given} {family}".strip() if given else family
                authors.append(name)

        citations = item.get("is-referenced-by-count", 0)

        licenses = item.get("license", [])
        is_open_license = any(
            "creativecommons" in lic.get("URL", "").lower() or "cc-by" in lic.get("URL", "").lower()
            for lic in licenses
        )

        if is_open_license:
            access_type = "open_access"
            has_full_text = True
            access_note = "Open access via Creative Commons license"
        else:
            access_type = "metadata_only"
            has_full_text = False
            access_note = "Metadata and abstract only - full text may require subscription"

        article = {
            "title": title,
            "url": url,
            "snippet": abstract[:500],
            "source": "CrossRef",
            "doi": doi,
            "type": item.get("type", "N/A"),
            "year": str(year),
            "citations": citations,
            "publisher": item.get("publisher", "N/A"),
            "authors": ", ".join(authors) if authors else "N/A",
            "access_type": access_type,
            "has_full_text": has_full_text,
            "access_note": access_note
        }

        articles.append(article)

    logger.info("crossref_search_end", articles_count=len(articles), query=query)
    return articles


@RETRY_STRATEGY
async def get_citation_count(doi: str) -> Optional[int]:
    """
    Retrieve the citation count for a given DOI.

    Args:
        doi: Publication DOI

    Returns:
        Number of citations or None if error
    """
    url = f"https://api.crossref.org/works/{doi}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url, headers=_HEADERS)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("is-referenced-by-count", 0)
    except Exception as e:
        logger.error("crossref_citations_error", doi=doi, detail=str(e))
        return None
