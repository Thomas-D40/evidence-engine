"""
Research agent for Europe PMC (Europe PubMed Central).

Europe PMC is a comprehensive life sciences literature database
providing access to biomedical and life sciences research, with better
full-text access than PubMed for European content.

API Documentation: https://europepmc.org/RestfulWebService
Rate Limits: Generous limits for reasonable usage
"""
from typing import List, Dict

import httpx

from ...logger import get_logger
from ..retry import RETRY_STRATEGY

logger = get_logger(__name__)

# Headers shared across all requests
_HEADERS = {"User-Agent": "VideoAnalyzerWorkflow/1.0 (Research Tool)"}


@RETRY_STRATEGY
async def search_europepmc(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search for biomedical and life sciences articles on Europe PMC.

    Europe PMC provides better full-text access than PubMed, especially
    for European research content, and includes preprints and other sources.

    Args:
        query: Search query (ideally in English)
        max_results: Maximum number of results (default: 5)

    Returns:
        List of dictionaries containing:
        - title: Article title
        - url: URL to the article
        - snippet: Article abstract/summary
        - source: "Europe PMC"
        - year: Publication year
        - authors: List of authors
        - journal: Journal name
        - pmid: PubMed ID if available
        - pmcid: PMC ID if available
        - doi: DOI if available
        - access_type: "open_access" or "abstract_only"
        - has_full_text: Boolean indicating full text availability
        - access_note: Full text availability note

    Raises:
        httpx.HTTPStatusError: On HTTP 5xx / 429 after retries
        httpx.TimeoutException: On timeout after retries
    """
    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        "query": query,
        "pageSize": max_results,
        "format": "json",
        "resultType": "core",
        "sort": "RELEVANCE"
    }

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(base_url, params=params, headers=_HEADERS)
        response.raise_for_status()
        data = response.json()

    result_list = data.get("resultList", {})
    results = result_list.get("result", [])

    if not results:
        logger.debug("europepmc_no_results", query=query)
        return []

    articles = []
    for item in results:
        title = item.get("title", "Untitled")
        abstract = item.get("abstractText", "")
        year = item.get("pubYear", "N/A")

        author_string = item.get("authorString", "")
        authors_list = [a.strip() for a in author_string.split(",")[:3]] if author_string else []

        journal = item.get("journalTitle", "N/A")
        if not journal or journal == "N/A":
            journal = item.get("bookOrReportDetails", {}).get("publisher", "N/A")

        pmid = item.get("pmid", "")
        pmcid = item.get("pmcid", "")
        doi = item.get("doi", "")

        # Build URL (prefer PMC, then DOI, then PubMed)
        url = ""
        if pmcid:
            url = f"https://europepmc.org/article/PMC/{pmcid.replace('PMC', '')}"
        elif doi:
            url = f"https://doi.org/{doi}"
        elif pmid:
            url = f"https://europepmc.org/article/MED/{pmid}"
        else:
            source = item.get("source", "")
            id_val = item.get("id", "")
            if source and id_val:
                url = f"https://europepmc.org/article/{source}/{id_val}"

        is_open_access = item.get("isOpenAccess", "") == "Y"
        has_pmc = bool(pmcid)
        in_epmc = item.get("inEPMC", "") == "Y"
        has_pdf = item.get("hasPDF", "") == "Y"

        if is_open_access or has_pmc or in_epmc:
            access_type = "open_access"
            has_full_text = True
            if has_pdf:
                access_note = "Full text PDF available (open access)"
            elif has_pmc:
                access_note = f"Full text available via PMC (ID: {pmcid})"
            else:
                access_note = "Full text HTML available (open access)"
        else:
            access_type = "abstract_only"
            has_full_text = False
            access_note = "Abstract only - full text may require subscription"

        if not abstract or len(abstract.strip()) < 20:
            abstract = f"Article from {journal} ({year})"

        article = {
            "title": title,
            "url": url if url else f"https://europepmc.org/search?query={query}",
            "snippet": abstract[:500],
            "source": "Europe PMC",
            "year": str(year),
            "authors": ", ".join(authors_list) if authors_list else "N/A",
            "journal": journal,
            "pmid": pmid if pmid else "N/A",
            "pmcid": pmcid if pmcid else "N/A",
            "doi": doi if doi else "N/A",
            "access_type": access_type,
            "has_full_text": has_full_text,
            "access_note": access_note
        }

        articles.append(article)

    logger.info("europepmc_search_end", articles_count=len(articles), query=query)
    return articles


@RETRY_STRATEGY
async def get_fulltext_xml(pmcid: str) -> str:
    """
    Retrieve full-text XML for an article from Europe PMC.

    Args:
        pmcid: PMC ID (with or without 'PMC' prefix)

    Returns:
        XML string of the full text, or empty string on error
    """
    clean_id = pmcid.replace("PMC", "")
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/PMC{clean_id}/fullTextXML"

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(url, headers=_HEADERS)
            response.raise_for_status()
            return response.text
    except Exception as e:
        logger.error("europepmc_fulltext_error", pmcid=pmcid, detail=str(e))
        return ""
