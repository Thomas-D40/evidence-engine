"""
Full-text fetching agent using httpx async.

Retrieves complete content from academic PDFs and HTML pages
to enhance analysis quality beyond abstracts.
All sources are fetched concurrently via asyncio.gather().
"""
import asyncio
import logging
from typing import Optional, Dict, List

import httpx

from .common import (
    get_cached_content,
    save_to_cache,
    detect_source_type,
    truncate_content,
)

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

FULLTEXT_MAX_CHARS = 50_000
FULLTEXT_TRUNCATION_SUFFIX = "\n\n[... truncated ...]"
FULLTEXT_USER_AGENT = "EvidenceEngine/1.0 (Research Tool)"
FULLTEXT_DEFAULT_TIMEOUT = 30


# ============================================================================
# URL Resolution by Source Type
# ============================================================================

def _resolve_arxiv_url(source: Dict) -> Optional[str]:
    url = source.get("url", "")
    if "arxiv.org/abs/" in url:
        return url.replace("/abs/", "/pdf/") + ".pdf"
    return None


def _resolve_pubmed_url(source: Dict) -> Optional[str]:
    pmcid = source.get("pmcid")
    if pmcid:
        return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
    return None


def _resolve_semantic_scholar_url(source: Dict) -> Optional[str]:
    if source.get("access_type") == "open_access":
        return source.get("pdf_url") or source.get("open_access_pdf")
    return None


def _resolve_core_url(source: Dict) -> Optional[str]:
    return source.get("downloadUrl") or source.get("download_url")


def _resolve_doaj_url(source: Dict) -> Optional[str]:
    return source.get("fulltext_url") or source.get("link")


def determine_fetch_url(source: Dict, source_type: str = None) -> Optional[str]:
    """Determine the best URL to fetch full text from."""
    if source_type is None:
        source_type = detect_source_type(source)

    resolvers = {
        "arxiv":           _resolve_arxiv_url,
        "pubmed":          _resolve_pubmed_url,
        "europepmc":       _resolve_pubmed_url,
        "semantic_scholar": _resolve_semantic_scholar_url,
        "core":            _resolve_core_url,
        "doaj":            _resolve_doaj_url,
    }

    resolver = resolvers.get(source_type)
    if resolver:
        url = resolver(source)
        if url:
            return url

    # Fallback: try main URL if it looks like a PDF
    url = source.get("url", "")
    if url and (".pdf" in url.lower() or "/pdf/" in url.lower()):
        return url

    return None


# ============================================================================
# HTTP Fetch
# ============================================================================

async def _fetch_url(client: httpx.AsyncClient, url: str) -> Optional[Dict]:
    """Fetch a single URL using the provided httpx client."""
    try:
        response = await client.get(url, headers={"User-Agent": FULLTEXT_USER_AGENT})
        response.raise_for_status()
        content = response.text

        if len(content) > FULLTEXT_MAX_CHARS:
            content = content[:FULLTEXT_MAX_CHARS] + FULLTEXT_TRUNCATION_SUFFIX

        return {"content": content, "length": len(content), "format": "html"}

    except httpx.HTTPStatusError as e:
        logger.warning("fulltext_http_error", url=url[:70], status=e.response.status_code)
        return None
    except httpx.TimeoutException:
        logger.warning("fulltext_timeout", url=url[:70])
        return None
    except Exception as e:
        logger.error("fulltext_fetch_error", url=url[:70], detail=str(e))
        return None


# ============================================================================
# Source Enhancement
# ============================================================================

async def _enhance_single_source(
    client: httpx.AsyncClient,
    source: Dict,
    source_type: Optional[str]
) -> Dict:
    """Fetch full text and enhance a single source object."""
    if source_type is None:
        source_type = detect_source_type(source)

    fetch_url = determine_fetch_url(source, source_type)
    if not fetch_url:
        return source

    cached = get_cached_content(fetch_url)
    if cached:
        result = {"content": cached, "length": len(cached), "format": "cached"}
    else:
        result = await _fetch_url(client, fetch_url)
        if result and result.get("content"):
            save_to_cache(fetch_url, result["content"])

    if result and result.get("content"):
        source["fulltext"] = result["content"]
        source["fulltext_length"] = result["length"]
        source["fulltext_format"] = result["format"]
        source["access_type"] = "full_text_retrieved"
        source["has_full_text"] = True
        source["access_note"] = f"Full text retrieved ({result['length']} chars, {result['format']})"

    return source


async def fetch_fulltext_for_sources(
    sources: List[Dict],
    source_types: Optional[List[str]] = None
) -> List[Dict]:
    """
    Fetch full text for multiple sources concurrently.

    All sources are fetched in parallel using asyncio.gather().

    Args:
        sources: List of source dictionaries
        source_types: Optional list of source types (parallel to sources)

    Returns:
        List of enhanced sources
    """
    if not sources:
        return []

    logger.info("fulltext_batch_start", sources_count=len(sources))

    async with httpx.AsyncClient(timeout=FULLTEXT_DEFAULT_TIMEOUT, follow_redirects=True) as client:
        tasks = [
            _enhance_single_source(
                client,
                source,
                source_types[i] if source_types and i < len(source_types) else None
            )
            for i, source in enumerate(sources)
        ]
        enhanced = await asyncio.gather(*tasks)

    success_count = sum(1 for s in enhanced if "fulltext" in s)
    logger.info("fulltext_batch_end", retrieved=success_count, requested=len(sources))

    return list(enhanced)
