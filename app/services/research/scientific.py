"""
Service de recherche ArXiv - API client wrapper.

Ce service recherche des publications scientifiques (pré-publications)
sur ArXiv. Il attend une requête optimisée générée par l'orchestrateur.

Note: The arxiv library uses synchronous HTTP internally; it is wrapped
with asyncio.to_thread() to avoid blocking the event loop.
"""
import asyncio
from typing import List, Dict

import arxiv

from ...logger import get_logger

logger = get_logger(__name__)


def _search_arxiv_sync(query: str, max_results: int) -> List[Dict[str, str]]:
    """Synchronous arxiv search — called via asyncio.to_thread()."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    articles = []
    for result in client.results(search):
        articles.append({
            "title": result.title,
            "summary": result.summary.replace("\n", " "),
            "authors": ", ".join([a.name for a in result.authors]),
            "url": result.entry_id,
            "published": result.published.strftime("%Y-%m-%d"),
            "source": "arxiv",
            "access_type": "open_access",
            "has_full_text": True,
            "access_note": "Full text available as PDF"
        })

    return articles


async def search_arxiv(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Recherche des articles scientifiques sur ArXiv.

    Args:
        query: Requête de recherche optimisée (en anglais, avec opérateurs AND/OR)
        max_results: Nombre maximum de résultats

    Returns:
        Liste d'articles avec titre, résumé, auteurs, url, date.
    """
    if not query or len(query.strip()) < 5:
        return []

    logger.info("arxiv_search_start", query_preview=query[:50])

    try:
        articles = await asyncio.to_thread(_search_arxiv_sync, query, max_results)
    except Exception as e:
        logger.error("arxiv_search_error", detail=str(e))
        return []

    return articles
