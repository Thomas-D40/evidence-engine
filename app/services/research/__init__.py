"""
Research Services - External API client wrappers.

This package contains API clients for various research databases:
- Medical/Biomedical: PubMed, Europe PMC
- Academic: ArXiv, Semantic Scholar, CrossRef, CORE, DOAJ
- Statistical: OECD, World Bank
- News: NewsAPI, GNews
- Fact-Check: Google Fact Check, ClaimBuster

Note: These are NOT LLM-based agents - they are pure API clients.
Query generation is handled by app.agents.orchestration.query_generator.
"""
from .scientific import search_arxiv
from .statistical import search_world_bank_data
from .pubmed import search_pubmed
from .semantic_scholar import search_semantic_scholar
from .crossref import search_crossref
from .oecd import search_oecd_data
from .core import search_core
from .doaj import search_doaj
from .europepmc import search_europepmc
from .news import search_newsapi
from .gnews import search_gnews
from .factcheck import search_google_factcheck
from .claimbuster import search_claimbuster, score_claim_claimbuster

__all__ = [
    "search_arxiv",
    "search_world_bank_data",
    "search_pubmed",
    "search_semantic_scholar",
    "search_crossref",
    "search_oecd_data",
    "search_core",
    "search_doaj",
    "search_europepmc",
    "search_newsapi",
    "search_gnews",
    "search_google_factcheck",
    "search_claimbuster",
    "score_claim_claimbuster",
]
