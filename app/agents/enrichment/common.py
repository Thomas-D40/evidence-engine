"""
Common utilities for enrichment agents.

Shared functionality for screening and full-text fetching.
"""
import hashlib
import logging
import time
from pathlib import Path
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

# Cache directory for retrieved full texts
CACHE_DIR = Path(".cache/fulltexts")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Caching Utilities
# ============================================================================

def get_cache_key(url: str) -> str:
    """Generate cache key from URL hash."""
    return hashlib.md5(url.encode()).hexdigest()


def get_cached_content(url: str) -> Optional[str]:
    """Retrieve cached content by URL."""
    cache_file = CACHE_DIR / f"{get_cache_key(url)}.txt"
    if cache_file.exists():
        try:
            return cache_file.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"[Cache] Read error: {e}")
    return None


def save_to_cache(url: str, content: str) -> None:
    """Save content to cache."""
    cache_file = CACHE_DIR / f"{get_cache_key(url)}.txt"
    try:
        cache_file.write_text(content, encoding="utf-8")
    except Exception as e:
        logger.warning(f"[Cache] Write error: {e}")


def clear_cache(older_than_days: Optional[int] = None) -> int:
    """Clear cached content. Returns number of files cleared."""
    if not CACHE_DIR.exists():
        return 0

    cleared = 0
    now = time.time()

    for cache_file in CACHE_DIR.glob("*.txt"):
        try:
            if older_than_days is not None:
                file_age_days = (now - cache_file.stat().st_mtime) / 86400
                if file_age_days < older_than_days:
                    continue
            cache_file.unlink()
            cleared += 1
        except Exception as e:
            logger.warning(f"[Cache] Error clearing file: {e}")

    logger.info(f"[Cache] Cleared {cleared} files")
    return cleared


# ============================================================================
# Source Content Extraction
# ============================================================================

def extract_source_content(source: Dict, prefer_fulltext: bool = True) -> str:
    """
    Extract the best available content from a source.

    Priority: fulltext → snippet → abstract → summary → ""
    """
    if prefer_fulltext and "fulltext" in source:
        return source["fulltext"]

    for field in ["snippet", "abstract", "summary"]:
        if field in source and source[field]:
            return source[field]

    return ""


def truncate_content(content: str, max_length: int) -> str:
    """Truncate content to maximum length with ellipsis."""
    if len(content) <= max_length:
        return content
    return content[:max_length] + "..."


# ============================================================================
# Source Type Detection
# ============================================================================

def detect_source_type(source: Dict) -> str:
    """Detect source type from source dictionary."""
    source_name = source.get("source", "").lower()

    type_mapping = {
        "arxiv":           "arxiv",
        "pubmed":          "pubmed",
        "pmc":             "pubmed",
        "europe pmc":      "europepmc",
        "europepmc":       "europepmc",
        "semantic scholar": "semantic_scholar",
        "crossref":        "crossref",
        "core":            "core",
        "doaj":            "doaj",
        "oecd":            "oecd",
        "world bank":      "world_bank",
    }

    for key, value in type_mapping.items():
        if key in source_name:
            return value

    return "unknown"


# ============================================================================
# Batch Processing Helpers
# ============================================================================

def batch_items(items: List, batch_size: int) -> List[List]:
    """Split items into batches of the given size."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
