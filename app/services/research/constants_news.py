"""
Constants for news and fact-checking API services.
"""

# ============================================================================
# NEWS API CONFIGURATION
# ============================================================================

# NewsAPI (newsapi.org)
NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"
NEWSAPI_DEFAULT_LANGUAGE = "en"
NEWSAPI_DEFAULT_SORT = "relevancy"
NEWSAPI_MAX_RESULTS_FREE = 100
NEWSAPI_MAX_DAYS_BACK_FREE = 30
NEWSAPI_TIMEOUT = 10

# GNews (gnews.io)
GNEWS_BASE_URL = "https://gnews.io/api/v4/search"
GNEWS_DEFAULT_LANGUAGE = "en"
GNEWS_DEFAULT_SORT = "relevance"
GNEWS_MAX_RESULTS_FREE = 10
GNEWS_TIMEOUT = 10

# ============================================================================
# FACT-CHECK API CONFIGURATION
# ============================================================================

# Google Fact Check API
GOOGLE_FACTCHECK_BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
GOOGLE_FACTCHECK_DEFAULT_LANGUAGE = "en"
GOOGLE_FACTCHECK_MAX_RESULTS = 10
GOOGLE_FACTCHECK_TIMEOUT = 10

# ClaimBuster API
CLAIMBUSTER_BASE_URL = "https://idir.uta.edu/claimbuster/api/v2"
CLAIMBUSTER_SCORE_ENDPOINT = "/score/text/"
CLAIMBUSTER_SEARCH_ENDPOINT = "/search/"
CLAIMBUSTER_MIN_SCORE = 0.5  # Minimum score to consider as checkworthy claim
CLAIMBUSTER_TIMEOUT = 10

# ============================================================================
# DEFAULTS
# ============================================================================

# Default number of results to return
DEFAULT_NEWS_MAX_RESULTS = 5
DEFAULT_FACTCHECK_MAX_RESULTS = 5

# Default time range for news searches (in days)
DEFAULT_NEWS_DAYS_BACK = 30
