"""
Retry and Backoff Configuration Constants.
"""

DEFAULT_RETRY_MAX_ATTEMPTS = 3
"""Default maximum number of retry attempts."""

DEFAULT_RETRY_BASE_DELAY = 1.0
"""Default base delay between retries (seconds)."""

DEFAULT_RETRY_BACKOFF_FACTOR = 2.0
"""Default exponential backoff multiplier."""

DEFAULT_RETRY_MAX_DELAY = 60.0
"""Default maximum delay between retries (seconds)."""

QUERY_GENERATOR_MAX_RETRY_ATTEMPTS = 2
"""Maximum retry attempts for query generation."""

QUERY_GENERATOR_BASE_DELAY = 1.0
"""Base delay for query generation retries (seconds)."""

WORLD_BANK_MAX_RETRY_ATTEMPTS = 3
"""Maximum retry attempts for World Bank search."""

WORLD_BANK_BASE_DELAY = 1.0
"""Base delay for World Bank search retries (seconds)."""

WORLD_BANK_FETCH_MAX_RETRY_ATTEMPTS = 2
"""Maximum retry attempts for World Bank data fetch."""

WORLD_BANK_FETCH_BASE_DELAY = 1.0
"""Base delay for World Bank fetch retries (seconds)."""
