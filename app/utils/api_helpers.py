"""
Utility functions for API calls with retry logic and error handling.

Provides decorators and helpers for robust API interactions:
- Exponential backoff retry
- Circuit breaker pattern
- Request throttling
- Error categorization
"""
import time
import asyncio
from typing import Callable, Any, Optional, Dict, List
from functools import wraps
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base exception for API-related errors."""
    pass


class TransientAPIError(APIError):
    """Transient error that should be retried."""
    pass


class PermanentAPIError(APIError):
    """Permanent error that should not be retried."""
    pass


class RateLimitError(TransientAPIError):
    """Rate limit exceeded."""
    pass


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for API calls.

    Prevents repeated calls to a failing service by opening the circuit
    after a threshold of failures. Automatically attempts recovery after
    a cooldown period.
    """

    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open

    def call(self, func: Callable, *args, **kwargs) -> Any:
        if self.state == "open":
            if self.last_failure_time and \
               datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout):
                self.state = "half-open"
                logger.info(f"Circuit breaker entering half-open state for {func.__name__}")
            else:
                raise PermanentAPIError(f"Circuit breaker open for {func.__name__}")

        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                logger.info(f"Circuit breaker closing for {func.__name__}")
                self.state = "closed"
                self.failure_count = 0
            return result

        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.warning(f"Circuit breaker opening for {func.__name__} after {self.failure_count} failures")
            raise


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, calls_per_second: float = 1.0):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time: Optional[float] = None

    def wait_if_needed(self):
        if self.last_call_time is not None:
            elapsed = time.time() - self.last_call_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
        self.last_call_time = time.time()


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    exceptions: tuple = (TransientAPIError, ConnectionError, TimeoutError)
):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each failure
        max_delay: Maximum delay between retries
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}: {e}")

                except PermanentAPIError as e:
                    logger.error(f"Permanent error in {func.__name__}: {e}")
                    raise

            raise last_exception

        return wrapper
    return decorator


def safe_api_call(func: Callable, *args, default=None, error_message: str = None, **kwargs) -> Any:
    """Safely call an API function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        msg = error_message or f"Error calling {func.__name__}"
        logger.error(f"{msg}: {e}")
        return default


# Global circuit breakers for different services
circuit_breakers = {
    "oecd": CircuitBreaker(failure_threshold=5, recovery_timeout=300),
    "world_bank": CircuitBreaker(failure_threshold=5, recovery_timeout=300),
    "arxiv": CircuitBreaker(failure_threshold=5, recovery_timeout=180),
    "pubmed": CircuitBreaker(failure_threshold=5, recovery_timeout=180),
    "semantic_scholar": CircuitBreaker(failure_threshold=5, recovery_timeout=180),
}

# Global rate limiters for different services
rate_limiters = {
    "oecd": RateLimiter(calls_per_second=1.0),
    "world_bank": RateLimiter(calls_per_second=2.0),
    "arxiv": RateLimiter(calls_per_second=1.0),
    "pubmed": RateLimiter(calls_per_second=3.0),
    "semantic_scholar": RateLimiter(calls_per_second=1.0),
}
