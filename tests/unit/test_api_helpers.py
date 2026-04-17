"""
Unit tests for API utility helpers.
Covers CircuitBreaker, RateLimiter, retry_with_backoff, and safe_api_call.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from app.utils.api_helpers import (
    CircuitBreaker,
    RateLimiter,
    retry_with_backoff,
    safe_api_call,
    TransientAPIError,
    PermanentAPIError,
)


class TestCircuitBreaker:
    def test_initial_state_is_closed(self):
        cb = CircuitBreaker(failure_threshold=5)
        assert cb.state == "closed"

    def test_failure_below_threshold_stays_closed(self):
        cb = CircuitBreaker(failure_threshold=5)
        failing = MagicMock(side_effect=Exception("err"))
        for _ in range(4):
            try:
                cb.call(failing)
            except Exception:
                pass
        assert cb.state == "closed"

    def test_failure_at_threshold_opens_circuit(self):
        cb = CircuitBreaker(failure_threshold=5)
        failing = MagicMock(side_effect=Exception("err"))
        for _ in range(5):
            try:
                cb.call(failing)
            except Exception:
                pass
        assert cb.state == "open"

    def test_open_circuit_raises_permanent_error(self):
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        cb.state = "open"
        cb.last_failure_time = datetime.now()
        # Use a named function so func.__name__ is available in log messages
        def ok_func():
            return "ok"
        with pytest.raises(PermanentAPIError):
            cb.call(ok_func)

    def test_recovery_timeout_transitions_to_half_open(self):
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        cb.state = "open"
        cb.last_failure_time = datetime.now() - timedelta(seconds=61)
        def ok_func():
            return "ok"
        result = cb.call(ok_func)
        # After a successful call in "half-open", circuit closes
        assert cb.state == "closed"
        assert result == "ok"

    def test_success_in_half_open_closes_circuit(self):
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        cb.state = "half-open"
        cb.failure_count = 3
        def ok_func():
            return "result"
        result = cb.call(ok_func)
        assert cb.state == "closed"
        assert cb.failure_count == 0
        assert result == "result"


class TestRateLimiter:
    def test_first_call_no_sleep(self):
        rl = RateLimiter(calls_per_second=1.0)
        with patch("time.sleep") as mock_sleep:
            rl.wait_if_needed()
        mock_sleep.assert_not_called()

    def test_second_call_within_interval_sleeps(self):
        rl = RateLimiter(calls_per_second=1.0)
        # Simulate a very recent last call
        import time
        rl.last_call_time = time.time()
        with patch("time.sleep") as mock_sleep:
            rl.wait_if_needed()
        mock_sleep.assert_called_once()
        sleep_duration = mock_sleep.call_args[0][0]
        assert sleep_duration > 0


class TestRetryWithBackoff:
    def test_success_on_first_attempt(self):
        call_count = [0]

        @retry_with_backoff(max_attempts=3, base_delay=0.01)
        def succeed():
            call_count[0] += 1
            return "ok"

        with patch("time.sleep"):
            result = succeed()
        assert result == "ok"
        assert call_count[0] == 1

    def test_success_after_two_failures(self):
        call_count = [0]

        @retry_with_backoff(max_attempts=3, base_delay=0.01)
        def fail_twice():
            call_count[0] += 1
            if call_count[0] < 3:
                raise TransientAPIError("transient")
            return "ok"

        with patch("time.sleep") as mock_sleep:
            result = fail_twice()
        assert result == "ok"
        assert call_count[0] == 3
        assert mock_sleep.call_count == 2

    def test_exceeds_max_attempts_raises(self):
        @retry_with_backoff(max_attempts=3, base_delay=0.01)
        def always_fail():
            raise TransientAPIError("always")

        with patch("time.sleep"):
            with pytest.raises(TransientAPIError):
                always_fail()

    def test_permanent_error_not_retried(self):
        call_count = [0]

        @retry_with_backoff(max_attempts=3, base_delay=0.01)
        def permanent_fail():
            call_count[0] += 1
            raise PermanentAPIError("permanent")

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(PermanentAPIError):
                permanent_fail()
        assert call_count[0] == 1
        mock_sleep.assert_not_called()


class TestSafeApiCall:
    def test_safe_api_call_success(self):
        result = safe_api_call(lambda: 42)
        assert result == 42

    def test_safe_api_call_exception_returns_default(self):
        def fail():
            raise ValueError("boom")

        result = safe_api_call(fail, default="fallback")
        assert result == "fallback"
