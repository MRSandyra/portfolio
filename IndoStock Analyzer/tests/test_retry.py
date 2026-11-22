import pytest

from indostock_analyzer.retry import RateLimitError, RetryPolicy, retry_call


def test_rate_limit_retry_uses_retry_after_delay():
    attempts = 0
    delays = []

    def operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RateLimitError("HTTP 429", retry_after=3.0)
        return "ok"

    result = retry_call(operation, RetryPolicy(retries=1), sleep=delays.append)

    assert result == "ok"
    assert attempts == 2
    assert delays == [3.0]


def test_retry_stops_after_configured_attempts():
    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1
        raise TimeoutError("source timeout")

    try:
        retry_call(operation, RetryPolicy(retries=2, backoff_seconds=0), sleep=lambda _: None)
    except TimeoutError:
        pass
    else:
        raise AssertionError("TimeoutError was not propagated")

    assert attempts == 3


def test_non_retryable_error_is_propagated_immediately():
    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1
        raise ValueError("invalid source payload")

    with pytest.raises(ValueError):
        retry_call(
            operation,
            RetryPolicy(retries=5),
            should_retry=lambda exc: not isinstance(exc, ValueError),
            sleep=lambda _: None,
        )

    assert attempts == 1
