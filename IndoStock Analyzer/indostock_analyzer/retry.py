from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Callable, TypeVar

from .logging_utils import log_event


T = TypeVar("T")
LOGGER = logging.getLogger(__name__)


class RateLimitError(RuntimeError):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass(frozen=True)
class RetryPolicy:
    retries: int = 2
    backoff_seconds: float = 1.0


def retry_call(
    operation: Callable[[], T],
    policy: RetryPolicy,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    should_retry: Callable[[BaseException], bool] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    operation_name: str = "external_operation",
) -> T:
    """Retry an operation with bounded exponential backoff."""
    last_error: BaseException | None = None
    for attempt in range(policy.retries + 1):
        try:
            return operation()
        except retry_on as exc:
            last_error = exc
            if (should_retry is not None and not should_retry(exc)) or attempt >= policy.retries:
                raise
            retry_after = getattr(exc, "retry_after", None)
            delay = retry_after if retry_after is not None else policy.backoff_seconds * (2**attempt)
            log_event(
                LOGGER,
                logging.WARNING,
                "source_retry_scheduled",
                operation=operation_name,
                attempt=attempt + 1,
                max_attempts=policy.retries + 1,
                delay_seconds=max(0.0, delay),
                error_type=type(exc).__name__,
            )
            sleep(max(0.0, delay))
    assert last_error is not None
    raise last_error


class RateLimiter:
    def __init__(
        self,
        min_interval_seconds: float,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self._monotonic = monotonic
        self._sleep = sleep
        self._last_request: float | None = None

    def wait(self) -> None:
        now = self._monotonic()
        if self._last_request is not None:
            remaining = self.min_interval_seconds - (now - self._last_request)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request = self._monotonic()
