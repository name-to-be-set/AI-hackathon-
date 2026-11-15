from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from threading import Condition
from time import monotonic


class RequestLimitExceeded(RuntimeError):
    """Raised when a session or the service exceeds its request limits."""


class FairRequestLimiter:
    def __init__(
        self,
        *,
        max_concurrent: int = 5,
        max_requests_per_minute: int = 60,
        max_requests_per_session_per_minute: int = 10,
        wait_timeout_seconds: float = 15,
    ) -> None:
        if min(
            max_concurrent,
            max_requests_per_minute,
            max_requests_per_session_per_minute,
        ) < 1:
            raise ValueError("Request limits must be positive.")
        if wait_timeout_seconds <= 0:
            raise ValueError("The queue timeout must be positive.")

        self.max_concurrent = max_concurrent
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_session_per_minute = (
            max_requests_per_session_per_minute
        )
        self.wait_timeout_seconds = wait_timeout_seconds
        self._condition = Condition()
        self._active_sessions: set[str] = set()
        self._waiters: deque[str] = deque()
        self._requests: deque[tuple[float, str]] = deque()

    def _remove_waiter(self, session_id: str) -> None:
        self._waiters.remove(session_id)
        self._condition.notify_all()

    def _prune_requests(self, now: float) -> None:
        cutoff = now - 60
        while self._requests and self._requests[0][0] <= cutoff:
            self._requests.popleft()

    def _acquire(self, session_id: str) -> None:
        deadline = monotonic() + self.wait_timeout_seconds
        with self._condition:
            if (
                session_id in self._active_sessions
                or session_id in self._waiters
            ):
                raise RequestLimitExceeded(
                    "A request from this chat is already in progress."
                )
            self._waiters.append(session_id)

            while True:
                now = monotonic()
                self._prune_requests(now)
                session_request_count = sum(
                    request_session == session_id
                    for _, request_session in self._requests
                )
                if (
                    session_request_count
                    >= self.max_requests_per_session_per_minute
                ):
                    self._remove_waiter(session_id)
                    raise RequestLimitExceeded(
                        "This chat has reached its short-term request limit. "
                        "Please wait a minute and try again."
                    )
                if len(self._requests) >= self.max_requests_per_minute:
                    self._remove_waiter(session_id)
                    raise RequestLimitExceeded(
                        "The assistant is handling many requests right now. "
                        "Please wait a minute and try again."
                    )
                if (
                    self._waiters[0] == session_id
                    and len(self._active_sessions) < self.max_concurrent
                ):
                    self._waiters.popleft()
                    self._active_sessions.add(session_id)
                    self._requests.append((now, session_id))
                    self._condition.notify_all()
                    return

                remaining = deadline - now
                if remaining <= 0:
                    self._remove_waiter(session_id)
                    raise RequestLimitExceeded(
                        "The assistant is busy. Please try again shortly."
                    )
                self._condition.wait(remaining)

    def _release(self, session_id: str) -> None:
        with self._condition:
            if session_id not in self._active_sessions:
                raise RuntimeError("Cannot release a request that is not active.")
            self._active_sessions.remove(session_id)
            self._condition.notify_all()

    @contextmanager
    def request_slot(self, session_id: str) -> Generator[None, None, None]:
        self._acquire(session_id)
        try:
            yield
        finally:
            self._release(session_id)


REQUEST_LIMITER = FairRequestLimiter()
