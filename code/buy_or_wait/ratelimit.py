from __future__ import annotations

import threading
import time
from collections import deque


class GeminiPacer:
    """Stay near Flash-Lite free-tier 15 RPM and 250k TPM. Thread-safe for batched runs."""

    def __init__(self, rpm: int = 12, tpm: int = 200_000):
        self.min_interval = 60.0 / max(1, rpm)
        self.tpm = tpm
        self._last = 0.0
        self._tokens: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def wait(self, upcoming_tokens: int = 0) -> None:
        with self._lock:
            now = time.monotonic()
            gap = self.min_interval - (now - self._last)
            sleep_for = gap if gap > 0 else 0.0
            cutoff = time.monotonic() - 60.0
            while self._tokens and self._tokens[0][0] < cutoff:
                self._tokens.popleft()
            used = sum(n for _, n in self._tokens)
            tpm_sleep = 0.0
            if used + upcoming_tokens > self.tpm:
                oldest = self._tokens[0][0] if self._tokens else time.monotonic()
                tpm_sleep = max(0.0, 60.0 - (time.monotonic() - oldest) + 0.5)
            self._last = time.monotonic() + max(sleep_for, tpm_sleep)
        delay = max(sleep_for, tpm_sleep)
        if delay > 0:
            time.sleep(delay)

    def record(self, tokens: int) -> None:
        with self._lock:
            self._tokens.append((time.monotonic(), max(0, tokens)))


PACER = GeminiPacer()


def is_rpm_error(exc: Exception) -> bool:
    text = str(exc).lower()
    if is_rpd_error(exc):
        return False
    return any(
        tok in text
        for tok in (
            "429",
            "rate limit",
            "ratelimit",
            "per minute",
            "perminute",
            "per_minute",
            "resource exhausted",
            "resource_exhausted",
            "quota",
            "failed_precondition",
            "unavailable",
            "deadline exceeded",
        )
    )


def is_rpd_error(exc: Exception) -> bool:
    text = str(exc).lower()
    if any(tok in text for tok in ("per minute", "perminute", "per_minute")):
        return False
    return any(
        tok in text
        for tok in (
            "per day",
            "perday",
            "per_day",
            "daily",
            "rpd",
            "requestsperday",
            "generate_content_free_tier_requests",
            "limit: 0",
            "quota exceeded",
            "resource exhausted",
            "resource_exhausted",
        )
    )
