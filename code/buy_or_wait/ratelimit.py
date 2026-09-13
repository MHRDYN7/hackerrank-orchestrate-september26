from __future__ import annotations

import time
from collections import deque


class GeminiPacer:
    """Stay under Flash-Lite free-tier 15 RPM and 250k TPM."""

    def __init__(self, rpm: int = 12, tpm: int = 200_000):
        self.min_interval = 60.0 / max(1, rpm)
        self.tpm = tpm
        self._last = 0.0
        self._tokens: deque[tuple[float, int]] = deque()

    def wait(self, upcoming_tokens: int = 0) -> None:
        now = time.monotonic()
        gap = self.min_interval - (now - self._last)
        if gap > 0:
            time.sleep(gap)
        cutoff = time.monotonic() - 60.0
        while self._tokens and self._tokens[0][0] < cutoff:
            self._tokens.popleft()
        used = sum(n for _, n in self._tokens)
        if used + upcoming_tokens > self.tpm:
            oldest = self._tokens[0][0] if self._tokens else time.monotonic()
            time.sleep(max(0.0, 60.0 - (time.monotonic() - oldest) + 0.5))
        self._last = time.monotonic()

    def record(self, tokens: int) -> None:
        self._tokens.append((time.monotonic(), max(0, tokens)))


PACER = GeminiPacer()


def is_rpm_error(exc: Exception) -> bool:
    text = str(exc).lower()
    if "per day" in text or "perday" in text or "daily" in text:
        return False
    return any(
        tok in text
        for tok in (
            "429",
            "rate limit",
            "ratelimit",
            "per minute",
            "perminute",
            "resource exhausted",
            "resource_exhausted",
            "quota",
        )
    )


def is_rpd_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(tok in text for tok in ("per day", "perday", "daily", "rpd"))
