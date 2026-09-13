from __future__ import annotations

import os
import re
import threading


class GeminiPacer:
    """No preemptive RPM/TPM sleeps.

    Parallel workers used to stack 5s RPM gaps and 60s TPM waits onto a shared
    clock, which turned ~40s sequential thinking time into 200s+ wall time.
    Rate limits are handled by waiting on real 429 responses in graph.py.
    In-flight Gemini calls are capped so a two-request batch stays near one thinking pass.
    """

    def wait(self, upcoming_tokens: int = 0) -> None:
        return

    def record(self, tokens: int) -> None:
        return


PACER = GeminiPacer()

_SLOT: threading.BoundedSemaphore | None = None
_SLOT_LOCK = threading.Lock()


def max_inflight() -> int:
    try:
        n = int(os.getenv("GEMINI_MAX_INFLIGHT", "2"))
    except ValueError:
        n = 2
    return max(1, min(n, 4))


def gemini_slot() -> threading.BoundedSemaphore:
    global _SLOT
    with _SLOT_LOCK:
        if _SLOT is None:
            _SLOT = threading.BoundedSemaphore(max_inflight())
        return _SLOT


class GeminiPacer:
    """No preemptive RPM/TPM sleeps.

    Parallel workers used to stack 5s RPM gaps and 60s TPM waits onto a shared
    clock, which turned ~40s sequential thinking time into 200s+ wall time.
    Rate limits are handled by waiting on real 429 responses in graph.py.
    """

    def wait(self, upcoming_tokens: int = 0) -> None:
        return

    def record(self, tokens: int) -> None:
        return


PACER = GeminiPacer()

_MODEL_IN_ERROR = re.compile(r"Error calling model '([^']+)'")


def model_from_exc(exc: Exception) -> str:
    match = _MODEL_IN_ERROR.search(str(exc))
    return match.group(1) if match else ""


_RETRY_SECONDS = re.compile(r"retry_delay\s*\{\s*seconds:\s*(\d+)", re.I)
_RETRY_DELAY_S = re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", re.I)


def retry_seconds(exc: Exception, fallback: float) -> float:
    text = str(exc)
    match = _RETRY_SECONDS.search(text) or _RETRY_DELAY_S.search(text)
    if match:
        return max(1.0, float(match.group(1)) + 0.5)
    return max(1.0, float(fallback))


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
