from __future__ import annotations

import re


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
