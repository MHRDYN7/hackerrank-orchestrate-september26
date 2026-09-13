from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class UsageTracker:
    provider: str = "Google"
    model: str = "gemini-3.5-flash-lite"
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    per_request: dict[str, dict] = field(default_factory=dict)

    def record(self, request_id: str, in_tok: int, out_tok: int, calls: int = 1) -> None:
        self.calls += calls
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        bucket = self.per_request.setdefault(request_id, {"calls": 0, "input": 0, "output": 0})
        bucket["calls"] += calls
        bucket["input"] += in_tok
        bucket["output"] += out_tok

    def write_report(self, path, n_requests: int, estimated_rate_per_m: float = 0.10) -> None:
        total = self.input_tokens + self.output_tokens
        avg = total / n_requests if n_requests else 0
        # Flash-Lite is cheap; keep a conservative placeholder rate.
        cost = (total / 1_000_000.0) * estimated_rate_per_m
        per_req_cost = cost / n_requests if n_requests else 0
        lines = [
            "# Token usage report",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "## Models",
            "",
            f"- Provider: {self.provider}",
            f"- Model: {self.model}",
            "",
            "## Final full-dataset run",
            "",
            f"- Requests: {n_requests}",
            f"- Model calls: {self.calls}",
            f"- Input tokens: {self.input_tokens}",
            f"- Output tokens: {self.output_tokens}",
            f"- Total tokens: {total}",
            f"- Average tokens per request: {avg:.2f}",
            f"- Estimated total cost (USD): {cost:.6f}",
            f"- Estimated cost per request (USD): {per_req_cost:.6f}",
            "",
            "If no API keys were configured, this run used the deterministic engine only and model calls are zero.",
            "",
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")


TRACKER = UsageTracker()
