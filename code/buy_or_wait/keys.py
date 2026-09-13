from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from .paths import ENV_PATHS

FLASH_LITE_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]


@dataclass
class KeyRing:
    keys: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=lambda: list(FLASH_LITE_MODELS))
    key_index: int = 0
    model_index: int = 0

    @property
    def model(self) -> str:
        return self.models[min(self.model_index, len(self.models) - 1)]

    @model.setter
    def model(self, value: str) -> None:
        name = (value or "").strip()
        if name and "flash-lite" not in name:
            return
        if name in self.models:
            self.model_index = self.models.index(name)
        elif name:
            self.models = [name] + [m for m in self.models if m != name]
            self.model_index = 0

    def current(self) -> str | None:
        if not self.keys:
            return None
        return self.keys[self.key_index % len(self.keys)]

    def rotate(self) -> str | None:
        if not self.keys:
            return None
        self.key_index = (self.key_index + 1) % len(self.keys)
        return self.current()

    def note_quota(self) -> str:
        """3.5 Flash-Lite RPD exhausted -> 3.1 Flash-Lite, same key. Then next key or stop."""
        if self.model_index < len(self.models) - 1:
            self.model_index += 1
            return f"switched_model:{self.model}"
        if len(self.keys) > 1:
            self.rotate()
            self.model_index = 0
            return f"rotated_key:{self.model}"
        return "exhausted"

    def has_keys(self) -> bool:
        return bool(self.keys)


def load_env() -> None:
    loaded = False
    for path in ENV_PATHS:
        if path.exists():
            load_dotenv(path, override=True)
            loaded = True
    if not loaded:
        load_dotenv()
    # LangGraph/LangChain pick up either name.
    if os.getenv("LANGSMITH_TRACING", "").strip().lower() in {"true", "1", "yes"}:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    project = os.getenv("LANGSMITH_PROJECT", "").strip().strip('"')
    if project:
        os.environ["LANGSMITH_PROJECT"] = project
        os.environ.setdefault("LANGCHAIN_PROJECT", project)
    endpoint = os.getenv("LANGSMITH_ENDPOINT", "").strip()
    if endpoint:
        os.environ.setdefault("LANGCHAIN_ENDPOINT", endpoint)


def load_key_ring() -> KeyRing:
    load_env()
    keys: list[str] = []
    for i in range(1, 8):
        val = os.getenv(f"GEMINI_API_KEY_{i}", "").strip()
        if val:
            keys.append(val)
    single = os.getenv("GEMINI_API_KEY", "").strip()
    if single and single not in keys:
        keys.append(single)
    preferred = os.getenv("GEMINI_MODEL", "").strip()
    ring = KeyRing(keys=keys)
    if preferred:
        ring.model = preferred
    return ring


def is_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        tok in text
        for tok in (
            "429",
            "resource exhausted",
            "resource_exhausted",
            "quota",
            "rate limit",
            "ratelimit",
        )
    )
