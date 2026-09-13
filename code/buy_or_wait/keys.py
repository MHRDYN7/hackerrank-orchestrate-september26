from __future__ import annotations

import os
import threading
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
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def model(self) -> str:
        with self._lock:
            return self.models[min(self.model_index, len(self.models) - 1)]

    @model.setter
    def model(self, value: str) -> None:
        name = (value or "").strip()
        if name and "flash-lite" not in name:
            return
        with self._lock:
            if name in self.models:
                self.model_index = self.models.index(name)
            elif name:
                self.models = [name] + [m for m in self.models if m != name]
                self.model_index = 0

    def current(self) -> str | None:
        with self._lock:
            if not self.keys:
                return None
            return self.keys[self.key_index % len(self.keys)]

    def rotate(self) -> str | None:
        with self._lock:
            if not self.keys:
                return None
            self.key_index = (self.key_index + 1) % len(self.keys)
            return self.keys[self.key_index % len(self.keys)]

    def ingest_keys(self, keys: list[str]) -> int:
        """Append newly provided keys without dropping the current one."""
        added = 0
        with self._lock:
            for key in keys:
                if key and key not in self.keys:
                    self.keys.append(key)
                    added += 1
        return added

    def note_quota(self) -> str:
        """3.5 Flash-Lite RPD exhausted -> 3.1 Flash-Lite, same key. Then next key's 3.5."""
        with self._lock:
            if self.model_index < len(self.models) - 1:
                self.model_index += 1
                return f"switched_model:{self.models[self.model_index]}"
            if len(self.keys) > 1:
                self.key_index = (self.key_index + 1) % len(self.keys)
                self.model_index = 0
                return f"rotated_key:{self.models[0]}"
            return "exhausted"

    def activate_new_keys(self, keys: list[str]) -> str | None:
        """If the user added keys to the environment, switch to the newest one on 3.5."""
        added = self.ingest_keys(keys)
        if not added:
            return None
        with self._lock:
            self.key_index = len(self.keys) - 1
            self.model_index = 0
            return f"new_key:{self.models[0]}"

    def has_keys(self) -> bool:
        with self._lock:
            return bool(self.keys)


LANGSMITH_PROJECT_NAME = "hackerrank"


def load_env() -> None:
    loaded = False
    for path in ENV_PATHS:
        if path.exists():
            load_dotenv(path, override=True)
            loaded = True
    if not loaded:
        load_dotenv()
    # Traces MUST go to the contest LangSmith project, never a default workspace name.
    os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT_NAME
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT_NAME
    tracing = os.getenv("LANGSMITH_TRACING", "").strip().lower()
    if tracing in {"true", "1", "yes"}:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_TRACING"] = "true"
    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    if api_key:
        os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
    endpoint = os.getenv("LANGSMITH_ENDPOINT", "").strip()
    if endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint


def collect_keys() -> list[str]:
    keys: list[str] = []
    for i in range(1, 8):
        val = os.getenv(f"GEMINI_API_KEY_{i}", "").strip()
        if val:
            keys.append(val)
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY"):
        single = os.getenv(name, "").strip()
        if single and single not in keys:
            keys.append(single)
    return keys


def load_key_ring() -> KeyRing:
    load_env()
    keys = collect_keys()
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
