from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field

from dotenv import load_dotenv

from .paths import ENV_PATHS

FLASH_LITE_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]


@dataclass
class KeySlot:
    key: str
    model_index: int = 0
    exhausted: bool = False


@dataclass
class KeyRing:
    keys: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=lambda: list(FLASH_LITE_MODELS))
    slots: list[KeySlot] = field(default_factory=list)
    rr: int = 0
    last_key: str | None = None
    last_model: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.keys and not self.slots:
            self.slots = [KeySlot(key) for key in self.keys]

    @property
    def model(self) -> str:
        with self._lock:
            if self.last_model:
                return self.last_model
            live = [s for s in self.slots if not s.exhausted]
            slot = live[0] if live else (self.slots[0] if self.slots else None)
            if slot is None:
                return self.models[0]
            return self.models[min(slot.model_index, len(self.models) - 1)]

    @model.setter
    def model(self, value: str) -> None:
        name = (value or "").strip()
        if name and "flash-lite" not in name:
            return
        with self._lock:
            if name in self.models:
                idx = self.models.index(name)
            elif name:
                self.models = [name] + [m for m in self.models if m != name]
                idx = 0
            else:
                return
            for slot in self.slots:
                if not slot.exhausted:
                    slot.model_index = idx

    def current(self) -> str | None:
        key, _model = self.checkout()
        return key

    def checkout(self) -> tuple[str | None, str]:
        """Round-robin a live key so several free-tier keys share RPM."""
        with self._lock:
            live = [s for s in self.slots if not s.exhausted]
            if not live:
                return None, self.models[0]
            slot = live[self.rr % len(live)]
            self.rr += 1
            model = self.models[min(slot.model_index, len(self.models) - 1)]
            self.last_key = slot.key
            self.last_model = model
            return slot.key, model

    def rotate(self) -> str | None:
        key, _model = self.checkout()
        return key

    def ingest_keys(self, keys: list[str]) -> int:
        added = 0
        with self._lock:
            have = {s.key for s in self.slots}
            for key in keys:
                if key and key not in have:
                    self.keys.append(key)
                    self.slots.append(KeySlot(key))
                    have.add(key)
                    added += 1
        return added

    def live_count(self) -> int:
        with self._lock:
            return sum(1 for s in self.slots if not s.exhausted)

    def note_quota(self, failed_model: str | None = None, failed_key: str | None = None) -> str:
        """Per-key: 3.5 Flash-Lite RPD -> 3.1 Flash-Lite, then retire that key."""
        failed = (failed_model or "").strip()
        failed_key = failed_key or self.last_key
        with self._lock:
            slot = next((s for s in self.slots if s.key == failed_key), None)
            if slot is None:
                live = [s for s in self.slots if not s.exhausted]
                return f"already_on:{self.models[0]}" if live else "exhausted"
            current = self.models[min(slot.model_index, len(self.models) - 1)]
            if failed and failed != current:
                return f"already_on:{current}"
            if slot.model_index < len(self.models) - 1:
                slot.model_index += 1
                return f"switched_model:{self.models[slot.model_index]}"
            slot.exhausted = True
            live = [s for s in self.slots if not s.exhausted]
            if live:
                return f"retired_key remaining={len(live)}"
            return "exhausted"

    def activate_new_keys(self, keys: list[str]) -> str | None:
        added = self.ingest_keys(keys)
        if not added:
            return None
        return f"new_keys:{added}"

    def has_keys(self) -> bool:
        with self._lock:
            return any(not s.exhausted for s in self.slots)


LANGSMITH_PROJECT_NAME = "hackerrank"


def load_env() -> None:
    loaded = False
    for path in ENV_PATHS:
        if path.exists():
            load_dotenv(path, override=True)
            loaded = True
    if not loaded:
        load_dotenv()
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
    """Prefer GOOGLE_API_KEY_2..N (fresh) before the base GOOGLE_API_KEY (often exhausted)."""
    keys: list[str] = []
    seen: set[str] = set()

    def add(val: str | None) -> None:
        text = (val or "").strip()
        if text and text not in seen:
            seen.add(text)
            keys.append(text)

    for i in range(2, 8):
        add(os.getenv(f"GOOGLE_API_KEY_{i}"))
        add(os.getenv(f"GEMINI_API_KEY_{i}"))
    add(os.getenv("GOOGLE_API_KEY_1"))
    add(os.getenv("GEMINI_API_KEY_1"))
    add(os.getenv("GOOGLE_API_KEY"))
    add(os.getenv("GEMINI_API_KEY"))
    add(os.getenv("GOOGLE_GENERATIVE_AI_API_KEY"))
    return keys


def load_key_ring() -> KeyRing:
    load_env()
    keys = collect_keys()
    ring = KeyRing(keys=keys)
    # Leftover GEMINI_MODEL=gemini-3.1-flash-lite from an exhausted-key run must not
    # skip 3.5 on fresh keys.
    if len(keys) <= 1:
        preferred = os.getenv("GEMINI_MODEL", "").strip()
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
