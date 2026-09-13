from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from .paths import ENV_PATH


@dataclass
class KeyRing:
    keys: list[str] = field(default_factory=list)
    model: str = "gemini-3.5-flash-lite"
    index: int = 0

    def current(self) -> str | None:
        if not self.keys:
            return None
        return self.keys[self.index % len(self.keys)]

    def rotate(self) -> str | None:
        if not self.keys:
            return None
        self.index = (self.index + 1) % len(self.keys)
        return self.current()

    def has_keys(self) -> bool:
        return bool(self.keys)


def load_key_ring() -> KeyRing:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    else:
        load_dotenv()
    keys: list[str] = []
    for i in range(1, 8):
        val = os.getenv(f"GEMINI_API_KEY_{i}", "").strip()
        if val:
            keys.append(val)
    single = os.getenv("GEMINI_API_KEY", "").strip()
    if single and single not in keys:
        keys.append(single)
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
    return KeyRing(keys=keys, model=model)
