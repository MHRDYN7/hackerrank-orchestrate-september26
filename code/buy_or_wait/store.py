from __future__ import annotations

from dataclasses import dataclass, field

from .engine import Series
from .fx import FxBook
from .messages import Amendment


@dataclass
class Store:
    profiles: dict[str, dict]
    requests: dict[str, dict]
    samples: dict[str, dict]
    options: dict[str, list[dict]]
    events: dict[str, list[dict]]
    messages: dict[str, list[dict]]
    amendments: dict[str, Amendment]
    series: dict[str, list[Series]]
    image_meta: dict[str, dict]
    fx: FxBook
    all_request_ids: list[str] = field(default_factory=list)
