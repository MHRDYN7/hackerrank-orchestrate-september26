from __future__ import annotations

from collections import defaultdict

from .db import write_sqlite
from .engine import build_series, normalize_events
from .fx import FxBook
from .images import amounts_by_event, load_image_amounts, save_image_amounts
from .load import load_raw
from .messages import parse_user_messages
from .paths import DATA_DIR
from .store import Store


def build_store() -> Store:
    raw = load_raw()
    fx = FxBook(raw["rates"])
    image_meta = load_image_amounts()
    save_image_amounts(image_meta)
    image_by_event = amounts_by_event(raw["images"])

    profiles = {r["user_id"]: r for r in raw["profiles"]}
    requests = {r["request_id"]: r for r in raw["requests"]}
    samples = {r["request_id"]: r for r in raw["samples"]}
    options: dict[str, list] = defaultdict(list)
    for row in raw["options"]:
        options[row["request_id"]].append(row)

    events_by_user: dict[str, list] = defaultdict(list)
    for row in raw["events"]:
        events_by_user[row["user_id"]].append(row)

    messages_by_user: dict[str, list] = defaultdict(list)
    for row in raw["messages"]:
        messages_by_user[row["user_id"]].append(row)

    normalized: dict[str, list] = {}
    series_map = {}
    amendments = {}
    for user_id, rows in events_by_user.items():
        home = profiles[user_id]["home_currency"]
        norm = normalize_events(rows, home, fx, image_by_event)
        normalized[user_id] = norm
        series_map[user_id] = build_series(norm, user_id)
        amendments[user_id] = parse_user_messages(messages_by_user.get(user_id, []))

    all_ids = [r["request_id"] for r in raw["requests"]]
    store = Store(
        profiles=profiles,
        requests=requests,
        samples=samples,
        options=dict(options),
        events=normalized,
        messages=dict(messages_by_user),
        amendments=amendments,
        series=series_map,
        image_meta=image_meta,
        fx=fx,
        all_request_ids=all_ids,
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_sqlite(store)
    return store
