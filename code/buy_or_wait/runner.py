from __future__ import annotations

from .engine import Decision, decision_row, evaluate_state, make_state
from .messages import Amendment
from .store import Store


def request_record(store: Store, request_id: str) -> dict:
    if request_id in store.requests:
        return store.requests[request_id]
    if request_id in store.samples:
        return store.samples[request_id]
    raise KeyError(request_id)


def decide(store: Store, request_id: str) -> Decision:
    req = request_record(store, request_id)
    user_id = req["user_id"]
    state = make_state(
        profile=store.profiles[user_id],
        request=req,
        options=store.options.get(request_id, []),
        events=store.events.get(user_id, []),
        amendment=store.amendments.get(user_id) or Amendment(),
        series=store.series.get(user_id, []),
    )
    return evaluate_state(state)


def decide_row(store: Store, request_id: str) -> dict[str, str]:
    return decision_row(decide(store, request_id))
