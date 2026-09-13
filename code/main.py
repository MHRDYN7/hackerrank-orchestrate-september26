from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Allow `uv run python main.py` from code/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from buy_or_wait.keys import load_env

load_env()

from buy_or_wait.graph import build_graph, run_request
from buy_or_wait.keys import load_key_ring
from buy_or_wait.paths import OUTPUT_PATH, USAGE_REPORT_PATH
from buy_or_wait.preprocess import build_store
from buy_or_wait.runner import decide_row
from buy_or_wait.tools import bind_store
from buy_or_wait.usage import TRACKER
from buy_or_wait.validate import OUTPUT_COLUMNS, validate_row


def write_output(rows: list[dict[str, str]], path=OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in OUTPUT_COLUMNS})


def score_samples(store, use_agent: bool = True) -> None:
    ids = sorted(store.samples.keys())
    exact = 0
    ring = load_key_ring()
    app = None
    if use_agent and ring.has_keys():
        app = build_graph(ring)
        print(f"Scoring {len(ids)} sample requests with {ring.model} (key loaded, thinking_level=high)")
    else:
        print(f"Scoring {len(ids)} sample requests (engine only)")
    for rid in ids:
        gold = store.samples[rid]
        if app is not None:
            pred = run_request(app, rid, ring)
        else:
            pred = decide_row(store, rid)
        fields = [
            "amount_safe_to_pay",
            "affordability_status",
            "recommended_payment_method",
            "payment_plan",
            "earliest_date_for_full_payment",
            "spending_changes_needed",
        ]
        mismatches = [f for f in fields if str(pred.get(f, "")) != str(gold.get(f, ""))]
        match = not mismatches
        if match:
            exact += 1
            flag = "OK"
        else:
            flag = "MISS"
        print(
            f"{flag} {rid} pred=({pred['affordability_status']},{pred['recommended_payment_method']},"
            f"{pred['amount_safe_to_pay']},{pred['earliest_date_for_full_payment']},{pred['spending_changes_needed']})"
            f" gold=({gold['affordability_status']},{gold['recommended_payment_method']},"
            f"{gold['amount_safe_to_pay']},{gold['earliest_date_for_full_payment']},{gold['spending_changes_needed']})"
        )
        if flag == "MISS":
            print(f"    mismatches={mismatches}")
            print(f"    pred_plan={pred['payment_plan']}")
            print(f"    gold_plan={gold['payment_plan']}")
    print(f"Exact field match (except explanation): {exact}/{len(ids)}")


def run_eval(store, use_agent: bool) -> list[dict[str, str]]:
    ring = load_key_ring()
    TRACKER.model = ring.model
    bind_store(store)
    app = None
    if use_agent and ring.has_keys():
        app = build_graph(ring)
    rows = []
    ids = store.all_request_ids
    for i, rid in enumerate(ids, 1):
        if use_agent and app is not None:
            row = run_request(app, rid, ring)
        else:
            row = decide_row(store, rid)
        req = store.requests[rid]
        errs = validate_row(row, req, store.options.get(rid, []))
        if errs:
            # fall back to pure engine if the agent row is invalid
            row = decide_row(store, rid)
        rows.append(row)
        if i % 25 == 0 or i == len(ids):
            print(f"processed {i}/{len(ids)}")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Buy or Wait? agent")
    parser.add_argument("--samples", action="store_true", help="Score the 25 public samples")
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--engine-only", action="store_true", help="Skip the LangGraph loop")
    args = parser.parse_args(argv)

    print("Building store from dataset/ ...")
    store = build_store()
    bind_store(store)
    print(f"Users={len(store.profiles)} eval_requests={len(store.all_request_ids)}")

    if args.preprocess_only:
        return 0
    if args.samples:
        score_samples(store, use_agent=not args.engine_only)
        return 0

    rows = run_eval(store, use_agent=not args.engine_only)
    write_output(rows)
    TRACKER.write_report(USAGE_REPORT_PATH, n_requests=len(rows))
    print(f"Wrote {OUTPUT_PATH} ({len(rows)} rows)")
    print(f"Wrote {USAGE_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
