from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Allow `uv run python main.py` from code/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from buy_or_wait.keys import load_env

load_env()

from buy_or_wait.graph import build_graph, run_request
from buy_or_wait.keys import load_key_ring
from buy_or_wait.paths import OUTPUT_PATH, SAMPLE_PREDICTIONS_PATH, USAGE_REPORT_PATH
from buy_or_wait.preprocess import build_store
from buy_or_wait.runner import decide_row
from buy_or_wait.tools import bind_store
from buy_or_wait.usage import TRACKER
from buy_or_wait.validate import OUTPUT_COLUMNS, validate_row

SCORE_FIELDS = [
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
]

# Iterate the known misses before the already-matching samples.
FLAWED_FIRST = [
    "request_11",
    "request_17",
    "request_18",
    "request_19",
    "request_21",
    "request_02",
    "request_03",
    "request_04",
    "request_05",
    "request_06",
    "request_07",
    "request_08",
    "request_10",
    "request_13",
    "request_14",
    "request_15",
    "request_20",
    "request_22",
    "request_23",
    "request_24",
    "request_25",
    "request_01",
    "request_09",
    "request_12",
    "request_16",
]


def write_output(rows: list[dict[str, str]], path=OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in OUTPUT_COLUMNS})


def _print_score(rid: str, pred: dict, gold: dict) -> bool:
    mismatches = [f for f in SCORE_FIELDS if str(pred.get(f, "")) != str(gold.get(f, ""))]
    match = not mismatches
    flag = "OK" if match else "MISS"
    print(
        f"{flag} {rid} pred=({pred['affordability_status']},{pred['recommended_payment_method']},"
        f"{pred['amount_safe_to_pay']},{pred['earliest_date_for_full_payment']},{pred['spending_changes_needed']})"
        f" gold=({gold['affordability_status']},{gold['recommended_payment_method']},"
        f"{gold['amount_safe_to_pay']},{gold['earliest_date_for_full_payment']},{gold['spending_changes_needed']})",
        flush=True,
    )
    if not match:
        print(f"    mismatches={mismatches}", flush=True)
        print(f"    pred_plan={pred['payment_plan']}", flush=True)
        print(f"    gold_plan={gold['payment_plan']}", flush=True)
        print(f"    pred_x={(pred.get('decision_explanation') or '')[:240]}", flush=True)
        print(f"    gold_x={(gold.get('decision_explanation') or '')[:240]}", flush=True)
    return match


def _sample_ids(store, requested: list[str] | None) -> list[str]:
    available = set(store.samples)
    if requested:
        ids = [rid.strip() for rid in requested if rid.strip() in available]
        return ids
    ordered = [rid for rid in FLAWED_FIRST if rid in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def score_samples(store, use_agent: bool = True, ids: list[str] | None = None, concurrency: int = 10) -> None:
    ids = _sample_ids(store, ids)
    ring = load_key_ring()
    app = None
    if use_agent and ring.has_keys():
        app = build_graph(ring)
        print(
            f"Scoring {len(ids)} sample requests with {ring.model} "
            f"(keys={len(ring.keys)}, concurrency={concurrency}, thinking_level=high)",
            flush=True,
        )
    else:
        print(f"Scoring {len(ids)} sample requests (engine only)", flush=True)

    def one(rid: str) -> tuple[str, dict, dict]:
        gold = store.samples[rid]
        if app is not None:
            pred = run_request(app, rid, ring)
        else:
            pred = decide_row(store, rid)
        return rid, pred, gold

    exact = 0
    results: dict[str, tuple[dict, dict]] = {}
    workers = max(1, concurrency if app is not None else 1)
    if workers == 1:
        for rid in ids:
            _, pred, gold = one(rid)
            results[rid] = (pred, gold)
            if _print_score(rid, pred, gold):
                exact += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(one, rid): rid for rid in ids}
            for fut in as_completed(futs):
                rid, pred, gold = fut.result()
                results[rid] = (pred, gold)
                if _print_score(rid, pred, gold):
                    exact += 1
    print(f"Exact field match (except explanation): {exact}/{len(ids)}", flush=True)
    field_hits = {f: 0 for f in SCORE_FIELDS}
    for rid in ids:
        pred, gold = results[rid]
        for f in SCORE_FIELDS:
            if str(pred.get(f, "")) == str(gold.get(f, "")):
                field_hits[f] += 1
    n = max(1, len(ids))
    for f in SCORE_FIELDS:
        print(f"  {f}: {field_hits[f]}/{len(ids)}", flush=True)
    write_output(
        [{"request_id": rid, **results[rid][0]} for rid in ids],
        SAMPLE_PREDICTIONS_PATH,
    )
    print(f"Wrote {SAMPLE_PREDICTIONS_PATH}", flush=True)


def run_eval(store, use_agent: bool, concurrency: int = 10) -> list[dict[str, str]]:
    ring = load_key_ring()
    TRACKER.model = ring.model
    bind_store(store)
    app = None
    if use_agent and ring.has_keys():
        app = build_graph(ring)
    ids = store.all_request_ids
    rows_by_id: dict[str, dict[str, str]] = {}

    def one(rid: str) -> tuple[str, dict[str, str]]:
        if use_agent and app is not None:
            row = run_request(app, rid, ring)
        else:
            row = decide_row(store, rid)
        req = store.requests[rid]
        errs = validate_row(row, req, store.options.get(rid, []))
        if errs and app is not None:
            print(f"invalid_row {rid} {errs}", flush=True)
        return rid, row

    workers = max(1, concurrency if app is not None else 1)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(one, rid): rid for rid in ids}
        for fut in as_completed(futs):
            rid, row = fut.result()
            rows_by_id[rid] = row
            done += 1
            if done % 10 == 0 or done == len(ids):
                print(f"processed {done}/{len(ids)}", flush=True)
    return [rows_by_id[rid] for rid in ids]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Buy or Wait? agent")
    parser.add_argument("--samples", action="store_true", help="Score the 25 public samples")
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--engine-only", action="store_true", help="Skip the LangGraph loop")
    parser.add_argument("--ids", default="", help="Comma-separated request ids to score")
    parser.add_argument("--concurrency", type=int, default=10, help="Parallel requests (429s are retried)")
    args = parser.parse_args(argv)

    print("Building store from dataset/ ...", flush=True)
    store = build_store()
    bind_store(store)
    print(f"Users={len(store.profiles)} eval_requests={len(store.all_request_ids)}", flush=True)

    if args.preprocess_only:
        return 0
    ids = [part.strip() for part in args.ids.split(",") if part.strip()] or None
    if args.samples:
        score_samples(store, use_agent=not args.engine_only, ids=ids, concurrency=args.concurrency)
        return 0

    rows = run_eval(store, use_agent=not args.engine_only, concurrency=args.concurrency)
    write_output(rows)
    TRACKER.write_report(USAGE_REPORT_PATH, n_requests=len(rows))
    print(f"Wrote {OUTPUT_PATH} ({len(rows)} rows)")
    print(f"Wrote {USAGE_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
