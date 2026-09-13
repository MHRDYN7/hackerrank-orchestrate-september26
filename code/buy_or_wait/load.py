from __future__ import annotations

import csv
from pathlib import Path

from .paths import DATASET_DIR


def read_csv(name: str) -> list[dict[str, str]]:
    path = DATASET_DIR / name
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_raw() -> dict[str, list[dict[str, str]]]:
    return {
        "profiles": read_csv("financial_profiles.csv"),
        "events": read_csv("financial_events.csv"),
        "requests": read_csv("requests.csv"),
        "samples": read_csv("sample_requests.csv"),
        "options": read_csv("request_payment_options.csv"),
        "rates": read_csv("exchange_rates.csv"),
        "messages": read_csv("messages.csv"),
        "images": read_csv("images.csv"),
    }
