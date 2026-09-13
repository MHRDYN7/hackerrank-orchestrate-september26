from __future__ import annotations

import json
from pathlib import Path

from .paths import DATA_DIR, IMAGE_AMOUNTS_PATH, MEDIA_DIR

# Hand-read amounts for the 16 blank-amount events. Used when Gemini is
# unavailable or as a cache seed. Keys are image_id.
FALLBACK_AMOUNTS: dict[str, dict] = {
    "image_01": {"event_id": "event_253", "amount": 4365000.0, "caption": "Net payable IDR 4,365,000"},
    "image_02": {"event_id": "event_1442", "amount": 100000.0, "caption": "Outstanding rent balance due INR 100,000"},
    "image_03": {"event_id": "event_1545", "amount": 41272.0, "caption": "Grocery cash paid INR 41,272"},
    "image_04": {"event_id": "event_1700", "amount": 2854.0, "caption": "Grocery delivery item bill INR 2,854"},
    "image_05": {"event_id": "event_1786", "amount": 704.05, "caption": "Telecom amount due INR 704.05"},
    "image_06": {"event_id": "event_3051", "amount": 1995.0, "caption": "Grocery invoice total INR 1,995"},
    "image_07": {"event_id": "event_3231", "amount": 8528.0, "caption": "Restaurant grand total INR 8,528"},
    "image_08": {"event_id": "event_4535", "amount": 15339.0, "caption": "Maintenance received INR 15,339"},
    "image_09": {"event_id": "event_5170", "amount": 723.0, "caption": "Water bill received INR 723"},
    "image_10": {"event_id": "event_6033", "amount": 79679.26, "caption": "Grocery invoice total INR 79,679.26"},
    "image_11": {"event_id": "event_6859", "amount": 3650.0, "caption": "Hospital amount payable INR 3,650"},
    "image_12": {"event_id": "event_7307", "amount": 33.50, "caption": "Taxi total USD 33.50"},
    "image_13": {"event_id": "event_7941", "amount": 2298.0, "caption": "Tote bag order total INR 2,298"},
    "image_14": {"event_id": "event_9421", "amount": 4543.0, "caption": "Pharmacy total INR 4,543"},
    "image_15": {"event_id": "event_9806", "amount": 9968.0, "caption": "Airline grand total INR 9,968"},
    "image_16": {"event_id": "event_10521", "amount": 393.22, "caption": "EV charging total INR 393.22"},
}


def load_image_amounts() -> dict[str, dict]:
    if IMAGE_AMOUNTS_PATH.exists():
        with IMAGE_AMOUNTS_PATH.open(encoding="utf-8") as handle:
            cached = json.load(handle)
        if cached:
            return cached
    return {k: dict(v) for k, v in FALLBACK_AMOUNTS.items()}


def save_image_amounts(data: dict[str, dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with IMAGE_AMOUNTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def image_path(image_id: str) -> Path:
    return MEDIA_DIR / f"{image_id}.png"


def amounts_by_event(image_rows: list[dict[str, str]] | None = None) -> dict[str, float]:
    extracted = load_image_amounts()
    by_event: dict[str, float] = {}
    for item in extracted.values():
        by_event[item["event_id"]] = float(item["amount"])
    if image_rows:
        for row in image_rows:
            image_id = row["image_id"]
            event_id = row.get("related_event_id") or ""
            if image_id in extracted and event_id:
                by_event[event_id] = float(extracted[image_id]["amount"])
    return by_event
