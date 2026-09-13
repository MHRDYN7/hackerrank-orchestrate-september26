from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP


def parse_date(value: str | date | None) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def iso(d: date | None) -> str:
    return d.isoformat() if d else ""


def parse_float(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return float(text.replace(",", ""))


def money(value: float | int | None) -> float:
    if value is None:
        return 0.0
    q = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return float(q)


def fmt_amount(value: float | int | None) -> str:
    if value is None:
        return "0"
    q = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if q == q.to_integral():
        return str(int(q))
    return f"{q:.2f}"


def split_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def parse_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}
