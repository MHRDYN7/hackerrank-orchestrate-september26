from __future__ import annotations

from datetime import date

from .formatters import parse_date


class FxBook:
    def __init__(self, rows: list[dict[str, str]]):
        self.direct: dict[tuple[date, str, str], float] = {}
        for row in rows:
            day = parse_date(row["rate_date"])
            if day is None:
                continue
            pair = (day, row["from_currency"].strip(), row["to_currency"].strip())
            self.direct[pair] = float(row["rate"])

    def convert(self, amount: float, from_ccy: str, to_ccy: str, on: date) -> float:
        if from_ccy == to_ccy:
            return amount
        rate = self._rate(from_ccy, to_ccy, on)
        if rate is None:
            return amount
        return amount * rate

    def _rate(self, src: str, dst: str, on: date) -> float | None:
        direct = self._lookup(src, dst, on)
        if direct is not None:
            return direct
        inverse = self._lookup(dst, src, on)
        if inverse not in (None, 0):
            return 1.0 / inverse
        if src != "USD" and dst != "USD":
            a = self._rate(src, "USD", on)
            b = self._rate("USD", dst, on)
            if a is not None and b is not None:
                return a * b
        if src != "EUR" and dst != "EUR":
            a = self._rate(src, "EUR", on)
            b = self._rate("EUR", dst, on)
            if a is not None and b is not None:
                return a * b
        return None

    def _lookup(self, src: str, dst: str, on: date) -> float | None:
        if (on, src, dst) in self.direct:
            return self.direct[(on, src, dst)]
        # nearest on-or-before, then nearest after
        earlier = [d for (d, a, b) in self.direct if a == src and b == dst and d <= on]
        if earlier:
            return self.direct[(max(earlier), src, dst)]
        later = [d for (d, a, b) in self.direct if a == src and b == dst and d > on]
        if later:
            return self.direct[(min(later), src, dst)]
        return None
