from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from calendar import monthrange
from statistics import median

from .formatters import fmt_amount, fmt_plan_amount, iso, money, parse_bool, parse_date, parse_float, split_list
from .fx import FxBook
from .messages import Amendment


FORECAST_DAYS = 90
VARIABLE_CATEGORIES = {
    "groceries",
    "transport",
    "dining",
    "shopping",
    "entertainment",
}
FIXED_CATEGORIES = {
    "rent",
    "housing",
    "utilities",
    "insurance",
    "debt_repayment",
    "education",
    "healthcare",
    "family_support",
    "salary",
    "cloud_storage",
    "streaming",
    "music_subscription",
    "delivery_membership",
    "gym",
}
INCOME_IGNORE_CATEGORIES = {"windfall"}
CREDIT_IGNORE_TYPES = {"refund"}
NON_CASH_STATUSES = {"unrealized"}
IGNORE_STATUSES = {"cancelled", "failed", "unrealized"}
SPECULATIVE_INCOME_TOKENS = (
    "commission",
    "bonus",
    "incentive",
    "performance",
    "overtime",
    "tip",
)
SALARY_STALE_DAYS = 48
FLEX_STALE_DAYS_MONTHLY = 45
FLEX_STALE_DAYS_WEEKLY = 18


def add_months(d: date, months: int) -> date:
    month = d.month - 1 + months
    year = d.year + month // 12
    month = month % 12 + 1
    day = min(d.day, monthrange(year, month)[1])
    return date(year, month, day)


@dataclass
class CashItem:
    item_date: date
    amount: float  # positive credit, negative debit
    kind: str
    category: str
    description: str
    event_id: str = ""
    series_id: str = ""
    essential: bool = True
    flexibility: str = "fixed"
    min_allowed: float | None = None
    source: str = "event"


@dataclass
class Series:
    series_id: str
    user_id: str
    description: str
    category: str
    direction: str
    amount: float
    last_date: date
    period_days: int
    monthly: bool
    event_id: str
    flexibility: str
    min_allowed: float | None
    event_type: str
    typical_amount: float = 0.0


@dataclass
class UserState:
    profile: dict
    request: dict
    options: list[dict]
    events: list[dict]
    amendment: Amendment
    series: list[Series]
    home: str
    min_balance: float
    start_balance: float
    request_date: date
    deadline: date
    requested: float
    allows_partial: bool
    methods: list[str]
    max_inst_months: int | None
    protect: set[str]
    reduce_ok: set[str]
    stop_ok: set[str]


@dataclass
class Candidate:
    candidate_id: str
    method: str
    status: str
    payments: list[tuple[date, float]]
    payment_plan: str
    spending_changes: str
    total_paid: float
    starts: date | None
    n_payments: int
    option_id: str
    completes_by_deadline: bool
    safe: bool
    explanation: str = ""


@dataclass
class Decision:
    request_id: str
    amount_safe_to_pay: float
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: str
    spending_changes_needed: str
    decision_explanation: str
    candidates: list[Candidate] = field(default_factory=list)


def normalize_events(
    events: list[dict],
    home: str,
    fx: FxBook,
    image_amounts: dict[str, float],
) -> list[dict]:
    out = []
    for raw in events:
        row = dict(raw)
        amount = parse_float(row.get("amount"))
        if amount is None:
            amount = image_amounts.get(row["event_id"])
        row["amount_raw"] = amount
        ccy = (row.get("currency") or home).strip()
        settle = parse_date(row.get("settlement_date") or row.get("event_date"))
        ev_date = parse_date(row.get("event_date")) or settle
        row["event_date_p"] = ev_date
        row["settlement_date_p"] = settle or ev_date
        if amount is None:
            row["amount_home"] = None
        else:
            row["amount_home"] = money(fx.convert(amount, ccy, home, row["settlement_date_p"]))
        row["min_allowed"] = parse_float(row.get("minimum_allowed_amount"))
        out.append(row)
    return out


def build_series(events: list[dict], user_id: str) -> list[Series]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in events:
        if row.get("status") != "settled":
            continue
        if row.get("amount_home") is None:
            continue
        if row.get("direction") == "non_cash":
            continue
        key = (row.get("description") or "", row.get("category") or "", row.get("direction") or "")
        groups[key].append(row)

    series: list[Series] = []
    for (desc, cat, direction), rows in groups.items():
        rows = sorted(rows, key=lambda r: r["event_date_p"])
        if len(rows) < 2:
            continue
        gaps = [
            (rows[i]["event_date_p"] - rows[i - 1]["event_date_p"]).days
            for i in range(1, len(rows))
            if rows[i]["event_date_p"] and rows[i - 1]["event_date_p"]
        ]
        gaps = [g for g in gaps if g > 0]
        if not gaps:
            continue
        med = median(gaps)
        monthly = 25 <= med <= 36 or (20 <= med <= 40 and len(rows) >= 3)
        weekly = (
            (not monthly)
            and 6 <= med <= 10
            and len(rows) >= 3
            and cat in {"dining", "entertainment", "streaming", "cloud_storage"}
        )
        last = rows[-1]
        cat_l = cat
        flex = last.get("flexibility") or "fixed"
        amounts = [float(r["amount_home"]) for r in rows if r.get("amount_home") is not None]
        typical_amount = median(amounts) if amounts else float(last["amount_home"])
        # Only treat weekend-food as weekly when the gaps actually look weekly.
        if "weekend food" in desc.lower() and flex != "fixed" and 5 <= med <= 14:
            weekly = True
            monthly = False
        if not monthly and not weekly:
            continue
        if weekly and flex == "fixed":
            continue
        if cat_l in VARIABLE_CATEGORIES and flex == "fixed":
            continue
        if cat_l not in FIXED_CATEGORIES and flex == "fixed" and direction != "credit":
            continue
        if direction == "credit" and _is_speculative_income(desc, cat):
            continue
        sid = f"{user_id}:{desc}:{cat}:{direction}"
        series.append(
            Series(
                series_id=sid,
                user_id=user_id,
                description=desc,
                category=cat,
                direction=direction,
                amount=float(last["amount_home"]),
                last_date=last["event_date_p"],
                period_days=7 if weekly and "weekend food" in desc.lower() else int(round(med)),
                monthly=monthly,
                event_id=last["event_id"],
                flexibility=last.get("flexibility") or "fixed",
                min_allowed=last.get("min_allowed"),
                event_type=last.get("event_type") or "",
                typical_amount=float(typical_amount),
            )
        )
    return series


def series_from_event(
    events: list[dict],
    event_id: str,
    user_id: str,
    amount_mode: str = "auto",
) -> Series | None:
    """Project a recurrence from one event using other settled rows in the same category.

    amount_mode is last, typical, or auto. Auto uses last for flexible series and typical
    for fixed spend, and always typical when the last ticket is an outlier versus the median.
    """
    found = None
    for row in events:
        if row.get("event_id") == event_id:
            found = row
            break
    if not found or found.get("direction") != "debit":
        return None
    cat = found.get("category") or ""
    rows = [
        r
        for r in events
        if r.get("status") == "settled"
        and r.get("direction") == "debit"
        and (r.get("category") or "") == cat
        and r.get("event_date_p") is not None
        and r.get("amount_home") is not None
    ]
    rows = sorted(rows, key=lambda r: r["event_date_p"])
    if len(rows) < 2:
        return None
    gaps = [(rows[i]["event_date_p"] - rows[i - 1]["event_date_p"]).days for i in range(1, len(rows))]
    gaps = [g for g in gaps if 0 < g < 60]
    med = median(gaps) if gaps else 30
    last = found if found.get("event_date_p") else rows[-1]
    amounts = [float(r["amount_home"]) for r in rows]
    typical = float(median(amounts)) if amounts else float(last.get("amount_home") or 0)
    last_amt = float(last.get("amount_home") or 0)
    flex = (last.get("flexibility") or "fixed").strip()
    mode = (amount_mode or "auto").strip().lower()
    if mode == "median":
        mode = "typical"
    if mode == "typical":
        amount = typical
    elif mode == "last":
        amount = last_amt
    else:
        outlier = typical > 0 and last_amt > 1.6 * typical
        if outlier:
            amount = typical
        elif flex in {"reducible", "stoppable", "reducible_or_stoppable"}:
            amount = last_amt
        else:
            amount = typical
    return Series(
        series_id=f"{user_id}:from:{event_id}",
        user_id=user_id,
        description=last.get("description") or cat,
        category=cat,
        direction="debit",
        amount=amount,
        last_date=last["event_date_p"],
        period_days=int(round(med)) if med else 30,
        monthly=25 <= float(med) <= 36,
        event_id=event_id,
        flexibility=flex,
        min_allowed=last.get("min_allowed"),
        event_type=last.get("event_type") or "",
        typical_amount=typical,
    )


def _is_speculative_income(desc: str, category: str = "") -> bool:
    blob = f"{desc} {category}".lower()
    return any(tok in blob for tok in SPECULATIVE_INCOME_TOKENS)


def _next_dates(last: date, monthly: bool, period_days: int, start: date, end: date) -> list[date]:
    dates = []
    if monthly:
        cur = last
        # walk forward month by month until past start-1
        guard = 0
        while cur <= end and guard < 24:
            cur = add_months(cur, 1)
            guard += 1
            if cur >= start and cur <= end:
                dates.append(cur)
        return dates
    cur = last
    guard = 0
    while guard < 80:
        cur = cur + timedelta(days=period_days)
        guard += 1
        if cur > end:
            break
        if cur >= start:
            dates.append(cur)
    return dates


def _should_project_income(series: Series, amendment: Amendment, request_date: date | None = None) -> bool:
    if series.direction != "credit":
        return True
    if _is_speculative_income(series.description, series.category):
        return False
    if series.category != "salary" and series.event_type != "income":
        return False
    if amendment.employment_ended or amendment.seasonal_ended:
        return False
    if amendment.ignore_prize or amendment.ignore_scam:
        if series.category in {"windfall"}:
            return False
    if request_date and series.category == "salary":
        confirmed = bool(
            amendment.salary_amount
            or amendment.salary_date
            or amendment.first_salary
            or amendment.remaining_salary
            or amendment.confirmed_invoice
        )
        if not confirmed and (request_date - series.last_date).days > SALARY_STALE_DAYS:
            return False
    return True


def _series_is_stale(series: Series, request_date: date) -> bool:
    if series.flexibility == "fixed" or series.direction == "credit":
        return False
    gap = (request_date - series.last_date).days
    limit = FLEX_STALE_DAYS_MONTHLY if series.monthly else FLEX_STALE_DAYS_WEEKLY
    return gap > limit


def build_forecast(
    state: UserState,
    spending_changes: dict[str, float | None] | None = None,
) -> list[CashItem]:
    """spending_changes maps event_id -> new amount (None means stop)."""
    spending_changes = spending_changes or {}
    start = state.request_date
    end = start + timedelta(days=FORECAST_DAYS)
    items: list[CashItem] = []
    seen_keys: set[tuple] = set()

    def add(item: CashItem):
        if item.item_date < start or item.item_date > end:
            return
        key = (item.item_date, item.category, item.description, round(item.amount, 2), item.kind)
        if key in seen_keys:
            return
        seen_keys.add(key)
        items.append(item)

    # Explicit future / pending / scheduled rows
    for row in state.events:
        status = row.get("status") or ""
        direction = row.get("direction") or ""
        settle = row["settlement_date_p"]
        if settle is None or settle < start:
            continue
        if status in IGNORE_STATUSES:
            continue
        amt = row.get("amount_home")
        if amt is None:
            continue
        etype = row.get("event_type") or ""
        cat = row.get("category") or ""

        if direction == "non_cash" or status == "unrealized":
            continue
        if direction == "credit":
            if status == "pending":
                continue
            if etype in CREDIT_IGNORE_TYPES:
                continue
            if cat in INCOME_IGNORE_CATEGORIES and status != "settled":
                continue
            if etype == "investment_valuation":
                continue
            if status == "scheduled" and etype == "income":
                signed = amt
            elif status == "settled" and settle >= start:
                signed = amt
            else:
                continue
        else:
            if status in {"pending", "scheduled"}:
                signed = -amt
            elif status == "settled" and settle >= start:
                # already in current balance if request_date is as-of today
                continue
            else:
                continue

        if state.amendment.internal_transfer and "transfer" in (row.get("description") or "").lower():
            continue

        add(
            CashItem(
                item_date=settle,
                amount=signed,
                kind=etype or direction,
                category=cat,
                description=row.get("description") or "",
                event_id=row.get("event_id") or "",
                essential=cat in state.protect or (row.get("flexibility") == "fixed"),
                flexibility=row.get("flexibility") or "fixed",
                min_allowed=row.get("min_allowed"),
                source="explicit",
            )
        )

    # Recurring projections
    for ser in state.series:
        if not _should_project_income(ser, state.amendment, start):
            continue
        if _series_is_stale(ser, start):
            continue
        if ser.event_id in spending_changes and spending_changes[ser.event_id] is None:
            continue
        amount = ser.amount
        if ser.event_id in spending_changes and spending_changes[ser.event_id] is not None:
            amount = float(spending_changes[ser.event_id])

        if ser.direction == "credit" and ser.category == "salary" and not _is_speculative_income(ser.description):
            override = state.amendment.remaining_salary or state.amendment.salary_amount
            if override is not None and (ser.amount <= 0 or 0.2 * ser.amount <= override <= 5 * ser.amount):
                amount = override

        if ser.category == "rent" and state.amendment.rent_increase_pct:
            amount = amount * (1 + state.amendment.rent_increase_pct / 100.0)

        dates = _next_dates(ser.last_date, ser.monthly, ser.period_days, start, end)
        if ser.direction == "credit":
            raw_next = add_months(ser.last_date, 1) if ser.monthly else ser.last_date + timedelta(days=ser.period_days)
            if raw_next < start:
                # A payday was already missed; do not invent a restarted cadence.
                dates = []
        if ser.direction != "credit" and not any(
            it.category == "salary" and it.amount > 0 for it in items
        ) and not (
            state.amendment.salary_amount or state.amendment.salary_date or state.amendment.first_salary
        ):
            dates = dates[:2]
        if ser.direction == "credit" and ser.category == "salary":
            if state.amendment.salary_date:
                moved = state.amendment.salary_date
                dates = []
                cur = moved
                if cur < start:
                    while cur < start:
                        cur = add_months(cur, 1)
                while cur <= end:
                    dates.append(cur)
                    cur = add_months(cur, 1)
            if state.amendment.first_salary and state.amendment.salary_date:
                dates = [d for d in dates if d >= state.amendment.salary_date]

        if ser.direction == "credit" and (state.amendment.employment_ended or state.amendment.seasonal_ended):
            dates = []

        signed_base = amount if ser.direction == "credit" else -amount
        first_salary_date = min(dates) if dates and ser.direction == "credit" else None
        for d in dates:
            # skip if explicit item already covers this payday
            if any(
                it.item_date == d
                and it.category == ser.category
                and (it.description == ser.description or ser.category == "salary")
                for it in items
            ):
                continue
            pay_amt = signed_base
            if (
                ser.direction == "credit"
                and ser.category == "salary"
                and state.amendment.salary_increase_from
                and d >= state.amendment.salary_increase_from
                and state.amendment.salary_amount is not None
            ):
                pay_amt = state.amendment.salary_amount
            if (
                ser.direction == "credit"
                and ser.category == "salary"
                and state.amendment.salary_temporary
                and state.amendment.salary_amount is not None
            ):
                if first_salary_date is not None and d == first_salary_date:
                    pay_amt = state.amendment.salary_amount
                else:
                    pay_amt = ser.typical_amount or ser.amount
            add(
                CashItem(
                    item_date=d,
                    amount=pay_amt,
                    kind="recurring",
                    category=ser.category,
                    description=ser.description,
                    event_id=ser.event_id,
                    series_id=ser.series_id,
                    essential=ser.category in state.protect or ser.flexibility == "fixed",
                    flexibility=ser.flexibility,
                    min_allowed=ser.min_allowed,
                    source="series",
                )
            )

    # Confirmed invoice (message) if no matching credit
    if state.amendment.confirmed_invoice and state.amendment.invoice_date:
        if start <= state.amendment.invoice_date <= end:
            add(
                CashItem(
                    item_date=state.amendment.invoice_date,
                    amount=state.amendment.confirmed_invoice,
                    kind="invoice",
                    category="salary",
                    description="Confirmed invoice settlement",
                    source="message",
                )
            )

    # One-time arrears with next salary
    if state.amendment.arrears_amount and state.amendment.salary_date:
        if start <= state.amendment.salary_date <= end:
            add(
                CashItem(
                    item_date=state.amendment.salary_date,
                    amount=state.amendment.arrears_amount,
                    kind="arrears",
                    category="salary",
                    description="One-time arrears adjustment",
                    source="message",
                )
            )

    _project_future_salary(state, items, start, end, add)
    _add_variable_envelopes(state, items, start, end, add)

    items.sort(key=lambda i: (i.item_date, i.amount))
    return items


def _project_future_salary(state: UserState, items: list[CashItem], start: date, end: date, add):
    """Extend an already-confirmed in-horizon salary cadence. Do not invent a first payday."""
    if state.amendment.employment_ended or state.amendment.seasonal_ended:
        return
    credits = [it for it in items if it.category == "salary" and it.amount > 0]
    if not credits:
        return
    last_credit = max(credits, key=lambda it: it.item_date)
    last_date = last_credit.item_date
    amount = last_credit.amount
    if state.amendment.remaining_salary is not None:
        amount = state.amendment.remaining_salary
    elif state.amendment.salary_amount is not None:
        amount = state.amendment.salary_amount
    cur = last_date
    for _ in range(3):
        cur = add_months(cur, 1)
        if cur > end:
            break
        if cur < start:
            continue
        add(
            CashItem(
                item_date=cur,
                amount=amount,
                kind="salary",
                category="salary",
                description="Projected salary",
                source="salary_proj",
            )
        )


def _add_variable_envelopes(state: UserState, items: list[CashItem], start: date, end: date, add):
    last_month_end = start.replace(day=1) - timedelta(days=1)
    months = []
    cursor = last_month_end.replace(day=1)
    for _ in range(3):
        months.append((cursor.year, cursor.month))
        cursor = (cursor - timedelta(days=1)).replace(day=1)
    by_cat_month: dict[tuple, float] = defaultdict(float)
    spent_this_month: dict[str, float] = defaultdict(float)
    for row in state.events:
        if row.get("status") != "settled" or row.get("direction") != "debit":
            continue
        cat = row.get("category") or ""
        if cat not in VARIABLE_CATEGORIES:
            continue
        d = row["event_date_p"]
        if d is None:
            continue
        amt = float(row.get("amount_home") or 0)
        if (d.year, d.month) in months:
            by_cat_month[(d.year, d.month, cat)] += amt
        if d.year == start.year and d.month == start.month and d < start:
            spent_this_month[cat] += amt

    for cat in VARIABLE_CATEGORIES:
        month_totals = [by_cat_month[(y, m, cat)] for (y, m) in months if by_cat_month[(y, m, cat)] > 0]
        if not month_totals:
            continue
        mean_amt = sum(month_totals) / len(month_totals)
        # Commute spend is lumpy; use the recent peak. Other variable spend uses the mean.
        conservative = max(month_totals) if cat == "transport" else mean_amt
        if conservative < 1:
            continue
        if cat not in state.protect:
            continue
        already_series = sum(-it.amount for it in items if it.category == cat and it.source == "series")
        if already_series >= conservative * 0.7:
            continue
        remaining = max(0.0, conservative - spent_this_month[cat])
        day = 18
        first_mid = date(start.year, start.month, min(day, monthrange(start.year, start.month)[1]))
        next_salaries = [
            it.item_date
            for it in items
            if it.category == "salary" and it.amount > 0 and it.item_date > start
        ]
        next_sal = min(next_salaries) if next_salaries else None
        if remaining >= 1:
            if cat == "transport":
                when = start
            elif next_sal:
                when = max(start, next_sal - timedelta(days=1))
            elif any(it.category == "salary" and it.amount > 0 for it in items):
                when = max(first_mid, start)
            else:
                when = start
            if when <= end:
                add(
                    CashItem(
                        item_date=when,
                        amount=-remaining,
                        kind="envelope",
                        category=cat,
                        description=f"Conservative {cat} spend",
                        essential=True,
                        source="envelope",
                    )
                )
        cur = add_months(first_mid, 1)
        if cur.day != day:
            cur = date(cur.year, cur.month, min(day, monthrange(cur.year, cur.month)[1]))
        while cur <= end:
            add(
                CashItem(
                    item_date=cur,
                    amount=-conservative,
                    kind="envelope",
                    category=cat,
                    description=f"Conservative {cat} spend",
                    essential=True,
                    source="envelope",
                )
            )
            cur = add_months(cur, 1)
            if cur.day != day:
                cur = date(cur.year, cur.month, min(day, monthrange(cur.year, cur.month)[1]))


def simulate(
    state: UserState,
    items: list[CashItem],
    extra_debits: list[tuple[date, float]] | None = None,
) -> tuple[bool, float, date | None]:
    extra_debits = extra_debits or []
    by_day: dict[date, list[tuple[int, float]]] = defaultdict(list)
    # Same-day order: credits first (salary on payday can fund the payment), then plan, then other debits.
    for it in items:
        rank = 0 if it.amount >= 0 else 2
        by_day[it.item_date].append((rank, it.amount))
    for d, amt in extra_debits:
        by_day[d].append((1, -abs(amt)))
    balance = state.start_balance
    trough = balance
    breach: date | None = None
    horizon_end = state.request_date + timedelta(days=FORECAST_DAYS)
    for d in sorted(by_day):
        if d < state.request_date or d > horizon_end:
            continue
        for _, amt in sorted(by_day[d], key=lambda x: x[0]):
            balance = money(balance + amt)
            if balance < trough:
                trough = balance
            if balance + 1e-8 < state.min_balance and breach is None:
                breach = d
    return breach is None, trough, breach


def amount_safe_to_pay(state: UserState, items: list[CashItem]) -> float:
    requested = state.requested
    ok, _, _ = simulate(state, items, [(state.request_date, requested)])
    if ok:
        return money(requested)
    lo, hi = 0.0, requested
    best = 0.0
    for _ in range(40):
        mid = money((lo + hi) / 2)
        ok, _, _ = simulate(state, items, [(state.request_date, mid)])
        if ok:
            best = mid
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.02:
            break
    return money(min(best, requested))


def earliest_full_payment(state: UserState, items: list[CashItem]) -> date | None:
    """First date a single full payment is safe on this ledger, with no spending changes."""
    end = state.request_date + timedelta(days=FORECAST_DAYS)
    d = state.request_date
    while d <= end:
        ok, _, _ = simulate(state, items, [(d, state.requested)])
        if ok:
            return d
        d += timedelta(days=1)
    return None


def expand_option(option: dict) -> list[tuple[date, float]]:
    n = int(float(option["number_of_payments"] or 1))
    first = parse_date(option["first_payment_date"])
    raw = str(option["payment_amount"]).strip()
    amt = float(raw)
    pays: list[tuple[date, float]] = []
    if n <= 1 or not option.get("payment_frequency_days"):
        pays = [(first, amt)]
    else:
        freq = int(float(option["payment_frequency_days"]))
        pays = [(first + timedelta(days=freq * k), amt) for k in range(n)]
    for i, (d, a) in enumerate(pays):
        pays[i] = (d, a)
    option["_amount_text"] = raw
    return pays


def installment_months(option: dict) -> int:
    n = int(float(option["number_of_payments"] or 1))
    freq = parse_float(option.get("payment_frequency_days")) or 0
    if n <= 1:
        return 1
    if freq >= 27:
        return n
    return max(1, int(round(n * freq / 30.0)))


def plan_text(payments: list[tuple[date, float]], amount_text: str | None = None) -> str:
    if not payments:
        return "none"
    bits = []
    for d, a in payments:
        bits.append(f"{d.isoformat()}:{amount_text if amount_text is not None else fmt_amount(a)}")
    return "|".join(bits)


def _fmt_ccy(home: str, amt: float) -> str:
    return f"{home} {fmt_amount(amt)}"


def explain(state: UserState, cand: Candidate) -> str:
    home = state.home
    mn = _fmt_ccy(home, state.min_balance)
    req = _fmt_ccy(home, state.requested)
    if cand.method == "full_payment":
        prefix = ""
        if cand.spending_changes != "none":
            prefix = _explain_changes(state, cand.spending_changes) + ", then "
        if cand.payments and cand.payments[0][0] == state.request_date:
            return (
                f"{prefix}Pay {req} today. This leaves at least {mn} available over the next 90 days."
                if not prefix
                else f"{prefix}pay {req} today. This leaves at least {mn} available."
            )
        day = cand.payments[0][0].isoformat() if cand.payments else ""
        return f"Pay {req} in full on {day}. Paying earlier would take the balance below the {mn} minimum."
    if cand.method == "wait":
        day = cand.payments[0][0].isoformat() if cand.payments else ""
        return f"Pay {req} in full on {day}. Paying earlier would take the balance below the {home} {fmt_amount(state.min_balance)} minimum."
    if cand.method == "partial_payment":
        a, b = cand.payments
        return (
            f"Pay {home} {fmt_amount(a[1])} today and the remaining {home} {fmt_amount(b[1])} on {b[0].isoformat()}. "
            f"This completes the full request and keeps the {mn} minimum protected."
        )
    if cand.method == "installments":
        each = cand.payments[0][1] if cand.payments else 0
        start = ""
        if cand.payments:
            start = cand.payments[0][0].strftime("%d %B %Y").lstrip("0")
        return (
            f"Use {len(cand.payments)} installments of {home} {fmt_amount(each)}, starting {start}. "
            f"This leaves at least {mn} available."
        )
    deadline = state.deadline.isoformat()
    return (
        f"Do not make this payment by {deadline}. None of the available options keeps the {mn} minimum protected."
    )


def _explain_changes(state: UserState, blob: str) -> str:
    parts = []
    for bit in blob.split("|"):
        if bit.startswith("stop:"):
            eid = bit.split(":", 1)[1]
            desc = next((s.description for s in state.series if s.event_id == eid), "a flexible expense")
            parts.append(f"Stop the {desc.lower()}")
        elif bit.startswith("reduce_to:"):
            _, eid, amt = bit.split(":", 2)
            desc = next((s.description for s in state.series if s.event_id == eid), "a flexible expense")
            parts.append(f"Reduce the {desc.lower()} to {state.home} {amt}")
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _change_blob(changes: dict[str, float | None]) -> str:
    if not changes:
        return "none"
    bits = []
    for eid, val in changes.items():
        if val is None:
            bits.append(f"stop:{eid}")
        else:
            bits.append(f"reduce_to:{eid}:{fmt_amount(val)}")
    return "|".join(bits)


def _legal_change_sets(state: UserState) -> list[dict[str, float | None]]:
    stops = []
    reduces = []
    for ser in state.series:
        if ser.category in state.protect:
            continue
        flex = ser.flexibility
        if flex in {"stoppable", "reducible_or_stoppable"} and ser.category in state.stop_ok:
            stops.append(ser)
        if flex in {"reducible", "reducible_or_stoppable"} and ser.category in state.reduce_ok:
            if ser.min_allowed is not None and ser.min_allowed < ser.amount:
                reduces.append(ser)
    # largest first
    stops.sort(key=lambda s: s.amount, reverse=True)
    reduces.sort(key=lambda s: s.amount - (s.min_allowed or 0), reverse=True)
    sets: list[dict[str, float | None]] = [{}]
    for s in stops[:6]:
        sets.append({s.event_id: None})
    for s in reduces[:6]:
        sets.append({s.event_id: float(s.min_allowed)})
    # pairs
    for s in stops[:4]:
        for r in reduces[:4]:
            if s.event_id == r.event_id:
                continue
            sets.append({s.event_id: None, r.event_id: float(r.min_allowed)})
        for s2 in stops[:4]:
            if s2.event_id <= s.event_id:
                continue
            sets.append({s.event_id: None, s2.event_id: None})
    # unique
    uniq = []
    seen = set()
    for ch in sets:
        key = tuple(sorted((k, str(v)) for k, v in ch.items()))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(ch)
    return uniq[:24]


def _status_for(method: str, changes: dict, request_date: date, payments: list[tuple[date, float]]) -> str:
    if method == "not_recommended":
        return "not_affordable"
    if method == "wait":
        return "affordable_later"
    if method == "full_payment" and not changes and payments and payments[0][0] == request_date:
        return "affordable_now"
    return "affordable_with_plan"


def evaluate_state(state: UserState) -> Decision:
    base_items = build_forecast(state, {})
    safe_today = amount_safe_to_pay(state, base_items)
    earliest = earliest_full_payment(state, base_items)
    candidates: list[Candidate] = []
    req_style = str(state.request.get("requested_amount") or "")

    def consider(
        method: str,
        payments: list[tuple[date, float]],
        changes: dict,
        option_id: str = "",
        amount_text: str | None = None,
    ) -> None:
        items = base_items if not changes else build_forecast(state, changes)
        ok, _, _ = simulate(state, items, [(d, a) for d, a in payments])
        last_pay = payments[-1][0] if payments else None
        completes = bool(last_pay and last_pay <= state.deadline)
        total = sum(a for _, a in payments)
        if method == "installments":
            plan = plan_text(payments, amount_text)
        elif method in {"full_payment", "wait"} and payments:
            plan = plan_text(payments, fmt_plan_amount(payments[0][1], req_style))
        else:
            plan = plan_text(payments)
        cand = Candidate(
            candidate_id=f"{method}:{option_id}:{_change_blob(changes)}",
            method=method,
            status=_status_for(method, changes, state.request_date, payments),
            payments=payments,
            payment_plan=plan,
            spending_changes=_change_blob(changes),
            total_paid=total,
            starts=payments[0][0] if payments else None,
            n_payments=len(payments),
            option_id=option_id,
            completes_by_deadline=completes,
            safe=ok,
        )
        cand.explanation = explain(state, cand)
        # A legal plan must finish by the deadline. Late but cash-safe schedules are not recommended.
        if ok and completes:
            candidates.append(cand)

    methods = set(state.methods)
    change_sets = [{}]
    no_change_completes = False

    def generate_for(changes: dict):
        if "full_payment" in methods:
            consider("full_payment", [(state.request_date, state.requested)], changes)
        if "full_payment" in methods:
            items = base_items if not changes else build_forecast(state, changes)
            early = earliest_full_payment(state, items)
            if early and early > state.request_date:
                consider("wait", [(early, state.requested)], changes)
        if "partial_payment" in methods and state.allows_partial:
            items = base_items if not changes else build_forecast(state, changes)
            safe = amount_safe_to_pay(state, items)
            early = earliest_full_payment(state, items)
            if 0 < safe < state.requested - 0.005 and early and early <= state.deadline:
                remainder = money(state.requested - safe)
                consider(
                    "partial_payment",
                    [(state.request_date, safe), (early, remainder)],
                    changes,
                )
        if "installments" in methods and not changes:
            for opt in state.options:
                if opt.get("payment_method") != "installments":
                    continue
                months = installment_months(opt)
                if state.max_inst_months is not None and months > state.max_inst_months:
                    continue
                pays = expand_option(opt)
                consider(
                    "installments",
                    pays,
                    changes,
                    opt["payment_option_id"],
                    opt.get("_amount_text"),
                )

    generate_for({})
    no_change_completes = any(c.completes_by_deadline and c.spending_changes == "none" for c in candidates)
    if not no_change_completes:
        for changes in _legal_change_sets(state):
            if not changes:
                continue
            if "full_payment" in methods:
                consider("full_payment", [(state.request_date, state.requested)], changes)
            if "full_payment" in methods:
                items = build_forecast(state, changes)
                early = earliest_full_payment(state, items)
                if early and early > state.request_date:
                    consider("wait", [(early, state.requested)], changes)

    def _change_cut(blob: str) -> float:
        if blob == "none":
            return 0.0
        total = 0.0
        for bit in blob.split("|"):
            if bit.startswith("stop:"):
                eid = bit.split(":", 1)[1]
                ser = next((s for s in state.series if s.event_id == eid), None)
                if ser:
                    total += ser.amount
            elif bit.startswith("reduce_to:"):
                _, eid, amt = bit.split(":", 2)
                ser = next((s for s in state.series if s.event_id == eid), None)
                if ser:
                    total += max(0.0, ser.amount - float(amt))
        return total

    def _change_penalty(blob: str) -> tuple[int, int]:
        if blob == "none":
            return (0, 0)
        bits = [b for b in blob.split("|") if b]
        lifestyle = 0
        for bit in bits:
            eid = bit.split(":")[1]
            ser = next((s for s in state.series if s.event_id == eid), None)
            if ser and ser.category in {"shopping", "dining", "entertainment"}:
                lifestyle += 1
        return (lifestyle, len(bits))

    def rank_key(c: Candidate):
        # Completing installment/partial/full beats wait even when wait is cheaper.
        return (
            0 if c.method != "wait" else 1,
            0 if c.spending_changes == "none" else 1,
            c.total_paid,
            c.starts.toordinal() if c.starts else 10**9,
            c.n_payments,
            _change_cut(c.spending_changes),
            _change_penalty(c.spending_changes),
            c.option_id or "zzz",
        )

    safe_cands = [c for c in candidates if c.safe]
    safe_cands.sort(key=rank_key)

    if safe_cands:
        best = safe_cands[0]
        earliest_text = iso(earliest) if earliest else ""
        if best.status == "affordable_now":
            earliest_text = iso(state.request_date)
        return Decision(
            request_id=state.request["request_id"],
            amount_safe_to_pay=safe_today,
            affordability_status=best.status,
            recommended_payment_method=best.method,
            payment_plan=best.payment_plan,
            earliest_date_for_full_payment=earliest_text,
            spending_changes_needed=best.spending_changes,
            decision_explanation=best.explanation,
            candidates=safe_cands,
        )

    return Decision(
        request_id=state.request["request_id"],
        amount_safe_to_pay=safe_today,
        affordability_status="not_affordable",
        recommended_payment_method="not_recommended",
        payment_plan="none",
        earliest_date_for_full_payment=iso(earliest) if earliest else "",
        spending_changes_needed="none",
        decision_explanation=explain(
            state,
            Candidate(
                "none",
                "not_recommended",
                "not_affordable",
                [],
                "none",
                "none",
                0,
                None,
                0,
                "",
                False,
                False,
            ),
        ),
        candidates=[],
    )


def make_state(
    profile: dict,
    request: dict,
    options: list[dict],
    events: list[dict],
    amendment: Amendment,
    series: list[Series],
) -> UserState:
    max_m = parse_float(profile.get("max_installment_months"))
    return UserState(
        profile=profile,
        request=request,
        options=options,
        events=events,
        amendment=amendment,
        series=series,
        home=profile["home_currency"],
        min_balance=float(profile["minimum_balance_to_keep"]),
        start_balance=float(profile["current_available_balance"]),
        request_date=parse_date(request["request_date"]),
        deadline=parse_date(request["desired_completion_date"]),
        requested=float(request["requested_amount"]),
        allows_partial=parse_bool(request.get("allows_partial_payment")),
        methods=split_list(profile.get("payment_methods_user_will_consider")),
        max_inst_months=int(max_m) if max_m is not None else None,
        protect=set(split_list(profile.get("expense_categories_to_protect"))),
        reduce_ok=set(split_list(profile.get("expense_categories_user_is_willing_to_reduce"))),
        stop_ok=set(split_list(profile.get("expense_categories_user_is_willing_to_stop"))),
    )


def decision_row(dec: Decision) -> dict[str, str]:
    return {
        "request_id": dec.request_id,
        "amount_safe_to_pay": fmt_amount(dec.amount_safe_to_pay),
        "affordability_status": dec.affordability_status,
        "recommended_payment_method": dec.recommended_payment_method,
        "payment_plan": dec.payment_plan,
        "earliest_date_for_full_payment": dec.earliest_date_for_full_payment,
        "spending_changes_needed": dec.spending_changes_needed,
        "decision_explanation": dec.decision_explanation.replace("\n", " "),
    }
