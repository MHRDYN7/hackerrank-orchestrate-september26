from __future__ import annotations

import sqlite3

from .paths import DB_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    home_currency TEXT,
    current_available_balance REAL,
    minimum_balance_to_keep REAL,
    payment_methods TEXT,
    max_installment_months TEXT
);
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    user_id TEXT,
    request_date TEXT,
    requested_amount REAL,
    desired_completion_date TEXT,
    allows_partial_payment TEXT,
    request_type TEXT
);
CREATE TABLE IF NOT EXISTS options (
    payment_option_id TEXT PRIMARY KEY,
    request_id TEXT,
    payment_method TEXT,
    payment_amount REAL,
    number_of_payments INTEGER,
    first_payment_date TEXT,
    payment_frequency_days TEXT,
    financing_fee REAL,
    total_payable_amount REAL
);
CREATE TABLE IF NOT EXISTS events_normalized (
    event_id TEXT PRIMARY KEY,
    user_id TEXT,
    event_type TEXT,
    description TEXT,
    category TEXT,
    direction TEXT,
    amount_home REAL,
    event_date TEXT,
    settlement_date TEXT,
    status TEXT,
    flexibility TEXT,
    linked_event_id TEXT
);
CREATE TABLE IF NOT EXISTS messages_parsed (
    user_id TEXT PRIMARY KEY,
    notes TEXT,
    salary_amount REAL,
    salary_date TEXT
);
CREATE TABLE IF NOT EXISTS recurring_series (
    series_id TEXT PRIMARY KEY,
    user_id TEXT,
    description TEXT,
    category TEXT,
    direction TEXT,
    amount REAL,
    last_date TEXT,
    event_id TEXT,
    flexibility TEXT
);
CREATE TABLE IF NOT EXISTS image_amounts (
    image_id TEXT PRIMARY KEY,
    event_id TEXT,
    amount REAL,
    caption TEXT
);
"""


def write_sqlite(store) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO users VALUES (?,?,?,?,?,?)",
            [
                (
                    p["user_id"],
                    p["home_currency"],
                    float(p["current_available_balance"]),
                    float(p["minimum_balance_to_keep"]),
                    p.get("payment_methods_user_will_consider") or "",
                    p.get("max_installment_months") or "",
                )
                for p in store.profiles.values()
            ],
        )
        reqs = list(store.requests.values()) + [
            {k: v for k, v in s.items() if k in {
                "request_id", "user_id", "request_date", "requested_amount",
                "desired_completion_date", "allows_partial_payment", "request_type",
            }}
            for s in store.samples.values()
            if s["request_id"] not in store.requests
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO requests VALUES (?,?,?,?,?,?,?)",
            [
                (
                    r["request_id"],
                    r["user_id"],
                    r["request_date"],
                    float(r["requested_amount"]),
                    r["desired_completion_date"],
                    r.get("allows_partial_payment") or "",
                    r.get("request_type") or "",
                )
                for r in reqs
            ],
        )
        opt_rows = []
        for opts in store.options.values():
            for o in opts:
                opt_rows.append(
                    (
                        o["payment_option_id"],
                        o["request_id"],
                        o["payment_method"],
                        float(o["payment_amount"]),
                        int(float(o["number_of_payments"] or 1)),
                        o["first_payment_date"],
                        o.get("payment_frequency_days") or "",
                        float(o.get("financing_fee") or 0),
                        float(o.get("total_payable_amount") or 0),
                    )
                )
        conn.executemany("INSERT OR REPLACE INTO options VALUES (?,?,?,?,?,?,?,?,?)", opt_rows)
        ev_rows = []
        for user_id, rows in store.events.items():
            for e in rows:
                ev_rows.append(
                    (
                        e["event_id"],
                        user_id,
                        e.get("event_type") or "",
                        e.get("description") or "",
                        e.get("category") or "",
                        e.get("direction") or "",
                        e.get("amount_home"),
                        str(e.get("event_date_p") or ""),
                        str(e.get("settlement_date_p") or ""),
                        e.get("status") or "",
                        e.get("flexibility") or "",
                        e.get("linked_event_id") or "",
                    )
                )
        conn.executemany(
            "INSERT OR REPLACE INTO events_normalized VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            ev_rows,
        )
        conn.executemany(
            "INSERT OR REPLACE INTO messages_parsed VALUES (?,?,?,?)",
            [
                (
                    uid,
                    "|".join(a.notes),
                    a.salary_amount,
                    a.salary_date.isoformat() if a.salary_date else "",
                )
                for uid, a in store.amendments.items()
            ],
        )
        conn.executemany(
            "INSERT OR REPLACE INTO recurring_series VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (
                    s.series_id,
                    s.user_id,
                    s.description,
                    s.category,
                    s.direction,
                    s.amount,
                    s.last_date.isoformat(),
                    s.event_id,
                    s.flexibility,
                )
                for rows in store.series.values()
                for s in rows
            ],
        )
        conn.executemany(
            "INSERT OR REPLACE INTO image_amounts VALUES (?,?,?,?)",
            [
                (iid, meta["event_id"], float(meta["amount"]), meta.get("caption") or "")
                for iid, meta in store.image_meta.items()
            ],
        )
        conn.commit()
    finally:
        conn.close()


def query_cash_items(user_id: str, status: str | None = None, category: str | None = None, upcoming_only: bool = False, limit: int = 40) -> list[dict]:
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        sql = "SELECT * FROM events_normalized WHERE user_id = ?"
        args: list = [user_id]
        if status:
            sql += " AND status = ?"
            args.append(status)
        if category:
            sql += " AND category = ?"
            args.append(category)
        if upcoming_only:
            sql += " AND status IN ('pending','scheduled')"
        sql += " ORDER BY settlement_date LIMIT ?"
        args.append(limit)
        return [dict(r) for r in conn.execute(sql, args)]
    finally:
        conn.close()
