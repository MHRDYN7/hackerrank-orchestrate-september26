from __future__ import annotations

from .formatters import parse_date, parse_float


OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}


def validate_row(row: dict, request: dict, options: list[dict]) -> list[str]:
    errors = []
    for col in OUTPUT_COLUMNS:
        if col not in row:
            errors.append(f"missing {col}")
    if errors:
        return errors
    requested = float(request["requested_amount"])
    safe = parse_float(row["amount_safe_to_pay"])
    if safe is None or safe < -1e-6 or safe > requested + 1e-6:
        errors.append("amount_safe_to_pay out of bounds")
    if row["affordability_status"] not in STATUSES:
        errors.append("bad status")
    if row["recommended_payment_method"] not in METHODS:
        errors.append("bad method")
    if row["affordability_status"] == "affordable_now":
        if row["earliest_date_for_full_payment"] != request["request_date"]:
            errors.append("affordable_now earliest_date mismatch")
    if row["recommended_payment_method"] == "installments" and row["payment_plan"] != "none":
        if not _matches_option(row["payment_plan"], options):
            errors.append("installment plan does not match an option")
    if row["recommended_payment_method"] == "partial_payment":
        parts = row["payment_plan"].split("|")
        if len(parts) != 2:
            errors.append("partial_payment must have exactly two payments")
        else:
            try:
                a = float(parts[0].split(":")[1])
                b = float(parts[1].split(":")[1])
                if abs(a + b - requested) > 0.05:
                    errors.append("partial payments do not sum to requested")
            except (IndexError, ValueError):
                errors.append("bad partial plan")
    changes = row["spending_changes_needed"]
    if changes != "none":
        bits = changes.split("|")
        if len(bits) > 3:
            errors.append("too many spending changes")
    return errors


def _matches_option(plan: str, options: list[dict]) -> bool:
    from .engine import expand_option, plan_text

    for opt in options:
        if opt.get("payment_method") != "installments":
            continue
        if plan_text(expand_option(opt)) == plan:
            return True
    return False
