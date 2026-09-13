from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from .formatters import parse_date


MONEY_RE = re.compile(
    r"(?:USD|EUR|INR|IDR|ZAR|GBP)?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)",
    re.I,
)
DATE_RE = re.compile(r"(20[0-9]{2}-[0-9]{2}-[0-9]{2})")


def _first_money(text: str) -> float | None:
    match = MONEY_RE.search(text.replace(" ", ""))
    if not match:
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def _all_dates(text: str) -> list[date]:
    out = []
    for raw in DATE_RE.findall(text):
        parsed = parse_date(raw)
        if parsed:
            out.append(parsed)
    return out


def _first_date(text: str) -> date | None:
    dates = _all_dates(text)
    return dates[0] if dates else None


@dataclass
class Amendment:
    salary_amount: float | None = None
    salary_date: date | None = None
    salary_temporary: bool = False
    salary_increase_from: date | None = None
    first_salary: bool = False
    employment_ended: bool = False
    seasonal_ended: bool = False
    household_ended: bool = False
    remaining_salary: float | None = None
    arrears_amount: float | None = None
    rent_increase_pct: float | None = None
    confirmed_invoice: float | None = None
    invoice_date: date | None = None
    ignore_pending_credit: bool = False
    ignore_unrealized: bool = False
    ignore_prize: bool = False
    ignore_scam: bool = False
    internal_transfer: bool = False
    failed_debit_open: bool = False
    dispute_open: bool = False
    fx_settlement: bool = False
    childcare_starts: bool = False
    notes: list[str] = field(default_factory=list)


def parse_message(text: str) -> Amendment:
    raw = text or ""
    low = raw.lower()
    amd = Amendment()

    if "pay the release" in low or "pay the processing charge" in low or "avoid losing the claim" in low:
        amd.ignore_scam = True
        amd.ignore_prize = True
        amd.notes.append("scam_prize")
        return amd

    if "displayed market value" in low or "displayed value of the investment" in low or "unrealized" in low:
        amd.ignore_unrealized = True
        amd.notes.append("unrealized_investment")
    if "no units have been sold" in low or "holding has not been sold" in low:
        amd.ignore_unrealized = True

    if "refund has been initiated but has not reached" in low or "refund is still processing" in low:
        amd.ignore_pending_credit = True
        amd.notes.append("pending_refund")
    if "prize claim has been verified and is still in payment processing" in low:
        amd.ignore_pending_credit = True
        amd.ignore_prize = True
        amd.notes.append("pending_prize")
    if "payout is still pending" in low or "isn't withdrawable" in low or "belum dapat ditarik" in low:
        amd.ignore_pending_credit = True
        amd.notes.append("pending_payout")
    if "bonus" in low and ("not been approved" in low or "still subject" in low or "masih menunggu" in low):
        amd.ignore_pending_credit = True
        amd.notes.append("pending_bonus")
    if "commission" in low and ("pending" in low or "belum" in low or "still pending" in low or "open deals" in low):
        amd.ignore_pending_credit = True
        amd.notes.append("pending_commission")

    if "transfer between your two accounts" in low or "transfer antara dua rekening" in low:
        amd.internal_transfer = True
        amd.notes.append("internal_transfer")

    if "previous debit attempt failed" in low or "debit attempt failed" in low:
        amd.failed_debit_open = True
        amd.notes.append("failed_debit_retry")
    if "reversal has not been posted" in low or "dispute is open" in low:
        amd.dispute_open = True
        amd.ignore_pending_credit = True
        amd.notes.append("open_dispute")

    if "settlement-date" in low or "settlement date" in low or "kurs pada tanggal" in low:
        amd.fx_settlement = True
        amd.notes.append("use_settlement_fx")

    if "increases monthly rent by 12%" in low or "menaikkan biaya sewa bulanan sebesar 12%" in low:
        amd.rent_increase_pct = 12.0
        amd.notes.append("rent_plus_12")

    if "employment has ended" in low or "hubungan kerja anda telah berakhir" in low:
        amd.employment_ended = True
        amd.notes.append("employment_ended")
    if "seasonal contract has ended" in low or "kontrak musiman saat ini telah berakhir" in low:
        amd.seasonal_ended = True
        amd.notes.append("seasonal_ended")
    if "household employment record has ended" in low or "pendapatan kerja rumah tangga telah berakhir" in low:
        amd.household_ended = True
        amd.notes.append("household_ended")
        amt = _amount_after(raw, ["remaining confirmed monthly salary is", "sisa gaji bulanan yang dikonfirmasi adalah"])
        if amt is not None:
            amd.remaining_salary = amt
            amd.salary_amount = amt

    if "childcare" in low:
        amd.childcare_starts = True
        amd.notes.append("childcare_starts")

    if "first salary" in low or "gaji pertama" in low:
        amd.first_salary = True
        amd.salary_amount = _amount_after(
            raw,
            [
                "first salary will be",
                "first salary from the new employer is",
                "first salary of",
                "gaji pertama akan menjadi",
                "gaji pertama dari perusahaan baru adalah",
                "gaji pertama anda sebesar",
                "gaji pertama anda adalah",
            ],
        ) or _first_money(raw)
        amd.salary_date = _date_after(
            raw,
            ["confirmed credit date is", "confirmed for", "scheduled for", "tanggal kredit yang dikonfirmasi adalah", "diketahui untuk", "pada"],
        ) or _first_date(raw)
        amd.notes.append("first_salary")

    if "temporary monthly pay is" in low or "gaji bulanan sementara anda adalah" in low:
        amd.salary_temporary = True
        amd.salary_amount = _amount_after(raw, ["temporary monthly pay is", "gaji bulanan sementara anda adalah"])
        amd.notes.append("temp_pay")

    if "next salary is reduced to" in low or "next salary is reduced" in low or "gaji berikutnya dikurangi" in low:
        amd.salary_temporary = True
        amd.salary_amount = _amount_after(raw, ["next salary is reduced to", "reduced to"])
        amd.notes.append("salary_reduced")

    if "monthly salary has increased to" in low or "gaji bulanan anda naik menjadi" in low:
        amd.salary_amount = _amount_after(raw, ["monthly salary has increased to", "gaji bulanan anda naik menjadi"])
        amd.salary_increase_from = _date_after(raw, ["applies from", "berlaku mulai"])
        amd.notes.append("salary_increase")

    if "confirmed salary is now expected on" in low or "kini diperkirakan masuk pada" in low:
        amd.salary_date = _date_after(raw, ["expected on", "masuk pada"])
        amd.notes.append("salary_date_moved")

    if "regular salary of" in low and "resumes on" in low:
        amd.salary_amount = _amount_after(raw, ["regular salary of"])
        amd.salary_date = _date_after(raw, ["resumes on"])
        amd.notes.append("salary_resumes")

    if "confirmed base salary is" in low or "gaji pokok yang dikonfirmasi adalah" in low:
        amd.salary_amount = _amount_after(raw, ["confirmed base salary is", "gaji pokok yang dikonfirmasi adalah"])
        amd.notes.append("base_salary")

    if "regular salary for the next payroll is" in low or "gaji rutin anda untuk penggajian berikutnya adalah" in low:
        amd.salary_amount = _amount_after(
            raw, ["regular salary for the next payroll is", "gaji rutin anda untuk penggajian berikutnya adalah"]
        )
        amd.notes.append("next_regular_salary")
        arrears = _amount_after(raw, ["one-time arrears adjustment of", "penyesuaian tunggakan satu kali sebesar"])
        if arrears is not None:
            amd.arrears_amount = arrears

    if "client approved an invoice payment of" in low or "menyetujui pembayaran faktur sebesar" in low:
        amd.confirmed_invoice = _amount_after(
            raw, ["invoice payment of", "pembayaran faktur sebesar"]
        )
        amd.invoice_date = _date_after(raw, ["settlement is expected on", "penyelesaian diperkirakan pada"])
        amd.notes.append("confirmed_invoice")

    if amd.salary_amount is None and ("salary" in low or "gaji" in low) and not amd.employment_ended:
        # last-resort salary figure if a payroll message mentioned a number
        if "payroll" in low or "penggajian" in low:
            pass

    return amd


def _amount_after(text: str, markers: list[str]) -> float | None:
    low = text.lower()
    for marker in markers:
        idx = low.find(marker.lower())
        if idx < 0:
            continue
        window = text[idx + len(marker) : idx + len(marker) + 80]
        match = re.search(
            r"(USD|EUR|INR|IDR|ZAR)?\s*([0-9]{4,}(?:\.[0-9]+)?|[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)",
            window,
            re.I,
        )
        if match:
            return float(match.group(2).replace(",", ""))
    return None


def _date_after(text: str, markers: list[str]) -> date | None:
    low = text.lower()
    for marker in markers:
        idx = low.find(marker.lower())
        if idx < 0:
            continue
        window = text[idx : idx + 80]
        match = DATE_RE.search(window)
        if match:
            return parse_date(match.group(1))
    return None


def parse_user_messages(rows: list[dict[str, str]]) -> Amendment:
    merged = Amendment()
    for row in sorted(rows, key=lambda r: r.get("sent_at") or ""):
        part = parse_message(row.get("message_text") or "")
        for name in (
            "salary_amount",
            "salary_date",
            "salary_increase_from",
            "remaining_salary",
            "arrears_amount",
            "rent_increase_pct",
            "confirmed_invoice",
            "invoice_date",
        ):
            val = getattr(part, name)
            if val is not None:
                setattr(merged, name, val)
        for name in (
            "salary_temporary",
            "first_salary",
            "employment_ended",
            "seasonal_ended",
            "household_ended",
            "ignore_pending_credit",
            "ignore_unrealized",
            "ignore_prize",
            "ignore_scam",
            "internal_transfer",
            "failed_debit_open",
            "dispute_open",
            "fx_settlement",
            "childcare_starts",
        ):
            if getattr(part, name):
                setattr(merged, name, True)
        merged.notes.extend(part.notes)
    return merged
