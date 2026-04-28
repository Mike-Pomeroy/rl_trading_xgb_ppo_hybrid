"""
Rebalance guard.

Purpose:
- Prevent accidental duplicate order submissions during the same rebalance period.
- Records successful submit attempts in a local CSV file.
- Used by Alpaca submit scripts before placing orders.

This does NOT connect to Alpaca.
This does NOT submit orders.
"""

from pathlib import Path
from datetime import datetime
import pandas as pd


GUARD_DIR = Path("rebalance_guard_logs")
GUARD_DIR.mkdir(exist_ok=True)

GUARD_LOG_PATH = GUARD_DIR / "rebalance_submissions.csv"


def get_rebalance_period(signal_date: str | pd.Timestamp) -> str:
    """
    Convert a signal date into a rebalance period like '2026-04'.

    Example:
        2026-04-27 -> 2026-04
    """
    dt = pd.Timestamp(signal_date)
    return dt.strftime("%Y-%m")


def get_next_rebalance_dates(signal_date: str | pd.Timestamp) -> dict:
    """
    Estimate the next monthly rebalance dates.

    Operational rule:
    - Preview date: first business day of the next calendar month.
    - Submit date: next business day after preview date.

    Note:
    This uses pandas business days, which handles weekends but not market holidays.
    For our safety warning this is good enough. The preview script itself will
    still use actual available market data dates.
    """
    dt = pd.Timestamp(signal_date).normalize()

    next_month_start = (dt + pd.offsets.MonthBegin(1)).normalize()

    # If next_month_start is a weekend, move to next business day.
    preview_date = pd.bdate_range(
        start=next_month_start,
        periods=1,
    )[0]

    # Submit on the next business day after preview.
    submit_date = pd.bdate_range(
        start=preview_date + pd.Timedelta(days=1),
        periods=1,
    )[0]

    return {
        "next_rebalance_period": preview_date.strftime("%Y-%m"),
        "next_preview_date": preview_date.strftime("%Y-%m-%d"),
        "next_submit_date": submit_date.strftime("%Y-%m-%d"),
        "suggested_preview_time": "9:35–9:45 AM Eastern",
        "suggested_submit_time": "9:35–9:45 AM Eastern",
    }


def load_guard_log() -> pd.DataFrame:
    if not GUARD_LOG_PATH.exists():
        return pd.DataFrame(
            columns=[
                "strategy_name",
                "rebalance_period",
                "signal_date",
                "submitted_at",
                "mode",
                "notes",
            ]
        )

    return pd.read_csv(GUARD_LOG_PATH)


def already_submitted(
    strategy_name: str,
    rebalance_period: str,
    mode: str = "paper",
) -> bool:
    log = load_guard_log()

    if log.empty:
        return False

    matches = log[
        (log["strategy_name"].astype(str) == strategy_name)
        & (log["rebalance_period"].astype(str) == rebalance_period)
        & (log["mode"].astype(str) == mode)
    ]

    return not matches.empty


def get_existing_submission_details(
    strategy_name: str,
    rebalance_period: str,
    mode: str = "paper",
) -> pd.DataFrame:
    log = load_guard_log()

    if log.empty:
        return log

    matches = log[
        (log["strategy_name"].astype(str) == strategy_name)
        & (log["rebalance_period"].astype(str) == rebalance_period)
        & (log["mode"].astype(str) == mode)
    ].copy()

    return matches


def assert_not_already_submitted(
    strategy_name: str,
    signal_date: str | pd.Timestamp,
    mode: str = "paper",
    allow_resubmit: bool = False,
) -> str:
    """
    Raises RuntimeError if this strategy already submitted for the period.

    Returns:
        rebalance_period
    """
    rebalance_period = get_rebalance_period(signal_date)
    next_dates = get_next_rebalance_dates(signal_date)

    if allow_resubmit:
        print(
            f"WARNING: allow_resubmit=True. "
            f"Duplicate guard bypassed for {strategy_name} {rebalance_period}."
        )
        return rebalance_period

    if already_submitted(strategy_name, rebalance_period, mode=mode):
        existing = get_existing_submission_details(
            strategy_name=strategy_name,
            rebalance_period=rebalance_period,
            mode=mode,
        )

        existing_text = ""

        if not existing.empty:
            existing_text = "\nExisting recorded submission(s):\n"
            existing_text += existing.to_string(index=False)

        raise RuntimeError(
            f"Duplicate submission blocked.\n"
            f"Strategy: {strategy_name}\n"
            f"Mode: {mode}\n"
            f"Current rebalance period: {rebalance_period}\n"
            f"Current signal date: {pd.Timestamp(signal_date).date()}\n"
            f"{existing_text}\n\n"
            f"Next expected rebalance period: {next_dates['next_rebalance_period']}\n"
            f"Next preview date: {next_dates['next_preview_date']} "
            f"around {next_dates['suggested_preview_time']}\n"
            f"Next submit date: {next_dates['next_submit_date']} "
            f"around {next_dates['suggested_submit_time']}\n\n"
            f"This period already has a recorded submission in:\n"
            f"{GUARD_LOG_PATH}\n\n"
            f"If this is intentional, set ALLOW_RESUBMIT = True in the submit script, "
            f"but only after checking Alpaca open orders and positions."
        )

    return rebalance_period


def record_submission(
    strategy_name: str,
    signal_date: str | pd.Timestamp,
    mode: str = "paper",
    notes: str = "",
) -> None:
    rebalance_period = get_rebalance_period(signal_date)

    log = load_guard_log()

    new_row = {
        "strategy_name": strategy_name,
        "rebalance_period": rebalance_period,
        "signal_date": str(pd.Timestamp(signal_date).date()),
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "notes": notes,
    }

    log = pd.concat([log, pd.DataFrame([new_row])], ignore_index=True)
    log.to_csv(GUARD_LOG_PATH, index=False)

    next_dates = get_next_rebalance_dates(signal_date)

    print(
        f"\nRecorded rebalance submission guard:\n"
        f"Strategy: {strategy_name}\n"
        f"Period:   {rebalance_period}\n"
        f"Mode:     {mode}\n"
        f"Log:      {GUARD_LOG_PATH}\n\n"
        f"Next expected preview date: {next_dates['next_preview_date']} "
        f"around {next_dates['suggested_preview_time']}\n"
        f"Next expected submit date:  {next_dates['next_submit_date']} "
        f"around {next_dates['suggested_submit_time']}"
    )