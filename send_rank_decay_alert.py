"""
Send short Rank Decay Monitor push notification through Pushover.

Reads:
    rank_decay_monitor/holdings_status.csv

Sends:
    Short Pushover notification.

Does NOT submit orders.
"""

import os
from pathlib import Path

import pandas as pd
import requests


MONITOR_DIR = Path("rank_decay_monitor")
HOLDINGS_STATUS_PATH = MONITOR_DIR / "holdings_status.csv"

PUSHOVER_API_URL = "https://api.pushover.net/1/messages.json"

ALERT_ON_WATCH = os.getenv("ALERT_ON_WATCH", "false").strip().lower() == "true"
ALERT_ALWAYS = os.getenv("ALERT_ALWAYS", "false").strip().lower() == "true"


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def short_status(status_value) -> str:
    text = str(status_value).upper()

    if "REVIEW" in text:
        return "REVIEW"
    if "WATCH" in text:
        return "WATCH"
    if "HOLD" in text:
        return "HOLD"

    return "UNK"


def short_rank(value) -> str:
    if pd.isna(value):
        return "NA"

    try:
        return str(int(float(value)))
    except Exception:
        return "NA"


def build_message() -> tuple[str, str] | None:
    if not HOLDINGS_STATUS_PATH.exists():
        raise FileNotFoundError(f"Missing required file: {HOLDINGS_STATUS_PATH}")

    status_df = pd.read_csv(HOLDINGS_STATUS_PATH)

    if status_df.empty:
        print("No holdings status rows found. No alert sent.")
        return None

    review_df = status_df[
        status_df["status"].astype(str).str.upper() == "REVIEW_REPLACEMENT"
    ].copy()

    watch_df = status_df[
        status_df["status"].astype(str).str.upper() == "WATCH_OUTSIDE_BAND"
    ].copy()

    if review_df.empty and not (ALERT_ON_WATCH and not watch_df.empty) and not ALERT_ALWAYS:
        print("No REVIEW_REPLACEMENT status found. No alert sent.")
        print(status_df.to_string(index=False))
        return None

    if not review_df.empty:
        headline = "REVIEW"
        priority = 1
    elif not watch_df.empty:
        headline = "WATCH"
        priority = 0
    else:
        headline = "HOLD"
        priority = 0

    parts = []

    for _, row in status_df.iterrows():
        symbol = str(row.get("symbol", "")).upper()
        rank = short_rank(row.get("rank"))
        days = row.get("consecutive_outside_days", 0)
        status = short_status(row.get("status", ""))

        try:
            days = int(float(days))
        except Exception:
            days = 0

        parts.append(f"{symbol} r{rank} d{days} {status}")

    title = f"RankDecay {headline}"
    message = " | ".join(parts) + " | Read-only. No orders."

    return title, message, priority


def send_pushover(title: str, message: str, priority: int) -> None:
    api_token = require_env("PUSHOVER_API_TOKEN")
    user_key = require_env("PUSHOVER_USER_KEY")

    response = requests.post(
        PUSHOVER_API_URL,
        data={
            "token": api_token,
            "user": user_key,
            "title": title,
            "message": message,
            "priority": priority,
        },
        timeout=30,
    )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Pushover notification failed: status={response.status_code}, body={response.text}"
        )

    print("Pushover notification sent successfully.")


def main() -> None:
    print("\n===== SEND RANK DECAY PUSHOVER ALERT =====")

    built = build_message()

    if not built:
        return

    title, message, priority = built

    print("\nAlert title:")
    print(title)
    print("\nAlert message:")
    print(message)
    print(f"Message length: {len(message)}")
    print(f"Priority: {priority}")

    send_pushover(title=title, message=message, priority=priority)


if __name__ == "__main__":
    main()