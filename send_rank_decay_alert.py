"""
Send short Rank Decay Monitor SMS alert.

Reads:
    rank_decay_monitor/holdings_status.csv

Sends:
    Very short plain GSM-style SMS.

Does NOT submit orders.
"""

import os
from pathlib import Path

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth


MONITOR_DIR = Path("rank_decay_monitor")
HOLDINGS_STATUS_PATH = MONITOR_DIR / "holdings_status.csv"

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


def build_message() -> str | None:
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
    elif not watch_df.empty:
        headline = "WATCH"
    else:
        headline = "HOLD"

    parts = [f"RD {headline}"]

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

    parts.append("NOORDERS")

    message = " ".join(parts)

    # Hard cap to avoid trial segment issues.
    if len(message) > 150:
        message = message[:150]

    return message


def send_sms(message: str) -> None:
    account_sid = require_env("TWILIO_ACCOUNT_SID")
    auth_token = require_env("TWILIO_AUTH_TOKEN")
    from_number = require_env("TWILIO_FROM_NUMBER")
    to_number = require_env("ALERT_TO_NUMBER")

    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"

    response = requests.post(
        url,
        data={
            "From": from_number,
            "To": to_number,
            "Body": message,
        },
        auth=HTTPBasicAuth(account_sid, auth_token),
        timeout=30,
    )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Twilio SMS failed: status={response.status_code}, body={response.text}"
        )

    print("SMS alert sent successfully.")


def main() -> None:
    print("\n===== SEND RANK DECAY ALERT =====")

    message = build_message()

    if not message:
        return

    print("\nAlert message:")
    print(message)
    print(f"Message length: {len(message)}")

    send_sms(message)


if __name__ == "__main__":
    main()