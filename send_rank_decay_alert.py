"""
Send Rank Decay Monitor Alert

Purpose:
- Read rank_decay_monitor/holdings_status.csv and latest_rankings.csv.
- If any holding has status REVIEW_REPLACEMENT, send an SMS alert through Twilio.
- Otherwise, print HOLD / no alert.
- Does NOT submit orders.

Required GitHub Secrets / environment variables:
    TWILIO_ACCOUNT_SID
    TWILIO_AUTH_TOKEN
    TWILIO_FROM_NUMBER
    ALERT_TO_NUMBER

Optional:
    ALERT_ON_WATCH = "true" to also alert on WATCH_OUTSIDE_BAND
"""

import os
from pathlib import Path

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth


MONITOR_DIR = Path("rank_decay_monitor")
HOLDINGS_STATUS_PATH = MONITOR_DIR / "holdings_status.csv"
RANKINGS_PATH = MONITOR_DIR / "latest_rankings.csv"
REPORT_PATH = MONITOR_DIR / "rank_decay_monitor_report.txt"

ALERT_ON_WATCH = os.getenv("ALERT_ON_WATCH", "false").strip().lower() == "true"
ALERT_ALWAYS = os.getenv("ALERT_ALWAYS", "false").strip().lower() == "true"

def require_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")

    return value


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    if path.stat().st_size == 0:
        raise RuntimeError(f"Required file is empty: {path}")

    return pd.read_csv(path)


def build_message() -> str | None:
    status_df = read_csv_required(HOLDINGS_STATUS_PATH)

    if status_df.empty:
        print("No holdings status rows found. No alert sent.")
        return None

    review_df = status_df[
        status_df["status"].astype(str).str.upper() == "REVIEW_REPLACEMENT"
    ].copy()

    watch_df = status_df[
        status_df["status"].astype(str).str.upper() == "WATCH_OUTSIDE_BAND"
    ].copy()

    # if review_df.empty and not (ALERT_ON_WATCH and not watch_df.empty):
    if review_df.empty and not (ALERT_ON_WATCH and not watch_df.empty) and not ALERT_ALWAYS:    
        print("No REVIEW_REPLACEMENT status found. No alert sent.")
        print(status_df.to_string(index=False))
        return None

    rankings_df = pd.DataFrame()

    if RANKINGS_PATH.exists() and RANKINGS_PATH.stat().st_size > 0:
        rankings_df = pd.read_csv(RANKINGS_PATH)

    if not rankings_df.empty:
        top_rankings = rankings_df.head(5)
        top_text = "\n".join(
            f"{int(row['rank'])}. {row['tic']} score={float(row['score']):.4f}"
            for _, row in top_rankings.iterrows()
        )
    else:
        top_text = "Rankings file not available."


    if not review_df.empty:
        headline = "REVIEW_REPLACEMENT triggered"
        trigger_df = review_df
    elif ALERT_ON_WATCH and not watch_df.empty:
        headline = "WATCH_OUTSIDE_BAND alert"
        trigger_df = watch_df
    else:
        headline = "Daily HOLD status"
        trigger_df = status_df



    trigger_lines = []

    for _, row in trigger_df.iterrows():
        trigger_lines.append(
            f"{row['symbol']}: rank={row.get('rank')}, "
            f"outside_days={row.get('consecutive_outside_days')}/"
            f"{row.get('confirm_days')}, status={row.get('status')}"
        )

    all_status_lines = []

    for _, row in status_df.iterrows():
        all_status_lines.append(
            f"{row['symbol']}: rank={row.get('rank')}, "
            f"days={row.get('consecutive_outside_days')}, "
            f"{row.get('status')}"
        )

    message = (
        f"Rank Decay Monitor: {headline}\n\n"
        f"Triggered:\n" + "\n".join(trigger_lines) + "\n\n"
        f"All holdings:\n" + "\n".join(all_status_lines) + "\n\n"
        f"Top rankings:\n{top_text}\n\n"
        f"Read-only alert. No Alpaca orders submitted."
    )

    # Keep SMS reasonably short. Full report is in GitHub Actions artifact.
    if len(message) > 1400:
        message = message[:1350] + "\n\n...truncated. See GitHub Actions artifact."

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

    send_sms(message)


if __name__ == "__main__":
    main()