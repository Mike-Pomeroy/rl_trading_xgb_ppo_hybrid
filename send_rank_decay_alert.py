"""
Send Rank Decay Monitor notification through Pushover.

Reads:
    rank_decay_monitor/holdings_status.csv
    rank_decay_monitor/latest_rankings.csv

Sends:
    A readable Pushover/email status message.

Does NOT submit orders.
"""

import os
from pathlib import Path

import pandas as pd
import requests


MONITOR_DIR = Path("rank_decay_monitor")
HOLDINGS_STATUS_PATH = MONITOR_DIR / "holdings_status.csv"
RANKINGS_PATH = MONITOR_DIR / "latest_rankings.csv"

PUSHOVER_API_URL = "https://api.pushover.net/1/messages.json"

ALERT_ON_WATCH = os.getenv("ALERT_ON_WATCH", "false").strip().lower() == "true"
ALERT_ALWAYS = os.getenv("ALERT_ALWAYS", "false").strip().lower() == "true"


def require_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")

    return value


def rank_text(value) -> str:
    if pd.isna(value):
        return "N/A"

    try:
        return str(int(float(value)))
    except Exception:
        return "N/A"


def build_message() -> tuple[str, str, int] | None:
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
        headline = "REVIEW_REPLACEMENT"
        title = "RankDecay REVIEW"
        priority = 1
    elif not watch_df.empty:
        headline = "WATCH"
        title = "RankDecay WATCH"
        priority = 0
    else:
        headline = "HOLD"
        title = "RankDecay HOLD"
        priority = 0

    holding_lines = []

    for _, row in status_df.iterrows():
        symbol = str(row.get("symbol", "")).upper()
        rank = rank_text(row.get("rank"))
        outside_days = row.get("consecutive_outside_days", 0)
        confirm_days = row.get("confirm_days", 3)
        status = str(row.get("status", ""))

        try:
            outside_days = int(float(outside_days))
        except Exception:
            outside_days = 0

        try:
            confirm_days = int(float(confirm_days))
        except Exception:
            confirm_days = 3

        holding_lines.append(
            f"- {symbol}: rank {rank}, outside {outside_days}/{confirm_days}, {status}"
        )

    top_lines = []

    if RANKINGS_PATH.exists() and RANKINGS_PATH.stat().st_size > 0:
        rankings_df = pd.read_csv(RANKINGS_PATH)

        if not rankings_df.empty:
            for _, row in rankings_df.head(12).iterrows():
                rank = rank_text(row.get("rank"))
                ticker = str(row.get("tic", "")).upper()

                try:
                    score = float(row.get("score"))
                    score_text = f"{score:.4f}"
                except Exception:
                    score_text = "N/A"

                top_lines.append(f"{rank}. {ticker} score {score_text}")

    if not top_lines:
        top_lines.append("Top rankings unavailable.")

    message = (
        f"Status: {headline}\n\n"
        f"Holdings:\n"
        + "\n".join(holding_lines)
        + "\n\nTop 12 rankings:\n"
        + "\n".join(top_lines)
        + "\n\nRead-only alert. No Alpaca orders submitted."
    )

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

    print(f"\nMessage length: {len(message)}")
    print(f"Priority: {priority}")

    send_pushover(title=title, message=message, priority=priority)


if __name__ == "__main__":
    main()