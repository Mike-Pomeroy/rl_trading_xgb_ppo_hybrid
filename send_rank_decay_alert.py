"""
Send Rank Decay Monitor Alert

Purpose:
- Read rank_decay_monitor/holdings_status.csv.
- Send a short SMS alert through Twilio.
- Can send either:
    - only REVIEW_REPLACEMENT alerts, or
    - daily status messages when ALERT_ALWAYS=true.
- Does NOT submit orders.

Required GitHub Secrets / environment variables:
    TWILIO_ACCOUNT_SID
    TWILIO_AUTH_TOKEN
    TWILIO_FROM_NUMBER
    ALERT_TO_NUMBER

Optional:
    ALERT_ON_WATCH = "true" to also alert on WATCH_OUTSIDE_BAND
    ALERT_ALWAYS = "true" to send a daily status message even when HOLD
"""

import os
from pathlib import Path

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth


MONITOR_DIR = Path("rank_decay_monitor")
HOLDINGS_STATUS_PATH = MONITOR_DIR / "holdings_status.csv"

ALERT_ON_WATCH = os.getenv("ALERT_ON_WATCH", "false").strip().lower() == "true"
ALERT_ALWAYS = os.getenv("ALERT_ALWAYS", "false").strip().lower