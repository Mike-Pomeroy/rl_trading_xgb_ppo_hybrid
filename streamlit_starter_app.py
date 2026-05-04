import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Trading Starter Dashboard",
    page_icon="📈",
    layout="wide",
)


BASE_DIR = Path(__file__).resolve().parent

PREVIEW_DIR = BASE_DIR / "alpaca_preview_orders_hybrid"
PROPOSED_ORDERS_PATH = PREVIEW_DIR / "proposed_orders.csv"
CURRENT_POSITIONS_PATH = PREVIEW_DIR / "current_positions.csv"
MODEL_SCORES_PATH = PREVIEW_DIR / "model_scores.csv"
OPEN_ORDERS_PATH = PREVIEW_DIR / "open_orders.csv"

ACCOUNT_RECONCILIATION_SUMMARY_PATH = (
    BASE_DIR / "account_reconciliation_reports" / "account_reconciliation_summary.txt"
)
TICKER_RANKING_PATH = BASE_DIR / "ticker_ranking_results" / "ticker_ranking.csv"

STALE_AFTER_HOURS = 24


def get_secret_or_env(name: str, default=None):
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return pd.DataFrame()


def file_modified_time(path: Path):
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime)


def file_age_label(path: Path) -> str:
    modified = file_modified_time(path)
    if modified is None:
        return "Missing"

    age = datetime.now() - modified

    if age < timedelta(minutes=1):
        return "Updated less than 1 minute ago"
    if age < timedelta(hours=1):
        minutes = int(age.total_seconds() // 60)
        return f"Updated {minutes} minute(s) ago"
    if age < timedelta(days=1):
        hours = int(age.total_seconds() // 3600)
        return f"Updated {hours} hour(s) ago"

    days = age.days
    return f"Updated {days} day(s) ago"


def is_file_stale(path: Path, stale_after_hours: int = STALE_AFTER_HOURS) -> bool:
    modified = file_modified_time(path)
    if modified is None:
        return True

    return datetime.now() - modified > timedelta(hours=stale_after_hours)


def show_dataframe(title: str, df: pd.DataFrame, empty_message: str):
    st.subheader(title)
    if df.empty:
        st.info(empty_message)
    else:
        st.dataframe(df, width='stretch')


def show_text_file(title: str, path: Path, empty_message: str):
    st.subheader(title)
    if not path.exists():
        st.info(empty_message)
        return

    try:
        text = path.read_text()
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return

    if not text.strip():
        st.info(empty_message)
    else:
        st.text(text)


st.title("📈 Trading Starter Dashboard")

st.warning(
    "Read-only dashboard. This app does not submit, cancel, replace, or close orders. "
    "Use your existing Cursor trading workflow for aual trade submission."
)

trading_mode = get_secret_or_env("TRADING_MODE", "paper")
enable_live_trading = str(get_secret_or_env("ENABLE_LIVE_TRADING", "false")).lower() == "true"
require_manual_approval = str(get_secret_or_env("REQUIRE_MANUAL_APPROVAL", "true")).lower() == "true"
cash_buffer_pct = get_secret_or_env("CASH_BUFFER_PCT", "0.15")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Trading Mode", str(trading_mode).upper())

with col2:
    st.metric("Live Trading Enabled", "YES" if enable_live_trading else "NO")

with col3:
    st.metric("Manual Approval", "YES" if require_manual_approval else "NO")

with col4:
    st.metric("Cash Buffer", str(cash_buffer_pct))

if str(trading_mode).lower() == "live" or enable_live_trading:
    st.error(
        "Live trading appears to be enabled in configuration. "
        "This dashboard is still read-only, but verify your settings carefully."
    )
else:
    st.success("Paper/safe mode configuration detected.")

st.divider()

st.header("Preview File Freshness")

preview_files = {
    "Proposed Orders": PROPOSED_ORDERS_PATH,
    "Current Positions": CURRENT_POSITIONS_PATH,
    "Model Scores": MODEL_SCORES_PATH,
    "Open Orders": OPEN_ORDERS_PATH,
}

freshness_rows = []

for label, path in preview_files.items():
    freshness_rows.append(
        {
            "File": label,
            "Path": str(path.relative_to(BASE_DIR)),
            "Status": "Exists" if path.exists() else "Missing",
            "Last Updated": file_age_label(path),
            "Stale": "YES" if is_file_stale(path) else "NO",
        }
    )

freshness_df = pd.DataFrame(freshness_rows)
st.dataframe(freshness_df, width='stretch')

missing_files = [label for label, path in preview_files.items() if not path.exists()]
stale_files = [label for label, path in preview_files.items() if is_file_stale(path)]

if missing_files:
    st.error(
        "Some preview files are missing. Run the normal preview workflow before relying on this dashboard."
    )
elif stale_files:
    st.warning(
        f"Some preview files appear older than {STALE_AFTER_HOURS} hours. "
        "Run a fresh preview before submitting trades."
    )
else:
    st.success("Preview files look fresh.")

st.divider()

proposed_orders = load_csv(PROPOSED_ORDERS_PATH)
current_positions = load_csv(CURRENT_POSITIONS_PATH)
model_scores = load_csv(MODEL_SCORES_PATH)
open_orders = load_csv(OPEN_ORDERS_PATH)
ticker_rankings = load_csv(TICKER_RANKING_PATH)

st.header("Portfolio / Order Preview Files")

col_a, col_b, col_c, col_d = st.columns(4)

with col_a:
    st.metric("Proposed Orders", len(proposed_orders))

with col_b:
    st.metric("Current Positions", len(current_positions))

with col_c:
    st.metric("Model Score Rows", len(model_scores))

with col_d:
    st.metric("Open Orders", len(open_orders))

st.divider()

show_dataframe(
    "Proposed Orders — Preview Only",
    proposed_orders,
    "No proposed orders file found or no proposed orders available.",
)

show_dataframe(
    "Current Positions",
    current_positions,
    "No current positions file found.",
)

show_dataframe(
    "Model Scores",
    model_scores,
    "No model scores file found.",
)

show_dataframe(
    "Open Orders",
    open_orders,
    "No open orders file found.",
)

st.divider()

show_text_file(
    "Account Reconciliation Summary",
    ACCOUNT_RECONCILIATION_SUMMARY_PATH,
    "No account reconciliation summary found.",
)

show_dataframe(
    "Latest Ticker Rankings",
    ticker_rankings,
    "No ticker ranking file found.",
)

st.divider()

st.header("Safety Checklist")

st.markdown(
    """
- This dashboard only reads local CSV/text output files.
- It does not connect to Alpaca directly.
- It does not submit orders.
- It does not cancel orders.
- It does not close positions.
- It does not modify your trading system.
- Check file freshness before relying on proposed orders.
- Actual order submission should continue through your existing tested workflow.
"""
)
