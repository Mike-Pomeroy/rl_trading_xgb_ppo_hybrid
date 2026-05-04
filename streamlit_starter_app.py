import os
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


def show_dataframe(title: str, df: pd.DataFrame, empty_message: str):
    st.subheader(title)
    if df.empty:
        st.info(empty_message)
    else:
        st.dataframe(df, use_container_width=True)


st.title("📈 Trading Starter Dashboard")

st.warning(
    "Read-only dashboard. This app does not submit, cancel, replace, or close orders. "
    "Use your existing Cursor trading workflow for actual trade submission."
)

trading_mode = get_secret_or_env("TRADING_MODE", "paper")
enable_live_trading = str(get_secret_or_env("ENABLE_LIVE_TRADING", "false")).lower() == "true"
require_manual_approval = str(get_secret_or_env("REQUIRE_MANUAL_APPROVAL", "true")).lower() == "true"
max_position_pct = get_secret_or_env("MAX_POSITION_PCT", "0.25")
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

proposed_orders = load_csv(PROPOSED_ORDERS_PATH)
current_positions = load_csv(CURRENT_POSITIONS_PATH)
model_scores = load_csv(MODEL_SCORES_PATH)
open_orders = load_csv(OPEN_ORDERS_PATH)

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

st.header("Safety Checklist")

st.markdown(
    """
- This dashboard only reads local CSV output files.
- It does not connect to Alpaca directly.
- It does not submit orders.
- It does not cancel orders.
- It does not close positions.
- It does not modify your trading system.
- Actual order submission should continue through your existing tested workflow.
"""
)
