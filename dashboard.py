"""
Read-only Streamlit dashboard for the hybrid Alpaca trading system.

Purpose:
- View Alpaca paper/live account overview.
- View current positions.
- View recent/open orders.
- View latest hybrid preview output.
- View rebalance guard status.
- View account reconciliation report.
- View ticker ranking files and reports.
- Download generated CSV/PDF/text files.

IMPORTANT:
- This dashboard is READ ONLY.
- It does NOT submit trades.
- It does NOT call Alpaca order-submit functions.
"""

import os
from decimal import Decimal
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Hybrid Trading Dashboard",
    layout="wide",
)

load_dotenv()

API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"


# ============================================================
# PATHS
# ============================================================

HYBRID_PREVIEW_DIR = Path("alpaca_preview_orders_hybrid")
HYBRID_SUBMITTED_DIR = Path("alpaca_submitted_orders_hybrid")
RANKING_DIR = Path("ticker_ranking_results")
REPORTS_DIR = Path(".")
REBALANCE_GUARD_DIR = Path("rebalance_guard_logs")
RECONCILIATION_DIR = Path("account_reconciliation_reports")

HYBRID_PROPOSED_ORDERS_PATH = HYBRID_PREVIEW_DIR / "proposed_orders.csv"
HYBRID_MODEL_SCORES_PATH = HYBRID_PREVIEW_DIR / "model_scores.csv"
HYBRID_CURRENT_POSITIONS_PATH = HYBRID_PREVIEW_DIR / "current_positions.csv"
HYBRID_OPEN_ORDERS_PATH = HYBRID_PREVIEW_DIR / "open_orders.csv"
HYBRID_SCREENED_ADDITIONS_PATH = HYBRID_PREVIEW_DIR / "screened_additions_scores.csv"
HYBRID_UNIVERSE_PATH = HYBRID_PREVIEW_DIR / "hybrid_universe.csv"
HYBRID_SUBMITTED_ORDERS_PATH = HYBRID_SUBMITTED_DIR / "submitted_orders.csv"

RANKING_CSV_PATH = RANKING_DIR / "ticker_ranking.csv"
RANKING_PDF_PATH = RANKING_DIR / "ticker_ranking_report.pdf"

REBALANCE_GUARD_LOG_PATH = REBALANCE_GUARD_DIR / "rebalance_submissions.csv"

RECONCILIATION_CSV_PATH = RECONCILIATION_DIR / "account_reconciliation.csv"
RECONCILIATION_SUMMARY_TXT_PATH = RECONCILIATION_DIR / "account_reconciliation_summary.txt"

BACKTEST_REPORT_PATH = REPORTS_DIR / "xgboost_hybrid_backtest_robustness_report.pdf"


# ============================================================
# FORMAT HELPERS
# ============================================================

def money(value) -> str:
    try:
        return f"${Decimal(str(value)):,.2f}"
    except Exception:
        return str(value)


def pct(value) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return str(value)


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    if path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as exc:
        st.error(f"Could not read file: {path}")
        st.exception(exc)
        return pd.DataFrame()


def safe_read_text(path: Path) -> str:
    if not path.exists():
        return ""

    if path.stat().st_size == 0:
        return ""

    try:
        return path.read_text()
    except Exception as exc:
        st.error(f"Could not read file: {path}")
        st.exception(exc)
        return ""


def file_download_button(path: Path, label: str, mime: str, key: str) -> None:
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as f:
            st.download_button(
                label=label,
                data=f.read(),
                file_name=path.name,
                mime=mime,
                key=key,
            )
    else:
        st.info(f"File not found yet: {path}")


def display_dataframe_or_info(df: pd.DataFrame, empty_message: str) -> None:
    if df.empty:
        st.info(empty_message)
    else:
        st.dataframe(df, use_container_width=True)


# ============================================================
# ALPACA CONNECTION
# ============================================================

st.title("Hybrid Trading Dashboard")

if PAPER:
    st.success("Mode: Alpaca Paper Trading")
else:
    st.error("Mode: Alpaca LIVE Trading")

st.warning(
    "Read-only dashboard. This app does not place trades or submit Alpaca orders."
)

if not API_KEY or not SECRET_KEY:
    st.warning("Missing Alpaca API keys. Check your .env file.")
    st.stop()

try:
    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
    account = trading_client.get_account()
    positions = trading_client.get_all_positions()

    open_orders_request = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
        limit=50,
    )
    open_orders = trading_client.get_orders(filter=open_orders_request)

    recent_orders_request = GetOrdersRequest(
        status=QueryOrderStatus.ALL,
        limit=50,
    )
    recent_orders = trading_client.get_orders(filter=recent_orders_request)

except Exception as e:
    st.error("Could not connect to Alpaca.")
    st.exception(e)
    st.stop()


# ============================================================
# TABS
# ============================================================

(
    overview_tab,
    positions_tab,
    orders_tab,
    hybrid_preview_tab,
    rebalance_guard_tab,
    reconciliation_tab,
    rankings_tab,
    reports_tab,
    safety_tab,
) = st.tabs(
    [
        "Overview",
        "Positions",
        "Orders",
        "Hybrid Preview",
        "Rebalance Guard",
        "Reconciliation",
        "Rankings",
        "Reports",
        "Safety",
    ]
)


# ============================================================
# OVERVIEW TAB
# ============================================================

with overview_tab:
    st.subheader("Account Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Account Status", str(account.status))
    col2.metric("Equity", money(account.equity))
    col3.metric("Cash", money(account.cash))
    col4.metric("Buying Power", money(account.buying_power))

    st.divider()

    col5, col6, col7 = st.columns(3)
    col5.metric("Current Positions", len(positions))
    col6.metric("Open Orders", len(open_orders))
    col7.metric("Recent Orders Loaded", len(recent_orders))

    if open_orders:
        st.error("There are open Alpaca orders. Do not submit new orders until these are filled or canceled.")
    else:
        st.success("No open Alpaca orders detected.")


# ============================================================
# POSITIONS TAB
# ============================================================

with positions_tab:
    st.subheader("Open Positions")

    if not positions:
        st.info("No open positions.")
    else:
        position_rows = []

        for p in positions:
            position_rows.append(
                {
                    "Symbol": p.symbol,
                    "Qty": p.qty,
                    "Market Value": money(p.market_value),
                    "Avg Entry": money(p.avg_entry_price),
                    "Current Price": money(p.current_price),
                    "Unrealized P/L": money(p.unrealized_pl),
                    "Unrealized P/L %": pct(p.unrealized_plpc),
                }
            )

        positions_df = pd.DataFrame(position_rows)
        st.dataframe(positions_df, use_container_width=True)

        total_market_value = sum(
            float(getattr(p, "market_value", 0) or 0)
            for p in positions
        )

        st.metric("Total Market Value", money(total_market_value))


# ============================================================
# ORDERS TAB
# ============================================================

with orders_tab:
    st.subheader("Open Orders")

    if not open_orders:
        st.success("No open orders.")
    else:
        open_order_rows = []

        for o in open_orders:
            open_order_rows.append(
                {
                    "Submitted": str(o.submitted_at),
                    "Symbol": o.symbol,
                    "Side": str(o.side),
                    "Type": str(o.type),
                    "Qty": o.qty,
                    "Notional": money(o.notional) if getattr(o, "notional", None) else "",
                    "Filled Qty": o.filled_qty,
                    "Status": str(o.status),
                    "Limit Price": money(o.limit_price) if o.limit_price else "",
                    "Filled Avg Price": money(o.filled_avg_price) if o.filled_avg_price else "",
                }
            )

        st.dataframe(pd.DataFrame(open_order_rows), use_container_width=True)

    st.divider()

    st.subheader("Recent Orders")

    if not recent_orders:
        st.info("No recent orders found.")
    else:
        recent_order_rows = []

        for o in recent_orders:
            recent_order_rows.append(
                {
                    "Submitted": str(o.submitted_at),
                    "Symbol": o.symbol,
                    "Side": str(o.side),
                    "Type": str(o.type),
                    "Qty": o.qty,
                    "Notional": money(o.notional) if getattr(o, "notional", None) else "",
                    "Filled Qty": o.filled_qty,
                    "Status": str(o.status),
                    "Limit Price": money(o.limit_price) if o.limit_price else "",
                    "Filled Avg Price": money(o.filled_avg_price) if o.filled_avg_price else "",
                }
            )

        st.dataframe(pd.DataFrame(recent_order_rows), use_container_width=True)


# ============================================================
# HYBRID PREVIEW TAB
# ============================================================

with hybrid_preview_tab:
    st.subheader("Latest Hybrid Preview")

    proposed_orders_df = safe_read_csv(HYBRID_PROPOSED_ORDERS_PATH)
    model_scores_df = safe_read_csv(HYBRID_MODEL_SCORES_PATH)
    current_positions_df = safe_read_csv(HYBRID_CURRENT_POSITIONS_PATH)
    preview_open_orders_df = safe_read_csv(HYBRID_OPEN_ORDERS_PATH)
    screened_additions_df = safe_read_csv(HYBRID_SCREENED_ADDITIONS_PATH)
    hybrid_universe_df = safe_read_csv(HYBRID_UNIVERSE_PATH)

    if proposed_orders_df.empty:
        st.info(
            "No hybrid preview file found yet. Run: "
            "python -u alpaca_order_preview_hybrid.py"
        )
    else:
        signal_date = (
            proposed_orders_df["signal_date"].dropna().astype(str).iloc[0]
            if "signal_date" in proposed_orders_df.columns and not proposed_orders_df["signal_date"].dropna().empty
            else "Unknown"
        )

        rebalance_period = (
            proposed_orders_df["rebalance_period"].dropna().astype(str).iloc[0]
            if "rebalance_period" in proposed_orders_df.columns and not proposed_orders_df["rebalance_period"].dropna().empty
            else "Unknown"
        )

        strategy_name = (
            proposed_orders_df["strategy_name"].dropna().astype(str).iloc[0]
            if "strategy_name" in proposed_orders_df.columns and not proposed_orders_df["strategy_name"].dropna().empty
            else "Unknown"
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Strategy", strategy_name)
        col2.metric("Signal Date", signal_date)
        col3.metric("Rebalance Period", rebalance_period)

        action_df = proposed_orders_df[
            proposed_orders_df["action"].astype(str).str.upper().isin(["BUY", "SELL"])
        ].copy()

        st.divider()

        st.subheader("Proposed Orders")

        if action_df.empty:
            st.success("No proposed BUY/SELL orders. Portfolio appears close to target.")
        else:
            st.warning("Proposed orders exist. Review carefully before running any submit script.")
            preferred_cols = [
                "symbol",
                "action",
                "selected",
                "is_screened_addition",
                "price_used",
                "current_value",
                "target_value",
                "dollar_delta",
                "notional_for_buy",
                "qty_for_sell",
            ]
            cols = [c for c in preferred_cols if c in action_df.columns]
            st.dataframe(action_df[cols], use_container_width=True)

        file_download_button(
            HYBRID_PROPOSED_ORDERS_PATH,
            "Download Proposed Orders CSV",
            "text/csv",
            "download_hybrid_proposed_orders",
        )

    st.divider()

    st.subheader("Hybrid Model Scores")

    if model_scores_df.empty:
        st.info("No model scores file found yet.")
    else:
        preferred_cols = ["rank", "tic", "score", "close"]
        cols = [c for c in preferred_cols if c in model_scores_df.columns]

        if cols:
            st.dataframe(model_scores_df[cols].head(30), use_container_width=True)
        else:
            st.dataframe(model_scores_df.head(30), use_container_width=True)

        file_download_button(
            HYBRID_MODEL_SCORES_PATH,
            "Download Model Scores CSV",
            "text/csv",
            "download_hybrid_model_scores",
        )

    st.divider()

    st.subheader("Hybrid Universe")

    if hybrid_universe_df.empty:
        st.info("No hybrid universe file found yet.")
    else:
        st.dataframe(hybrid_universe_df, use_container_width=True)

        if "is_screened_addition" in hybrid_universe_df.columns:
            additions = hybrid_universe_df[
                hybrid_universe_df["is_screened_addition"] == True
            ]

            if not additions.empty:
                st.write("Screened additions:")
                st.write(", ".join(additions["ticker"].astype(str).tolist()))

        file_download_button(
            HYBRID_UNIVERSE_PATH,
            "Download Hybrid Universe CSV",
            "text/csv",
            "download_hybrid_universe",
        )

    st.divider()

    st.subheader("Preview Open Orders File")

    if preview_open_orders_df.empty:
        st.success("Preview file shows no open orders, or no open-orders file exists.")
    else:
        st.error("Preview file contains open orders. Do not submit new orders.")
        st.dataframe(preview_open_orders_df, use_container_width=True)

    st.divider()

    st.subheader("Preview Current Positions File")

    display_dataframe_or_info(
        current_positions_df,
        "No current positions preview file found yet.",
    )


# ============================================================
# REBALANCE GUARD TAB
# ============================================================

with rebalance_guard_tab:
    st.subheader("Rebalance Guard Status")

    guard_df = safe_read_csv(REBALANCE_GUARD_LOG_PATH)

    if guard_df.empty:
        st.info("No rebalance submissions have been recorded yet.")
        st.write(f"Expected file: {REBALANCE_GUARD_LOG_PATH}")
    else:
        st.dataframe(guard_df.sort_values("submitted_at", ascending=False), use_container_width=True)

        latest = guard_df.sort_values("submitted_at", ascending=False).iloc[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Latest Strategy", str(latest.get("strategy_name", "")))
        col2.metric("Latest Period", str(latest.get("rebalance_period", "")))
        col3.metric("Mode", str(latest.get("mode", "")))

        st.info(
            "This log is used by the submit script to block accidental duplicate "
            "submissions for the same strategy, month, and mode."
        )

        file_download_button(
            REBALANCE_GUARD_LOG_PATH,
            "Download Rebalance Guard Log CSV",
            "text/csv",
            "download_rebalance_guard_log",
        )


# ============================================================
# RECONCILIATION TAB
# ============================================================

with reconciliation_tab:
    st.subheader("Account Reconciliation")

    reconciliation_df = safe_read_csv(RECONCILIATION_CSV_PATH)
    reconciliation_summary_text = safe_read_text(RECONCILIATION_SUMMARY_TXT_PATH)

    if reconciliation_df.empty and not reconciliation_summary_text:
        st.info(
            "No reconciliation report found yet. Run: "
            "python -u account_reconciliation_report.py"
        )
    else:
        if reconciliation_summary_text:
            status_line = ""
            for line in reconciliation_summary_text.splitlines():
                if line.startswith("Status:"):
                    status_line = line.replace("Status:", "").strip()
                    break

            if status_line:
                if status_line.startswith("ALIGNED"):
                    st.success(status_line)
                elif status_line.startswith("REVIEW"):
                    st.warning(status_line)
                elif status_line.startswith("NOT SAFE"):
                    st.error(status_line)
                else:
                    st.info(status_line)

            with st.expander("View Text Summary Report", expanded=False):
                st.text(reconciliation_summary_text)

            file_download_button(
                RECONCILIATION_SUMMARY_TXT_PATH,
                "Download Reconciliation Summary TXT",
                "text/plain",
                "download_reconciliation_summary_txt",
            )

        st.divider()

        st.subheader("Reconciliation Table")

        if reconciliation_df.empty:
            st.info("No reconciliation CSV rows found.")
        else:
            attention_df = reconciliation_df[
                reconciliation_df.get("needs_attention", False) == True
            ].copy()

            dust_df = reconciliation_df[
                reconciliation_df.get("is_dust_position", False) == True
            ].copy()

            selected_df = reconciliation_df[
                reconciliation_df.get("is_selected_target", False) == True
            ].copy()

            col1, col2, col3 = st.columns(3)
            col1.metric("Selected Target Positions", len(selected_df))
            col2.metric("Needs Attention", len(attention_df))
            col3.metric("Dust Positions", len(dust_df))

            preferred_cols = [
                "symbol",
                "is_selected_target",
                "alpaca_market_value",
                "target_value",
                "reconciled_dollar_delta",
                "is_dust_position",
                "needs_attention",
                "unrealized_pl",
                "unrealized_plpc",
            ]
            cols = [c for c in preferred_cols if c in reconciliation_df.columns]

            st.dataframe(reconciliation_df[cols], use_container_width=True)

            file_download_button(
                RECONCILIATION_CSV_PATH,
                "Download Reconciliation CSV",
                "text/csv",
                "download_reconciliation_csv",
            )


# ============================================================
# RANKINGS TAB
# ============================================================

with rankings_tab:
    st.subheader("Ticker Ranking Results")

    rankings_df = safe_read_csv(RANKING_CSV_PATH)

    if rankings_df.empty:
        st.info(
            "No rankings file found yet. Expected: "
            "ticker_ranking_results/ticker_ranking.csv"
        )
    else:
        st.dataframe(rankings_df, use_container_width=True)

        file_download_button(
            RANKING_CSV_PATH,
            "Download Ranking CSV",
            "text/csv",
            "download_ranking_csv",
        )

    st.divider()

    file_download_button(
        RANKING_PDF_PATH,
        "Download Ranking PDF",
        "application/pdf",
        "download_ranking_pdf",
    )


# ============================================================
# REPORTS TAB
# ============================================================

with reports_tab:
    st.subheader("Reports")

    file_download_button(
        RANKING_PDF_PATH,
        "Download Ticker Ranking PDF",
        "application/pdf",
        "reports_download_ranking_pdf",
    )

    file_download_button(
        BACKTEST_REPORT_PATH,
        "Download Hybrid Backtest / Robustness PDF",
        "application/pdf",
        "reports_download_backtest_pdf",
    )

    file_download_button(
        RECONCILIATION_SUMMARY_TXT_PATH,
        "Download Reconciliation Summary TXT",
        "text/plain",
        "reports_download_reconciliation_summary_txt",
    )

    file_download_button(
        RECONCILIATION_CSV_PATH,
        "Download Reconciliation CSV",
        "text/csv",
        "reports_download_reconciliation_csv",
    )

    st.divider()

    st.subheader("Submitted Hybrid Orders Log")

    submitted_df = safe_read_csv(HYBRID_SUBMITTED_ORDERS_PATH)

    if submitted_df.empty:
        st.info("No submitted hybrid orders log found yet.")
    else:
        st.dataframe(submitted_df, use_container_width=True)

        file_download_button(
            HYBRID_SUBMITTED_ORDERS_PATH,
            "Download Submitted Orders CSV",
            "text/csv",
            "download_submitted_orders_csv",
        )


# ============================================================
# SAFETY TAB
# ============================================================

with safety_tab:
    st.subheader("Safety Status")

    st.success("This dashboard is read-only.")
    st.write("It can view your Alpaca account, positions, orders, preview files, rankings, reconciliation reports, and reports.")
    st.write("It does not place trades.")
    st.write("It does not call any Alpaca submit/order placement code.")

    st.divider()

    st.subheader("Manual Trading Workflow")

    st.write("1. Run the hybrid preview script from Terminal/Cursor:")
    st.code("python -u alpaca_order_preview_hybrid.py", language="bash")

    st.write("2. Run the reconciliation report:")
    st.code("python -u account_reconciliation_report.py", language="bash")

    st.write("3. Review the dashboard Hybrid Preview, Reconciliation, and Rebalance Guard tabs.")

    st.write("4. Only if everything looks correct, run the paper submit script manually:")
    st.code("python -u alpaca_order_submit_paper_hybrid.py", language="bash")

    st.write("5. After orders fill, rerun preview and reconciliation, then confirm no new orders are proposed.")

    st.divider()

    if PAPER:
        st.success("Current mode is PAPER trading.")
    else:
        st.error("Current mode is LIVE trading. Be careful.")