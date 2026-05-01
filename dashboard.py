"""
Read-only Streamlit dashboard for the hybrid Alpaca trading system.

Purpose:
- View Alpaca paper/live account overview.
- View current positions.
- View recent/open orders.
- View latest hybrid preview output.
- View rebalance guard status.
- View account reconciliation report.
- View paper-submit readiness checklist.
- View ticker ranking files and reports.
- Download generated CSV/PDF/text files.

IMPORTANT:
- This dashboard is READ ONLY.
- It does NOT submit trades.
- It does NOT call Alpaca order-submit functions.
"""

import os
import subprocess
import sys
from datetime import datetime
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

def file_freshness(path: Path) -> str:
    """
    Return a readable last-modified timestamp for a local file.
    """
    if not path.exists():
        return "Missing"

    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime)
        return modified.strftime("%Y-%m-%d %I:%M:%S %p")
    except Exception:
        return "Unknown"


def run_read_only_script(script_name: str) -> None:
    """
    Run a local read-only helper script from the dashboard.

    Safety:
    - This should only be used for scripts that do NOT submit orders.
    - Do not use this for alpaca_order_submit_paper_hybrid.py.
    """
    script_path = Path(script_name)

    if not script_path.exists():
        st.error(f"Script not found: {script_name}")
        return

    with st.spinner(f"Running {script_name}..."):
        try:
            result = subprocess.run(
                [sys.executable, "-u", script_name],
                capture_output=True,
                text=True,
                timeout=900,
            )

            if result.returncode == 0:
                st.success(f"{script_name} completed successfully.")
            else:
                st.error(f"{script_name} finished with an error.")

            if result.stdout:
                with st.expander(f"{script_name} output", expanded=False):
                    st.code(result.stdout)

            if result.stderr:
                with st.expander(f"{script_name} errors / warnings", expanded=False):
                    st.code(result.stderr)

        except subprocess.TimeoutExpired:
            st.error(f"{script_name} timed out.")
        except Exception as exc:
            st.error(f"Could not run {script_name}.")
            st.exception(exc)


def get_latest_preview_metadata(proposed_orders_df: pd.DataFrame) -> dict:
    if proposed_orders_df.empty:
        return {
            "strategy_name": "Unknown",
            "signal_date": "Unknown",
            "rebalance_period": "Unknown",
        }

    def first_value(col: str) -> str:
        if col not in proposed_orders_df.columns:
            return "Unknown"

        values = proposed_orders_df[col].dropna().astype(str)

        if values.empty:
            return "Unknown"

        return values.iloc[0]

    return {
        "strategy_name": first_value("strategy_name"),
        "signal_date": first_value("signal_date"),
        "rebalance_period": first_value("rebalance_period"),
    }


def get_reconciliation_status(summary_text: str) -> str:
    if not summary_text:
        return "Unknown"

    for line in summary_text.splitlines():
        if line.startswith("Status:"):
            return line.replace("Status:", "").strip()

    return "Unknown"


def build_submit_readiness(
    paper_mode: bool,
    open_orders_count: int,
    proposed_orders_df: pd.DataFrame,
    guard_df: pd.DataFrame,
    reconciliation_df: pd.DataFrame,
    reconciliation_status: str,
) -> tuple[pd.DataFrame, bool, str]:
    """
    Build a read-only checklist for whether it is reasonable to consider
    manually running alpaca_order_submit_paper_hybrid.py.

    This does NOT submit orders.
    """
    rows = []

    metadata = get_latest_preview_metadata(proposed_orders_df)
    strategy_name = metadata["strategy_name"]
    rebalance_period = metadata["rebalance_period"]

    has_preview = not proposed_orders_df.empty

    if has_preview and "action" in proposed_orders_df.columns:
        action_df = proposed_orders_df[
            proposed_orders_df["action"].astype(str).str.upper().isin(["BUY", "SELL"])
        ].copy()
    else:
        action_df = pd.DataFrame()

    proposed_order_count = len(action_df)

    already_submitted = False

    if (
        not guard_df.empty
        and strategy_name != "Unknown"
        and rebalance_period != "Unknown"
        and {"strategy_name", "rebalance_period", "mode"}.issubset(guard_df.columns)
    ):
        already_submitted = not guard_df[
            (guard_df["strategy_name"].astype(str) == strategy_name)
            & (guard_df["rebalance_period"].astype(str) == rebalance_period)
            & (guard_df["mode"].astype(str) == "paper")
        ].empty

    has_reconciliation = not reconciliation_df.empty or reconciliation_status != "Unknown"

    needs_attention_count = 0

    if not reconciliation_df.empty and "needs_attention" in reconciliation_df.columns:
        needs_attention_count = int((reconciliation_df["needs_attention"] == True).sum())

    def add_check(name: str, passed: bool, detail: str) -> None:
        rows.append(
            {
                "Check": name,
                "Passed": bool(passed),
                "Status": "PASS" if passed else "REVIEW",
                "Detail": detail,
            }
        )

    add_check(
        "Paper mode is active",
        paper_mode,
        "Dashboard is connected in paper mode." if paper_mode else "Dashboard is not in paper mode.",
    )

    add_check(
        "No open Alpaca orders",
        open_orders_count == 0,
        f"Open Alpaca orders: {open_orders_count}",
    )

    add_check(
        "Hybrid preview file exists",
        has_preview,
        f"Preview strategy={strategy_name}, signal_date={metadata['signal_date']}, period={rebalance_period}",
    )

    add_check(
        "Proposed orders exist",
        proposed_order_count > 0,
        f"Proposed BUY/SELL orders: {proposed_order_count}",
    )

    add_check(
        "Rebalance guard has not already submitted this period",
        not already_submitted,
        (
            f"No paper submission found for {strategy_name} / {rebalance_period}."
            if not already_submitted
            else f"Already submitted for {strategy_name} / {rebalance_period}."
        ),
    )

    add_check(
        "Reconciliation report exists",
        has_reconciliation,
        f"Reconciliation status: {reconciliation_status}",
    )

    add_check(
        "Reconciliation expects action or review",
        reconciliation_status.startswith("NOT NEEDED / REVIEW") or needs_attention_count > 0,
        (
            f"Status={reconciliation_status}, needs_attention={needs_attention_count}. "
            "This is expected when preview has rebalance orders."
        ),
    )

    add_check(
        "Dashboard remains read-only",
        True,
        "No submit button exists in this dashboard.",
    )

    checklist_df = pd.DataFrame(rows)

    ready = (
        paper_mode
        and open_orders_count == 0
        and has_preview
        and proposed_order_count > 0
        and not already_submitted
        and has_reconciliation
    )

    if ready:
        message = "READY FOR MANUAL PAPER SUBMIT REVIEW"
    elif already_submitted:
        message = "NOT READY - already submitted for this rebalance period"
    elif open_orders_count > 0:
        message = "NOT READY - open Alpaca orders exist"
    elif not has_preview:
        message = "NOT READY - missing hybrid preview file"
    elif proposed_order_count == 0:
        message = "NOT READY - no proposed orders"
    elif not has_reconciliation:
        message = "NOT READY - missing reconciliation report"
    elif not paper_mode:
        message = "NOT READY - dashboard is not in paper mode"
    else:
        message = "REVIEW REQUIRED"

    return checklist_df, ready, message


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

# ============================================================
# READ-ONLY ACTION BUTTONS
# ============================================================


with st.sidebar:
    st.header("Read-Only Actions")

    st.caption(
        "These buttons refresh local files only. "
        "They do not submit Alpaca orders."
    )

    if st.button("Run Hybrid Preview", type="primary"):
        run_read_only_script("alpaca_order_preview_hybrid.py")

    if st.button("Run Reconciliation Report"):
        run_read_only_script("account_reconciliation_report.py")

    if st.button("Refresh Dashboard"):
        st.rerun()

    st.divider()

    st.subheader("File Freshness")

    freshness_rows = [
        {
            "File": "Hybrid Preview",
            "Updated": file_freshness(HYBRID_PROPOSED_ORDERS_PATH),
        },
        {
            "File": "Model Scores",
            "Updated": file_freshness(HYBRID_MODEL_SCORES_PATH),
        },
        {
            "File": "Reconciliation",
            "Updated": file_freshness(RECONCILIATION_CSV_PATH),
        },
        {
            "File": "Guard Log",
            "Updated": file_freshness(REBALANCE_GUARD_LOG_PATH),
        },
        {
            "File": "Submitted Orders",
            "Updated": file_freshness(HYBRID_SUBMITTED_ORDERS_PATH),
        },
    ]

    st.dataframe(
        pd.DataFrame(freshness_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.warning("No order-submit buttons are available in this dashboard.")



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
# LOAD LOCAL FILES ONCE
# ============================================================

proposed_orders_df = safe_read_csv(HYBRID_PROPOSED_ORDERS_PATH)
model_scores_df = safe_read_csv(HYBRID_MODEL_SCORES_PATH)
current_positions_df = safe_read_csv(HYBRID_CURRENT_POSITIONS_PATH)
preview_open_orders_df = safe_read_csv(HYBRID_OPEN_ORDERS_PATH)
screened_additions_df = safe_read_csv(HYBRID_SCREENED_ADDITIONS_PATH)
hybrid_universe_df = safe_read_csv(HYBRID_UNIVERSE_PATH)
guard_df = safe_read_csv(REBALANCE_GUARD_LOG_PATH)
reconciliation_df = safe_read_csv(RECONCILIATION_CSV_PATH)
reconciliation_summary_text = safe_read_text(RECONCILIATION_SUMMARY_TXT_PATH)
reconciliation_status = get_reconciliation_status(reconciliation_summary_text)


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
    readiness_tab,
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
        "Submit Readiness",
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

    if proposed_orders_df.empty:
        st.info(
            "No hybrid preview file found yet. Run: "
            "python -u alpaca_order_preview_hybrid.py"
        )
    else:
        metadata = get_latest_preview_metadata(proposed_orders_df)

        col1, col2, col3 = st.columns(3)
        col1.metric("Strategy", metadata["strategy_name"])
        col2.metric("Signal Date", metadata["signal_date"])
        col3.metric("Rebalance Period", metadata["rebalance_period"])

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

    if reconciliation_df.empty and not reconciliation_summary_text:
        st.info(
            "No reconciliation report found yet. Run: "
            "python -u account_reconciliation_report.py"
        )
    else:
        if reconciliation_status != "Unknown":
            if reconciliation_status.startswith("ALIGNED"):
                st.success(reconciliation_status)
            elif reconciliation_status.startswith("REVIEW"):
                st.warning(reconciliation_status)
            elif reconciliation_status.startswith("NOT SAFE"):
                st.error(reconciliation_status)
            else:
                st.info(reconciliation_status)

        if reconciliation_summary_text:
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
# SUBMIT READINESS TAB
# ============================================================

with readiness_tab:
    st.subheader("Manual Paper Submit Readiness")

    checklist_df, ready, readiness_message = build_submit_readiness(
        paper_mode=PAPER,
        open_orders_count=len(open_orders),
        proposed_orders_df=proposed_orders_df,
        guard_df=guard_df,
        reconciliation_df=reconciliation_df,
        reconciliation_status=reconciliation_status,
    )

    if ready:
        st.success(readiness_message)
    elif readiness_message.startswith("NOT READY"):
        st.error(readiness_message)
    else:
        st.warning(readiness_message)

    st.write(
        "This tab does not submit orders. It only tells you whether the manual "
        "paper submit script is reasonable to consider."
    )

    st.dataframe(checklist_df, use_container_width=True)

    st.divider()

    st.subheader("Manual Submit Command")

    st.code("python -u alpaca_order_submit_paper_hybrid.py", language="bash")

    st.warning(
        "Run this command only from Terminal/Cursor, only on the planned submit day, "
        "and only after reviewing Hybrid Preview, Reconciliation, and Rebalance Guard."
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

    st.write("1. Run the hybrid preview script from the dashboard sidebar or Terminal/Cursor:")
    st.code("python -u alpaca_order_preview_hybrid.py", language="bash")

    st.write("2. Run the reconciliation report from the dashboard sidebar or Terminal/Cursor:")
    st.code("python -u account_reconciliation_report.py", language="bash")

    st.write("3. Review the dashboard Hybrid Preview, Reconciliation, Rebalance Guard, and Submit Readiness tabs.")

    st.write("4. Only if everything looks correct, run the paper submit script manually:")
    st.code("python -u alpaca_order_submit_paper_hybrid.py", language="bash")

    st.write("5. After orders fill, rerun preview and reconciliation, then confirm no new orders are proposed.")

    st.divider()

    if PAPER:
        st.success("Current mode is PAPER trading.")
    else:
        st.error("Current mode is LIVE trading. Be careful.")