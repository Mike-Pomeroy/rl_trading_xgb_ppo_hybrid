"""
Account reconciliation report for the hybrid Alpaca paper trading system.

Purpose:
- Read current Alpaca positions and open orders.
- Read the latest hybrid preview target portfolio.
- Compare current Alpaca holdings against model target values.
- Show dollar drift by symbol.
- Show whether the account appears aligned.
- Show whether it is safe to consider a future submit.
- Save CSV and PDF-style text report outputs.

IMPORTANT:
- This script is READ ONLY.
- It does NOT submit orders.
- It does NOT modify Alpaca account state.

Run:
    python -u account_reconciliation_report.py

Required .env:
    APCA_API_KEY_ID
    APCA_API_SECRET_KEY
"""

import os
from pathlib import Path
from datetime import datetime
from decimal import Decimal
from typing import Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

PAPER = True

PREVIEW_DIR = Path("alpaca_preview_orders_hybrid")
OUTPUT_DIR = Path("account_reconciliation_reports")
OUTPUT_DIR.mkdir(exist_ok=True)

PROPOSED_ORDERS_PATH = PREVIEW_DIR / "proposed_orders.csv"
MODEL_SCORES_PATH = PREVIEW_DIR / "model_scores.csv"
HYBRID_UNIVERSE_PATH = PREVIEW_DIR / "hybrid_universe.csv"
PREVIEW_OPEN_ORDERS_PATH = PREVIEW_DIR / "open_orders.csv"

RECONCILIATION_CSV_PATH = OUTPUT_DIR / "account_reconciliation.csv"
SUMMARY_TXT_PATH = OUTPUT_DIR / "account_reconciliation_summary.txt"

# Ignore tiny leftover fractional dust positions.
DUST_DOLLAR_THRESHOLD = 25.0

# Position drift threshold for the selected holdings.
TARGET_DRIFT_DOLLAR_THRESHOLD = 75.0

# If proposed BUY/SELL orders exist above this value, account is not fully aligned.
MEANINGFUL_ORDER_THRESHOLD = 25.0


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


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    if path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


# ============================================================
# ALPACA HELPERS
# ============================================================

def get_trading_client() -> TradingClient:
    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")

    if not api_key or not secret_key:
        raise RuntimeError(
            "Missing Alpaca credentials. Check .env for "
            "APCA_API_KEY_ID and APCA_API_SECRET_KEY."
        )

    return TradingClient(
        api_key=api_key,
        secret_key=secret_key,
        paper=PAPER,
    )


def get_positions_df(client: TradingClient) -> pd.DataFrame:
    positions = client.get_all_positions()

    rows = []

    for pos in positions:
        rows.append({
            "symbol": str(getattr(pos, "symbol", "")).upper(),
            "qty": safe_float(getattr(pos, "qty", None), 0.0),
            "market_value": safe_float(getattr(pos, "market_value", None), 0.0),
            "current_price": safe_float(getattr(pos, "current_price", None), np.nan),
            "avg_entry_price": safe_float(getattr(pos, "avg_entry_price", None), np.nan),
            "unrealized_pl": safe_float(getattr(pos, "unrealized_pl", None), np.nan),
            "unrealized_plpc": safe_float(getattr(pos, "unrealized_plpc", None), np.nan),
        })

    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "qty",
                "market_value",
                "current_price",
                "avg_entry_price",
                "unrealized_pl",
                "unrealized_plpc",
            ]
        )

    return pd.DataFrame(rows)


def get_open_orders_df(client: TradingClient) -> pd.DataFrame:
    request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
    open_orders = client.get_orders(filter=request)

    rows = []

    for order in open_orders:
        rows.append({
            "symbol": str(getattr(order, "symbol", "")).upper(),
            "id": getattr(order, "id", None),
            "side": getattr(order, "side", None),
            "type": getattr(order, "type", None),
            "status": getattr(order, "status", None),
            "qty": getattr(order, "qty", None),
            "notional": getattr(order, "notional", None),
            "submitted_at": getattr(order, "submitted_at", None),
        })

    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "id",
                "side",
                "type",
                "status",
                "qty",
                "notional",
                "submitted_at",
            ]
        )

    return pd.DataFrame(rows)


# ============================================================
# RECONCILIATION
# ============================================================

def build_reconciliation(
    account_equity: float,
    positions_df: pd.DataFrame,
    proposed_orders_df: pd.DataFrame,
) -> pd.DataFrame:
    if proposed_orders_df.empty:
        raise RuntimeError(
            f"No proposed orders/target file found at {PROPOSED_ORDERS_PATH}. "
            "Run alpaca_order_preview_hybrid.py first."
        )

    target_cols = [
        "symbol",
        "selected",
        "is_screened_addition",
        "rank",
        "price_used",
        "current_value",
        "target_value",
        "dollar_delta",
        "action",
        "strategy_name",
        "signal_date",
        "rebalance_period",
    ]

    available_cols = [c for c in target_cols if c in proposed_orders_df.columns]
    target_df = proposed_orders_df[available_cols].copy()

    target_df["symbol"] = target_df["symbol"].astype(str).str.upper()

    positions = positions_df.copy()

    if positions.empty:
        positions = pd.DataFrame(
            columns=[
                "symbol",
                "qty",
                "market_value",
                "current_price",
                "avg_entry_price",
                "unrealized_pl",
                "unrealized_plpc",
            ]
        )

    positions["symbol"] = positions["symbol"].astype(str).str.upper()

    merged = target_df.merge(
        positions,
        on="symbol",
        how="outer",
        suffixes=("_preview", "_alpaca"),
    )

    for col in ["target_value", "market_value", "current_value"]:
        if col not in merged.columns:
            merged[col] = 0.0

    merged["target_value"] = merged["target_value"].fillna(0.0).astype(float)
    merged["alpaca_market_value"] = merged["market_value"].fillna(0.0).astype(float)

    # Prefer Alpaca live market value for current value.
    merged["reconciled_dollar_delta"] = (
        merged["target_value"] - merged["alpaca_market_value"]
    )

    merged["abs_reconciled_dollar_delta"] = merged["reconciled_dollar_delta"].abs()

    merged["is_selected_target"] = merged.get("selected", False).fillna(False).astype(bool)

    merged["is_dust_position"] = (
        (merged["target_value"].abs() < 1e-9)
        & (merged["alpaca_market_value"].abs() > 0)
        & (merged["alpaca_market_value"].abs() < DUST_DOLLAR_THRESHOLD)
    )

    merged["meaningful_drift"] = (
        merged["abs_reconciled_dollar_delta"] >= TARGET_DRIFT_DOLLAR_THRESHOLD
    )

    merged["needs_attention"] = (
        merged["meaningful_drift"]
        & ~merged["is_dust_position"]
    )

    output_cols = [
        "symbol",
        "is_selected_target",
        "is_screened_addition",
        "rank",
        "qty",
        "current_price",
        "alpaca_market_value",
        "target_value",
        "reconciled_dollar_delta",
        "abs_reconciled_dollar_delta",
        "action",
        "is_dust_position",
        "meaningful_drift",
        "needs_attention",
        "unrealized_pl",
        "unrealized_plpc",
        "strategy_name",
        "signal_date",
        "rebalance_period",
    ]

    output_cols = [c for c in output_cols if c in merged.columns]

    merged = merged[output_cols].sort_values(
        ["needs_attention", "is_selected_target", "abs_reconciled_dollar_delta", "symbol"],
        ascending=[False, False, False, True],
    )

    return merged.reset_index(drop=True)


def summarize_status(
    account,
    positions_df: pd.DataFrame,
    open_orders_df: pd.DataFrame,
    proposed_orders_df: pd.DataFrame,
    reconciliation_df: pd.DataFrame,
) -> Dict[str, object]:
    action_orders = pd.DataFrame()

    if not proposed_orders_df.empty and "action" in proposed_orders_df.columns:
        action_orders = proposed_orders_df[
            proposed_orders_df["action"].astype(str).str.upper().isin(["BUY", "SELL"])
        ].copy()

    meaningful_orders = pd.DataFrame()

    if not action_orders.empty and "dollar_delta" in action_orders.columns:
        meaningful_orders = action_orders[
            action_orders["dollar_delta"].astype(float).abs() >= MEANINGFUL_ORDER_THRESHOLD
        ].copy()

    needs_attention = reconciliation_df[
        reconciliation_df.get("needs_attention", False) == True
    ].copy()

    dust_positions = reconciliation_df[
        reconciliation_df.get("is_dust_position", False) == True
    ].copy()

    selected_positions = reconciliation_df[
        reconciliation_df.get("is_selected_target", False) == True
    ].copy()

    safe_to_submit = (
        len(open_orders_df) == 0
        and meaningful_orders.empty
        and needs_attention.empty
    )

    if len(open_orders_df) > 0:
        status = "NOT SAFE - open Alpaca orders exist"
    elif not meaningful_orders.empty:
        status = "NOT NEEDED / REVIEW - preview has proposed orders"
    elif not needs_attention.empty:
        status = "REVIEW - account drift exceeds threshold"
    else:
        status = "ALIGNED - no submit needed"

    return {
        "status": status,
        "safe_to_submit": safe_to_submit,
        "account_equity": safe_float(getattr(account, "equity", None), 0.0),
        "cash": safe_float(getattr(account, "cash", None), 0.0),
        "buying_power": safe_float(getattr(account, "buying_power", None), 0.0),
        "position_count": len(positions_df),
        "open_order_count": len(open_orders_df),
        "meaningful_proposed_order_count": len(meaningful_orders),
        "needs_attention_count": len(needs_attention),
        "dust_position_count": len(dust_positions),
        "selected_position_count": len(selected_positions),
    }


def write_summary_report(
    summary: Dict[str, object],
    reconciliation_df: pd.DataFrame,
    open_orders_df: pd.DataFrame,
    proposed_orders_df: pd.DataFrame,
) -> None:
    lines: List[str] = []

    lines.append("HYBRID ACCOUNT RECONCILIATION REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Mode: {'PAPER' if PAPER else 'LIVE'}")
    lines.append("")
    lines.append("SUMMARY")
    lines.append("-" * 80)
    lines.append(f"Status: {summary['status']}")
    lines.append(f"Account equity: {money(summary['account_equity'])}")
    lines.append(f"Cash: {money(summary['cash'])}")
    lines.append(f"Buying power: {money(summary['buying_power'])}")
    lines.append(f"Positions: {summary['position_count']}")
    lines.append(f"Open orders: {summary['open_order_count']}")
    lines.append(f"Meaningful proposed orders: {summary['meaningful_proposed_order_count']}")
    lines.append(f"Positions needing attention: {summary['needs_attention_count']}")
    lines.append(f"Dust positions: {summary['dust_position_count']}")
    lines.append(f"Selected target positions: {summary['selected_position_count']}")
    lines.append("")

    if not proposed_orders_df.empty:
        strategy_name = proposed_orders_df.get("strategy_name", pd.Series(["Unknown"])).dropna().astype(str)
        signal_date = proposed_orders_df.get("signal_date", pd.Series(["Unknown"])).dropna().astype(str)
        rebalance_period = proposed_orders_df.get("rebalance_period", pd.Series(["Unknown"])).dropna().astype(str)

        lines.append("LATEST PREVIEW")
        lines.append("-" * 80)
        lines.append(f"Strategy: {strategy_name.iloc[0] if not strategy_name.empty else 'Unknown'}")
        lines.append(f"Signal date: {signal_date.iloc[0] if not signal_date.empty else 'Unknown'}")
        lines.append(f"Rebalance period: {rebalance_period.iloc[0] if not rebalance_period.empty else 'Unknown'}")
        lines.append("")

    lines.append("OPEN ORDERS")
    lines.append("-" * 80)

    if open_orders_df.empty:
        lines.append("No open Alpaca orders.")
    else:
        lines.append(open_orders_df.to_string(index=False))

    lines.append("")
    lines.append("RECONCILIATION")
    lines.append("-" * 80)

    if reconciliation_df.empty:
        lines.append("No reconciliation rows.")
    else:
        display_cols = [
            "symbol",
            "is_selected_target",
            "alpaca_market_value",
            "target_value",
            "reconciled_dollar_delta",
            "is_dust_position",
            "needs_attention",
        ]
        display_cols = [c for c in display_cols if c in reconciliation_df.columns]

        lines.append(
            reconciliation_df[display_cols].to_string(
                index=False,
                float_format=lambda x: f"{x:,.2f}",
            )
        )

    lines.append("")
    lines.append("FILES")
    lines.append("-" * 80)
    lines.append(f"Reconciliation CSV: {RECONCILIATION_CSV_PATH}")
    lines.append(f"Summary TXT: {SUMMARY_TXT_PATH}")

    SUMMARY_TXT_PATH.write_text("\n".join(lines))


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("\n===== HYBRID ACCOUNT RECONCILIATION REPORT =====")
    print(f"PAPER: {PAPER}")
    print(f"Preview file: {PROPOSED_ORDERS_PATH}")

    client = get_trading_client()
    account = client.get_account()
    positions_df = get_positions_df(client)
    open_orders_df = get_open_orders_df(client)
    proposed_orders_df = safe_read_csv(PROPOSED_ORDERS_PATH)

    reconciliation_df = build_reconciliation(
        account_equity=safe_float(getattr(account, "equity", None), 0.0),
        positions_df=positions_df,
        proposed_orders_df=proposed_orders_df,
    )

    summary = summarize_status(
        account=account,
        positions_df=positions_df,
        open_orders_df=open_orders_df,
        proposed_orders_df=proposed_orders_df,
        reconciliation_df=reconciliation_df,
    )

    reconciliation_df.to_csv(RECONCILIATION_CSV_PATH, index=False)

    write_summary_report(
        summary=summary,
        reconciliation_df=reconciliation_df,
        open_orders_df=open_orders_df,
        proposed_orders_df=proposed_orders_df,
    )

    print("\n===== SUMMARY =====")
    print(f"Status: {summary['status']}")
    print(f"Account equity: {money(summary['account_equity'])}")
    print(f"Cash: {money(summary['cash'])}")
    print(f"Positions: {summary['position_count']}")
    print(f"Open orders: {summary['open_order_count']}")
    print(f"Meaningful proposed orders: {summary['meaningful_proposed_order_count']}")
    print(f"Positions needing attention: {summary['needs_attention_count']}")
    print(f"Dust positions: {summary['dust_position_count']}")

    print("\n===== RECONCILIATION =====")

    display_cols = [
        "symbol",
        "is_selected_target",
        "alpaca_market_value",
        "target_value",
        "reconciled_dollar_delta",
        "is_dust_position",
        "needs_attention",
    ]
    display_cols = [c for c in display_cols if c in reconciliation_df.columns]

    print(
        reconciliation_df[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:,.2f}",
        )
    )

    print("\n===== SAVED FILES =====")
    print(f"Reconciliation CSV: {RECONCILIATION_CSV_PATH}")
    print(f"Summary TXT:        {SUMMARY_TXT_PATH}")


if __name__ == "__main__":
    main()