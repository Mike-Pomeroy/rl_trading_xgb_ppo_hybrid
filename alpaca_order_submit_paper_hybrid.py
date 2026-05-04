"""
Submit Alpaca PAPER orders from the HYBRID preview file.

IMPORTANT:
- This script is PAPER ONLY.
- It reads alpaca_preview_orders_hybrid/proposed_orders.csv.
- It checks the rebalance guard before submitting.
- It checks Alpaca for existing open orders and skips those symbols.
- It submits SELL orders first, then BUY orders.
- It saves a frozen monthly target snapshot after a submission attempt.
- It refuses to run unless:
    PAPER = True
    SUBMIT_ORDERS = True
    CONFIRM_PAPER_ONLY = "YES_PAPER_ONLY"

Run:
    python -u alpaca_order_submit_paper_hybrid.py

Required .env:
    APCA_API_KEY_ID
    APCA_API_SECRET_KEY
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest

from rebalance_guard import assert_not_already_submitted, record_submission
from target_snapshot import save_target_snapshot


# ============================================================
# SAFETY CONFIG
# ============================================================

load_dotenv()

PAPER = True
SUBMIT_ORDERS = True
CONFIRM_PAPER_ONLY = "YES_PAPER_ONLY"

STRATEGY_NAME = "hybrid_plus_5"
MODE = "paper"
ALLOW_RESUBMIT = False

PREVIEW_PATH = Path("alpaca_preview_orders_hybrid/proposed_orders.csv")
OUTPUT_DIR = Path("alpaca_submitted_orders_hybrid")
OUTPUT_DIR.mkdir(exist_ok=True)

SUBMITTED_ORDERS_PATH = OUTPUT_DIR / "submitted_orders.csv"

# Avoid accidental tiny/noisy orders.
MIN_BUY_NOTIONAL = 10.0
MIN_SELL_QTY = 0.000001

# Reduce sell quantity slightly to avoid Alpaca insufficient-quantity errors
# caused by rounding differences between preview file and actual position qty.
SELL_QTY_SAFETY_MULTIPLIER = 0.9999

# Pause between sell phase and buy phase.
WAIT_AFTER_SELLS_SECONDS = 5

# Submit sell orders before buys.
SELL_FIRST = True


# ============================================================
# HELPERS
# ============================================================

def get_trading_client() -> TradingClient:
    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")

    if not api_key or not secret_key:
        raise RuntimeError(
            "Missing Alpaca credentials. Check your .env file for "
            "APCA_API_KEY_ID and APCA_API_SECRET_KEY."
        )

    return TradingClient(
        api_key=api_key,
        secret_key=secret_key,
        paper=PAPER,
    )


def get_open_order_symbols(client: TradingClient) -> Set[str]:
    """
    Return symbols that already have open Alpaca orders.

    This prevents duplicate orders when a previous order is still pending,
    accepted, partially filled, or otherwise open.
    """
    request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
    open_orders = client.get_orders(filter=request)

    symbols = {
        str(getattr(order, "symbol", "")).upper()
        for order in open_orders
        if getattr(order, "symbol", None)
    }

    if symbols:
        print("\nOpen Alpaca orders already exist for:")
        print(", ".join(sorted(symbols)))

    return symbols


def get_position_qty_map(client: TradingClient) -> Dict[str, float]:
    """
    Return current Alpaca position quantities by symbol.

    Used to cap SELL orders so we never request more shares than Alpaca shows
    as currently held.
    """
    positions = client.get_all_positions()

    qty_map = {}

    for position in positions:
        symbol = str(getattr(position, "symbol", "")).upper()

        if not symbol:
            continue

        try:
            qty_map[symbol] = float(position.qty)
        except Exception:
            qty_map[symbol] = 0.0

    return qty_map


def safety_checks() -> None:
    if not PAPER:
        raise RuntimeError("PAPER must be True. This script is paper-only.")

    if not SUBMIT_ORDERS:
        raise RuntimeError(
            "SUBMIT_ORDERS is False. Review proposed_orders.csv first. "
            "When ready for PAPER orders, set SUBMIT_ORDERS = True."
        )

    if CONFIRM_PAPER_ONLY != "YES_PAPER_ONLY":
        raise RuntimeError(
            'CONFIRM_PAPER_ONLY must be exactly "YES_PAPER_ONLY" before submitting.'
        )

    if not PREVIEW_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {PREVIEW_PATH}. Run alpaca_order_preview_hybrid.py first."
        )


def load_proposed_orders() -> pd.DataFrame:
    df = pd.read_csv(PREVIEW_PATH)

    required_cols = [
        "symbol",
        "action",
        "side",
        "order_type",
        "notional_for_buy",
        "qty_for_sell",
        "dollar_delta",
        "signal_date",
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"proposed_orders.csv is missing columns: {missing}")

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["action"] = df["action"].astype(str).str.upper()
    df["side"] = df["side"].astype(str).str.lower()

    orders = df[df["action"].isin(["BUY", "SELL"])].copy()

    if orders.empty:
        print("No BUY/SELL orders found in proposed_orders.csv.")
        return orders

    return orders.reset_index(drop=True)


def get_signal_date_from_preview(orders_df: pd.DataFrame) -> str:
    """
    Read signal_date from proposed_orders.csv.

    Requires alpaca_order_preview_hybrid.py to write signal_date.
    """
    if "signal_date" not in orders_df.columns:
        raise RuntimeError(
            "proposed_orders.csv does not contain signal_date. "
            "Rerun alpaca_order_preview_hybrid.py after adding the signal_date columns."
        )

    signal_dates = orders_df["signal_date"].dropna().astype(str).unique()

    if len(signal_dates) != 1:
        raise RuntimeError(
            f"Expected exactly one signal_date in proposed_orders.csv, got: {signal_dates}"
        )

    return signal_dates[0]


def validate_order_row(
    row: pd.Series,
    position_qty_map: Dict[str, float],
) -> Dict[str, object]:
    symbol = str(row["symbol"]).upper()
    action = str(row["action"]).upper()

    if action not in {"BUY", "SELL"}:
        raise ValueError(f"Invalid action for {symbol}: {action}")

    if action == "BUY":
        notional = float(row.get("notional_for_buy", np.nan))

        if not np.isfinite(notional) or notional < MIN_BUY_NOTIONAL:
            return {
                "valid": False,
                "reason": f"BUY notional below minimum or invalid: {notional}",
            }

        return {
            "valid": True,
            "symbol": symbol,
            "side": OrderSide.BUY,
            "notional": round(notional, 2),
            "qty": None,
        }

    preview_qty = float(row.get("qty_for_sell", np.nan))

    if not np.isfinite(preview_qty) or preview_qty < MIN_SELL_QTY:
        return {
            "valid": False,
            "reason": f"SELL qty below minimum or invalid: {preview_qty}",
        }

    actual_qty = float(position_qty_map.get(symbol, 0.0))

    if actual_qty < MIN_SELL_QTY:
        return {
            "valid": False,
            "reason": f"No Alpaca position quantity available to sell: {actual_qty}",
        }

    # Sell the lower of preview qty and actual Alpaca qty, then apply a tiny
    # safety multiplier to avoid insufficient-quantity errors.
    safe_qty = min(preview_qty, actual_qty) * SELL_QTY_SAFETY_MULTIPLIER
    safe_qty = round(safe_qty, 6)

    if safe_qty < MIN_SELL_QTY:
        return {
            "valid": False,
            "reason": (
                f"Safe SELL qty below minimum after cap/buffer. "
                f"preview_qty={preview_qty}, actual_qty={actual_qty}, safe_qty={safe_qty}"
            ),
        }

    return {
        "valid": True,
        "symbol": symbol,
        "side": OrderSide.SELL,
        "notional": None,
        "qty": safe_qty,
        "preview_qty": preview_qty,
        "actual_qty": actual_qty,
    }


def submit_one_order(client: TradingClient, order_info: Dict[str, object]):
    symbol = order_info["symbol"]
    side = order_info["side"]
    notional = order_info["notional"]
    qty = order_info["qty"]

    if side == OrderSide.BUY:
        request = MarketOrderRequest(
            symbol=symbol,
            notional=notional,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
    else:
        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )

    return client.submit_order(order_data=request)


def order_to_row(
    original_row: pd.Series,
    status: str,
    message: str,
    alpaca_order=None,
    submitted_qty=None,
    actual_position_qty=None,
) -> Dict[str, object]:
    return {
        "symbol": original_row.get("symbol"),
        "action": original_row.get("action"),
        "side": original_row.get("side"),
        "signal_date": original_row.get("signal_date"),
        "rebalance_period": original_row.get("rebalance_period"),
        "dollar_delta": original_row.get("dollar_delta"),
        "notional_for_buy": original_row.get("notional_for_buy"),
        "qty_for_sell": original_row.get("qty_for_sell"),
        "submitted_qty": submitted_qty,
        "actual_position_qty": actual_position_qty,
        "status": status,
        "message": message,
        "alpaca_order_id": getattr(alpaca_order, "id", None) if alpaca_order is not None else None,
        "alpaca_order_status": getattr(alpaca_order, "status", None) if alpaca_order is not None else None,
        "submitted_at": getattr(alpaca_order, "submitted_at", None) if alpaca_order is not None else None,
    }


def submit_orders_phase(
    client: TradingClient,
    orders_df: pd.DataFrame,
    action: str,
    position_qty_map: Dict[str, float],
) -> List[Dict[str, object]]:
    submitted_rows = []

    phase_orders = orders_df[orders_df["action"] == action].copy()

    if phase_orders.empty:
        print(f"No {action} orders to submit.")
        return submitted_rows

    print(f"\n===== SUBMITTING {action} ORDERS =====")

    for _, row in phase_orders.iterrows():
        symbol = row["symbol"]
        order_info = validate_order_row(row, position_qty_map)

        if not order_info["valid"]:
            print(f"SKIP {action} {symbol}: {order_info['reason']}")
            submitted_rows.append(
                order_to_row(row, status="skipped", message=order_info["reason"])
            )
            continue

        try:
            if action == "BUY":
                print(f"Submitting BUY {symbol}: notional=${order_info['notional']:,.2f}")
            else:
                print(
                    f"Submitting SELL {symbol}: qty={order_info['qty']} "
                    f"(preview_qty={order_info.get('preview_qty')}, "
                    f"actual_qty={order_info.get('actual_qty')})"
                )

            alpaca_order = submit_one_order(client, order_info)

            print(
                f"Submitted {action} {symbol}: "
                f"id={getattr(alpaca_order, 'id', None)} "
                f"status={getattr(alpaca_order, 'status', None)}"
            )

            submitted_rows.append(
                order_to_row(
                    row,
                    status="submitted",
                    message="submitted",
                    alpaca_order=alpaca_order,
                    submitted_qty=order_info.get("qty"),
                    actual_position_qty=order_info.get("actual_qty"),
                )
            )

        except Exception as exc:
            print(f"ERROR submitting {action} {symbol}: {exc}")
            submitted_rows.append(
                order_to_row(row, status="error", message=str(exc))
            )

    return submitted_rows


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("\n===== ALPACA HYBRID PAPER ORDER SUBMIT =====")
    print(f"PAPER: {PAPER}")
    print(f"SUBMIT_ORDERS: {SUBMIT_ORDERS}")
    print(f"CONFIRM_PAPER_ONLY: {CONFIRM_PAPER_ONLY}")
    print(f"STRATEGY_NAME: {STRATEGY_NAME}")
    print(f"MODE: {MODE}")
    print(f"ALLOW_RESUBMIT: {ALLOW_RESUBMIT}")
    print(f"Preview file: {PREVIEW_PATH}")

    safety_checks()

    client = get_trading_client()
    account = client.get_account()

    print("\nAccount check:")
    print(f"Account status: {getattr(account, 'status', None)}")
    print(f"Equity: {getattr(account, 'equity', None)}")
    print(f"Buying power: {getattr(account, 'buying_power', None)}")

    orders_df = load_proposed_orders()

    if orders_df.empty:
        print("Nothing to submit.")
        return

    signal_date = get_signal_date_from_preview(orders_df)

    rebalance_period = assert_not_already_submitted(
        strategy_name=STRATEGY_NAME,
        signal_date=signal_date,
        mode=MODE,
        allow_resubmit=ALLOW_RESUBMIT,
    )

    print("\nRebalance guard passed.")
    print(f"Strategy: {STRATEGY_NAME}")
    print(f"Signal date: {signal_date}")
    print(f"Rebalance period: {rebalance_period}")

    open_order_symbols = get_open_order_symbols(client)

    if open_order_symbols:
        before_count = len(orders_df)

        # Exclude symbols that already have open orders.
        orders_df = orders_df[
            ~orders_df["symbol"].astype(str).str.upper().isin(open_order_symbols)
        ].copy()

        skipped_count = before_count - len(orders_df)

        print(
            f"\nSkipped {skipped_count} proposed order(s) because those symbols "
            f"already have open Alpaca orders."
        )

    if orders_df.empty:
        print("All proposed orders were skipped due to existing open orders.")
        return

    position_qty_map = get_position_qty_map(client)

    print("\nOrders loaded from preview:")
    cols = [
        "symbol",
        "action",
        "dollar_delta",
        "notional_for_buy",
        "qty_for_sell",
    ]
    print(orders_df[cols].to_string(index=False))

    submitted_rows = []

    if SELL_FIRST:
        submitted_rows.extend(
            submit_orders_phase(
                client=client,
                orders_df=orders_df,
                action="SELL",
                position_qty_map=position_qty_map,
            )
        )

        if WAIT_AFTER_SELLS_SECONDS > 0:
            print(f"\nWaiting {WAIT_AFTER_SELLS_SECONDS} seconds after sells...")
            time.sleep(WAIT_AFTER_SELLS_SECONDS)

        submitted_rows.extend(
            submit_orders_phase(
                client=client,
                orders_df=orders_df,
                action="BUY",
                position_qty_map=position_qty_map,
            )
        )
    else:
        submitted_rows.extend(
            submit_orders_phase(
                client=client,
                orders_df=orders_df,
                action="BUY",
                position_qty_map=position_qty_map,
            )
        )
        submitted_rows.extend(
            submit_orders_phase(
                client=client,
                orders_df=orders_df,
                action="SELL",
                position_qty_map=position_qty_map,
            )
        )

    submitted_df = pd.DataFrame(submitted_rows)
    submitted_df.to_csv(SUBMITTED_ORDERS_PATH, index=False)

    successful_or_attempted = submitted_df[
        submitted_df["status"].isin(["submitted", "error"])
    ]

    if not successful_or_attempted.empty:
        record_submission(
            strategy_name=STRATEGY_NAME,
            signal_date=signal_date,
            mode=MODE,
            notes=f"Submitted from {PREVIEW_PATH}",
        )

        print("\nRecorded rebalance submission guard:")
        print(f"Strategy: {STRATEGY_NAME}")
        print(f"Period:   {rebalance_period}")
        print(f"Mode:     {MODE}")

        snapshot_path = save_target_snapshot(
            proposed_orders_path=PREVIEW_PATH,
            strategy_name=STRATEGY_NAME,
            rebalance_period=rebalance_period,
            mode=MODE,
        )

        print("\nFrozen monthly target snapshot saved:")
        print(f"Snapshot: {snapshot_path}")

    else:
        print("\nNo submitted/error orders recorded, so rebalance guard was not updated.")
        print("No frozen monthly target snapshot was saved.")

    print("\n===== SAVED =====")
    print(f"Submitted order log: {SUBMITTED_ORDERS_PATH}")

    print("\nReview this in Alpaca paper dashboard before doing anything else.")


if __name__ == "__main__":
    main()