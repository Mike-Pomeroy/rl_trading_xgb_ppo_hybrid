"""
Alpaca paper-order preview for the XGBoost Top-K strategy.

Purpose:
- Connect to Alpaca paper account.
- Read current account equity and positions.
- Fetch split-adjusted data through data_module / xgb_topk_strategy.
- Train model using data before the latest signal date.
- Select current Top-K stocks.
- Calculate target portfolio using small-account rules.
- Print proposed orders.
- Save proposed_orders.csv.
- Submit NO orders.

Run:
    python alpaca_order_preview.py

Environment variables required:
    APCA_API_KEY_ID
    APCA_API_SECRET_KEY
"""
import os

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient

import xgb_topk_strategy as strat


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

OUTPUT_DIR = Path("alpaca_preview_orders")
OUTPUT_DIR.mkdir(exist_ok=True)

PAPER = True
SUBMIT_ORDERS = False  # SAFETY: keep False. This script previews only.

INITIAL_FALLBACK_EQUITY = 3000.0

DATA_LIST = [
    "AAPL", "MSFT", "SPY", "GOOGL", "AMZN",
    "META", "TSLA", "NVDA", "JPM", "JNJ",
    "AVGO", "LLY", "UNH", "COST", "V",
    "MA", "HD", "PG", "XOM", "AMD",
]

TRADE_LIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN",
    "META", "TSLA", "NVDA", "JPM", "JNJ",
    "AVGO", "LLY", "UNH", "COST", "V",
    "MA", "HD", "PG", "XOM", "AMD",
]

TOP_K = 3
TARGET_HORIZON = 30
DATA_ADJUSTMENT = "split"

CASH_BUFFER = 0.20
TRANSACTION_COST_ESTIMATE = 0.005

MIN_DOLLARS_PER_POSITION = 500.0
ALLOW_FRACTIONAL_SHARES = True

TRAIN_START_DATE: Optional[str] = None

# Conservative order controls.
MIN_ORDER_DOLLARS = 10.0
SELL_POSITIONS_NOT_IN_TRADE_LIST = False
SELL_UNSELECTED_TRADE_LIST_POSITIONS = True


# ============================================================
# ALPACA HELPERS
# ============================================================
def get_trading_client() -> TradingClient:
    """
    Create Alpaca trading client.

    Uses paper=True by default.
    Reads credentials from:
    - APCA_API_KEY_ID
    - APCA_API_SECRET_KEY
    """
    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")

    if not api_key or not secret_key:
        raise RuntimeError(
            "Missing Alpaca credentials. Check your .env file has "
            "APCA_API_KEY_ID and APCA_API_SECRET_KEY."
        )

    return TradingClient(
        api_key=api_key,
        secret_key=secret_key,
        paper=PAPER,
    )

def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def get_account_equity(client: TradingClient) -> float:
    account = client.get_account()

    equity = safe_float(getattr(account, "equity", None), 0.0)

    if equity <= 0:
        print(
            f"WARNING: Could not read positive Alpaca equity. "
            f"Using fallback ${INITIAL_FALLBACK_EQUITY:,.2f}."
        )
        equity = INITIAL_FALLBACK_EQUITY

    return equity


def get_positions_df(client: TradingClient) -> pd.DataFrame:
    positions = client.get_all_positions()

    rows = []

    for pos in positions:
        symbol = str(getattr(pos, "symbol", "")).upper()

        rows.append({
            "symbol": symbol,
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


# ============================================================
# MODEL / TARGET PORTFOLIO
# ============================================================

def configure_strategy_module() -> None:
    strat.DATA_LIST = DATA_LIST
    strat.TRADE_LIST = TRADE_LIST
    strat.TOP_K = TOP_K
    strat.TARGET_HORIZON = TARGET_HORIZON
    strat.DATA_ADJUSTMENT = DATA_ADJUSTMENT
    strat.CASH_BUFFER = CASH_BUFFER
    strat.TRANSACTION_COST = TRANSACTION_COST_ESTIMATE
    strat.MIN_DOLLARS_PER_POSITION = MIN_DOLLARS_PER_POSITION
    strat.ALLOW_FRACTIONAL_SHARES = ALLOW_FRACTIONAL_SHARES

def latest_signal_day(full_df: pd.DataFrame) -> pd.Timestamp:
    model_df = full_df[full_df["tic"].isin(TRADE_LIST)].copy()
    model_df = model_df.dropna(subset=strat.FEATURES + ["close"])

    if model_df.empty:
        raise ValueError("No usable model rows after feature cleaning.")

    counts = (
        model_df.groupby("date")["tic"]
        .nunique()
        .sort_index()
    )

    required_count = len(TRADE_LIST)

    complete_dates = counts[counts >= required_count]

    if complete_dates.empty:
        print("\nWARNING: No date has all trade tickers available.")
        print("Recent scoreable ticker counts:")
        print(counts.tail(10).to_string())

        # Conservative fallback: use the latest date with the most available tickers.
        max_count = counts.max()
        best_dates = counts[counts == max_count]
        signal_date = best_dates.index.max()

        print(
            f"Using latest best-coverage date: {pd.Timestamp(signal_date).date()} "
            f"with {int(max_count)} / {required_count} tickers."
        )

        return pd.Timestamp(signal_date)

    signal_date = complete_dates.index.max()

    print(
        f"Using latest full-coverage signal date: {pd.Timestamp(signal_date).date()} "
        f"with {required_count} / {required_count} tickers."
    )

    return pd.Timestamp(signal_date)


def train_current_model(
    full_df: pd.DataFrame,
    signal_date: pd.Timestamp,
):
    target_col = f"future_return_{TARGET_HORIZON}"

    model_df = full_df[full_df["tic"].isin(TRADE_LIST)].copy()

    train_mask = model_df["date"] < signal_date

    if TRAIN_START_DATE is not None:
        train_mask &= model_df["date"] >= pd.Timestamp(TRAIN_START_DATE)

    model = strat.train_model(
        model_df.loc[train_mask],
        strat.FEATURES,
        target_col,
    )

    if model is None:
        raise ValueError(
            "Model training failed. Not enough clean training rows. "
            "Check data history and MIN_TRAIN_ROWS."
        )

    return model

def score_latest_day(
    full_df: pd.DataFrame,
    model,
    signal_date: pd.Timestamp,
) -> pd.DataFrame:
    signal_day = (
        full_df[
            (full_df["date"] == signal_date)
            & (full_df["tic"].isin(TRADE_LIST))
        ]
        .copy()
    )

    before = set(signal_day["tic"].unique())

    signal_day = signal_day.dropna(subset=strat.FEATURES + ["close"]).copy()

    after = set(signal_day["tic"].unique())
    missing = sorted(set(TRADE_LIST) - after)

    print(
        f"Scoreable tickers on {signal_date.date()}: "
        f"{len(after)} / {len(TRADE_LIST)}"
    )

    if missing:
        print("Missing/unscoreable tickers:")
        print(", ".join(missing))

    if signal_day.empty:
        raise ValueError(f"No usable signal rows for {signal_date.date()}.")

    signal_day["score"] = model.predict(signal_day[strat.FEATURES])
    signal_day = signal_day.sort_values("score", ascending=False).reset_index(drop=True)
    signal_day["rank"] = np.arange(1, len(signal_day) + 1)

    return signal_day


def make_target_weights_for_preview(
    selected: List[str],
    tickers: List[str],
    prices: np.ndarray,
    account_equity: float,
) -> np.ndarray:
    return strat.make_target_weights(
        selected=selected,
        tickers=tickers,
        px=prices,
        invested_fraction=max(0.0, 1.0 - CASH_BUFFER),
        portfolio_value=account_equity,
        min_dollars_per_position=MIN_DOLLARS_PER_POSITION,
        allow_fractional_shares=ALLOW_FRACTIONAL_SHARES,
    )


def build_current_holdings_map(positions_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    holdings = {}

    for _, row in positions_df.iterrows():
        symbol = str(row["symbol"]).upper()

        holdings[symbol] = {
            "qty": safe_float(row.get("qty"), 0.0),
            "market_value": safe_float(row.get("market_value"), 0.0),
            "current_price": safe_float(row.get("current_price"), np.nan),
        }

    return holdings


def build_order_preview(
    account_equity: float,
    positions_df: pd.DataFrame,
    signal_day: pd.DataFrame,
    selected: List[str],
) -> pd.DataFrame:
    latest_prices = (
        signal_day.set_index("tic")["close"]
        .reindex(TRADE_LIST)
        .astype(float)
        .values
    )

    target_weights = make_target_weights_for_preview(
        selected=selected,
        tickers=TRADE_LIST,
        prices=latest_prices,
        account_equity=account_equity,
    )

    target_dollars_by_symbol = {
        ticker: account_equity * target_weights[i]
        for i, ticker in enumerate(TRADE_LIST)
    }

    close_by_symbol = {
        row["tic"]: float(row["close"])
        for _, row in signal_day.iterrows()
    }

    score_by_symbol = {
        row["tic"]: float(row["score"])
        for _, row in signal_day.iterrows()
    }

    rank_by_symbol = {
        row["tic"]: int(row["rank"])
        for _, row in signal_day.iterrows()
    }

    holdings = build_current_holdings_map(positions_df)

    symbols_to_consider = set(TRADE_LIST)

    if SELL_POSITIONS_NOT_IN_TRADE_LIST:
        symbols_to_consider |= set(holdings.keys())

    rows = []

    for symbol in sorted(symbols_to_consider):
        in_trade_list = symbol in TRADE_LIST
        is_selected = symbol in selected

        current_qty = holdings.get(symbol, {}).get("qty", 0.0)
        current_value = holdings.get(symbol, {}).get("market_value", 0.0)

        price = close_by_symbol.get(
            symbol,
            holdings.get(symbol, {}).get("current_price", np.nan),
        )

        target_value = 0.0

        if in_trade_list:
            target_value = float(target_dollars_by_symbol.get(symbol, 0.0))

        if not in_trade_list and not SELL_POSITIONS_NOT_IN_TRADE_LIST:
            continue

        if in_trade_list and not is_selected and not SELL_UNSELECTED_TRADE_LIST_POSITIONS:
            target_value = current_value

        dollar_delta = target_value - current_value

        action = "HOLD"
        order_type = "none"
        side = ""
        notional = np.nan
        qty = np.nan

        if abs(dollar_delta) >= MIN_ORDER_DOLLARS:
            if dollar_delta > 0:
                action = "BUY"
                side = "buy"
                order_type = "market_notional_day"
                notional = round(float(dollar_delta), 2)
            else:
                action = "SELL"
                side = "sell"
                order_type = "market_fractional_qty_day"

                if np.isfinite(price) and price > 0:
                    qty = abs(float(dollar_delta)) / price
                    qty = round(qty, 6)
                else:
                    qty = abs(current_qty)
                    qty = round(qty, 6)

        rows.append({
            "symbol": symbol,
            "in_trade_list": in_trade_list,
            "selected": is_selected,
            "rank": rank_by_symbol.get(symbol, np.nan),
            "score": score_by_symbol.get(symbol, np.nan),
            "price_used": price,
            "current_qty": current_qty,
            "current_value": current_value,
            "target_value": target_value,
            "dollar_delta": dollar_delta,
            "action": action,
            "side": side,
            "order_type": order_type,
            "notional_for_buy": notional,
            "qty_for_sell": qty,
        })

    orders_df = pd.DataFrame(rows)

    if not orders_df.empty:
        action_order = {"SELL": 0, "BUY": 1, "HOLD": 2}
        orders_df["action_sort"] = orders_df["action"].map(action_order).fillna(9)
        orders_df = (
            orders_df
            .sort_values(["action_sort", "selected", "rank", "symbol"], ascending=[True, False, True, True])
            .drop(columns=["action_sort"])
            .reset_index(drop=True)
        )

    return orders_df


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if SUBMIT_ORDERS:
        raise RuntimeError(
            "SUBMIT_ORDERS is True. This preview script is designed to submit nothing. "
            "Keep SUBMIT_ORDERS = False."
        )

    configure_strategy_module()

    print("\n===== ALPACA ORDER PREVIEW ONLY =====")
    print(f"PAPER: {PAPER}")
    print(f"SUBMIT_ORDERS: {SUBMIT_ORDERS}")
    print(f"TOP_K: {TOP_K}")
    print(f"TARGET_HORIZON: {TARGET_HORIZON}")
    print(f"DATA_ADJUSTMENT: {DATA_ADJUSTMENT}")
    print(f"CASH_BUFFER: {CASH_BUFFER}")
    print(f"TRANSACTION_COST_ESTIMATE: {TRANSACTION_COST_ESTIMATE}")
    print(f"MIN_DOLLARS_PER_POSITION: {MIN_DOLLARS_PER_POSITION}")
    print(f"TRADE TICKERS: {len(TRADE_LIST)}")

    client = get_trading_client()

    print("\nReading Alpaca paper account...")
    account_equity = get_account_equity(client)
    positions_df = get_positions_df(client)

    print(f"Account equity: ${account_equity:,.2f}")
    print(f"Current positions: {len(positions_df)}")

    if not positions_df.empty:
        print("\nCurrent Alpaca positions:")
        print(
            positions_df[[
                "symbol",
                "qty",
                "market_value",
                "current_price",
                "unrealized_pl",
                "unrealized_plpc",
            ]].to_string(index=False)
        )
    else:
        print("\nCurrent Alpaca positions: none")

    print("\nPreparing model data...")
    full_df = strat.prepare_full_df()
    full_df = strat.normalize_date_column(full_df, "date")

    signal_date = latest_signal_day(full_df)

    print(f"Latest usable signal date: {signal_date.date()}")

    model = train_current_model(full_df, signal_date)
    signal_day = score_latest_day(full_df, model, signal_date)

    selected = signal_day.head(TOP_K)["tic"].tolist()

    print("\n===== MODEL SELECTION =====")
    print(f"Selected Top-{TOP_K}: {', '.join(selected)}")

    print("\nTop scored tickers:")
    print(
        signal_day[["rank", "tic", "score", "close"]]
        .head(10)
        .to_string(index=False)
    )

    orders_df = build_order_preview(
        account_equity=account_equity,
        positions_df=positions_df,
        signal_day=signal_day,
        selected=selected,
    )

    proposed_orders = orders_df[orders_df["action"].isin(["BUY", "SELL"])].copy()

    print("\n===== PROPOSED ORDERS - PREVIEW ONLY =====")

    if proposed_orders.empty:
        print("No proposed BUY/SELL orders. Portfolio is already close to target.")
    else:
        cols = [
            "symbol",
            "selected",
            "action",
            "price_used",
            "current_value",
            "target_value",
            "dollar_delta",
            "notional_for_buy",
            "qty_for_sell",
        ]
        print(proposed_orders[cols].to_string(index=False))

    print("\n===== TARGET PORTFOLIO =====")
    target_cols = [
        "symbol",
        "selected",
        "rank",
        "price_used",
        "current_value",
        "target_value",
        "dollar_delta",
        "action",
    ]
    print(orders_df[target_cols].to_string(index=False))

    orders_path = OUTPUT_DIR / "proposed_orders.csv"
    selected_path = OUTPUT_DIR / "model_scores.csv"
    positions_path = OUTPUT_DIR / "current_positions.csv"

    orders_df.to_csv(orders_path, index=False)
    signal_day.to_csv(selected_path, index=False)
    positions_df.to_csv(positions_path, index=False)

    print("\n===== SAVED FILES =====")
    print(f"Proposed orders:    {orders_path}")
    print(f"Model scores:       {selected_path}")
    print(f"Current positions:  {positions_path}")

    print("\nNo orders were submitted.")


if __name__ == "__main__":
    main()