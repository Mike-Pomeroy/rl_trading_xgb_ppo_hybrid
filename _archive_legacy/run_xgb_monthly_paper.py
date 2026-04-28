import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import time

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from alpaca_trade_api.rest import REST

from config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_BASE_URL,
    XGB_SYMBOLS,
    XGB_TOP_K,
    XGB_CASH_BUFFER,
    XGB_TARGET_HORIZON,
    XGB_STATE_FILE,
    XGB_DRY_RUN,
)
from data_module import prepare_data, INDICATORS

NY_TZ = ZoneInfo("America/New_York")

FEATURE_COLS = INDICATORS + [
    "volatility_30",
    "return_5",
    "return_10",
    "price_vs_sma30",
    "spy_trend",
]


def now_ny() -> datetime:
    return datetime.now(NY_TZ)


def load_state(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def save_state(path: str, state: dict) -> None:
    Path(path).write_text(json.dumps(state, indent=2))


def already_rebalanced_this_month(state: dict) -> bool:
    last_month = state.get("last_rebalance_month")
    current_month = now_ny().strftime("%Y-%m")
    return last_month == current_month


def mark_rebalanced(state_path: str, selected: list[str], weights: dict) -> None:
    state = load_state(state_path)
    state["last_rebalance_month"] = now_ny().strftime("%Y-%m")
    state["last_rebalance_time"] = now_ny().isoformat()
    state["last_selected"] = selected
    state["last_weights"] = weights
    save_state(state_path, state)


def build_feature_table_from_local_data(symbols: list[str], target_horizon: int) -> pd.DataFrame:
    df = prepare_data()
    df = df[df["tic"].isin(symbols)].copy()
    df = df.sort_values(["tic", "date"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("prepare_data() returned no rows for XGB_SYMBOLS")

    df["date"] = pd.to_datetime(df["date"], utc=True).dt.normalize()
    df["future_return"] = (
        df.groupby("tic")["close"].shift(-target_horizon) / df["close"] - 1.0
    )

    needed_cols = ["date", "tic", "close"] + FEATURE_COLS + ["future_return"]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=needed_cols).copy()

    return df


def build_training_and_latest(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df.dropna(subset=feature_cols + ["future_return"]).copy()

    latest_rows = []
    for sym in df["tic"].unique():
        sub = df[df["tic"] == sym].sort_values("date").copy()
        latest = sub.dropna(subset=feature_cols).tail(1)
        if not latest.empty:
            latest_rows.append(latest)

    if not latest_rows:
        raise ValueError("No latest rows available for scoring.")

    latest_df = pd.concat(latest_rows, ignore_index=True)
    return train_df, latest_df


def compute_target_values(
    symbols: list[str],
    selected: list[str],
    portfolio_value: float,
    cash_buffer: float,
) -> np.ndarray:
    invest_fraction = 1.0 - cash_buffer
    per_name = (portfolio_value * invest_fraction) / len(selected)

    target = []
    for sym in symbols:
        target.append(per_name if sym in selected else 0.0)

    return np.array(target, dtype=np.float64)


def get_live_positions(api: REST) -> dict[str, dict]:
    positions_raw = api.list_positions()
    positions = {}

    for p in positions_raw:
        positions[p.symbol] = {
            "qty": float(p.qty),
            "market_value": float(p.market_value),
            "side": p.side,
        }

    return positions


def build_orders(
    symbols: list[str],
    target_values: np.ndarray,
    prices: dict[str, float],
    positions: dict[str, dict],
) -> list[dict]:
    orders = []

    for i, sym in enumerate(symbols):
        target_value = float(target_values[i])
        price = float(prices[sym])

        if price <= 0:
            continue

        current_qty = float(positions.get(sym, {}).get("qty", 0.0))
        current_value = current_qty * price

        diff_value = target_value - current_value
        raw_qty = diff_value / price

        if abs(raw_qty) < 0.01:
            continue

        if raw_qty > 0:
            side = "buy"
            order_qty = round(raw_qty, 6)
        else:
            side = "sell"
            order_qty = round(min(abs(raw_qty), current_qty), 6)
            if order_qty <= 0:
                continue

        orders.append(
            {
                "symbol": sym,
                "qty": order_qty,
                "side": side,
                "type": "market",
                "time_in_force": "day",
            }
        )

    return orders


def split_orders_sells_first(orders: list[dict]) -> list[dict]:
    sells = [o for o in orders if o["side"] == "sell"]
    buys = [o for o in orders if o["side"] == "buy"]
    return sells + buys


def main() -> None:
    if already_rebalanced_this_month(load_state(XGB_STATE_FILE)):
        print("Already rebalanced this month. No action taken.")
        return

    api = REST(
        ALPACA_API_KEY.strip(),
        ALPACA_SECRET_KEY.strip(),
        ALPACA_BASE_URL.strip(),
    )

    df = build_feature_table_from_local_data(
        symbols=XGB_SYMBOLS,
        target_horizon=XGB_TARGET_HORIZON,
    )
    train_df, latest_df = build_training_and_latest(df, FEATURE_COLS)

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(train_df[FEATURE_COLS], train_df["future_return"])

    latest_df = latest_df.copy()
    latest_df["score"] = model.predict(latest_df[FEATURE_COLS])

    ranked = latest_df.sort_values("score", ascending=False)
    selected = ranked.head(XGB_TOP_K)["tic"].tolist()

    account = api.get_account()
    portfolio_value = float(account.portfolio_value)

    print("\nChecking open Alpaca orders...")
    open_orders = api.list_orders(status="open")

    if open_orders:
        print("Open orders found:")
        for o in open_orders:
            print(f"  {o.symbol} {o.side} qty={o.qty} status={o.status}")

        if XGB_DRY_RUN:
            print("DRY RUN: would cancel all open orders before rebalancing.")
        else:
            print("Cancelling open orders...")
            api.cancel_all_orders()
            time.sleep(2)
    else:
        print("No open orders found.")

    positions = get_live_positions(api)

    print("\nCurrent Alpaca positions:")
    if positions:
        for sym, pos in positions.items():
            print(sym, pos)
    else:
        print("  No current positions.")

    prices = {}
    for sym in XGB_SYMBOLS:
        trade = api.get_latest_trade(sym, feed="iex")
        prices[sym] = float(trade.price)

    target_values = compute_target_values(
        symbols=XGB_SYMBOLS,
        selected=selected,
        portfolio_value=portfolio_value,
        cash_buffer=XGB_CASH_BUFFER,
    )

    orders = build_orders(
        symbols=XGB_SYMBOLS,
        target_values=target_values,
        prices=prices,
        positions=positions,
    )
    orders = split_orders_sells_first(orders)

    latest_dates = latest_df[["tic", "date"]].sort_values("tic")

    print("\n===== XGB MONTHLY PAPER SIGNAL =====")
    print(f"Run Time: {now_ny().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Portfolio Value: ${portfolio_value:,.2f}")
    print(f"Selected top {XGB_TOP_K}: {selected}")
    print(f"Cash buffer: {XGB_CASH_BUFFER:.0%}")

    print("\nLatest feature row dates used for scoring:")
    for _, row in latest_dates.iterrows():
        print(f"  {row['tic']}: {pd.Timestamp(row['date']).date()}")

    print("\nScores:")
    for _, row in ranked[["tic", "score"]].iterrows():
        print(f"  {row['tic']}: {row['score']:.6f}")

    print("\nTarget allocations:")
    for sym, tv in zip(XGB_SYMBOLS, target_values):
        print(f"  {sym}: ${tv:,.2f}")

    if len(orders) == 0:
        print("\nNo orders needed.")
        mark_rebalanced(
            XGB_STATE_FILE,
            selected=selected,
            weights={sym: float(tv / portfolio_value) for sym, tv in zip(XGB_SYMBOLS, target_values)},
        )
        return

    print("\nOrders:")
    for order in orders:
        print(order)

    if XGB_DRY_RUN:
        print("\nDRY RUN ONLY. No paper orders sent.")
        return

    for o in orders:
        api.submit_order(
            symbol=o["symbol"],
            qty=o["qty"],
            side=o["side"],
            type=o["type"],
            time_in_force=o["time_in_force"],
        )

    mark_rebalanced(
        XGB_STATE_FILE,
        selected=selected,
        weights={sym: float(tv / portfolio_value) for sym, tv in zip(XGB_SYMBOLS, target_values)},
    )
    print("\nPaper orders sent successfully.")


if __name__ == "__main__":
    main()