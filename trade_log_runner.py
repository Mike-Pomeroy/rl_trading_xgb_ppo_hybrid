"""
Trade log runner for the XGBoost Top-K strategy.

Purpose:
- Use the current best realistic setup.
- Generate a monthly trade log.
- Show selected tickers, target dollars, shares bought/sold, cash, costs, and turnover.
- Save CSV files for review before any Alpaca paper/live trading.

Run:
    python trade_log_runner.py
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import xgb_topk_strategy as strat


# ============================================================
# REALISTIC SMALL-ACCOUNT CONFIG
# ============================================================

OUTPUT_DIR = Path("trade_log_results")
OUTPUT_DIR.mkdir(exist_ok=True)

INITIAL_AMOUNT = 3000.0

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
TRANSACTION_COST = 0.005

MIN_DOLLARS_PER_POSITION = 500.0
ALLOW_FRACTIONAL_SHARES = True

TRAIN_START_DATE = None

TEST_START_DATE = "2021-01-01"
TEST_END_DATE = "2025-01-01"

WALK_FORWARD = True


# ============================================================
# HELPERS
# ============================================================

def make_realistic_target_weights(
    selected: List[str],
    tickers: List[str],
    px: np.ndarray,
    invested_fraction: float,
    portfolio_value: float,
    min_dollars_per_position: float,
    allow_fractional_shares: bool,
) -> np.ndarray:
    """
    Conservative small-account allocation.

    - Equal-weight selected valid tickers.
    - Skip positions below minimum dollars.
    - If fractional shares are disabled, skip names where target dollars
      cannot buy at least one whole share.
    - Re-equal-weight across surviving selected tickers.
    - If none survive, stay in cash.
    """
    valid_px = np.isfinite(px) & (px > 0)

    selected_indices = [
        i for i, ticker in enumerate(tickers)
        if ticker in selected and valid_px[i]
    ]

    weights = np.zeros(len(tickers), dtype=np.float64)

    if not selected_indices:
        return weights

    candidate_indices = selected_indices.copy()

    while candidate_indices:
        equal_weight = invested_fraction / len(candidate_indices)
        target_dollars = portfolio_value * equal_weight

        valid_candidate_indices = []

        for idx in candidate_indices:
            price = px[idx]

            if target_dollars < min_dollars_per_position:
                continue

            if not allow_fractional_shares and target_dollars < price:
                continue

            valid_candidate_indices.append(idx)

        if len(valid_candidate_indices) == len(candidate_indices):
            for idx in valid_candidate_indices:
                weights[idx] = equal_weight

            return weights

        if len(valid_candidate_indices) == 0:
            return weights

        candidate_indices = valid_candidate_indices

    return weights


def rebalance_with_trade_log(
    cash: float,
    shares: np.ndarray,
    px: np.ndarray,
    target_weights: np.ndarray,
    transaction_cost: float,
    tickers: List[str],
    signal_date: pd.Timestamp,
    execution_date: pd.Timestamp,
    selected: List[str],
) -> Dict[str, object]:
    """
    Rebalance portfolio and return:
    - new cash
    - new shares
    - transaction cost
    - turnover
    - trade rows
    - before/after portfolio values
    """
    current_values = strat.current_position_values(shares, px)
    portfolio_value_before = cash + float(np.sum(current_values))

    if portfolio_value_before <= 0:
        return {
            "cash": 0.0,
            "shares": np.zeros_like(shares),
            "transaction_cost_paid": 0.0,
            "turnover": 0.0,
            "portfolio_value_before": 0.0,
            "portfolio_value_after": 0.0,
            "trade_rows": [],
        }

    current_weights = current_values / (portfolio_value_before + 1e-12)

    turnover = float(np.sum(np.abs(target_weights - current_weights)))
    transaction_cost_paid = transaction_cost * turnover * portfolio_value_before

    investable_value = max(portfolio_value_before - transaction_cost_paid, 0.0)

    new_shares = np.zeros_like(shares)
    target_dollars = investable_value * target_weights

    valid_px = np.isfinite(px) & (px > 0)

    for i in range(len(tickers)):
        if target_weights[i] > 0 and valid_px[i]:
            new_shares[i] = target_dollars[i] / px[i]

    new_position_values = strat.current_position_values(new_shares, px)
    new_cash = investable_value - float(np.sum(new_position_values))
    portfolio_value_after = new_cash + float(np.sum(new_position_values))

    trade_rows = []

    for i, ticker in enumerate(tickers):
        old_shares = float(shares[i])
        new_share_qty = float(new_shares[i])
        share_delta = new_share_qty - old_shares

        old_value = float(current_values[i])
        new_value = float(new_position_values[i])
        dollar_delta = new_value - old_value

        action = "HOLD"

        if share_delta > 1e-8:
            action = "BUY"
        elif share_delta < -1e-8:
            action = "SELL"

        # Log meaningful trades, current holdings, and selected target names.
        should_log = (
            action != "HOLD"
            or ticker in selected
            or abs(old_value) > 1e-6
            or abs(new_value) > 1e-6
        )

        if should_log:
            trade_rows.append({
                "signal_date": signal_date,
                "execution_date": execution_date,
                "ticker": ticker,
                "selected": ticker in selected,
                "action": action,
                "price": float(px[i]) if valid_px[i] else np.nan,
                "old_shares": old_shares,
                "new_shares": new_share_qty,
                "share_delta": share_delta,
                "old_value": old_value,
                "target_value": new_value,
                "dollar_delta": dollar_delta,
                "target_weight": float(target_weights[i]),
                "portfolio_value_before": portfolio_value_before,
                "portfolio_value_after": portfolio_value_after,
                "cash_after_rebalance": new_cash,
                "turnover": turnover,
                "transaction_cost_paid": transaction_cost_paid,
            })

    return {
        "cash": new_cash,
        "shares": new_shares,
        "transaction_cost_paid": transaction_cost_paid,
        "turnover": turnover,
        "portfolio_value_before": portfolio_value_before,
        "portfolio_value_after": portfolio_value_after,
        "trade_rows": trade_rows,
    }


def compute_position_snapshot(
    date: pd.Timestamp,
    cash: float,
    shares: np.ndarray,
    px: np.ndarray,
    tickers: List[str],
) -> List[Dict[str, object]]:
    rows = []
    values = strat.current_position_values(shares, px)
    portfolio_value = cash + float(np.sum(values))

    for i, ticker in enumerate(tickers):
        if abs(values[i]) > 1e-6:
            rows.append({
                "date": date,
                "ticker": ticker,
                "shares": float(shares[i]),
                "price": float(px[i]) if np.isfinite(px[i]) else np.nan,
                "position_value": float(values[i]),
                "weight": float(values[i] / (portfolio_value + 1e-12)),
                "cash": cash,
                "portfolio_value": portfolio_value,
            })

    return rows


# ============================================================
# MAIN STRATEGY RUN WITH TRADE LOG
# ============================================================

def run_trade_log_strategy() -> None:
    # Make imported strategy module use this runner's config.
    strat.DATA_LIST = DATA_LIST
    strat.TRADE_LIST = TRADE_LIST
    strat.TOP_K = TOP_K
    strat.TARGET_HORIZON = TARGET_HORIZON
    strat.CASH_BUFFER = CASH_BUFFER
    strat.TRANSACTION_COST = TRANSACTION_COST
    strat.MIN_DOLLARS_PER_POSITION = MIN_DOLLARS_PER_POSITION
    strat.ALLOW_FRACTIONAL_SHARES = ALLOW_FRACTIONAL_SHARES
    strat.DATA_ADJUSTMENT = DATA_ADJUSTMENT

    print("Preparing data...")
    full_df = strat.prepare_full_df()
    full_df = strat.normalize_date_column(full_df, "date")

    model_df = full_df[full_df["tic"].isin(TRADE_LIST)].copy()

    test_mask = (
        (model_df["date"] >= pd.Timestamp(TEST_START_DATE)) &
        (model_df["date"] < pd.Timestamp(TEST_END_DATE))
    )

    test_df = model_df.loc[test_mask].dropna(
        subset=strat.FEATURES + ["close"]
    ).copy()

    if test_df.empty:
        raise ValueError("No test rows found. Check dates and ticker data.")

    price_matrix = strat.build_price_matrix(test_df, TRADE_LIST)
    dates = list(price_matrix.index.unique())
    signal_dates = strat.get_monthly_signal_dates(dates)

    target_col = f"future_return_{TARGET_HORIZON}"

    cash = float(INITIAL_AMOUNT)
    shares = np.zeros(len(TRADE_LIST), dtype=np.float64)

    trade_log_rows = []
    monthly_rows = []
    daily_history_rows = []
    selection_rows = []
    score_rows = []
    snapshot_rows = []

    pending_rebalances: Dict[pd.Timestamp, Dict[str, object]] = {}

    print(f"Running monthly walk-forward strategy from {TEST_START_DATE} to {TEST_END_DATE}...")

    for signal_date in signal_dates:
        execution_date = strat.next_trading_date(dates, signal_date)

        if execution_date is None:
            continue

        signal_day = (
            test_df[test_df["date"] == signal_date]
            .dropna(subset=strat.FEATURES)
            .copy()
        )

        if signal_day.empty:
            continue

        train_mask = model_df["date"] < signal_date

        if TRAIN_START_DATE is not None:
            train_mask &= model_df["date"] >= pd.Timestamp(TRAIN_START_DATE)

        model = strat.train_model(
            model_df.loc[train_mask],
            strat.FEATURES,
            target_col,
        )

        if model is None:
            print(f"Skipping {signal_date.date()} - not enough training rows.")
            continue

        signal_day["score"] = model.predict(signal_day[strat.FEATURES])
        signal_day = signal_day.sort_values("score", ascending=False)

        selected = signal_day.head(TOP_K)["tic"].tolist()

        for rank, (_, row) in enumerate(signal_day.iterrows(), start=1):
            score_rows.append({
                "signal_date": signal_date,
                "ticker": row["tic"],
                "rank": rank,
                "score": float(row["score"]),
                target_col: row.get(target_col, np.nan),
                "selected": row["tic"] in selected,
            })

        exec_px = price_matrix.loc[execution_date].astype(float).values

        current_values = strat.current_position_values(shares, exec_px)
        current_portfolio_value = cash + float(np.sum(current_values))

        target_weights = make_realistic_target_weights(
            selected=selected,
            tickers=TRADE_LIST,
            px=exec_px,
            invested_fraction=max(0.0, 1.0 - CASH_BUFFER),
            portfolio_value=current_portfolio_value,
            min_dollars_per_position=MIN_DOLLARS_PER_POSITION,
            allow_fractional_shares=ALLOW_FRACTIONAL_SHARES,
        )

        pending_rebalances[execution_date] = {
            "signal_date": signal_date,
            "selected": selected,
            "target_weights": target_weights,
        }

        selection_rows.append({
            "signal_date": signal_date,
            "execution_date": execution_date,
            "selected": ",".join(selected),
            "target_weight_sum": float(np.sum(target_weights)),
            "expected_cash_weight": float(1.0 - np.sum(target_weights)),
            "portfolio_value_at_signal": current_portfolio_value,
        })

    for dt in dates:
        px = price_matrix.loc[dt].astype(float).values

        if dt in pending_rebalances:
            rebalance_info = pending_rebalances[dt]

            result = rebalance_with_trade_log(
                cash=cash,
                shares=shares,
                px=px,
                target_weights=rebalance_info["target_weights"],
                transaction_cost=TRANSACTION_COST,
                tickers=TRADE_LIST,
                signal_date=rebalance_info["signal_date"],
                execution_date=dt,
                selected=rebalance_info["selected"],
            )

            cash = result["cash"]
            shares = result["shares"]

            trade_log_rows.extend(result["trade_rows"])

            monthly_rows.append({
                "signal_date": rebalance_info["signal_date"],
                "execution_date": dt,
                "selected": ",".join(rebalance_info["selected"]),
                "portfolio_value_before": result["portfolio_value_before"],
                "portfolio_value_after": result["portfolio_value_after"],
                "cash_after_rebalance": cash,
                "cash_weight_after_rebalance": cash / (result["portfolio_value_after"] + 1e-12),
                "turnover": result["turnover"],
                "transaction_cost_paid": result["transaction_cost_paid"],
                "target_weight_sum": float(np.sum(rebalance_info["target_weights"])),
                "num_positions_after": int(np.sum(shares > 1e-8)),
            })

            snapshot_rows.extend(
                compute_position_snapshot(
                    date=dt,
                    cash=cash,
                    shares=shares,
                    px=px,
                    tickers=TRADE_LIST,
                )
            )

        values = strat.current_position_values(shares, px)
        portfolio_value = cash + float(np.sum(values))

        daily_history_rows.append({
            "date": dt,
            "portfolio_value": portfolio_value,
            "cash": cash,
            "invested_value": float(np.sum(values)),
            "num_positions": int(np.sum(shares > 1e-8)),
        })

    history_df = pd.DataFrame(daily_history_rows)
    monthly_df = pd.DataFrame(monthly_rows)
    trade_log_df = pd.DataFrame(trade_log_rows)
    selections_df = pd.DataFrame(selection_rows)
    scores_df = pd.DataFrame(score_rows)
    snapshots_df = pd.DataFrame(snapshot_rows)

    stats = strat.compute_stats(
        history_df["portfolio_value"].values,
        INITIAL_AMOUNT,
    )

    # Save files.
    history_path = OUTPUT_DIR / "trade_log_daily_history.csv"
    monthly_path = OUTPUT_DIR / "trade_log_monthly_summary.csv"
    trade_log_path = OUTPUT_DIR / "trade_log_orders.csv"
    selections_path = OUTPUT_DIR / "trade_log_selections.csv"
    scores_path = OUTPUT_DIR / "trade_log_scores.csv"
    snapshots_path = OUTPUT_DIR / "trade_log_position_snapshots.csv"

    history_df.to_csv(history_path, index=False)
    monthly_df.to_csv(monthly_path, index=False)
    trade_log_df.to_csv(trade_log_path, index=False)
    selections_df.to_csv(selections_path, index=False)
    scores_df.to_csv(scores_path, index=False)
    snapshots_df.to_csv(snapshots_path, index=False)

    print("\n===== TRADE LOG STRATEGY RESULT =====")
    print(f"Initial Amount: ${INITIAL_AMOUNT:,.2f}")
    print(f"Final Portfolio: ${stats['final_portfolio']:,.2f}")
    print(f"Total Return: {stats['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {stats['sharpe']:.3f}")
    print(f"Max Drawdown: {stats['max_drawdown_pct']:.2f}%")

    print("\n===== CONFIG =====")
    print(f"TOP_K: {TOP_K}")
    print(f"TARGET_HORIZON: {TARGET_HORIZON}")
    print(f"DATA_ADJUSTMENT: {DATA_ADJUSTMENT}")
    print(f"CASH_BUFFER: {CASH_BUFFER}")
    print(f"TRANSACTION_COST: {TRANSACTION_COST}")
    print(f"MIN_DOLLARS_PER_POSITION: {MIN_DOLLARS_PER_POSITION}")
    print(f"ALLOW_FRACTIONAL_SHARES: {ALLOW_FRACTIONAL_SHARES}")
    print(f"TRADE TICKERS: {len(TRADE_LIST)}")

    print("\n===== SAVED FILES =====")
    print(f"Daily history:        {history_path}")
    print(f"Monthly summary:      {monthly_path}")
    print(f"Trade orders:         {trade_log_path}")
    print(f"Selections:           {selections_path}")
    print(f"Scores:               {scores_path}")
    print(f"Position snapshots:   {snapshots_path}")

    print("\n===== RECENT MONTHLY REBALANCES =====")
    if not monthly_df.empty:
        print(
            monthly_df.tail(12)[[
                "signal_date",
                "execution_date",
                "selected",
                "portfolio_value_before",
                "portfolio_value_after",
                "cash_after_rebalance",
                "turnover",
                "transaction_cost_paid",
                "num_positions_after",
            ]].to_string(index=False)
        )
    else:
        print("No monthly rebalances generated.")

    print("\n===== RECENT TRADE ORDERS =====")
    if not trade_log_df.empty:
        print(
            trade_log_df.tail(30)[[
                "execution_date",
                "ticker",
                "selected",
                "action",
                "price",
                "old_shares",
                "new_shares",
                "share_delta",
                "old_value",
                "target_value",
                "dollar_delta",
            ]].to_string(index=False)
        )
    else:
        print("No trade orders generated.")


if __name__ == "__main__":
    run_trade_log_strategy()