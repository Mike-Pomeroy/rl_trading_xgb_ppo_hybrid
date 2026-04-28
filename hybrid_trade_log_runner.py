"""
Hybrid trade log runner.

Purpose:
- Backtest the best hybrid universe setup:
    Fixed current 20 + screened top 5 additions
- Each month:
    1. Use only past data to pick screened additions.
    2. Combine fixed 20 + additions.
    3. Train XGBoost on that month's hybrid universe.
    4. Pick Top 3.
    5. Rebalance next trading day.
    6. Log actual trades, turnover, cash, costs, and selected universe.

This does NOT trade.
This does NOT submit Alpaca orders.

Run:
    python -u hybrid_trade_log_runner.py
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import xgb_topk_strategy as strat


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = Path("hybrid_trade_log_results")
OUTPUT_DIR.mkdir(exist_ok=True)

INITIAL_AMOUNT = 3000.0

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

HYBRID_ADD_COUNT = 5

CURRENT_20_DATA_LIST = [
    "AAPL", "MSFT", "SPY", "GOOGL", "AMZN",
    "META", "TSLA", "NVDA", "JPM", "JNJ",
    "AVGO", "LLY", "UNH", "COST", "V",
    "MA", "HD", "PG", "XOM", "AMD",
]

CURRENT_20_TRADE_LIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN",
    "META", "TSLA", "NVDA", "JPM", "JNJ",
    "AVGO", "LLY", "UNH", "COST", "V",
    "MA", "HD", "PG", "XOM", "AMD",
]

CANDIDATE_UNIVERSE = sorted(set([
    # Current 20
    "AAPL", "MSFT", "SPY", "GOOGL", "AMZN",
    "META", "TSLA", "NVDA", "JPM", "JNJ",
    "AVGO", "LLY", "UNH", "COST", "V",
    "MA", "HD", "PG", "XOM", "AMD",

    # Additional large/liquid candidates
    "WMT", "ORCL", "NFLX", "BAC",
    "KO", "PEP", "MRK", "ABBV", "CVX",
    "CRM", "ADBE", "CSCO", "TMO", "MCD",
    "ABT", "WFC", "LIN", "ACN", "DIS",
    "IBM", "QCOM", "INTU", "TXN", "NOW",
    "GE", "PM", "CAT", "ISRG", "AMAT",
    "NEE", "UBER", "BKNG", "RTX", "HON",
    "LOW", "PFE", "GS", "AXP", "BLK",
    "AMGN", "SPGI", "PLD", "SYK", "C",
    "SCHW", "DE", "MDT", "LMT", "TJX",
    "ELV", "ADP", "VRTX", "GILD", "ADI",
    "MU", "PANW", "KLAC", "LRCX", "REGN",
    "CB", "MMC", "BSX", "ETN", "FI",
    "SO", "DUK", "COP", "SLB", "BA",
]))

EXCLUDE_FROM_SELECTION = {"SPY"}

MIN_HISTORY_DAYS = 750
MIN_PRICE = 10.0
MIN_AVG_DOLLAR_VOLUME_60D = 50_000_000

MOMENTUM_6M_DAYS = 126
MOMENTUM_12M_DAYS = 252
VOL_DAYS = 60
DRAWDOWN_DAYS = 252


# ============================================================
# SCREENING HELPERS
# ============================================================

def max_drawdown(values: pd.Series) -> float:
    values = values.dropna().astype(float)

    if values.empty:
        return np.nan

    peak = values.cummax()
    drawdown = values / peak - 1.0

    return float(drawdown.min())


def safe_pct_change(series: pd.Series, periods: int) -> float:
    series = series.dropna().astype(float)

    if len(series) <= periods:
        return np.nan

    start = series.iloc[-periods - 1]
    end = series.iloc[-1]

    if not np.isfinite(start) or start <= 0:
        return np.nan

    return float(end / start - 1.0)


def compute_screen_metrics_asof(
    full_df: pd.DataFrame,
    signal_date: pd.Timestamp,
    candidate_tickers: List[str],
) -> pd.DataFrame:
    hist = full_df[
        (full_df["date"] <= signal_date)
        & (full_df["tic"].isin(candidate_tickers))
    ].copy()

    rows = []

    for tic, g in hist.groupby("tic"):
        g = g.sort_values("date").copy()

        close = g["close"].astype(float)
        volume = g["volume"].astype(float)

        if close.empty:
            continue

        daily_returns = close.pct_change()

        latest_close = close.iloc[-1]
        history_days = int(close.notna().sum())

        avg_dollar_volume_60d = (
            (close * volume)
            .tail(60)
            .replace([np.inf, -np.inf], np.nan)
            .mean()
        )

        momentum_6m = safe_pct_change(close, MOMENTUM_6M_DAYS)
        momentum_12m = safe_pct_change(close, MOMENTUM_12M_DAYS)

        volatility_60d = (
            daily_returns
            .tail(VOL_DAYS)
            .replace([np.inf, -np.inf], np.nan)
            .std()
        )

        drawdown_1y = max_drawdown(close.tail(DRAWDOWN_DAYS))

        rows.append({
            "signal_date": signal_date,
            "ticker": tic,
            "latest_close": float(latest_close) if pd.notna(latest_close) else np.nan,
            "history_days": history_days,
            "avg_dollar_volume_60d": (
                float(avg_dollar_volume_60d)
                if pd.notna(avg_dollar_volume_60d)
                else np.nan
            ),
            "momentum_6m": momentum_6m,
            "momentum_12m": momentum_12m,
            "volatility_60d": (
                float(volatility_60d)
                if pd.notna(volatility_60d)
                else np.nan
            ),
            "drawdown_1y": drawdown_1y,
        })

    return pd.DataFrame(rows)


def add_universe_score(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()

    out["eligible"] = True
    out.loc[out["ticker"].isin(EXCLUDE_FROM_SELECTION), "eligible"] = False
    out.loc[out["history_days"] < MIN_HISTORY_DAYS, "eligible"] = False
    out.loc[out["latest_close"] < MIN_PRICE, "eligible"] = False
    out.loc[out["avg_dollar_volume_60d"] < MIN_AVG_DOLLAR_VOLUME_60D, "eligible"] = False

    required_cols = [
        "momentum_6m",
        "momentum_12m",
        "volatility_60d",
        "drawdown_1y",
    ]

    for col in required_cols:
        out.loc[out[col].isna(), "eligible"] = False

    eligible = out["eligible"]

    out["rank_momentum_6m"] = np.nan
    out["rank_momentum_12m"] = np.nan
    out["rank_liquidity"] = np.nan
    out["rank_low_volatility"] = np.nan
    out["rank_low_drawdown"] = np.nan

    out.loc[eligible, "rank_momentum_6m"] = out.loc[eligible, "momentum_6m"].rank(pct=True)
    out.loc[eligible, "rank_momentum_12m"] = out.loc[eligible, "momentum_12m"].rank(pct=True)
    out.loc[eligible, "rank_liquidity"] = out.loc[eligible, "avg_dollar_volume_60d"].rank(pct=True)

    out.loc[eligible, "rank_low_volatility"] = (
        -out.loc[eligible, "volatility_60d"]
    ).rank(pct=True)

    out.loc[eligible, "rank_low_drawdown"] = out.loc[eligible, "drawdown_1y"].rank(pct=True)

    out["universe_score"] = (
        0.30 * out["rank_momentum_6m"]
        + 0.25 * out["rank_momentum_12m"]
        + 0.20 * out["rank_liquidity"]
        + 0.15 * out["rank_low_volatility"]
        + 0.10 * out["rank_low_drawdown"]
    )

    out.loc[~eligible, "universe_score"] = np.nan

    out = out.sort_values(
        ["eligible", "universe_score", "avg_dollar_volume_60d"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    out["universe_rank"] = np.nan

    if out["eligible"].any():
        out.loc[out["eligible"], "universe_rank"] = np.arange(
            1,
            int(out["eligible"].sum()) + 1,
        )

    return out


def select_screened_additions_asof(
    full_df: pd.DataFrame,
    signal_date: pd.Timestamp,
    add_count: int,
) -> Tuple[List[str], pd.DataFrame]:
    candidate_trade_list = sorted(
        set(CANDIDATE_UNIVERSE)
        - {"SPY"}
        - set(CURRENT_20_TRADE_LIST)
    )

    metrics = compute_screen_metrics_asof(
        full_df=full_df,
        signal_date=signal_date,
        candidate_tickers=candidate_trade_list,
    )

    scored = add_universe_score(metrics)

    additions = (
        scored[scored["eligible"]]
        .head(add_count)["ticker"]
        .tolist()
    )

    return additions, scored


# ============================================================
# TRADE LOG HELPERS
# ============================================================

def rebalance_with_trade_log(
    cash: float,
    shares: np.ndarray,
    px: np.ndarray,
    target_weights: np.ndarray,
    transaction_cost: float,
    tickers: List[str],
    signal_date: pd.Timestamp,
    execution_date: pd.Timestamp,
    selected_top3: List[str],
    additions: List[str],
    hybrid_universe: List[str],
) -> Dict[str, object]:
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

        should_log = (
            action != "HOLD"
            or ticker in selected_top3
            or abs(old_value) > 1e-6
            or abs(new_value) > 1e-6
        )

        if should_log:
            trade_rows.append({
                "signal_date": signal_date,
                "execution_date": execution_date,
                "ticker": ticker,
                "in_hybrid_universe": ticker in hybrid_universe,
                "is_screened_addition": ticker in additions,
                "selected_top3": ticker in selected_top3,
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
                "selected_top3_list": ",".join(selected_top3),
                "screened_additions": ",".join(additions),
                "hybrid_universe": ",".join(hybrid_universe),
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


# ============================================================
# MAIN RUN
# ============================================================

def configure_strategy_module() -> None:
    strat.DATA_LIST = sorted(set(CANDIDATE_UNIVERSE))
    strat.TRADE_LIST = sorted(set(CANDIDATE_UNIVERSE) - {"SPY"})
    strat.TOP_K = TOP_K
    strat.TARGET_HORIZON = TARGET_HORIZON
    strat.DATA_ADJUSTMENT = DATA_ADJUSTMENT
    strat.CASH_BUFFER = CASH_BUFFER
    strat.TRANSACTION_COST = TRANSACTION_COST
    strat.MIN_DOLLARS_PER_POSITION = MIN_DOLLARS_PER_POSITION
    strat.ALLOW_FRACTIONAL_SHARES = ALLOW_FRACTIONAL_SHARES


def run_hybrid_trade_log() -> None:
    configure_strategy_module()

    all_trade_tickers = sorted(set(CANDIDATE_UNIVERSE) - {"SPY"})

    print("\n===== HYBRID TRADE LOG RUNNER =====")
    print(f"Initial amount:       ${INITIAL_AMOUNT:,.2f}")
    print(f"Hybrid add count:     {HYBRID_ADD_COUNT}")
    print(f"All trade tickers:    {len(all_trade_tickers)}")
    print(f"Top K:                {TOP_K}")
    print(f"Target horizon:       {TARGET_HORIZON}")
    print(f"Data adjustment:      {DATA_ADJUSTMENT}")
    print(f"Cash buffer:          {CASH_BUFFER}")
    print(f"Transaction cost:     {TRANSACTION_COST}")
    print(f"Test:                 {TEST_START_DATE} to {TEST_END_DATE}")

    print("\nPreparing data...")
    full_df = strat.prepare_full_df()
    full_df = strat.normalize_date_column(full_df, "date")

    target_col = f"future_return_{TARGET_HORIZON}"

    model_df = full_df[full_df["tic"].isin(all_trade_tickers)].copy()

    test_mask = (
        (model_df["date"] >= pd.Timestamp(TEST_START_DATE))
        & (model_df["date"] < pd.Timestamp(TEST_END_DATE))
    )

    test_df = model_df.loc[test_mask].dropna(
        subset=strat.FEATURES + ["close"]
    ).copy()

    if test_df.empty:
        raise ValueError("No test rows found. Check date range and data.")

    price_matrix = strat.build_price_matrix(test_df, all_trade_tickers)
    dates = list(price_matrix.index.unique())
    signal_dates = strat.get_monthly_signal_dates(dates)

    cash = float(INITIAL_AMOUNT)
    shares = np.zeros(len(all_trade_tickers), dtype=np.float64)

    pending_rebalances: Dict[pd.Timestamp, Dict[str, object]] = {}

    monthly_rows = []
    trade_rows = []
    daily_history_rows = []
    selection_rows = []
    score_rows = []
    screen_rows = []

    print("\nBuilding monthly signals...")

    for signal_date in signal_dates:
        execution_date = strat.next_trading_date(dates, signal_date)

        if execution_date is None:
            continue

        additions, screened = select_screened_additions_asof(
            full_df=full_df,
            signal_date=signal_date,
            add_count=HYBRID_ADD_COUNT,
        )

        hybrid_universe = sorted(set(CURRENT_20_TRADE_LIST) | set(additions))

        if not screened.empty:
            for _, row in screened.iterrows():
                screen_rows.append({
                    "signal_date": signal_date,
                    "ticker": row["ticker"],
                    "eligible": row["eligible"],
                    "universe_rank": row["universe_rank"],
                    "universe_score": row["universe_score"],
                    "in_additions": row["ticker"] in additions,
                    "add_count": HYBRID_ADD_COUNT,
                })

        train_mask = model_df["date"] < signal_date

        if TRAIN_START_DATE is not None:
            train_mask &= model_df["date"] >= pd.Timestamp(TRAIN_START_DATE)

        train_df = model_df.loc[
            train_mask
            & model_df["tic"].isin(hybrid_universe)
        ].copy()

        model = strat.train_model(
            train_df=train_df,
            features=strat.FEATURES,
            target_col=target_col,
        )

        if model is None:
            print(f"Skipping {signal_date.date()} - not enough training rows.")
            continue

        signal_day = (
            model_df[
                (model_df["date"] == signal_date)
                & (model_df["tic"].isin(hybrid_universe))
            ]
            .dropna(subset=strat.FEATURES + ["close"])
            .copy()
        )

        if signal_day.empty:
            continue

        signal_day["score"] = model.predict(signal_day[strat.FEATURES])
        signal_day = signal_day.sort_values("score", ascending=False).reset_index(drop=True)
        signal_day["rank"] = np.arange(1, len(signal_day) + 1)

        selected_top3 = signal_day.head(TOP_K)["tic"].tolist()

        for _, row in signal_day.iterrows():
            score_rows.append({
                "signal_date": signal_date,
                "ticker": row["tic"],
                "rank": int(row["rank"]),
                "score": float(row["score"]),
                target_col: row.get(target_col, np.nan),
                "selected_top3": row["tic"] in selected_top3,
                "is_screened_addition": row["tic"] in additions,
                "hybrid_universe": ",".join(hybrid_universe),
            })

        exec_px = price_matrix.loc[execution_date].astype(float).values

        current_values = strat.current_position_values(shares, exec_px)
        current_portfolio_value = cash + float(np.sum(current_values))

        target_weights = strat.make_target_weights(
            selected=selected_top3,
            tickers=all_trade_tickers,
            px=exec_px,
            invested_fraction=max(0.0, 1.0 - CASH_BUFFER),
            portfolio_value=current_portfolio_value,
            min_dollars_per_position=MIN_DOLLARS_PER_POSITION,
            allow_fractional_shares=ALLOW_FRACTIONAL_SHARES,
        )

        pending_rebalances[execution_date] = {
            "signal_date": signal_date,
            "selected_top3": selected_top3,
            "additions": additions,
            "hybrid_universe": hybrid_universe,
            "target_weights": target_weights,
        }

        selection_rows.append({
            "signal_date": signal_date,
            "execution_date": execution_date,
            "selected_top3": ",".join(selected_top3),
            "screened_additions": ",".join(additions),
            "hybrid_universe": ",".join(hybrid_universe),
            "hybrid_universe_count": len(hybrid_universe),
            "target_weight_sum": float(np.sum(target_weights)),
            "expected_cash_weight": float(1.0 - np.sum(target_weights)),
        })

    print("\nRunning portfolio simulation...")

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
                tickers=all_trade_tickers,
                signal_date=rebalance_info["signal_date"],
                execution_date=dt,
                selected_top3=rebalance_info["selected_top3"],
                additions=rebalance_info["additions"],
                hybrid_universe=rebalance_info["hybrid_universe"],
            )

            cash = result["cash"]
            shares = result["shares"]
            trade_rows.extend(result["trade_rows"])

            monthly_rows.append({
                "signal_date": rebalance_info["signal_date"],
                "execution_date": dt,
                "selected_top3": ",".join(rebalance_info["selected_top3"]),
                "screened_additions": ",".join(rebalance_info["additions"]),
                "hybrid_universe_count": len(rebalance_info["hybrid_universe"]),
                "portfolio_value_before": result["portfolio_value_before"],
                "portfolio_value_after": result["portfolio_value_after"],
                "cash_after_rebalance": cash,
                "cash_weight_after_rebalance": cash / (result["portfolio_value_after"] + 1e-12),
                "turnover": result["turnover"],
                "transaction_cost_paid": result["transaction_cost_paid"],
                "target_weight_sum": float(np.sum(rebalance_info["target_weights"])),
                "num_positions_after": int(np.sum(shares > 1e-8)),
            })

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
    trades_df = pd.DataFrame(trade_rows)
    selections_df = pd.DataFrame(selection_rows)
    scores_df = pd.DataFrame(score_rows)
    screen_df = pd.DataFrame(screen_rows)

    stats = strat.compute_stats(
        history_df["portfolio_value"].values,
        INITIAL_AMOUNT,
    )

    history_path = OUTPUT_DIR / "hybrid_trade_log_daily_history.csv"
    monthly_path = OUTPUT_DIR / "hybrid_trade_log_monthly_summary.csv"
    trades_path = OUTPUT_DIR / "hybrid_trade_log_orders.csv"
    selections_path = OUTPUT_DIR / "hybrid_trade_log_selections.csv"
    scores_path = OUTPUT_DIR / "hybrid_trade_log_scores.csv"
    screen_path = OUTPUT_DIR / "hybrid_trade_log_screened_additions.csv"

    history_df.to_csv(history_path, index=False)
    monthly_df.to_csv(monthly_path, index=False)
    trades_df.to_csv(trades_path, index=False)
    selections_df.to_csv(selections_path, index=False)
    scores_df.to_csv(scores_path, index=False)
    screen_df.to_csv(screen_path, index=False)

    print("\n===== HYBRID TRADE LOG RESULT =====")
    print(f"Initial Amount: ${INITIAL_AMOUNT:,.2f}")
    print(f"Final Portfolio: ${stats['final_portfolio']:,.2f}")
    print(f"Total Return: {stats['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {stats['sharpe']:.3f}")
    print(f"Max Drawdown: {stats['max_drawdown_pct']:.2f}%")

    print("\n===== SAVED FILES =====")
    print(f"Daily history:        {history_path}")
    print(f"Monthly summary:      {monthly_path}")
    print(f"Trade orders:         {trades_path}")
    print(f"Selections:           {selections_path}")
    print(f"Scores:               {scores_path}")
    print(f"Screened additions:   {screen_path}")

    print("\n===== RECENT MONTHLY REBALANCES =====")
    if not monthly_df.empty:
        print(
            monthly_df.tail(12)[[
                "signal_date",
                "execution_date",
                "selected_top3",
                "screened_additions",
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
    if not trades_df.empty:
        print(
            trades_df.tail(30)[[
                "execution_date",
                "ticker",
                "is_screened_addition",
                "selected_top3",
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
    run_hybrid_trade_log()