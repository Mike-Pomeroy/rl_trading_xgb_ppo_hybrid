"""
Rank Decay Exit Research Runner

Purpose:
- Compare the current monthly XGBoost Top-3 strategy against daily rank-monitoring rules.
- Test multiple target horizons: 10, 15, 21, and 30 trading days.
- Monthly baseline: select Top 3 once per month and hold until next monthly rebalance.
- Rank-decay variant:
    1. Start with monthly Top 3.
    2. Each trading day, rank the universe using the latest walk-forward model.
    3. If a held ticker is outside the monitor band for N consecutive days,
       replace it with the highest-ranked non-held ticker.
    4. Limit replacements to MAX_REPLACEMENTS_PER_DAY.

This is research only.
It does not connect to Alpaca.
It does not submit orders.

Run:
    python -u rank_decay_exit_runner.py
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import xgb_topk_strategy as strat


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = Path("rank_decay_results")
OUTPUT_DIR.mkdir(exist_ok=True)

INITIAL_AMOUNT = 3000
TOP_K = 3

# Test multiple forward-return prediction horizons.
TARGET_HORIZONS = [15, 21, 30]

CASH_BUFFER = 0.20
TRANSACTION_COST = 0.005

TRAIN_START_DATE = None
TRAIN_END_DATE = "2022-01-01"

MIN_DOLLARS_PER_POSITION = 500.0
ALLOW_FRACTIONAL_SHARES = True

SPY_TREND_FILTER = False
WALK_FORWARD = True

# Current fixed 20, with SPY used for data/features but excluded from trading.
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

YEAR_RANGES: List[Tuple[str, str]] = [
    ("2021-01-01", "2022-01-01"),
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2025-01-01"),
    ("2025-01-01", "2026-05-01"),
    ("2021-01-01", "2026-05-01"),
]

# Experiments to compare.
# monitor_rank = acceptable band. Example: Top 7 or Top 10.
# confirm_days = consecutive days outside that band before replacement.
#
# Top-5 was noisy in the first test, so this set focuses on more forgiving rules.
EXPERIMENTS = [
    {"name": "rank_decay_top7_3days", "monitor_rank": 7, "confirm_days": 3},
    {"name": "rank_decay_top10_3days", "monitor_rank": 10, "confirm_days": 3},
    {"name": "rank_decay_top10_5days", "monitor_rank": 10, "confirm_days": 5},
    {"name": "rank_decay_top12_3days", "monitor_rank": 12, "confirm_days": 3},
    {"name": "rank_decay_top12_5days", "monitor_rank": 12, "confirm_days": 5},
]

MAX_REPLACEMENTS_PER_DAY = 1


# ============================================================
# HELPERS
# ============================================================

def clean_stats(stats: Dict[str, object]) -> Dict[str, float]:
    return {
        "final_portfolio": float(stats["final_portfolio"]),
        "total_return_pct": float(stats["total_return_pct"]),
        "sharpe": float(stats["sharpe"]),
        "max_drawdown_pct": float(stats["max_drawdown_pct"]),
    }

def ensure_future_return_columns(
    df: pd.DataFrame,
    horizons: List[int],
) -> pd.DataFrame:
    """
    Ensure full_df has future_return_X columns for every target horizon.

    future_return_X = close price X trading rows ahead for the same ticker
                      divided by current close, minus 1.
    """
    out = df.copy()
    out = strat.normalize_date_column(out, "date")
    out = out.sort_values(["tic", "date"]).reset_index(drop=True)

    if "tic" not in out.columns or "close" not in out.columns:
        raise RuntimeError("Dataframe must contain 'tic' and 'close' columns.")

    for horizon in horizons:
        col = f"future_return_{horizon}"

        if col in out.columns:
            continue

        out[col] = (
            out.groupby("tic")["close"]
            .shift(-horizon)
            .div(out["close"])
            .sub(1.0)
        )

        print(f"Added missing target column: {col}")

    return out


def make_equal_weight_targets(
    selected: List[str],
    tickers: List[str],
    px: np.ndarray,
    cash: float,
    shares: np.ndarray,
) -> np.ndarray:
    current_values = strat.current_position_values(shares, px)
    portfolio_value = cash + float(np.sum(current_values))

    return strat.make_target_weights(
        selected=selected,
        tickers=tickers,
        px=px,
        invested_fraction=max(0.0, 1.0 - CASH_BUFFER),
        portfolio_value=portfolio_value,
        min_dollars_per_position=MIN_DOLLARS_PER_POSITION,
        allow_fractional_shares=ALLOW_FRACTIONAL_SHARES,
    )


def rank_signal_day(
    model,
    signal_day: pd.DataFrame,
    features: List[str],
    target_col: str,
) -> pd.DataFrame:
    ranked = signal_day.dropna(subset=features).copy()

    if ranked.empty:
        return ranked

    ranked["score"] = model.predict(ranked[features])
    ranked = ranked.sort_values("score", ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)

    if target_col not in ranked.columns:
        ranked[target_col] = np.nan

    return ranked


def build_model_for_date(
    model_df: pd.DataFrame,
    signal_date: pd.Timestamp,
    features: List[str],
    target_col: str,
):
    train_mask = model_df["date"] < signal_date

    if TRAIN_START_DATE is not None:
        train_mask &= model_df["date"] >= pd.Timestamp(TRAIN_START_DATE)

    return strat.train_model(model_df.loc[train_mask], features, target_col)


# ============================================================
# BASELINE MONTHLY STRATEGY
# ============================================================

def run_monthly_baseline(
    full_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    target_horizon: int,
):
    return strat.run_xgb_topk_monthly_strategy(
        full_df=full_df,
        tickers=TRADE_LIST,
        features=strat.FEATURES,
        initial_amount=INITIAL_AMOUNT,
        test_start_date=start_date,
        test_end_date=end_date,
        train_end_date=TRAIN_END_DATE,
        train_start_date=TRAIN_START_DATE,
        top_k=TOP_K,
        transaction_cost=TRANSACTION_COST,
        target_horizon=target_horizon,
        cash_buffer=CASH_BUFFER,
        spy_trend_filter=SPY_TREND_FILTER,
        walk_forward=WALK_FORWARD,
    )


# ============================================================
# RANK DECAY STRATEGY
# ============================================================

def run_rank_decay_strategy(
    full_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    monitor_rank: int,
    confirm_days: int,
    target_horizon: int,
):
    """
    Simulate:
    - Monthly Top-3 entry.
    - Daily rank monitoring.
    - Exit/replace if held ticker is outside monitor_rank for confirm_days.
    """
    target_col = f"future_return_{target_horizon}"

    df = full_df.copy()
    df = strat.normalize_date_column(df, "date")

    model_df = df[df["tic"].isin(TRADE_LIST)].copy()

    test_mask = (
        (model_df["date"] >= pd.Timestamp(start_date))
        & (model_df["date"] < pd.Timestamp(end_date))
    )

    test_df = model_df.loc[test_mask].dropna(subset=strat.FEATURES + ["close"]).copy()

    if test_df.empty:
        raise ValueError(f"No test rows for {start_date} to {end_date}")

    price_matrix = strat.build_price_matrix(test_df, TRADE_LIST)
    dates = list(price_matrix.index.unique())

    monthly_signal_dates = set(strat.get_monthly_signal_dates(dates))

    cash = float(INITIAL_AMOUNT)
    shares = np.zeros(len(TRADE_LIST), dtype=np.float64)

    portfolio_values = []
    history_rows = []
    trade_rows = []
    rank_rows = []

    outside_counts: Dict[str, int] = {ticker: 0 for ticker in TRADE_LIST}
    current_selected: List[str] = []

    for dt in dates:
        px = price_matrix.loc[dt].astype(float).values

        # Rank the universe every trading day using only prior training data.
        signal_day = (
            test_df[test_df["date"] == dt]
            .dropna(subset=strat.FEATURES)
            .copy()
        )

        ranked = pd.DataFrame()

        if not signal_day.empty:
            model = build_model_for_date(
                model_df=model_df,
                signal_date=dt,
                features=strat.FEATURES,
                target_col=target_col,
            )

            if model is not None:
                ranked = rank_signal_day(
                    model=model,
                    signal_day=signal_day,
                    features=strat.FEATURES,
                    target_col=target_col,
                )

        if not ranked.empty:
            rank_map = {
                str(row["tic"]): int(row["rank"])
                for _, row in ranked.iterrows()
            }

            for _, row in ranked.iterrows():
                rank_rows.append({
                    "date": dt,
                    "tic": row["tic"],
                    "rank": int(row["rank"]),
                    "score": float(row["score"]),
                    "target_horizon": target_horizon,
                    target_col: row.get(target_col, np.nan),
                })
        else:
            rank_map = {}

        # Monthly reset: set portfolio to that day's Top 3.
        if dt in monthly_signal_dates and not ranked.empty:
            current_selected = ranked.head(TOP_K)["tic"].astype(str).tolist()
            target_weights = make_equal_weight_targets(
                selected=current_selected,
                tickers=TRADE_LIST,
                px=px,
                cash=cash,
                shares=shares,
            )

            cash, shares, cost = strat.rebalance_to_weights(
                cash=cash,
                shares=shares,
                px=px,
                target_weights=target_weights,
                transaction_cost=TRANSACTION_COST,
            )

            outside_counts = {ticker: 0 for ticker in TRADE_LIST}

            trade_rows.append({
                "date": dt,
                "event": "MONTHLY_RESET",
                "selected": ",".join(current_selected),
                "sold": "",
                "bought": ",".join(current_selected),
                "monitor_rank": monitor_rank,
                "confirm_days": confirm_days,
                "target_horizon": target_horizon,
                "transaction_cost_paid": cost,
            })

        # Daily rank-decay monitor.
        elif current_selected and not ranked.empty:
            replacements_done = 0
            sold_symbols = []
            bought_symbols = []

            held_before = list(current_selected)

            for held in held_before:
                rank = rank_map.get(held, 999999)

                if rank > monitor_rank:
                    outside_counts[held] = outside_counts.get(held, 0) + 1
                else:
                    outside_counts[held] = 0

                if (
                    outside_counts[held] >= confirm_days
                    and replacements_done < MAX_REPLACEMENTS_PER_DAY
                ):
                    candidates = [
                        ticker
                        for ticker in ranked["tic"].astype(str).tolist()
                        if ticker not in current_selected
                    ]

                    if not candidates:
                        continue

                    replacement = candidates[0]

                    current_selected.remove(held)
                    current_selected.append(replacement)

                    outside_counts[held] = 0
                    outside_counts[replacement] = 0

                    sold_symbols.append(held)
                    bought_symbols.append(replacement)

                    replacements_done += 1

            if replacements_done > 0:
                target_weights = make_equal_weight_targets(
                    selected=current_selected,
                    tickers=TRADE_LIST,
                    px=px,
                    cash=cash,
                    shares=shares,
                )

                cash, shares, cost = strat.rebalance_to_weights(
                    cash=cash,
                    shares=shares,
                    px=px,
                    target_weights=target_weights,
                    transaction_cost=TRANSACTION_COST,
                )

                trade_rows.append({
                    "date": dt,
                    "event": "RANK_DECAY_REPLACE",
                    "selected": ",".join(current_selected),
                    "sold": ",".join(sold_symbols),
                    "bought": ",".join(bought_symbols),
                    "monitor_rank": monitor_rank,
                    "confirm_days": confirm_days,
                    "target_horizon": target_horizon,
                    "transaction_cost_paid": cost,
                })
            else:
                cost = 0.0

        else:
            cost = 0.0

        values = strat.current_position_values(shares, px)
        portfolio_value = cash + float(np.sum(values))

        portfolio_values.append(portfolio_value)

        history_rows.append({
            "date": dt,
            "portfolio_value": portfolio_value,
            "cash": cash,
            "selected": ",".join(current_selected),
            "target_horizon": target_horizon,
            "transaction_cost_paid": cost,
        })

    return strat.StrategyResult(
        stats=strat.compute_stats(portfolio_values, INITIAL_AMOUNT),
        history=pd.DataFrame(history_rows),
        selections=pd.DataFrame(trade_rows),
        scored=pd.DataFrame(rank_rows),
    )


# ============================================================
# RUNNER
# ============================================================

def run_one_range(
    full_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    target_horizon: int,
):
    print("\n" + "=" * 100)
    print(
        f"Running rank-decay research: {start_date} to {end_date} "
        f"| target_horizon={target_horizon}"
    )
    print("=" * 100)

    strat.DATA_LIST = DATA_LIST
    strat.TRADE_LIST = TRADE_LIST
    strat.INITIAL_AMOUNT = INITIAL_AMOUNT
    strat.TOP_K = TOP_K
    strat.TARGET_HORIZON = target_horizon
    strat.CASH_BUFFER = CASH_BUFFER
    strat.TRANSACTION_COST = TRANSACTION_COST
    strat.TRAIN_START_DATE = TRAIN_START_DATE
    strat.TRAIN_END_DATE = TRAIN_END_DATE
    strat.MIN_DOLLARS_PER_POSITION = MIN_DOLLARS_PER_POSITION
    strat.ALLOW_FRACTIONAL_SHARES = ALLOW_FRACTIONAL_SHARES
    strat.SPY_TREND_FILTER = SPY_TREND_FILTER
    strat.WALK_FORWARD = WALK_FORWARD

    results = []

    baseline = run_monthly_baseline(
        full_df=full_df,
        start_date=start_date,
        end_date=end_date,
        target_horizon=target_horizon,
    )

    results.append({
        "start_date": start_date,
        "end_date": end_date,
        "target_horizon": target_horizon,
        "strategy": "monthly_top3_baseline",
        "monitor_rank": "",
        "confirm_days": "",
        "num_replacement_events": 0,
        "num_trade_events": len(baseline.selections),
        **clean_stats(baseline.stats),
    })

    all_history_frames = []
    all_trade_frames = []

    hist = baseline.history.copy()
    hist["strategy"] = "monthly_top3_baseline"
    hist["start_date"] = start_date
    hist["end_date"] = end_date
    hist["target_horizon"] = target_horizon
    all_history_frames.append(hist)

    trades = baseline.selections.copy()
    trades["strategy"] = "monthly_top3_baseline"
    trades["start_date"] = start_date
    trades["end_date"] = end_date
    trades["target_horizon"] = target_horizon
    all_trade_frames.append(trades)

    print(
        f"monthly_top3_baseline: "
        f"Return={baseline.stats['total_return_pct']:.2f}% "
        f"Sharpe={baseline.stats['sharpe']:.3f} "
        f"MaxDD={baseline.stats['max_drawdown_pct']:.2f}%"
    )

    for exp in EXPERIMENTS:
        name = exp["name"]
        monitor_rank = int(exp["monitor_rank"])
        confirm_days = int(exp["confirm_days"])

        result = run_rank_decay_strategy(
            full_df=full_df,
            start_date=start_date,
            end_date=end_date,
            monitor_rank=monitor_rank,
            confirm_days=confirm_days,
            target_horizon=target_horizon,
        )

        selections = result.selections.copy()

        replacement_events = 0
        if not selections.empty and "event" in selections.columns:
            replacement_events = int((selections["event"] == "RANK_DECAY_REPLACE").sum())

        results.append({
            "start_date": start_date,
            "end_date": end_date,
            "target_horizon": target_horizon,
            "strategy": name,
            "monitor_rank": monitor_rank,
            "confirm_days": confirm_days,
            "num_replacement_events": replacement_events,
            "num_trade_events": len(selections),
            **clean_stats(result.stats),
        })

        hist = result.history.copy()
        hist["strategy"] = name
        hist["start_date"] = start_date
        hist["end_date"] = end_date
        hist["target_horizon"] = target_horizon
        all_history_frames.append(hist)

        trades = selections.copy()
        trades["strategy"] = name
        trades["start_date"] = start_date
        trades["end_date"] = end_date
        trades["target_horizon"] = target_horizon
        all_trade_frames.append(trades)

        print(
            f"{name}: "
            f"Return={result.stats['total_return_pct']:.2f}% "
            f"Sharpe={result.stats['sharpe']:.3f} "
            f"MaxDD={result.stats['max_drawdown_pct']:.2f}% "
            f"Replacements={replacement_events}"
        )

    summary_df = pd.DataFrame(results)
    history_df = pd.concat(all_history_frames, ignore_index=True)
    trades_df = pd.concat(all_trade_frames, ignore_index=True)

    return summary_df, history_df, trades_df


def main() -> None:

    print("Preparing full data once...")
    strat.DATA_LIST = DATA_LIST
    strat.TRADE_LIST = TRADE_LIST
    full_df = strat.prepare_full_df()
    full_df = strat.normalize_date_column(full_df, "date")
    full_df = ensure_future_return_columns(full_df, TARGET_HORIZONS)    

   
   
    all_summaries = []
    all_histories = []
    all_trades = []

    for target_horizon in TARGET_HORIZONS:
        print("\n" + "#" * 100)
        print(f"TARGET HORIZON: {target_horizon}")
        print("#" * 100)

        for start_date, end_date in YEAR_RANGES:
            summary_df, history_df, trades_df = run_one_range(
                full_df=full_df,
                start_date=start_date,
                end_date=end_date,
                target_horizon=target_horizon,
            )

            all_summaries.append(summary_df)
            all_histories.append(history_df)
            all_trades.append(trades_df)

    summary = pd.concat(all_summaries, ignore_index=True)
    history = pd.concat(all_histories, ignore_index=True)
    trades = pd.concat(all_trades, ignore_index=True)

    summary_path = OUTPUT_DIR / "rank_decay_summary.csv"
    history_path = OUTPUT_DIR / "rank_decay_history.csv"
    trades_path = OUTPUT_DIR / "rank_decay_trades.csv"

    summary.to_csv(summary_path, index=False)
    history.to_csv(history_path, index=False)
    trades.to_csv(trades_path, index=False)

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Saved summary: {summary_path}")
    print(f"Saved history: {history_path}")
    print(f"Saved trades:  {trades_path}")

    print("\nSummary sorted by full-period Sharpe:")
    display = summary.copy()
    full_period = display[display["start_date"] == "2021-01-01"].copy()

    cols = [
        "target_horizon",
        "strategy",
        "start_date",
        "end_date",
        "total_return_pct",
        "sharpe",
        "max_drawdown_pct",
        "num_replacement_events",
        "num_trade_events",
    ]

    if full_period.empty:
        print(display[cols].to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    else:
        print(
            full_period[cols]
            .sort_values(
                ["sharpe", "total_return_pct"],
                ascending=[False, False],
            )
            .to_string(index=False, float_format=lambda x: f"{x:,.4f}")
        )


if __name__ == "__main__":
    main()