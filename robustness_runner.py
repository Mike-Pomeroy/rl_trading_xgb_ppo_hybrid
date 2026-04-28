"""
Run XGBoost top-k strategy robustness tests across:
- 10-ticker universe
- 15-ticker universe
- multiple year ranges

This script imports your existing xgb_topk_strategy.py and saves:
- robustness_summary.csv
- robustness_selections.csv
- robustness_annual_returns.csv

Run:
    python robustness_runner.py
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import xgb_topk_strategy as strat


# =========================
# TEST CONFIG
# =========================

OUTPUT_DIR = Path("robustness_results")
OUTPUT_DIR.mkdir(exist_ok=True)

INITIAL_AMOUNT = 3000
TOP_K = 3
TARGET_HORIZON = 30
CASH_BUFFER = 0.2
TRANSACTION_COST = 0.005
SPY_TREND_FILTER = False
WALK_FORWARD = True
MOMENTUM_LOOKBACK = 126

MIN_DOLLARS_PER_POSITION = 500.0
ALLOW_FRACTIONAL_SHARES = True
DATA_ADJUSTMENT = "split"


TRAIN_START_DATE = None
TRAIN_END_DATE = "2022-01-01"

YEAR_RANGES: List[Tuple[str, str]] = [
    ("2021-01-01", "2022-01-01"),
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2025-01-01"),
    ("2025-01-01", "2026-04-27"),
    ("2022-01-01", "2024-01-01"),
    ("2021-01-01", "2025-01-01"),
]

UNIVERSES: Dict[str, Dict[str, List[str]]] = {
    "15_ticker_no_spy_trade": {
        "data_list": [
            "AAPL", "MSFT", "SPY", "GOOGL", "AMZN",
            "META", "TSLA", "NVDA", "JPM", "JNJ",
            "AVGO", "LLY", "UNH", "COST", "V",
        ],
        "trade_list": [
            "AAPL", "MSFT", "GOOGL", "AMZN",
            "META", "TSLA", "NVDA", "JPM", "JNJ",
            "AVGO", "LLY", "UNH", "COST", "V",
        ],
    },

    "20_ticker_no_spy_trade": {
        "data_list": [
            "AAPL", "MSFT", "SPY", "GOOGL", "AMZN",
            "META", "TSLA", "NVDA", "JPM", "JNJ",
            "AVGO", "LLY", "UNH", "COST", "V",
            "MA", "HD", "PG", "XOM", "AMD",
        ],
        "trade_list": [
            "AAPL", "MSFT", "GOOGL", "AMZN",
            "META", "TSLA", "NVDA", "JPM", "JNJ",
            "AVGO", "LLY", "UNH", "COST", "V",
            "MA", "HD", "PG", "XOM", "AMD",
        ],
    },


    "screened_20_no_spy_trade": {
        "data_list": [
            "SPY",
            "GOOGL", "JNJ", "CAT", "MU", "XOM",
            "WMT", "LRCX", "NVDA", "CVX", "AMAT",
            "AMD", "CSCO", "ADI", "GS", "C",
            "AMZN", "MRK", "AVGO", "AAPL", "KLAC",
        ],
        "trade_list": [
            "GOOGL", "JNJ", "CAT", "MU", "XOM",
            "WMT", "LRCX", "NVDA", "CVX", "AMAT",
            "AMD", "CSCO", "ADI", "GS", "C",
            "AMZN", "MRK", "AVGO", "AAPL", "KLAC",
        ],
    },

}


# =========================
# HELPER FUNCTIONS
# =========================

def clean_stats(stats: Dict[str, object]) -> Dict[str, float]:
    """Remove large array fields from stats so results are CSV-friendly."""
    return {
        "final_portfolio": float(stats["final_portfolio"]),
        "total_return_pct": float(stats["total_return_pct"]),
        "sharpe": float(stats["sharpe"]),
        "max_drawdown_pct": float(stats["max_drawdown_pct"]),
    }


def ranking_metrics(scored: pd.DataFrame, target_col: str, top_k: int) -> Dict[str, float]:
    """Return ranking diagnostics without printing."""
    if scored is None or scored.empty or target_col not in scored.columns:
        return {
            "top_count": 0,
            "not_top_count": 0,
            "top_mean_future_return": np.nan,
            "not_top_mean_future_return": np.nan,
            "top_minus_rest_mean": np.nan,
            "monthly_hit_rate": np.nan,
            "monthly_spread_mean": np.nan,
            "monthly_spread_median": np.nan,
            "monthly_spread_min": np.nan,
            "monthly_spread_max": np.nan,
        }

    clean = scored.dropna(subset=["rank", target_col]).copy()

    if clean.empty:
        return {
            "top_count": 0,
            "not_top_count": 0,
            "top_mean_future_return": np.nan,
            "not_top_mean_future_return": np.nan,
            "top_minus_rest_mean": np.nan,
            "monthly_hit_rate": np.nan,
            "monthly_spread_mean": np.nan,
            "monthly_spread_median": np.nan,
            "monthly_spread_min": np.nan,
            "monthly_spread_max": np.nan,
        }

    top = clean[clean["rank"] <= top_k]
    rest = clean[clean["rank"] > top_k]

    top_mean = top[target_col].mean()
    rest_mean = rest[target_col].mean()

    monthly_spreads = []

    for _, month_df in clean.groupby("signal_date"):
        month_top = month_df[month_df["rank"] <= top_k][target_col]
        month_rest = month_df[month_df["rank"] > top_k][target_col]

        if len(month_top) > 0 and len(month_rest) > 0:
            monthly_spreads.append(month_top.mean() - month_rest.mean())

    monthly_spreads = pd.Series(monthly_spreads, dtype=float)

    if monthly_spreads.empty:
        hit_rate = np.nan
        spread_mean = np.nan
        spread_median = np.nan
        spread_min = np.nan
        spread_max = np.nan
    else:
        hit_rate = float((monthly_spreads > 0).mean())
        spread_mean = float(monthly_spreads.mean())
        spread_median = float(monthly_spreads.median())
        spread_min = float(monthly_spreads.min())
        spread_max = float(monthly_spreads.max())

    return {
        "top_count": int(len(top)),
        "not_top_count": int(len(rest)),
        "top_mean_future_return": float(top_mean) if pd.notna(top_mean) else np.nan,
        "not_top_mean_future_return": float(rest_mean) if pd.notna(rest_mean) else np.nan,
        "top_minus_rest_mean": (
            float(top_mean - rest_mean)
            if pd.notna(top_mean) and pd.notna(rest_mean)
            else np.nan
        ),
        "monthly_hit_rate": hit_rate,
        "monthly_spread_mean": spread_mean,
        "monthly_spread_median": spread_median,
        "monthly_spread_min": spread_min,
        "monthly_spread_max": spread_max,
    }


def annual_returns(history: pd.DataFrame) -> pd.DataFrame:
    hist = history.copy()
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("date")

    annual = (
        hist.groupby(hist["date"].dt.year)["portfolio_value"]
        .agg(["first", "last"])
        .reset_index()
        .rename(columns={"date": "year"})
    )

    annual["annual_return_pct"] = (annual["last"] / annual["first"] - 1.0) * 100.0

    return annual[["year", "annual_return_pct"]]


def run_one_test(
    full_df: pd.DataFrame,
    universe_name: str,
    data_list: List[str],
    trade_list: List[str],
    start_date: str,
    end_date: str,
) -> Tuple[List[Dict[str, object]], pd.DataFrame, List[Dict[str, object]]]:

    print("\n" + "=" * 90)
    print(f"Running {universe_name}: {start_date} to {end_date}")
    print("=" * 90)

    full_df = full_df[full_df["tic"].isin(data_list)].copy()
    full_df = strat.normalize_date_column(full_df, "date")
    
    print("\nDEBUG future return columns:")
    print([c for c in full_df.columns if c.startswith("future_return_")])


    test_df = full_df[
        (full_df["date"] >= pd.Timestamp(start_date)) &
        (full_df["date"] < pd.Timestamp(end_date))
    ].dropna(subset=strat.FEATURES + ["close"]).copy()

    trade_test_df = test_df[test_df["tic"].isin(trade_list)].copy()

    if test_df.empty:
        raise ValueError(f"No test rows for {universe_name} {start_date} to {end_date}")

    if trade_test_df.empty:
        raise ValueError(f"No trade rows for {universe_name} {start_date} to {end_date}")

    xgb_result = strat.run_xgb_topk_monthly_strategy(
        full_df=full_df,
        tickers=trade_list,
        features=strat.FEATURES,
        initial_amount=INITIAL_AMOUNT,
        test_start_date=start_date,
        test_end_date=end_date,
        train_end_date=TRAIN_END_DATE,
        train_start_date=TRAIN_START_DATE,
        top_k=TOP_K,
        transaction_cost=TRANSACTION_COST,
        target_horizon=TARGET_HORIZON,
        cash_buffer=CASH_BUFFER,
        spy_trend_filter=SPY_TREND_FILTER,
        walk_forward=WALK_FORWARD,
    )

    results = {
        f"XGBoost Top-{TOP_K}": xgb_result,
        "Buy & Hold Equal Weight": strat.run_buy_hold_equal_weight_benchmark(
            test_df=trade_test_df,
            tickers=trade_list,
            initial_amount=INITIAL_AMOUNT,
        ),
        "Monthly Equal Weight": strat.run_monthly_equal_weight_benchmark(
            test_df=trade_test_df,
            tickers=trade_list,
            initial_amount=INITIAL_AMOUNT,
            transaction_cost=TRANSACTION_COST,
            cash_buffer=0.0,
        ),
        f"Momentum Top-{TOP_K}": strat.run_momentum_topk_benchmark(
            test_df=trade_test_df,
            tickers=trade_list,
            initial_amount=INITIAL_AMOUNT,
            top_k=TOP_K,
            lookback=MOMENTUM_LOOKBACK,
            transaction_cost=TRANSACTION_COST,
            cash_buffer=CASH_BUFFER,
        ),
        "SPY Buy & Hold": strat.run_spy_benchmark(
            test_df=test_df,
            initial_amount=INITIAL_AMOUNT,
        ),
    }

    summary_rows = []
    annual_rows = []

    rank_stats = ranking_metrics(
        scored=xgb_result.scored,
        target_col=f"future_return_{TARGET_HORIZON}",
        top_k=TOP_K,
    )

    for strategy_name, result in results.items():
        row = {
            "universe": universe_name,
            "start_date": start_date,
            "end_date": end_date,
            "strategy": strategy_name,
            "num_data_tickers": len(data_list),
            "num_trade_tickers": len(trade_list),
            "top_k": TOP_K,
            "target_horizon": TARGET_HORIZON,
            "cash_buffer": CASH_BUFFER,
            "data_adjustment": DATA_ADJUSTMENT,
            "transaction_cost": TRANSACTION_COST,
            **clean_stats(result.stats),
        }

        if strategy_name == f"XGBoost Top-{TOP_K}":
            row.update(rank_stats)

        summary_rows.append(row)

        annual = annual_returns(result.history)

        for _, annual_row in annual.iterrows():
            annual_rows.append({
                "universe": universe_name,
                "start_date": start_date,
                "end_date": end_date,
                "strategy": strategy_name,
                "year": int(annual_row["year"]),
                "annual_return_pct": float(annual_row["annual_return_pct"]),
            })

    selections = xgb_result.selections.copy()

    if not selections.empty:
        selections["universe"] = universe_name
        selections["start_date"] = start_date
        selections["end_date"] = end_date

    print("\nSummary:")
    for row in summary_rows:
        print(
            f"{row['strategy']:<28} "
            f"Return={row['total_return_pct']:>8.2f}% "
            f"Sharpe={row['sharpe']:>6.3f} "
            f"MaxDD={row['max_drawdown_pct']:>7.2f}%"
        )

    print("\nRecent XGBoost selections:")
    if selections.empty:
        print("No selections.")
    else:
        print(selections.tail(5).to_string(index=False))

    return summary_rows, selections, annual_rows


def main() -> None:
    # Make sure the imported strategy module uses the same config as this runner.
    strat.INITIAL_AMOUNT = INITIAL_AMOUNT
    strat.TOP_K = TOP_K
    strat.TARGET_HORIZON = TARGET_HORIZON
    strat.CASH_BUFFER = CASH_BUFFER
    strat.TRANSACTION_COST = TRANSACTION_COST
    strat.SPY_TREND_FILTER = SPY_TREND_FILTER
    strat.WALK_FORWARD = WALK_FORWARD
    strat.MOMENTUM_LOOKBACK = MOMENTUM_LOOKBACK
    strat.TRAIN_START_DATE = TRAIN_START_DATE
    strat.TRAIN_END_DATE = TRAIN_END_DATE

    strat.MIN_DOLLARS_PER_POSITION = MIN_DOLLARS_PER_POSITION
    strat.ALLOW_FRACTIONAL_SHARES = ALLOW_FRACTIONAL_SHARES
    strat.DATA_ADJUSTMENT = DATA_ADJUSTMENT

    # Fetch/build data once using the largest universe, then filter down per test.
    largest_data_list = sorted(
        set(
            ticker
            for universe in UNIVERSES.values()
            for ticker in universe["data_list"]
        )
    )

    print("Preparing data once for largest universe:")
    print(largest_data_list)

    strat.DATA_LIST = largest_data_list

    full_df = strat.prepare_full_df()
    full_df = strat.normalize_date_column(full_df, "date")

    all_summary_rows = []
    all_selection_frames = []
    all_annual_rows = []

    for universe_name, universe in UNIVERSES.items():
        data_list = universe["data_list"]
        trade_list = universe["trade_list"]

        for start_date, end_date in YEAR_RANGES:
            summary_rows, selections, annual_rows = run_one_test(
                full_df=full_df,
                universe_name=universe_name,
                data_list=data_list,
                trade_list=trade_list,
                start_date=start_date,
                end_date=end_date,
            )

            all_summary_rows.extend(summary_rows)
            all_annual_rows.extend(annual_rows)

            if not selections.empty:
                all_selection_frames.append(selections)

    summary_df = pd.DataFrame(all_summary_rows)
    annual_df = pd.DataFrame(all_annual_rows)

    if all_selection_frames:
        selections_df = pd.concat(all_selection_frames, ignore_index=True)
    else:
        selections_df = pd.DataFrame()

    summary_path = OUTPUT_DIR / "robustness_summary.csv"
    selections_path = OUTPUT_DIR / "robustness_selections.csv"
    annual_path = OUTPUT_DIR / "robustness_annual_returns.csv"

    summary_df.to_csv(summary_path, index=False)
    selections_df.to_csv(selections_path, index=False)
    annual_df.to_csv(annual_path, index=False)

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)
    print(f"Saved summary to:    {summary_path}")
    print(f"Saved selections to: {selections_path}")
    print(f"Saved annuals to:    {annual_path}")

    print("\nXGBoost-only summary:")
    xgb_only = summary_df[summary_df["strategy"] == f"XGBoost Top-{TOP_K}"].copy()

    cols = [
        "universe",
        "start_date",
        "end_date",
        "total_return_pct",
        "sharpe",
        "max_drawdown_pct",
        "monthly_hit_rate",
        "top_minus_rest_mean",
    ]

    print(
        xgb_only[cols]
        .sort_values(["universe", "start_date", "end_date"])
        .to_string(index=False, float_format=lambda x: f"{x:,.4f}")
    )


if __name__ == "__main__":
    main()