"""
Hybrid universe runner.

Purpose:
- Keep the proven fixed current 20 universe.
- Each month, use only past data to screen extra candidates.
- Test whether adding screened names improves performance without replacing the core universe.

Tests:
- Fixed 20 only
- Fixed 20 + screened top 5
- Fixed 20 + screened top 10
- Fixed 20 + screened top 20

This does NOT trade.
This does NOT submit Alpaca orders.

Run:
    python -u hybrid_universe_runner.py
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import xgb_topk_strategy as strat


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = Path("hybrid_universe_results")
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

HYBRID_ADD_COUNTS = [0, 5, 10, 20]

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
) -> tuple[List[str], pd.DataFrame]:
    if add_count <= 0:
        return [], pd.DataFrame()

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
# PERFORMANCE HELPERS
# ============================================================

def clean_stats(stats: Dict[str, object]) -> Dict[str, float]:
    return {
        "final_portfolio": float(stats["final_portfolio"]),
        "total_return_pct": float(stats["total_return_pct"]),
        "sharpe": float(stats["sharpe"]),
        "max_drawdown_pct": float(stats["max_drawdown_pct"]),
    }


def ranking_metrics(scored: pd.DataFrame, target_col: str, top_k: int) -> Dict[str, float]:
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


# ============================================================
# HYBRID STRATEGY
# ============================================================

def run_hybrid_strategy(
    full_df: pd.DataFrame,
    add_count: int,
    start_date: str,
    end_date: str,
) -> Dict[str, object]:
    target_col = f"future_return_{TARGET_HORIZON}"

    df = full_df.copy()
    df = strat.normalize_date_column(df, "date")

    all_trade_tickers = sorted(set(CANDIDATE_UNIVERSE) - {"SPY"})
    model_df = df[df["tic"].isin(all_trade_tickers)].copy()

    test_mask = (
        (model_df["date"] >= pd.Timestamp(start_date))
        & (model_df["date"] < pd.Timestamp(end_date))
    )

    test_df = model_df.loc[test_mask].dropna(
        subset=strat.FEATURES + ["close"]
    ).copy()

    if test_df.empty:
        raise ValueError(f"No test rows for hybrid universe {start_date} to {end_date}")

    price_matrix = strat.build_price_matrix(test_df, all_trade_tickers)
    dates = list(price_matrix.index.unique())
    signal_dates = strat.get_monthly_signal_dates(dates)

    cash = float(INITIAL_AMOUNT)
    shares = np.zeros(len(all_trade_tickers), dtype=np.float64)

    history_rows = []
    selection_rows = []
    scored_rows = []
    screen_rows = []

    pending_weights_by_execution_date: Dict[pd.Timestamp, np.ndarray] = {}

    for signal_date in signal_dates:
        execution_date = strat.next_trading_date(dates, signal_date)

        if execution_date is None:
            continue

        additions, screened = select_screened_additions_asof(
            full_df=df,
            signal_date=signal_date,
            add_count=add_count,
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
                    "add_count": add_count,
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
            print(
                f"Skipping {signal_date.date()} add_count={add_count} - "
                f"not enough training rows."
            )
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
            scored_rows.append({
                "signal_date": signal_date,
                "ticker": row["tic"],
                "score": row["score"],
                "rank": row["rank"],
                target_col: row.get(target_col, np.nan),
                "selected_top3": row["tic"] in selected_top3,
                "hybrid_universe": ",".join(hybrid_universe),
                "add_count": add_count,
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

        pending_weights_by_execution_date[execution_date] = target_weights

        selection_rows.append({
            "signal_date": signal_date,
            "execution_date": execution_date,
            "add_count": add_count,
            "selected_top3": ",".join(selected_top3),
            "additions": ",".join(additions),
            "hybrid_universe": ",".join(hybrid_universe),
            "hybrid_universe_count": len(hybrid_universe),
            "target_weight_sum": float(np.sum(target_weights)),
        })

    for dt in dates:
        px = price_matrix.loc[dt].astype(float).values

        if dt in pending_weights_by_execution_date:
            cash, shares, cost = strat.rebalance_to_weights(
                cash=cash,
                shares=shares,
                px=px,
                target_weights=pending_weights_by_execution_date[dt],
                transaction_cost=TRANSACTION_COST,
            )
        else:
            cost = 0.0

        values = strat.current_position_values(shares, px)
        portfolio_value = cash + float(np.sum(values))

        history_rows.append({
            "date": dt,
            "portfolio_value": portfolio_value,
            "cash": cash,
            "invested_value": float(np.sum(values)),
            "transaction_cost_paid": cost,
            "num_positions": int(np.sum(shares > 1e-8)),
        })

    history_df = pd.DataFrame(history_rows)
    selections_df = pd.DataFrame(selection_rows)
    scored_df = pd.DataFrame(scored_rows)
    screen_df = pd.DataFrame(screen_rows)

    stats = strat.compute_stats(
        history_df["portfolio_value"].values,
        INITIAL_AMOUNT,
    )

    return {
        "stats": stats,
        "history": history_df,
        "selections": selections_df,
        "scored": scored_df,
        "screen": screen_df,
    }


# ============================================================
# MAIN
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


def main() -> None:
    configure_strategy_module()

    print("\n===== HYBRID UNIVERSE TEST =====")
    print(f"Candidate tickers including SPY: {len(strat.DATA_LIST)}")
    print(f"Candidate trade tickers:        {len(strat.TRADE_LIST)}")
    print(f"Hybrid add counts:              {HYBRID_ADD_COUNTS}")
    print(f"Trade Top K:                    {TOP_K}")
    print(f"Target horizon:                 {TARGET_HORIZON}")
    print(f"Data adjustment:                {DATA_ADJUSTMENT}")
    print(f"Cash buffer:                    {CASH_BUFFER}")
    print(f"Transaction cost:               {TRANSACTION_COST}")

    print("\nPreparing full data once...")
    full_df = strat.prepare_full_df()
    full_df = strat.normalize_date_column(full_df, "date")

    print("\nFuture return columns:")
    print([c for c in full_df.columns if c.startswith("future_return_")])

    summary_rows = []
    all_selections = []
    all_scored = []
    all_screen = []
    all_annual_rows = []

    for start_date, end_date in YEAR_RANGES:
        for add_count in HYBRID_ADD_COUNTS:
            print("\n" + "=" * 90)
            print(f"Running hybrid add_count={add_count}: {start_date} to {end_date}")
            print("=" * 90)

            result = run_hybrid_strategy(
                full_df=full_df,
                add_count=add_count,
                start_date=start_date,
                end_date=end_date,
            )

            stats = clean_stats(result["stats"])
            rank_stats = ranking_metrics(
                result["scored"],
                target_col=f"future_return_{TARGET_HORIZON}",
                top_k=TOP_K,
            )

            strategy_name = (
                "Fixed Current 20 Top-3"
                if add_count == 0
                else f"Fixed 20 + Screened {add_count} Top-3"
            )

            summary_rows.append({
                "strategy": strategy_name,
                "start_date": start_date,
                "end_date": end_date,
                "add_count": add_count,
                "candidate_count": len(strat.TRADE_LIST),
                "hybrid_base_count": len(CURRENT_20_TRADE_LIST),
                "top_k": TOP_K,
                "target_horizon": TARGET_HORIZON,
                "cash_buffer": CASH_BUFFER,
                "transaction_cost": TRANSACTION_COST,
                "data_adjustment": DATA_ADJUSTMENT,
                **stats,
                **rank_stats,
            })

            annual = annual_returns(result["history"])

            for _, row in annual.iterrows():
                all_annual_rows.append({
                    "strategy": strategy_name,
                    "start_date": start_date,
                    "end_date": end_date,
                    "add_count": add_count,
                    "year": int(row["year"]),
                    "annual_return_pct": float(row["annual_return_pct"]),
                })

            if not result["selections"].empty:
                s = result["selections"].copy()
                s["strategy"] = strategy_name
                s["start_date"] = start_date
                s["end_date"] = end_date
                all_selections.append(s)

            if not result["scored"].empty:
                sc = result["scored"].copy()
                sc["strategy"] = strategy_name
                sc["start_date"] = start_date
                sc["end_date"] = end_date
                all_scored.append(sc)

            if not result["screen"].empty:
                screen = result["screen"].copy()
                screen["strategy"] = strategy_name
                screen["start_date"] = start_date
                screen["end_date"] = end_date
                all_screen.append(screen)

            print("\nSummary:")
            print(
                f"{strategy_name:<32} "
                f"Return={stats['total_return_pct']:>8.2f}% "
                f"Sharpe={stats['sharpe']:>6.3f} "
                f"MaxDD={stats['max_drawdown_pct']:>7.2f}%"
            )

            print("\nRecent selections:")
            if result["selections"].empty:
                print("No selections.")
            else:
                print(
                    result["selections"]
                    .tail(5)[[
                        "signal_date",
                        "execution_date",
                        "selected_top3",
                        "additions",
                        "hybrid_universe_count",
                    ]]
                    .to_string(index=False)
                )

    summary_df = pd.DataFrame(summary_rows)
    annual_df = pd.DataFrame(all_annual_rows)

    selections_df = (
        pd.concat(all_selections, ignore_index=True)
        if all_selections
        else pd.DataFrame()
    )
    scored_df = (
        pd.concat(all_scored, ignore_index=True)
        if all_scored
        else pd.DataFrame()
    )
    screen_df = (
        pd.concat(all_screen, ignore_index=True)
        if all_screen
        else pd.DataFrame()
    )

    summary_path = OUTPUT_DIR / "hybrid_universe_summary.csv"
    annual_path = OUTPUT_DIR / "hybrid_universe_annual_returns.csv"
    selections_path = OUTPUT_DIR / "hybrid_universe_selections.csv"
    scored_path = OUTPUT_DIR / "hybrid_universe_scored.csv"
    screen_path = OUTPUT_DIR / "hybrid_screened_additions.csv"

    summary_df.to_csv(summary_path, index=False)
    annual_df.to_csv(annual_path, index=False)
    selections_df.to_csv(selections_path, index=False)
    scored_df.to_csv(scored_path, index=False)
    screen_df.to_csv(screen_path, index=False)

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)
    print(f"Saved summary to:     {summary_path}")
    print(f"Saved annuals to:     {annual_path}")
    print(f"Saved selections to:  {selections_path}")
    print(f"Saved scored rows to: {scored_path}")
    print(f"Saved screen rows to: {screen_path}")

    print("\nCompact summary:")
    cols = [
        "strategy",
        "start_date",
        "end_date",
        "total_return_pct",
        "sharpe",
        "max_drawdown_pct",
        "monthly_hit_rate",
        "top_minus_rest_mean",
    ]

    print(
        summary_df[cols]
        .sort_values(["start_date", "end_date", "add_count"])
        .to_string(index=False, float_format=lambda x: f"{x:,.4f}")
    )


if __name__ == "__main__":
    main()