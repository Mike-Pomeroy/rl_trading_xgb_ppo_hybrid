"""
Universe overlap runner.

Purpose:
- Start with your current 20-stock universe.
- Compare it against a larger candidate universe.
- Fetch split-adjusted historical data.
- Score candidates using simple objective metrics:
    - liquidity proxy
    - 6-month momentum
    - 12-month momentum
    - volatility penalty
    - drawdown penalty
- Select a new Top 20 candidate universe.
- Show how many of your current 20 are still included.
- Save CSV reports.

This does NOT trade.
This does NOT submit Alpaca orders.

Run:
    python -u universe_overlap_runner.py
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from data_module import fetch_alpaca_daily_bars


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = Path("universe_overlap_results")
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_ADJUSTMENT = "split"
FEED = "sip"  # If your plan rejects this, change to "delayed_sip".

START_DATE = "2019-01-01"
END_DATE = None  # None uses data_module default/current date if supported below.

CURRENT_20 = [
    "AAPL", "MSFT", "SPY", "GOOGL", "AMZN",
    "META", "TSLA", "NVDA", "JPM", "JNJ",
    "AVGO", "LLY", "UNH", "COST", "V",
    "MA", "HD", "PG", "XOM", "AMD",
]

# SPY is useful as a benchmark/regime ticker, but we do not want it selected
# as a tradable stock in the new Top-20 list.
EXCLUDE_FROM_SELECTION = {"SPY"}

# Larger candidate list.
# This is intentionally hand-curated large-cap/liquid names to avoid jumping
# straight into a messy full-market screen.
CANDIDATE_UNIVERSE = sorted(set([
    # Current 20
    "AAPL", "MSFT", "SPY", "GOOGL", "AMZN",
    "META", "TSLA", "NVDA", "JPM", "JNJ",
    "AVGO", "LLY", "UNH", "COST", "V",
    "MA", "HD", "PG", "XOM", "AMD",

    # Additional large/liquid candidates
    "BRK.B", "WMT", "ORCL", "NFLX", "BAC",
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

TOP_N = 20

MIN_HISTORY_DAYS = 1000
MIN_PRICE = 10.0

# Liquidity proxy uses close * volume from daily bars.
# Since you are trading small size, this threshold is intentionally modest.
MIN_AVG_DOLLAR_VOLUME_60D = 50_000_000

MOMENTUM_6M_DAYS = 126
MOMENTUM_12M_DAYS = 252
VOL_DAYS = 60
DRAWDOWN_DAYS = 252


# ============================================================
# HELPERS
# ============================================================

def normalize_ticker_for_alpaca(ticker: str) -> str:
    """
    Alpaca may use BRK.B or BRK/B depending context.
    If BRK.B causes issues, change this function to return BRK/B.
    """
    return ticker


def max_drawdown(values: pd.Series) -> float:
    values = values.dropna().astype(float)

    if values.empty:
        return np.nan

    peak = values.cummax()
    dd = values / peak - 1.0

    return float(dd.min())


def safe_pct_change(series: pd.Series, periods: int) -> float:
    series = series.dropna().astype(float)

    if len(series) <= periods:
        return np.nan

    start = series.iloc[-periods - 1]
    end = series.iloc[-1]

    if not np.isfinite(start) or start <= 0:
        return np.nan

    return float(end / start - 1.0)


def compute_candidate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for tic, g in df.groupby("tic"):
        g = g.sort_values("date").copy()

        close = g["close"].astype(float)
        volume = g["volume"].astype(float)

        daily_returns = close.pct_change()

        latest_close = close.iloc[-1] if len(close) else np.nan
        history_days = int(close.notna().sum())

        avg_dollar_volume_60d = (
            (close * volume)
            .tail(60)
            .replace([np.inf, -np.inf], np.nan)
            .mean()
        )

        momentum_6m = safe_pct_change(close, MOMENTUM_6M_DAYS)
        momentum_12m = safe_pct_change(close, MOMENTUM_12M_DAYS)

        vol_60d = (
            daily_returns
            .tail(VOL_DAYS)
            .replace([np.inf, -np.inf], np.nan)
            .std()
        )

        drawdown_1y = max_drawdown(close.tail(DRAWDOWN_DAYS))

        rows.append({
            "ticker": tic,
            "latest_close": float(latest_close) if pd.notna(latest_close) else np.nan,
            "history_days": history_days,
            "avg_dollar_volume_60d": float(avg_dollar_volume_60d)
                if pd.notna(avg_dollar_volume_60d) else np.nan,
            "momentum_6m": momentum_6m,
            "momentum_12m": momentum_12m,
            "volatility_60d": float(vol_60d) if pd.notna(vol_60d) else np.nan,
            "drawdown_1y": drawdown_1y,
        })

    metrics = pd.DataFrame(rows)

    return metrics


def add_ranks_and_score(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()

    # Eligibility filters.
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

    # Percentile ranks where higher is better.
    out["rank_momentum_6m"] = np.nan
    out["rank_momentum_12m"] = np.nan
    out["rank_liquidity"] = np.nan
    out["rank_low_volatility"] = np.nan
    out["rank_low_drawdown"] = np.nan

    out.loc[eligible, "rank_momentum_6m"] = out.loc[eligible, "momentum_6m"].rank(pct=True)
    out.loc[eligible, "rank_momentum_12m"] = out.loc[eligible, "momentum_12m"].rank(pct=True)
    out.loc[eligible, "rank_liquidity"] = out.loc[eligible, "avg_dollar_volume_60d"].rank(pct=True)

    # Lower volatility is better.
    out.loc[eligible, "rank_low_volatility"] = (
        -out.loc[eligible, "volatility_60d"]
    ).rank(pct=True)

    # Drawdown is negative; less negative is better.
    out.loc[eligible, "rank_low_drawdown"] = out.loc[eligible, "drawdown_1y"].rank(pct=True)

    # Simple blended score.
    # This is intentionally not XGBoost. It is just an objective pre-screen.
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
    out.loc[out["eligible"], "universe_rank"] = (
        np.arange(1, int(out["eligible"].sum()) + 1)
    )

    return out


def build_overlap_report(selected_top: List[str]) -> pd.DataFrame:
    current_set = set(CURRENT_20)
    selected_set = set(selected_top)

    overlap = sorted(current_set & selected_set)
    dropped = sorted(current_set - selected_set)
    added = sorted(selected_set - current_set)

    rows = []

    for ticker in sorted(current_set | selected_set):
        rows.append({
            "ticker": ticker,
            "in_current_20": ticker in current_set,
            "in_new_top_20": ticker in selected_set,
            "overlap": ticker in current_set and ticker in selected_set,
            "new_addition": ticker in added,
            "dropped_from_current": ticker in dropped,
        })

    return pd.DataFrame(rows)


def print_summary(selected_top: List[str], scored: pd.DataFrame) -> None:
    current_set = set(CURRENT_20)
    selected_set = set(selected_top)

    overlap = sorted(current_set & selected_set)
    dropped = sorted(current_set - selected_set)
    added = sorted(selected_set - current_set)

    print("\n===== NEW TOP-20 UNIVERSE =====")
    print(", ".join(selected_top))

    print("\n===== OVERLAP SUMMARY =====")
    print(f"Current universe size: {len(CURRENT_20)}")
    print(f"New universe size:     {len(selected_top)}")
    print(f"Overlap count:         {len(overlap)}")
    print(f"Overlap percentage:    {len(overlap) / len(CURRENT_20):.1%}")

    print("\nStill included from current 20:")
    print(", ".join(overlap) if overlap else "None")

    print("\nNew additions:")
    print(", ".join(added) if added else "None")

    print("\nDropped from current 20:")
    print(", ".join(dropped) if dropped else "None")

    print("\n===== TOP 30 SCORED CANDIDATES =====")
    cols = [
        "universe_rank",
        "ticker",
        "universe_score",
        "momentum_6m",
        "momentum_12m",
        "volatility_60d",
        "drawdown_1y",
        "avg_dollar_volume_60d",
        "latest_close",
    ]

    print(
        scored[scored["eligible"]]
        .head(30)[cols]
        .to_string(index=False, float_format=lambda x: f"{x:,.4f}")
    )

    ineligible = scored[~scored["eligible"]].copy()

    if not ineligible.empty:
        print("\n===== INELIGIBLE / FILTERED OUT SAMPLE =====")
        print(
            ineligible[[
                "ticker",
                "latest_close",
                "history_days",
                "avg_dollar_volume_60d",
                "momentum_6m",
                "momentum_12m",
                "volatility_60d",
                "drawdown_1y",
            ]]
            .head(20)
            .to_string(index=False, float_format=lambda x: f"{x:,.4f}")
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    tickers = [normalize_ticker_for_alpaca(t) for t in CANDIDATE_UNIVERSE]

    print("\n===== UNIVERSE OVERLAP RUNNER =====")
    print(f"Candidate tickers: {len(tickers)}")
    print(f"Data adjustment:   {DATA_ADJUSTMENT}")
    print(f"Feed:              {FEED}")
    print(f"Start date:        {START_DATE}")
    print(f"Top N:             {TOP_N}")

    print("\nFetching Alpaca daily bars...")
    fetch_kwargs = {
        "tickers": tickers,
        "start_date": START_DATE,
        "adjustment": DATA_ADJUSTMENT,
        "feed": FEED,
    }

    if END_DATE is not None:
        fetch_kwargs["end_date"] = END_DATE

    raw_df = fetch_alpaca_daily_bars(**fetch_kwargs)

    raw_df["date"] = pd.to_datetime(raw_df["date"]).dt.normalize()

    print(f"Rows fetched: {len(raw_df):,}")
    print(f"Tickers fetched: {raw_df['tic'].nunique()}")

    missing_fetch = sorted(set(tickers) - set(raw_df["tic"].unique()))

    if missing_fetch:
        print("\nWARNING: These tickers were requested but not fetched:")
        print(", ".join(missing_fetch))

    metrics = compute_candidate_metrics(raw_df)
    scored = add_ranks_and_score(metrics)

    selected_top = (
        scored[scored["eligible"]]
        .head(TOP_N)["ticker"]
        .tolist()
    )

    overlap_df = build_overlap_report(selected_top)

    # Save outputs.
    scored_path = OUTPUT_DIR / "universe_scored_candidates.csv"
    overlap_path = OUTPUT_DIR / "universe_overlap_report.csv"
    selected_path = OUTPUT_DIR / "new_top20_universe.csv"

    scored.to_csv(scored_path, index=False)
    overlap_df.to_csv(overlap_path, index=False)
    pd.DataFrame({"ticker": selected_top}).to_csv(selected_path, index=False)

    print_summary(selected_top, scored)

    print("\n===== SAVED FILES =====")
    print(f"Scored candidates: {scored_path}")
    print(f"Overlap report:    {overlap_path}")
    print(f"New Top 20:        {selected_path}")


if __name__ == "__main__":
    main()