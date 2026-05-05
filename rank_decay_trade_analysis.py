"""
Rank Decay Trade Analysis

Purpose:
- Analyze rank_decay_exit_runner.py outputs.
- Focus on the best candidate rank-decay rules.
- Use ONLY the full-period test for detailed event analysis to avoid duplicate rows
  from overlapping yearly test windows.

Summarizes:
    1. Full-period performance
    2. Replacement counts by year/month
    3. Most commonly sold tickers
    4. Most commonly bought replacement tickers
    5. Sell -> buy replacement pairs
    6. Days between replacement events
    7. Approximate holding-period behavior
    8. META-specific events, if any

This is research only.
It does not connect to Alpaca.
It does not submit orders.

Run:
    python -u rank_decay_trade_analysis.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("rank_decay_results")
OUTPUT_DIR = Path("rank_decay_results/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = INPUT_DIR / "rank_decay_summary.csv"
TRADES_PATH = INPUT_DIR / "rank_decay_trades.csv"
HISTORY_PATH = INPUT_DIR / "rank_decay_history.csv"

FULL_START_DATE = "2021-01-01"
FULL_END_DATE = "2026-05-01"

# Candidate strategies to inspect more closely.
CANDIDATES = [
    {"target_horizon": 21, "strategy": "rank_decay_top12_3days"},
    {"target_horizon": 21, "strategy": "rank_decay_top12_5days"},
    {"target_horizon": 21, "strategy": "rank_decay_top10_5days"},
    {"target_horizon": 30, "strategy": "rank_decay_top12_3days"},
    {"target_horizon": 30, "strategy": "rank_decay_top10_3days"},
    {"target_horizon": 30, "strategy": "monthly_top3_baseline"},
]


# ============================================================
# HELPERS
# ============================================================

def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    if path.stat().st_size == 0:
        raise RuntimeError(f"Required file is empty: {path}")

    return pd.read_csv(path)


def normalize_dates(
    df: pd.DataFrame,
    columns=("date", "start_date", "end_date"),
) -> pd.DataFrame:
    out = df.copy()

    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    return out


def candidate_mask(df: pd.DataFrame, candidate: dict) -> pd.Series:
    return (
        (df["target_horizon"].astype(int) == int(candidate["target_horizon"]))
        & (df["strategy"].astype(str) == str(candidate["strategy"]))
    )


def full_period_mask(df: pd.DataFrame) -> pd.Series:
    return (
        (df["start_date"].dt.strftime("%Y-%m-%d") == FULL_START_DATE)
        & (df["end_date"].dt.strftime("%Y-%m-%d") == FULL_END_DATE)
    )


def split_symbols(value) -> list[str]:
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    return [
        part.strip().upper()
        for part in text.split(",")
        if part.strip()
    ]


def expand_symbol_column(
    df: pd.DataFrame,
    column: str,
    output_symbol_col: str,
) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        symbols = split_symbols(row.get(column, ""))

        for symbol in symbols:
            new_row = row.to_dict()
            new_row[output_symbol_col] = symbol
            rows.append(new_row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def summarize_days_between_replacements(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (target_horizon, strategy), group in trades.groupby(["target_horizon", "strategy"]):
        replacements = group[group["event"] == "RANK_DECAY_REPLACE"].copy()
        replacements = replacements.sort_values("date")

        if len(replacements) <= 1:
            rows.append({
                "target_horizon": target_horizon,
                "strategy": strategy,
                "replacement_events": len(replacements),
                "avg_days_between_replacements": np.nan,
                "median_days_between_replacements": np.nan,
                "min_days_between_replacements": np.nan,
                "max_days_between_replacements": np.nan,
            })
            continue

        diffs = replacements["date"].diff().dt.days.dropna()

        rows.append({
            "target_horizon": target_horizon,
            "strategy": strategy,
            "replacement_events": len(replacements),
            "avg_days_between_replacements": float(diffs.mean()),
            "median_days_between_replacements": float(diffs.median()),
            "min_days_between_replacements": float(diffs.min()),
            "max_days_between_replacements": float(diffs.max()),
        })

    return pd.DataFrame(rows)


def summarize_replacements_by_period(
    trades: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    replacements = trades[trades["event"] == "RANK_DECAY_REPLACE"].copy()

    if replacements.empty:
        return pd.DataFrame(), pd.DataFrame()

    replacements["year"] = replacements["date"].dt.year
    replacements["month"] = replacements["date"].dt.to_period("M").astype(str)

    by_year = (
        replacements
        .groupby(["target_horizon", "strategy", "year"])
        .size()
        .reset_index(name="replacement_events")
        .sort_values(["target_horizon", "strategy", "year"])
    )

    by_month = (
        replacements
        .groupby(["target_horizon", "strategy", "month"])
        .size()
        .reset_index(name="replacement_events")
        .sort_values(["target_horizon", "strategy", "month"])
    )

    return by_year, by_month


def summarize_sold_bought(
    trades: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    replacements = trades[trades["event"] == "RANK_DECAY_REPLACE"].copy()

    sold_expanded = expand_symbol_column(
        replacements,
        column="sold",
        output_symbol_col="sold_symbol",
    )

    bought_expanded = expand_symbol_column(
        replacements,
        column="bought",
        output_symbol_col="bought_symbol",
    )

    if sold_expanded.empty:
        sold_counts = pd.DataFrame()
    else:
        sold_counts = (
            sold_expanded
            .groupby(["target_horizon", "strategy", "sold_symbol"])
            .size()
            .reset_index(name="times_sold")
            .sort_values(
                ["target_horizon", "strategy", "times_sold"],
                ascending=[True, True, False],
            )
        )

    if bought_expanded.empty:
        bought_counts = pd.DataFrame()
    else:
        bought_counts = (
            bought_expanded
            .groupby(["target_horizon", "strategy", "bought_symbol"])
            .size()
            .reset_index(name="times_bought")
            .sort_values(
                ["target_horizon", "strategy", "times_bought"],
                ascending=[True, True, False],
            )
        )

    pair_rows = []

    for _, row in replacements.iterrows():
        sold_symbols = split_symbols(row.get("sold", ""))
        bought_symbols = split_symbols(row.get("bought", ""))

        # Usually one sold and one bought because MAX_REPLACEMENTS_PER_DAY = 1.
        max_len = max(len(sold_symbols), len(bought_symbols))

        for i in range(max_len):
            pair_rows.append({
                "target_horizon": row.get("target_horizon"),
                "strategy": row.get("strategy"),
                "date": row.get("date"),
                "sold_symbol": sold_symbols[i] if i < len(sold_symbols) else "",
                "bought_symbol": bought_symbols[i] if i < len(bought_symbols) else "",
            })

    pair_df = pd.DataFrame(pair_rows)

    if not pair_df.empty:
        pair_counts = (
            pair_df
            .groupby(["target_horizon", "strategy", "sold_symbol", "bought_symbol"])
            .size()
            .reset_index(name="pair_count")
            .sort_values(
                ["target_horizon", "strategy", "pair_count"],
                ascending=[True, True, False],
            )
        )
    else:
        pair_counts = pd.DataFrame()

    return sold_counts, bought_counts, pair_counts


def build_holding_period_summary(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Approximate holding periods from daily selected lists.

    This does not know exact intraday fills. It measures consecutive calendar-day
    stretches where a ticker appears in the selected list.
    """
    rows = []

    history = history.copy()
    history = history.sort_values(["target_horizon", "strategy", "date"])

    for (target_horizon, strategy), group in history.groupby(["target_horizon", "strategy"]):
        group = group.sort_values("date").copy()

        active = {}

        for _, row in group.iterrows():
            dt = row["date"]
            selected_symbols = set(split_symbols(row.get("selected", "")))

            # Close holdings that disappeared.
            for symbol in list(active.keys()):
                if symbol not in selected_symbols:
                    start_dt = active.pop(symbol)
                    rows.append({
                        "target_horizon": target_horizon,
                        "strategy": strategy,
                        "symbol": symbol,
                        "start_date": start_dt,
                        "end_date": dt,
                        "holding_days": int((dt - start_dt).days),
                    })

            # Open new holdings.
            for symbol in selected_symbols:
                if symbol not in active:
                    active[symbol] = dt

        # Close anything still active at final date.
        if not group.empty:
            final_dt = group["date"].max()

            for symbol, start_dt in active.items():
                rows.append({
                    "target_horizon": target_horizon,
                    "strategy": strategy,
                    "symbol": symbol,
                    "start_date": start_dt,
                    "end_date": final_dt,
                    "holding_days": int((final_dt - start_dt).days),
                })

    holding_periods = pd.DataFrame(rows)

    if holding_periods.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary = (
        holding_periods
        .groupby(["target_horizon", "strategy"])
        .agg(
            avg_holding_days=("holding_days", "mean"),
            median_holding_days=("holding_days", "median"),
            min_holding_days=("holding_days", "min"),
            max_holding_days=("holding_days", "max"),
            holding_segments=("holding_days", "count"),
        )
        .reset_index()
    )

    return summary, holding_periods


def find_symbol_events(trades: pd.DataFrame, symbol: str) -> pd.DataFrame:
    symbol = symbol.upper()

    mask = (
        trades["sold"].fillna("").astype(str).str.upper().str.contains(symbol)
        | trades["bought"].fillna("").astype(str).str.upper().str.contains(symbol)
        | trades["selected"].fillna("").astype(str).str.upper().str.contains(symbol)
    )

    return trades[mask].copy().sort_values(["target_horizon", "strategy", "date"])


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    summary = normalize_dates(read_required_csv(SUMMARY_PATH))
    trades = normalize_dates(read_required_csv(TRADES_PATH))
    history = normalize_dates(read_required_csv(HISTORY_PATH))

    # Keep only candidate rows across all periods for performance summaries.
    candidate_summary_frames = []
    candidate_trade_frames = []
    candidate_history_frames = []

    for candidate in CANDIDATES:
        s_mask = candidate_mask(summary, candidate)
        t_mask = candidate_mask(trades, candidate)
        h_mask = candidate_mask(history, candidate)

        candidate_summary_frames.append(summary[s_mask].copy())
        candidate_trade_frames.append(trades[t_mask].copy())
        candidate_history_frames.append(history[h_mask].copy())

    candidate_summary_all_periods = pd.concat(candidate_summary_frames, ignore_index=True)
    candidate_trades_all_periods = pd.concat(candidate_trade_frames, ignore_index=True)
    candidate_history_all_periods = pd.concat(candidate_history_frames, ignore_index=True)

    # Full-period summary is the key performance comparison.
    full_period_summary = candidate_summary_all_periods[
        full_period_mask(candidate_summary_all_periods)
    ].copy()

    full_period_summary = full_period_summary.sort_values(
        ["sharpe", "total_return_pct"],
        ascending=[False, False],
    )

    # IMPORTANT:
    # Detailed event analysis should use ONLY full-period trades/history.
    # Otherwise yearly windows plus the full window double-count the same events.
    full_period_trades = candidate_trades_all_periods[
        full_period_mask(candidate_trades_all_periods)
    ].copy()

    full_period_history = candidate_history_all_periods[
        full_period_mask(candidate_history_all_periods)
    ].copy()

    by_year, by_month = summarize_replacements_by_period(full_period_trades)
    sold_counts, bought_counts, pair_counts = summarize_sold_bought(full_period_trades)
    replacement_spacing = summarize_days_between_replacements(full_period_trades)
    holding_summary, holding_periods = build_holding_period_summary(full_period_history)
    
    meta_events = find_symbol_events(full_period_trades, "META")
    meta_events = meta_events[meta_events["date"].notna()].copy()
    
    # Save outputs.
    full_period_summary.to_csv(
        OUTPUT_DIR / "candidate_full_period_summary.csv",
        index=False,
    )
    candidate_summary_all_periods.to_csv(
        OUTPUT_DIR / "candidate_all_period_summary.csv",
        index=False,
    )
    full_period_trades.to_csv(
        OUTPUT_DIR / "candidate_full_period_trades.csv",
        index=False,
    )
    full_period_history.to_csv(
        OUTPUT_DIR / "candidate_full_period_history.csv",
        index=False,
    )

    if not by_year.empty:
        by_year.to_csv(OUTPUT_DIR / "replacement_events_by_year.csv", index=False)

    if not by_month.empty:
        by_month.to_csv(OUTPUT_DIR / "replacement_events_by_month.csv", index=False)

    if not sold_counts.empty:
        sold_counts.to_csv(OUTPUT_DIR / "most_sold_tickers.csv", index=False)

    if not bought_counts.empty:
        bought_counts.to_csv(OUTPUT_DIR / "most_bought_tickers.csv", index=False)

    if not pair_counts.empty:
        pair_counts.to_csv(OUTPUT_DIR / "replacement_pair_counts.csv", index=False)

    if not replacement_spacing.empty:
        replacement_spacing.to_csv(OUTPUT_DIR / "replacement_spacing.csv", index=False)

    if not holding_summary.empty:
        holding_summary.to_csv(OUTPUT_DIR / "holding_period_summary.csv", index=False)

    if not holding_periods.empty:
        holding_periods.to_csv(OUTPUT_DIR / "holding_period_details.csv", index=False)

    if not meta_events.empty:
        meta_events.to_csv(OUTPUT_DIR / "meta_events.csv", index=False)

    # Print useful summary.
    print("\n===== CANDIDATE FULL-PERIOD SUMMARY =====")
    cols = [
        "target_horizon",
        "strategy",
        "total_return_pct",
        "sharpe",
        "max_drawdown_pct",
        "num_replacement_events",
        "num_trade_events",
    ]
    cols = [c for c in cols if c in full_period_summary.columns]

    print(
        full_period_summary[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:,.4f}",
        )
    )

    print("\n===== REPLACEMENT SPACING - FULL PERIOD ONLY =====")
    if replacement_spacing.empty:
        print("No replacement spacing rows.")
    else:
        print(
            replacement_spacing.to_string(
                index=False,
                float_format=lambda x: f"{x:,.2f}",
            )
        )

    print("\n===== HOLDING PERIOD SUMMARY - FULL PERIOD ONLY =====")
    if holding_summary.empty:
        print("No holding period summary rows.")
    else:
        print(
            holding_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:,.2f}",
            )
        )

    print("\n===== MOST SOLD TICKERS - FULL PERIOD ONLY =====")
    if sold_counts.empty:
        print("No sold ticker rows.")
    else:
        print(sold_counts.head(30).to_string(index=False))

    print("\n===== MOST BOUGHT TICKERS - FULL PERIOD ONLY =====")
    if bought_counts.empty:
        print("No bought ticker rows.")
    else:
        print(bought_counts.head(30).to_string(index=False))

    print("\n===== REPLACEMENT PAIRS - FULL PERIOD ONLY =====")
    if pair_counts.empty:
        print("No replacement pair rows.")
    else:
        print(pair_counts.head(30).to_string(index=False))

    print("\n===== META EVENTS - FULL PERIOD ONLY =====")
    if meta_events.empty:
        print("No META events found for the selected candidates.")
    else:
        meta_cols = [
            "target_horizon",
            "strategy",
            "start_date",
            "end_date",
            "date",
            "event",
            "selected",
            "sold",
            "bought",
            "monitor_rank",
            "confirm_days",
        ]
        meta_cols = [c for c in meta_cols if c in meta_events.columns]

        print(meta_events[meta_cols].head(80).to_string(index=False))

    print("\n===== SAVED ANALYSIS FILES =====")
    print(f"Output folder: {OUTPUT_DIR}")
    for path in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"- {path}")


if __name__ == "__main__":
    main()