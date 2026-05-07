"""
Rank Decay Daily Monitor

Purpose:
- Read the current frozen monthly target holdings.
- Rank the trade universe using the XGBoost model with a 21-trading-day target horizon.
- Monitor whether current holdings remain inside the Top-12 rank band.
- Track consecutive days outside the Top-12 band.
- Print HOLD / REVIEW status.
- Save monitor state and reports.

IMPORTANT:
- This script is READ ONLY.
- It does NOT submit trades.
- It does NOT modify the Alpaca account.

Research rule:
    target_horizon = 21
    monitor_rank = Top 12
    confirm_days = 3
    max replacement suggestion = 1

Run:
    python -u rank_decay_daily_monitor.py
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import xgb_topk_strategy as strat


# ============================================================
# CONFIG
# ============================================================

STRATEGY_NAME = "hybrid_plus_5"
MODE = "paper"

TARGET_HORIZON = 21
MONITOR_RANK = 12
CONFIRM_DAYS = 3
MAX_REPLACEMENT_SUGGESTIONS = 1

TRAIN_START_DATE = None

OUTPUT_DIR = Path("rank_decay_monitor")
OUTPUT_DIR.mkdir(exist_ok=True)

STATE_PATH = OUTPUT_DIR / "rank_decay_monitor_state.csv"
RANKINGS_PATH = OUTPUT_DIR / "latest_rankings.csv"
REPORT_PATH = OUTPUT_DIR / "rank_decay_monitor_report.txt"
HOLDINGS_STATUS_PATH = OUTPUT_DIR / "holdings_status.csv"

TARGET_SNAPSHOT_DIR = Path("monthly_target_snapshots")
PREVIEW_PATH = Path("alpaca_preview_orders_hybrid/proposed_orders.csv")

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


# ============================================================
# HELPERS
# ============================================================

def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def ensure_future_return_column(
    df: pd.DataFrame,
    target_horizon: int,
) -> pd.DataFrame:
    out = df.copy()
    out = strat.normalize_date_column(out, "date")
    out = out.sort_values(["tic", "date"]).reset_index(drop=True)

    col = f"future_return_{target_horizon}"

    if col not in out.columns:
        out[col] = (
            out.groupby("tic")["close"]
            .shift(-target_horizon)
            .div(out["close"])
            .sub(1.0)
        )
        print(f"Added missing target column: {col}")

    return out


def find_latest_snapshot_path() -> Path | None:
    if not TARGET_SNAPSHOT_DIR.exists():
        return None

    pattern = f"{STRATEGY_NAME}_*_{MODE}_target.csv"
    paths = sorted(TARGET_SNAPSHOT_DIR.glob(pattern))

    if not paths:
        return None

    return paths[-1]


def load_current_target_holdings() -> Tuple[List[str], str, Path | None]:
    
    env_holdings = os.getenv("RANK_DECAY_HOLDINGS", "").strip()

    if env_holdings:
        holdings = [
            symbol.strip().upper()
            for symbol in env_holdings.split(",")
            if symbol.strip()
        ]

        if holdings:
            source = os.getenv(
                "RANK_DECAY_TARGET_SOURCE",
                "github_secret_holdings",
            )

            return holdings, source, None
    
    
    
    snapshot_path = find_latest_snapshot_path()

    if snapshot_path is not None:
        df = pd.read_csv(snapshot_path)

        if "selected" not in df.columns or "symbol" not in df.columns:
            raise RuntimeError(
                f"Snapshot file missing required columns selected/symbol: {snapshot_path}"
            )

        selected = df[df["selected"].apply(parse_bool)].copy()
        holdings = selected["symbol"].astype(str).str.upper().tolist()

        return holdings, f"frozen_snapshot:{snapshot_path}", snapshot_path

    if not PREVIEW_PATH.exists():
        raise FileNotFoundError(
            "No frozen target snapshot found and no preview file found. "
            "Run preview first or create a monthly target snapshot."
        )

    df = pd.read_csv(PREVIEW_PATH)

    if "selected" not in df.columns or "symbol" not in df.columns:
        raise RuntimeError(
            f"Preview file missing required columns selected/symbol: {PREVIEW_PATH}"
        )

    selected = df[df["selected"].apply(parse_bool)].copy()
    holdings = selected["symbol"].astype(str).str.upper().tolist()

    return holdings, f"latest_preview:{PREVIEW_PATH}", None


def load_state() -> pd.DataFrame:
    if not STATE_PATH.exists() or STATE_PATH.stat().st_size == 0:
        return pd.DataFrame(
            columns=[
                "symbol",
                "consecutive_outside_days",
                "last_monitor_date",
                "last_rank",
                "last_status",
            ]
        )

    return pd.read_csv(STATE_PATH)


def save_state(state_df: pd.DataFrame) -> None:
    state_df.to_csv(STATE_PATH, index=False)


def build_model_for_latest_signal(
    model_df: pd.DataFrame,
    signal_date: pd.Timestamp,
    target_col: str,
):
    train_mask = model_df["date"] < signal_date

    if TRAIN_START_DATE is not None:
        train_mask &= model_df["date"] >= pd.Timestamp(TRAIN_START_DATE)

    return strat.train_model(
        model_df.loc[train_mask],
        strat.FEATURES,
        target_col,
    )


def rank_latest_signal_day(full_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Timestamp]:
    target_col = f"future_return_{TARGET_HORIZON}"

    df = full_df.copy()
    df = strat.normalize_date_column(df, "date")

    model_df = df[df["tic"].isin(TRADE_LIST)].copy()
    usable = model_df.dropna(subset=strat.FEATURES + ["close"]).copy()

    if usable.empty:
        raise RuntimeError("No usable rows available for ranking.")

    latest_date = usable["date"].max()
    signal_day = usable[usable["date"] == latest_date].copy()

    model = build_model_for_latest_signal(
        model_df=model_df,
        signal_date=latest_date,
        target_col=target_col,
    )

    if model is None:
        raise RuntimeError(f"Could not train model for signal date {latest_date.date()}")

    ranked = signal_day.dropna(subset=strat.FEATURES).copy()
    ranked["score"] = model.predict(ranked[strat.FEATURES])
    ranked = ranked.sort_values("score", ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)

    keep_cols = ["date", "tic", "rank", "score", "close", target_col]
    keep_cols = [col for col in keep_cols if col in ranked.columns]

    return ranked[keep_cols].copy(), latest_date


def update_holding_state(
    current_holdings: List[str],
    ranked: pd.DataFrame,
    signal_date: pd.Timestamp,
) -> pd.DataFrame:
    state = load_state()

    if "symbol" not in state.columns:
        state = pd.DataFrame(
            columns=[
                "symbol",
                "consecutive_outside_days",
                "last_monitor_date",
                "last_rank",
                "last_status",
            ]
        )

    state["symbol"] = state["symbol"].astype(str).str.upper()

    state_map: Dict[str, dict] = {
        row["symbol"]: row.to_dict()
        for _, row in state.iterrows()
    }

    rank_map = {
        str(row["tic"]).upper(): int(row["rank"])
        for _, row in ranked.iterrows()
    }

    signal_date_str = signal_date.strftime("%Y-%m-%d")
    rows = []

    for symbol in current_holdings:
        symbol = symbol.upper()
        previous = state_map.get(symbol, {})

        previous_count = int(previous.get("consecutive_outside_days", 0) or 0)
        previous_monitor_date = str(previous.get("last_monitor_date", "") or "")

        rank = rank_map.get(symbol, 999999)
        is_inside_band = rank <= MONITOR_RANK

        already_counted_today = previous_monitor_date == signal_date_str

        if already_counted_today:
            new_count = previous_count
        elif is_inside_band:
            new_count = 0
        else:
            new_count = previous_count + 1

        if is_inside_band:
            status = "HOLD_IN_BAND"
        elif new_count >= CONFIRM_DAYS:
            status = "REVIEW_REPLACEMENT"
        else:
            status = "WATCH_OUTSIDE_BAND"

        rows.append({
            "symbol": symbol,
            "rank": rank if rank != 999999 else np.nan,
            "monitor_rank": MONITOR_RANK,
            "inside_band": is_inside_band,
            "consecutive_outside_days": new_count,
            "confirm_days": CONFIRM_DAYS,
            "status": status,
            "last_monitor_date": signal_date_str,
            "last_rank": rank if rank != 999999 else np.nan,
            "last_status": status,
        })

    state_df = pd.DataFrame(rows)

    save_cols = [
        "symbol",
        "consecutive_outside_days",
        "last_monitor_date",
        "last_rank",
        "last_status",
    ]

    save_state(state_df[save_cols].copy())

    return state_df


def suggest_replacements(
    holdings_status: pd.DataFrame,
    ranked: pd.DataFrame,
    current_holdings: List[str],
) -> pd.DataFrame:
    flagged = holdings_status[
        holdings_status["status"] == "REVIEW_REPLACEMENT"
    ].copy()

    if flagged.empty:
        return pd.DataFrame()

    flagged = flagged.sort_values(
        ["consecutive_outside_days", "rank"],
        ascending=[False, False],
    ).head(MAX_REPLACEMENT_SUGGESTIONS)

    held_set = {symbol.upper() for symbol in current_holdings}

    candidates = ranked[
        ~ranked["tic"].astype(str).str.upper().isin(held_set)
    ].copy()

    candidates = candidates.sort_values("rank").reset_index(drop=True)

    if candidates.empty:
        return pd.DataFrame()

    rows = []

    for _, held_row in flagged.iterrows():
        replacement = candidates.iloc[0]

        rows.append({
            "sell_candidate": held_row["symbol"],
            "sell_candidate_rank": held_row["rank"],
            "sell_candidate_consecutive_outside_days": held_row["consecutive_outside_days"],
            "buy_candidate": str(replacement["tic"]).upper(),
            "buy_candidate_rank": int(replacement["rank"]),
            "buy_candidate_score": float(replacement["score"]),
        })

    return pd.DataFrame(rows)


def write_report(
    signal_date: pd.Timestamp,
    target_source: str,
    current_holdings: List[str],
    holdings_status: pd.DataFrame,
    ranked: pd.DataFrame,
    replacement_suggestions: pd.DataFrame,
) -> None:
    lines = []

    lines.append("RANK DECAY DAILY MONITOR")
    lines.append("=" * 80)
    lines.append(f"Signal date: {signal_date.strftime('%Y-%m-%d')}")
    lines.append(f"Strategy: {STRATEGY_NAME}")
    lines.append(f"Mode: {MODE}")
    lines.append(f"Target source: {target_source}")
    lines.append(f"Target horizon: {TARGET_HORIZON}")
    lines.append(f"Monitor rule: outside Top {MONITOR_RANK} for {CONFIRM_DAYS} consecutive days")
    lines.append(f"Current holdings: {', '.join(current_holdings)}")
    lines.append("")

    lines.append("HOLDINGS STATUS")
    lines.append("-" * 80)

    if holdings_status.empty:
        lines.append("No holdings status rows.")
    else:
        lines.append(
            holdings_status[
                [
                    "symbol",
                    "rank",
                    "monitor_rank",
                    "inside_band",
                    "consecutive_outside_days",
                    "confirm_days",
                    "status",
                ]
            ].to_string(index=False)
        )

    lines.append("")
    lines.append("TOP 12 RANKINGS")
    lines.append("-" * 80)

    top_cols = ["rank", "tic", "score", "close"]
    top_cols = [col for col in top_cols if col in ranked.columns]

    lines.append(
        ranked.head(MONITOR_RANK)[top_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:,.6f}",
        )
    )

    lines.append("")
    lines.append("REPLACEMENT SUGGESTIONS")
    lines.append("-" * 80)

    if replacement_suggestions.empty:
        lines.append("No replacement review triggered. HOLD.")
    else:
        lines.append(
            replacement_suggestions.to_string(
                index=False,
                float_format=lambda x: f"{x:,.6f}",
            )
        )
        lines.append("")
        lines.append("This is a review signal only. It does not submit orders.")

    REPORT_PATH.write_text("\n".join(lines))


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("\n===== RANK DECAY DAILY MONITOR =====")
    print("READ ONLY - no Alpaca orders will be submitted.")
    print(f"Rule: 21-day target, Top-{MONITOR_RANK} monitor, {CONFIRM_DAYS}-day confirmation")

    current_holdings, target_source, snapshot_path = load_current_target_holdings()

    if not current_holdings:
        raise RuntimeError("No current target holdings found.")

    print(f"\nTarget source: {target_source}")
    print(f"Current target holdings: {', '.join(current_holdings)}")

    print("\nPreparing full data...")
    strat.DATA_LIST = DATA_LIST
    strat.TRADE_LIST = TRADE_LIST
    strat.TARGET_HORIZON = TARGET_HORIZON

    full_df = strat.prepare_full_df()
    full_df = strat.normalize_date_column(full_df, "date")
    full_df = ensure_future_return_column(full_df, TARGET_HORIZON)

    ranked, signal_date = rank_latest_signal_day(full_df)
    ranked.to_csv(RANKINGS_PATH, index=False)

    holdings_status = update_holding_state(
        current_holdings=current_holdings,
        ranked=ranked,
        signal_date=signal_date,
    )
    holdings_status.to_csv(HOLDINGS_STATUS_PATH, index=False)

    replacement_suggestions = suggest_replacements(
        holdings_status=holdings_status,
        ranked=ranked,
        current_holdings=current_holdings,
    )

    write_report(
        signal_date=signal_date,
        target_source=target_source,
        current_holdings=current_holdings,
        holdings_status=holdings_status,
        ranked=ranked,
        replacement_suggestions=replacement_suggestions,
    )

    print("\n===== MONITOR RESULT =====")
    print(f"Signal date: {signal_date.strftime('%Y-%m-%d')}")
    print(f"Current holdings: {', '.join(current_holdings)}")
    print("")

    display_cols = [
        "symbol",
        "rank",
        "monitor_rank",
        "inside_band",
        "consecutive_outside_days",
        "confirm_days",
        "status",
    ]

    print("Holdings status:")
    print(holdings_status[display_cols].to_string(index=False))

    print("\nTop 12 rankings:")
    top_cols = ["rank", "tic", "score", "close"]
    top_cols = [col for col in top_cols if col in ranked.columns]

    print(
        ranked.head(MONITOR_RANK)[top_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:,.6f}",
        )
    )

    print("\nReplacement suggestions:")
    if replacement_suggestions.empty:
        print("No replacement review triggered. HOLD.")
    else:
        print(replacement_suggestions.to_string(index=False))

    print("\n===== SAVED =====")
    print(f"State:           {STATE_PATH}")
    print(f"Holdings status: {HOLDINGS_STATUS_PATH}")
    print(f"Rankings:        {RANKINGS_PATH}")
    print(f"Report:          {REPORT_PATH}")

    print("\nReminder: this monitor is read-only. It does not submit orders.")


if __name__ == "__main__":
    main()