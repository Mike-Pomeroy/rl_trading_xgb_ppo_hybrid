"""
Alpaca paper-order preview for the HYBRID XGBoost Top-K strategy.

Hybrid strategy:
- Keep the fixed current 20 universe as the core.
- Each signal date, screen the larger candidate universe.
- Add the top 5 screened names to the fixed universe.
- Train XGBoost on the hybrid universe.
- Select Top 3.
- Calculate target paper portfolio.
- Print proposed orders.
- Save proposed_orders.csv.
- Submit NO orders.

Run:
    python -u alpaca_order_preview_hybrid.py

Environment variables required:
    APCA_API_KEY_ID
    APCA_API_SECRET_KEY
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

import xgb_topk_strategy as strat


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

OUTPUT_DIR = Path("alpaca_preview_orders_hybrid")
OUTPUT_DIR.mkdir(exist_ok=True)

PAPER = True
SUBMIT_ORDERS = False  # SAFETY: this script previews only.

TOP_K = 3
TARGET_HORIZON = 30
DATA_ADJUSTMENT = "split"

HYBRID_ADD_COUNT = 5

CASH_BUFFER = 0.20
TRANSACTION_COST_ESTIMATE = 0.005

MIN_DOLLARS_PER_POSITION = 500.0
ALLOW_FRACTIONAL_SHARES = True

TRAIN_START_DATE: Optional[str] = None

INITIAL_FALLBACK_EQUITY = 3000.0

MIN_ORDER_DOLLARS = 25.0

SELL_POSITIONS_NOT_IN_HYBRID_UNIVERSE = False
SELL_UNSELECTED_HYBRID_POSITIONS = True

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
# ALPACA HELPERS
# ============================================================

def get_trading_client() -> TradingClient:
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


def get_open_orders_df(client: TradingClient) -> pd.DataFrame:
    request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
    open_orders = client.get_orders(filter=request)

    rows = []

    for order in open_orders:
        rows.append({
            "symbol": str(getattr(order, "symbol", "")).upper(),
            "id": getattr(order, "id", None),
            "side": getattr(order, "side", None),
            "type": getattr(order, "type", None),
            "status": getattr(order, "status", None),
            "qty": getattr(order, "qty", None),
            "notional": getattr(order, "notional", None),
            "submitted_at": getattr(order, "submitted_at", None),
        })

    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "id",
                "side",
                "type",
                "status",
                "qty",
                "notional",
                "submitted_at",
            ]
        )

    return pd.DataFrame(rows)


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
# MODEL / PREVIEW HELPERS
# ============================================================

def configure_strategy_module() -> None:
    strat.DATA_LIST = sorted(set(CANDIDATE_UNIVERSE))
    strat.TRADE_LIST = sorted(set(CANDIDATE_UNIVERSE) - {"SPY"})
    strat.TOP_K = TOP_K
    strat.TARGET_HORIZON = TARGET_HORIZON
    strat.DATA_ADJUSTMENT = DATA_ADJUSTMENT
    strat.CASH_BUFFER = CASH_BUFFER
    strat.TRANSACTION_COST = TRANSACTION_COST_ESTIMATE
    strat.MIN_DOLLARS_PER_POSITION = MIN_DOLLARS_PER_POSITION
    strat.ALLOW_FRACTIONAL_SHARES = ALLOW_FRACTIONAL_SHARES


def latest_signal_day(full_df: pd.DataFrame) -> pd.Timestamp:
    """
    Use the latest date where the fixed core universe has full coverage.
    Additions are selected afterward from whatever candidates are eligible.
    """
    model_df = full_df[full_df["tic"].isin(CURRENT_20_TRADE_LIST)].copy()
    model_df = model_df.dropna(subset=strat.FEATURES + ["close"])

    if model_df.empty:
        raise ValueError("No usable core-universe rows after feature cleaning.")

    counts = model_df.groupby("date")["tic"].nunique().sort_index()
    required_count = len(CURRENT_20_TRADE_LIST)

    complete_dates = counts[counts >= required_count]

    if complete_dates.empty:
        print("\nWARNING: No date has full core-universe coverage.")
        print("Recent core scoreable ticker counts:")
        print(counts.tail(10).to_string())

        max_count = counts.max()
        best_dates = counts[counts == max_count]
        signal_date = best_dates.index.max()

        print(
            f"Using latest best core-coverage date: {pd.Timestamp(signal_date).date()} "
            f"with {int(max_count)} / {required_count} core tickers."
        )

        return pd.Timestamp(signal_date)

    signal_date = complete_dates.index.max()

    print(
        f"Using latest full core-coverage signal date: {pd.Timestamp(signal_date).date()} "
        f"with {required_count} / {required_count} core tickers."
    )

    return pd.Timestamp(signal_date)


def train_current_model(
    full_df: pd.DataFrame,
    signal_date: pd.Timestamp,
    hybrid_universe: List[str],
):
    target_col = f"future_return_{TARGET_HORIZON}"

    model_df = full_df[full_df["tic"].isin(hybrid_universe)].copy()

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
    hybrid_universe: List[str],
) -> pd.DataFrame:
    signal_day = (
        full_df[
            (full_df["date"] == signal_date)
            & (full_df["tic"].isin(hybrid_universe))
        ]
        .copy()
    )

    signal_day = signal_day.dropna(subset=strat.FEATURES + ["close"]).copy()

    scoreable = sorted(signal_day["tic"].unique())
    missing = sorted(set(hybrid_universe) - set(scoreable))

    print(
        f"Scoreable hybrid tickers on {signal_date.date()}: "
        f"{len(scoreable)} / {len(hybrid_universe)}"
    )

    if missing:
        print("Missing/unscoreable hybrid tickers:")
        print(", ".join(missing))

    if signal_day.empty:
        raise ValueError(f"No usable signal rows for {signal_date.date()}.")

    signal_day["score"] = model.predict(signal_day[strat.FEATURES])
    signal_day = signal_day.sort_values("score", ascending=False).reset_index(drop=True)
    signal_day["rank"] = np.arange(1, len(signal_day) + 1)

    return signal_day


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
    hybrid_universe: List[str],
    screened_additions: List[str],
) -> pd.DataFrame:
    latest_prices = (
        signal_day.set_index("tic")["close"]
        .reindex(hybrid_universe)
        .astype(float)
        .values
    )

    target_weights = strat.make_target_weights(
        selected=selected,
        tickers=hybrid_universe,
        px=latest_prices,
        invested_fraction=max(0.0, 1.0 - CASH_BUFFER),
        portfolio_value=account_equity,
        min_dollars_per_position=MIN_DOLLARS_PER_POSITION,
        allow_fractional_shares=ALLOW_FRACTIONAL_SHARES,
    )

    target_dollars_by_symbol = {
        ticker: account_equity * target_weights[i]
        for i, ticker in enumerate(hybrid_universe)
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

    symbols_to_consider = set(hybrid_universe)

    if SELL_POSITIONS_NOT_IN_HYBRID_UNIVERSE:
        symbols_to_consider |= set(holdings.keys())

    rows = []

    for symbol in sorted(symbols_to_consider):
        in_hybrid = symbol in hybrid_universe
        is_selected = symbol in selected
        is_screened_addition = symbol in screened_additions

        current_qty = holdings.get(symbol, {}).get("qty", 0.0)
        current_value = holdings.get(symbol, {}).get("market_value", 0.0)

        price = close_by_symbol.get(
            symbol,
            holdings.get(symbol, {}).get("current_price", np.nan),
        )

        target_value = 0.0

        if in_hybrid:
            target_value = float(target_dollars_by_symbol.get(symbol, 0.0))

        if not in_hybrid and not SELL_POSITIONS_NOT_IN_HYBRID_UNIVERSE:
            continue

        if in_hybrid and not is_selected and not SELL_UNSELECTED_HYBRID_POSITIONS:
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
            "in_hybrid_universe": in_hybrid,
            "is_screened_addition": is_screened_addition,
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
            .sort_values(
                ["action_sort", "selected", "rank", "symbol"],
                ascending=[True, False, True, True],
            )
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
            "SUBMIT_ORDERS is True. This hybrid preview script submits nothing. "
            "Keep SUBMIT_ORDERS = False."
        )

    configure_strategy_module()

    print("\n===== ALPACA HYBRID ORDER PREVIEW ONLY =====")
    print(f"PAPER: {PAPER}")
    print(f"SUBMIT_ORDERS: {SUBMIT_ORDERS}")
    print(f"TOP_K: {TOP_K}")
    print(f"TARGET_HORIZON: {TARGET_HORIZON}")
    print(f"DATA_ADJUSTMENT: {DATA_ADJUSTMENT}")
    print(f"HYBRID_ADD_COUNT: {HYBRID_ADD_COUNT}")
    print(f"CASH_BUFFER: {CASH_BUFFER}")
    print(f"TRANSACTION_COST_ESTIMATE: {TRANSACTION_COST_ESTIMATE}")
    print(f"MIN_DOLLARS_PER_POSITION: {MIN_DOLLARS_PER_POSITION}")
    print(f"CORE TRADE TICKERS: {len(CURRENT_20_TRADE_LIST)}")
    print(f"CANDIDATE TICKERS INCLUDING SPY: {len(CANDIDATE_UNIVERSE)}")

    client = get_trading_client()

    print("\nReading Alpaca paper account...")
    account_equity = get_account_equity(client)
    positions_df = get_positions_df(client)
    open_orders_df = get_open_orders_df(client)

    print(f"Account equity: ${account_equity:,.2f}")
    print(f"Current positions: {len(positions_df)}")
    print(f"Open orders: {len(open_orders_df)}")

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

    if not open_orders_df.empty:
        print("\nWARNING: Existing open Alpaca orders detected.")
        print("Do not submit new orders until these are filled or canceled.")
        print(open_orders_df.to_string(index=False))

    print("\nPreparing model data...")
    full_df = strat.prepare_full_df()
    full_df = strat.normalize_date_column(full_df, "date")

    signal_date = latest_signal_day(full_df)

    print(f"Latest usable signal date: {signal_date.date()}")

    screened_additions, screened_scores = select_screened_additions_asof(
        full_df=full_df,
        signal_date=signal_date,
        add_count=HYBRID_ADD_COUNT,
    )

    hybrid_universe = sorted(set(CURRENT_20_TRADE_LIST) | set(screened_additions))

    print("\n===== HYBRID UNIVERSE =====")
    print(f"Screened additions Top-{HYBRID_ADD_COUNT}: {', '.join(screened_additions)}")
    print(f"Hybrid universe count: {len(hybrid_universe)}")
    print(", ".join(hybrid_universe))

    model = train_current_model(
        full_df=full_df,
        signal_date=signal_date,
        hybrid_universe=hybrid_universe,
    )

    signal_day = score_latest_day(
        full_df=full_df,
        model=model,
        signal_date=signal_date,
        hybrid_universe=hybrid_universe,
    )

    selected = signal_day.head(TOP_K)["tic"].tolist()

    print("\n===== MODEL SELECTION =====")
    print(f"Selected Top-{TOP_K}: {', '.join(selected)}")

    print("\nTop scored hybrid tickers:")
    print(
        signal_day[["rank", "tic", "score", "close"]]
        .head(15)
        .to_string(index=False)
    )

    orders_df = build_order_preview(
        account_equity=account_equity,
        positions_df=positions_df,
        signal_day=signal_day,
        selected=selected,
        hybrid_universe=hybrid_universe,
        screened_additions=screened_additions,
    )

    orders_df["strategy_name"] = "hybrid_plus_5"
    orders_df["signal_date"] = str(signal_date.date())
    orders_df["rebalance_period"] = signal_date.strftime("%Y-%m")


    proposed_orders = orders_df[orders_df["action"].isin(["BUY", "SELL"])].copy()

    print("\n===== PROPOSED ORDERS - PREVIEW ONLY =====")

    if proposed_orders.empty:
        print("No proposed BUY/SELL orders. Portfolio is already close to target.")
    else:
        cols = [
            "symbol",
            "is_screened_addition",
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
        "is_screened_addition",
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
    open_orders_path = OUTPUT_DIR / "open_orders.csv"
    screened_path = OUTPUT_DIR / "screened_additions_scores.csv"
    hybrid_path = OUTPUT_DIR / "hybrid_universe.csv"

    orders_df.to_csv(orders_path, index=False)
    signal_day.to_csv(selected_path, index=False)
    positions_df.to_csv(positions_path, index=False)
    open_orders_df.to_csv(open_orders_path, index=False)
    screened_scores.to_csv(screened_path, index=False)
    pd.DataFrame({
        "ticker": hybrid_universe,
        "is_core_current_20": [t in CURRENT_20_TRADE_LIST for t in hybrid_universe],
        "is_screened_addition": [t in screened_additions for t in hybrid_universe],
    }).to_csv(hybrid_path, index=False)

    print("\n===== SAVED FILES =====")
    print(f"Proposed orders:          {orders_path}")
    print(f"Model scores:             {selected_path}")
    print(f"Current positions:        {positions_path}")
    print(f"Open orders:              {open_orders_path}")
    print(f"Screened addition scores: {screened_path}")
    print(f"Hybrid universe:          {hybrid_path}")

    print("\nNo orders were submitted.")


if __name__ == "__main__":
    main()