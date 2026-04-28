"""
XGBoost top-k monthly stock strategy with safer backtesting defaults.

Key behavior:
- DATA_LIST includes SPY so SPY-based regime features can be calculated.
- TRADE_LIST excludes SPY so SPY cannot be trained/ranked/selected/traded by XGBoost.
- Features are required for prediction, but future targets are required only for training.
- Trade decisions are made from signal date t and executed on the next available trading day.
- Optional walk-forward retraining is supported.
- Adds monthly equal-weight and simple momentum top-k benchmarks.
- Adds ranking diagnostics to check whether model scores separate future returns.
"""

import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from data_module import INDICATORS, prepare_data

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------
# CONFIG
# ----------------------------

# Used for downloading/building features.
# Keep SPY here because spy_trend, spy_sma_200, and spy_above_200sma need it.
DATA_LIST = [
    "AAPL", "MSFT", "SPY", "GOOGL", "AMZN",
    "META", "TSLA", "NVDA", "JPM", "JNJ"
]

# Used for model training, ranking, selection, portfolio construction, and tradable benchmarks.
# SPY is intentionally excluded here.
TRADE_LIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN",
    "META", "TSLA", "NVDA", "JPM", "JNJ"
]

FEATURES = INDICATORS + [
    "volatility_30",
    "return_5",
    "return_10",
    "price_vs_sma30",
    "spy_trend",
]

INITIAL_AMOUNT = 3000
TOP_K = 3
TRANSACTION_COST = 0.005

MIN_DOLLARS_PER_POSITION = 500.0
ALLOW_FRACTIONAL_SHARES = True
DATA_ADJUSTMENT = "split"

# 30 trading days was the strongest tested horizon so far.
TARGET_HORIZON = 30

CASH_BUFFER = 0.20
SPY_TREND_FILTER = False
TRAIN_START_DATE = None
TRAIN_END_DATE = "2022-01-01"

TEST_START_DATE = "2021-01-01"
TEST_END_DATE = "2025-01-01"

# Walk-forward retrains the model at each rebalance date using only past data.
# Set to False to mimic a single-model train/test split.
WALK_FORWARD = True

MIN_TRAIN_ROWS = 500

# About 6 trading months.
MOMENTUM_LOOKBACK = 126


@dataclass
class StrategyResult:
    stats: Dict[str, object]
    history: pd.DataFrame
    selections: pd.DataFrame
    scored: Optional[pd.DataFrame] = None


def make_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )


def compute_stats(
    portfolio_values: Iterable[float],
    initial_amount: float,
) -> Dict[str, object]:
    portfolio_values = np.array(list(portfolio_values), dtype=np.float64)

    if len(portfolio_values) == 0:
        raise ValueError("No portfolio values available for stats.")

    returns = portfolio_values[1:] / (portfolio_values[:-1] + 1e-12) - 1.0

    sharpe = (
        (np.mean(returns) / (np.std(returns, ddof=1) + 1e-12)) * np.sqrt(252)
        if len(returns) > 1
        else 0.0
    )

    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / (peak + 1e-12)

    return {
        "final_portfolio": float(portfolio_values[-1]),
        "total_return_pct": float(
            (portfolio_values[-1] / initial_amount - 1.0) * 100.0
        ),
        "sharpe": float(sharpe),
        "max_drawdown_pct": float(np.max(drawdown) * 100.0),
        "portfolio_values": portfolio_values,
    }


def print_stats(name: str, stats: Dict[str, object]) -> None:
    print(f"\n===== {name} =====")
    print(f"Final Portfolio: ${stats['final_portfolio']:,.2f}")
    print(f"Total Return: {stats['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {stats['sharpe']:.3f}")
    print(f"Max Drawdown: {stats['max_drawdown_pct']:.2f}%")


def add_future_return_targets(
    df: pd.DataFrame,
    horizons: Iterable[int],
) -> pd.DataFrame:
    out = df.sort_values(["tic", "date"]).copy()

    for horizon in horizons:
        out[f"future_return_{horizon}"] = (
            out.groupby("tic")["close"].shift(-horizon) / out["close"] - 1.0
        )

    return out


def normalize_date_column(df: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    df = df.copy()
    dates = pd.to_datetime(df[column])

    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert(None)

    df[column] = dates.dt.normalize()
    return df


def prepare_full_df() -> pd.DataFrame:
    df = prepare_data(
        tickers=DATA_LIST,
        adjustment=DATA_ADJUSTMENT,
    )

    df = df[df["tic"].isin(DATA_LIST)].copy()

    df = normalize_date_column(df, "date")
    df = df.sort_values(["tic", "date"]).reset_index(drop=True)

    df = add_future_return_targets(
        df,
        horizons=sorted(set([TARGET_HORIZON, 10, 21, 30, 45, 60])),
    )

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def build_price_matrix(df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    tmp = df.copy()
    tmp = normalize_date_column(tmp, "date")

    pivot = tmp.pivot_table(
        index="date",
        columns="tic",
        values="close",
        aggfunc="last",
    )

    pivot = pivot.reindex(columns=tickers).sort_index()
    pivot = pivot.ffill()
    pivot = pivot.replace([np.inf, -np.inf], np.nan)

    return pivot




def get_monthly_signal_dates(dates: Iterable[pd.Timestamp]) -> List[pd.Timestamp]:
    dts = (
        pd.to_datetime(pd.Series(list(dates)))
        .dt.normalize()
        .sort_values()
        .drop_duplicates()
    )

    if dts.empty:
        return []

    monthly = dts.groupby(dts.dt.to_period("M")).min()

    return list(pd.to_datetime(monthly).sort_values())


def next_trading_date(
    dates: List[pd.Timestamp],
    signal_date: pd.Timestamp,
) -> Optional[pd.Timestamp]:
    idx = np.searchsorted(dates, signal_date, side="right")

    if idx >= len(dates):
        return None

    return dates[idx]


def current_position_values(shares: np.ndarray, px: np.ndarray) -> np.ndarray:
    valid_px = np.isfinite(px) & (px > 0)

    values = np.zeros(len(shares), dtype=np.float64)
    values[valid_px] = shares[valid_px] * px[valid_px]

    return values


def rebalance_to_weights(
    cash: float,
    shares: np.ndarray,
    px: np.ndarray,
    target_weights: np.ndarray,
    transaction_cost: float,
) -> Tuple[float, np.ndarray, float]:
    values = current_position_values(shares, px)
    portfolio_value = cash + float(np.sum(values))

    if portfolio_value <= 0:
        return 0.0, np.zeros_like(shares), 0.0

    current_weights = values / (portfolio_value + 1e-12)

    # Turnover on risky assets only.
    turnover = float(np.sum(np.abs(target_weights - current_weights)))
    cost = transaction_cost * turnover * portfolio_value

    investable_value = max(portfolio_value - cost, 0.0)

    new_shares = np.zeros_like(shares)
    valid_px = np.isfinite(px) & (px > 0)

    for i, weight in enumerate(target_weights):
        if weight > 0 and valid_px[i]:
            new_shares[i] = (investable_value * weight) / px[i]

    new_cash = investable_value * max(0.0, 1.0 - float(np.sum(target_weights)))

    return new_cash, new_shares, cost


def train_model(
    train_df: pd.DataFrame,
    features: List[str],
    target_col: str,
) -> Optional[XGBRegressor]:
    clean = train_df.dropna(subset=features + [target_col]).copy()

    if len(clean) < MIN_TRAIN_ROWS:
        return None

    model = make_model()
    model.fit(clean[features], clean[target_col])

    return model


def make_target_weights(
    selected: List[str],
    tickers: List[str],
    px: np.ndarray,
    invested_fraction: float,
    portfolio_value: float,
    min_dollars_per_position: float = 0.0,
    allow_fractional_shares: bool = True,
) -> np.ndarray:
    """
    Build target weights for selected tickers using small-account constraints.

    Conservative behavior:
    - Start with equal weights across selected valid tickers.
    - Drop any position whose target dollars are below MIN_DOLLARS_PER_POSITION.
    - If fractional shares are disabled, also drop positions where target dollars
      cannot buy at least one share.
    - Re-equal-weight across remaining valid selected names.
    - If no valid names remain, hold cash.
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

        # If all current candidates pass, assign weights.
        if len(valid_candidate_indices) == len(candidate_indices):
            for idx in valid_candidate_indices:
                weights[idx] = equal_weight

            return weights

        # If some candidates were removed, recalculate equal weights over survivors.
        if len(valid_candidate_indices) == 0:
            return weights

        candidate_indices = valid_candidate_indices

    return weights


def run_xgb_topk_monthly_strategy(
    full_df: pd.DataFrame,
    tickers: List[str],
    features: List[str],
    initial_amount: float,
    test_start_date: str,
    test_end_date: str,
    train_end_date: str,
    train_start_date: Optional[str] = None,
    top_k: int = 4,
    transaction_cost: float = 0.001,
    target_horizon: int = 21,
    cash_buffer: float = 0.15,
    spy_trend_filter: bool = False,
    walk_forward: bool = True,
) -> StrategyResult:
    target_col = f"future_return_{target_horizon}"

    df = full_df.copy()
    df = normalize_date_column(df, "date")

    # IMPORTANT:
    # full_df may include SPY so we can calculate/use spy_trend features.
    # But the model should train, score, rank, and trade only the tradable tickers.
    model_df = df[df["tic"].isin(tickers)].copy()

    test_mask = (
        (model_df["date"] >= pd.Timestamp(test_start_date)) &
        (model_df["date"] < pd.Timestamp(test_end_date))
    )

    # For prediction, we need features and close, but not future_return.
    test_df = model_df.loc[test_mask].dropna(subset=features + ["close"]).copy()

    price_matrix = build_price_matrix(test_df, tickers)
    dates = list(price_matrix.index.unique())
    signal_dates = get_monthly_signal_dates(dates)

    cash = float(initial_amount)
    shares = np.zeros(len(tickers), dtype=np.float64)

    portfolio_values = []
    history_rows = []
    selection_rows = []
    scored_rows = []

    static_model = None

    if not walk_forward:
        train_mask = model_df["date"] < pd.Timestamp(train_end_date)

        if train_start_date is not None:
            train_mask &= model_df["date"] >= pd.Timestamp(train_start_date)

        static_model = train_model(model_df.loc[train_mask], features, target_col)

        if static_model is None:
            raise ValueError("Not enough clean training rows for the static model.")

    pending_weights_by_execution_date: Dict[pd.Timestamp, np.ndarray] = {}

    for signal_date in signal_dates:
        execution_date = next_trading_date(dates, signal_date)

        if execution_date is None:
            continue

        signal_day = (
            test_df[test_df["date"] == signal_date]
            .dropna(subset=features)
            .copy()
        )

        if signal_day.empty:
            continue

        if spy_trend_filter:
            # Since SPY is excluded from model_df when not tradable,
            # read the market regime from the original full dataframe.
            regime_row = df[df["date"] == signal_date]

            if "spy_above_200sma" in regime_row.columns and not regime_row.empty:
                spy_ok = bool(regime_row["spy_above_200sma"].dropna().max() > 0.5)
            else:
                spy_ok = False

            if not spy_ok:
                pending_weights_by_execution_date[execution_date] = np.zeros(
                    len(tickers),
                    dtype=np.float64,
                )

                selection_rows.append({
                    "signal_date": signal_date,
                    "execution_date": execution_date,
                    "selected": "CASH_SPY_FILTER",
                })

                continue

        if walk_forward:
            train_mask = model_df["date"] < signal_date

            if train_start_date is not None:
                train_mask &= model_df["date"] >= pd.Timestamp(train_start_date)

            model = train_model(model_df.loc[train_mask], features, target_col)

            if model is None:
                continue

        else:
            model = static_model

        signal_day["score"] = model.predict(signal_day[features])
        signal_day = signal_day.sort_values("score", ascending=False)

        selected = signal_day.head(top_k)["tic"].tolist()

        # Safety check: SPY should not be selected unless it is explicitly in tickers.
        invalid_selected = [ticker for ticker in selected if ticker not in tickers]

        if invalid_selected:
            raise RuntimeError(
                f"Model selected non-tradable tickers: {invalid_selected}. "
                f"Tradable universe is: {tickers}"
            )

        exec_px = price_matrix.loc[execution_date].astype(float).values

        current_values = current_position_values(shares, exec_px)
        current_portfolio_value = cash + float(np.sum(current_values))

        target_weights = make_target_weights(
            selected=selected,
            tickers=tickers,
            px=exec_px,
            invested_fraction=max(0.0, 1.0 - cash_buffer),
            portfolio_value=current_portfolio_value,
            min_dollars_per_position=MIN_DOLLARS_PER_POSITION,
            allow_fractional_shares=ALLOW_FRACTIONAL_SHARES,
        )

        pending_weights_by_execution_date[execution_date] = target_weights

        for rank, (_, row) in enumerate(signal_day.iterrows(), start=1):
            scored_rows.append({
                "signal_date": signal_date,
                "tic": row["tic"],
                "score": row["score"],
                "rank": rank,
                target_col: row.get(target_col, np.nan),
            })

        selection_rows.append({
            "signal_date": signal_date,
            "execution_date": execution_date,
            "selected": ",".join(selected),
        })

    for dt in dates:
        px = price_matrix.loc[dt].astype(float).values

        if dt in pending_weights_by_execution_date:
            cash, shares, cost = rebalance_to_weights(
                cash=cash,
                shares=shares,
                px=px,
                target_weights=pending_weights_by_execution_date[dt],
                transaction_cost=transaction_cost,
            )
        else:
            cost = 0.0

        values = current_position_values(shares, px)
        portfolio_value = cash + float(np.sum(values))

        portfolio_values.append(portfolio_value)

        history_rows.append({
            "date": dt,
            "portfolio_value": portfolio_value,
            "cash": cash,
            "transaction_cost_paid": cost,
        })

    stats = compute_stats(portfolio_values, initial_amount)

    return StrategyResult(
        stats=stats,
        history=pd.DataFrame(history_rows),
        selections=pd.DataFrame(selection_rows),
        scored=pd.DataFrame(scored_rows),
    )


def run_spy_benchmark(
    test_df: pd.DataFrame,
    initial_amount: float,
) -> StrategyResult:
    spy = (
        test_df[test_df["tic"] == "SPY"]
        .dropna(subset=["close"])
        .sort_values("date")
        .copy()
    )

    if spy.empty:
        raise ValueError("SPY benchmark failed: no valid close prices.")

    prices = spy["close"].astype(float).values
    shares = initial_amount / prices[0]
    values = shares * prices

    history = pd.DataFrame({
        "date": spy["date"].values,
        "portfolio_value": values,
    })

    return StrategyResult(
        stats=compute_stats(values, initial_amount),
        history=history,
        selections=pd.DataFrame(),
    )


def run_buy_hold_equal_weight_benchmark(
    test_df: pd.DataFrame,
    tickers: List[str],
    initial_amount: float,
) -> StrategyResult:
    prices = build_price_matrix(test_df, tickers).dropna(how="all")

    start_prices = prices.iloc[0].astype(float).values
    valid = np.isfinite(start_prices) & (start_prices > 0)

    if not np.any(valid):
        raise ValueError("Equal-weight benchmark failed: no valid starting prices.")

    shares = np.zeros(len(tickers), dtype=float)
    equal_weight = 1.0 / np.sum(valid)

    shares[valid] = (initial_amount * equal_weight) / start_prices[valid]

    values = []

    for _, row in prices.iterrows():
        px = row.astype(float).values
        values.append(float(np.sum(current_position_values(shares, px))))

    history = pd.DataFrame({
        "date": prices.index,
        "portfolio_value": values,
    })

    return StrategyResult(
        stats=compute_stats(values, initial_amount),
        history=history,
        selections=pd.DataFrame(),
    )


def run_monthly_equal_weight_benchmark(
    test_df: pd.DataFrame,
    tickers: List[str],
    initial_amount: float,
    transaction_cost: float = 0.001,
    cash_buffer: float = 0.0,
) -> StrategyResult:
    prices = build_price_matrix(test_df, tickers).dropna(how="all")
    dates = list(prices.index.unique())

    signal_dates = get_monthly_signal_dates(dates)

    execution_dates = {
        next_trading_date(dates, signal_date)
        for signal_date in signal_dates
    }

    execution_dates.discard(None)

    cash = initial_amount
    shares = np.zeros(len(tickers), dtype=float)

    values = []
    history_rows = []

    for dt in dates:
        px = prices.loc[dt].astype(float).values

        if dt in execution_dates:
            valid = np.isfinite(px) & (px > 0)

            weights = np.zeros(len(tickers), dtype=float)

            if np.any(valid):
                weights[valid] = (1.0 - cash_buffer) / np.sum(valid)

            cash, shares, cost = rebalance_to_weights(
                cash=cash,
                shares=shares,
                px=px,
                target_weights=weights,
                transaction_cost=transaction_cost,
            )
        else:
            cost = 0.0

        portfolio_value = cash + float(np.sum(current_position_values(shares, px)))

        values.append(portfolio_value)

        history_rows.append({
            "date": dt,
            "portfolio_value": portfolio_value,
            "transaction_cost_paid": cost,
        })

    history = pd.DataFrame(history_rows)

    return StrategyResult(
        stats=compute_stats(values, initial_amount),
        history=history,
        selections=pd.DataFrame(),
    )


def run_momentum_topk_benchmark(
    test_df: pd.DataFrame,
    tickers: List[str],
    initial_amount: float,
    top_k: int = 4,
    lookback: int = 126,
    transaction_cost: float = 0.001,
    cash_buffer: float = 0.15,
) -> StrategyResult:
    prices = build_price_matrix(test_df, tickers).dropna(how="all")
    dates = list(prices.index.unique())
    signal_dates = get_monthly_signal_dates(dates)

    cash = initial_amount
    shares = np.zeros(len(tickers), dtype=float)

    pending_weights_by_execution_date: Dict[pd.Timestamp, np.ndarray] = {}
    selection_rows = []

    for signal_date in signal_dates:
        execution_date = next_trading_date(dates, signal_date)

        if execution_date is None:
            continue

        signal_idx = prices.index.get_loc(signal_date)

        if isinstance(signal_idx, slice) or signal_idx < lookback:
            continue

        momentum = prices.iloc[signal_idx] / prices.iloc[signal_idx - lookback] - 1.0

        selected = (
            momentum.replace([np.inf, -np.inf], np.nan)
            .dropna()
            .sort_values(ascending=False)
            .head(top_k)
            .index
            .tolist()
        )

        exec_px = prices.loc[execution_date].astype(float).values
        
        current_values = current_position_values(shares, exec_px)
        current_portfolio_value = cash + float(np.sum(current_values))

        pending_weights_by_execution_date[execution_date] = make_target_weights(
            selected=selected,
            tickers=tickers,
            px=exec_px,
            invested_fraction=max(0.0, 1.0 - cash_buffer),
            portfolio_value=current_portfolio_value,
            min_dollars_per_position=MIN_DOLLARS_PER_POSITION,
            allow_fractional_shares=ALLOW_FRACTIONAL_SHARES,
        )

    values = []
    history_rows = []

    for dt in dates:
        px = prices.loc[dt].astype(float).values

        if dt in pending_weights_by_execution_date:
            cash, shares, cost = rebalance_to_weights(
                cash=cash,
                shares=shares,
                px=px,
                target_weights=pending_weights_by_execution_date[dt],
                transaction_cost=transaction_cost,
            )
        else:
            cost = 0.0

        portfolio_value = cash + float(np.sum(current_position_values(shares, px)))

        values.append(portfolio_value)

        history_rows.append({
            "date": dt,
            "portfolio_value": portfolio_value,
            "transaction_cost_paid": cost,
        })

    return StrategyResult(
        stats=compute_stats(values, initial_amount),
        history=pd.DataFrame(history_rows),
        selections=pd.DataFrame(selection_rows),
    )


def print_ranking_diagnostics(
    scored: pd.DataFrame,
    target_col: str,
) -> None:
    if scored is None or scored.empty or target_col not in scored.columns:
        print("\nNo ranking diagnostics available.")
        return

    clean = scored.dropna(subset=["rank", target_col]).copy()

    if clean.empty:
        print("\nNo realized future returns available for ranking diagnostics.")
        return

    clean["bucket"] = np.where(
        clean["rank"] <= TOP_K,
        f"Top {TOP_K}",
        f"Not Top {TOP_K}",
    )

    summary = clean.groupby("bucket")[target_col].agg(["count", "mean", "median"])

    print("\n===== RANKING DIAGNOSTICS =====")
    print(summary.to_string(float_format=lambda x: f"{x:,.4f}"))

    try:
        monthly = clean.groupby("signal_date", group_keys=False).apply(
            lambda x: x.loc[x["rank"] <= TOP_K, target_col].mean()
            - x.loc[x["rank"] > TOP_K, target_col].mean(),
            include_groups=False,
        )
    except TypeError:
        # Older pandas versions do not support include_groups.
        monthly = clean.groupby("signal_date", group_keys=False).apply(
            lambda x: x.loc[x["rank"] <= TOP_K, target_col].mean()
            - x.loc[x["rank"] > TOP_K, target_col].mean()
        )

    print("\nMonthly top-minus-rest spread:")
    print(monthly.describe().to_string(float_format=lambda x: f"{x:,.4f}"))

    hit_rate = (monthly > 0).mean()

    print(f"\nMonthly ranking hit rate: {hit_rate:.2%}")


def align_history_for_plot(results: Dict[str, StrategyResult]) -> pd.DataFrame:
    frames = []

    for name, result in results.items():
        hist = result.history[["date", "portfolio_value"]].copy()
        hist["strategy"] = name
        frames.append(hist)

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    full_df = prepare_full_df()
    full_df = normalize_date_column(full_df, "date")

    test_df = full_df[
        (full_df["date"] >= pd.Timestamp(TEST_START_DATE)) &
        (full_df["date"] < pd.Timestamp(TEST_END_DATE))
    ].dropna(subset=FEATURES + ["close"]).copy()

    trade_test_df = test_df[test_df["tic"].isin(TRADE_LIST)].copy()

    if test_df.empty:
        raise ValueError(
            "No usable test rows after feature cleaning. "
            "Check dates and data availability."
        )

    if trade_test_df.empty:
        raise ValueError(
            "No usable tradable test rows after feature cleaning. "
            "Check TRADE_LIST and data availability."
        )

    xgb_result = run_xgb_topk_monthly_strategy(
        full_df=full_df,
        tickers=TRADE_LIST,
        features=FEATURES,
        initial_amount=INITIAL_AMOUNT,
        test_start_date=TEST_START_DATE,
        test_end_date=TEST_END_DATE,
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
        f"XGBoost Top-{TOP_K} Monthly": xgb_result,
        "Buy & Hold Equal Weight": run_buy_hold_equal_weight_benchmark(
            test_df=trade_test_df,
            tickers=TRADE_LIST,
            initial_amount=INITIAL_AMOUNT,
        ),
        "Monthly Equal Weight": run_monthly_equal_weight_benchmark(
            test_df=trade_test_df,
            tickers=TRADE_LIST,
            initial_amount=INITIAL_AMOUNT,
            transaction_cost=TRANSACTION_COST,
            cash_buffer=0.0,
        ),
        f"Momentum Top-{TOP_K} Monthly": run_momentum_topk_benchmark(
            test_df=trade_test_df,
            tickers=TRADE_LIST,
            initial_amount=INITIAL_AMOUNT,
            top_k=TOP_K,
            lookback=MOMENTUM_LOOKBACK,
            transaction_cost=TRANSACTION_COST,
            cash_buffer=CASH_BUFFER,
        ),
        "SPY Buy & Hold": run_spy_benchmark(
            test_df=test_df,
            initial_amount=INITIAL_AMOUNT,
        ),
    }

    for name, result in results.items():
        print_stats(name, result.stats)

    print_ranking_diagnostics(
        scored=xgb_result.scored,
        target_col=f"future_return_{TARGET_HORIZON}",
    )

    print("\n===== RECENT XGBOOST SELECTIONS =====")

    if not xgb_result.selections.empty:
        print(xgb_result.selections.tail(12).to_string(index=False))
    else:
        print("No XGBoost selections were generated.")

    plot_df = align_history_for_plot(results)

    plt.figure(figsize=(12, 6))

    for name, grp in plot_df.groupby("strategy"):
        plt.plot(grp["date"], grp["portfolio_value"], label=name)

    plt.title("Equity Curve Comparison")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("equity_curve_comparison.png", dpi=150)
    print("Saved chart to equity_curve_comparison.png")


if __name__ == "__main__":
    main()