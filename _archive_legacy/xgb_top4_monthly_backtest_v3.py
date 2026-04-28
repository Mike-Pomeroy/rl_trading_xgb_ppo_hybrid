import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

from data_module import prepare_data, INDICATORS

# ----------------------------
# CONFIG
# ----------------------------
TICKER_LIST = [
    "AAPL", "MSFT", "SPY", "GOOGL", "AMZN",
    "META", "TSLA", "NVDA", "JPM", "JNJ"
]

FEATURES = INDICATORS + [
    "volatility_30",
    "return_5",
    "return_10",
    "price_vs_sma30",
    "spy_trend",
]

INITIAL_AMOUNT = 100000
TOP_K = 4
TRANSACTION_COST = 0.001
TARGET_HORIZON = 30

# best practical settings so far
CASH_BUFFER = 0.15
SPY_TREND_FILTER = False

TRAIN_END_DATE = "2022-01-01"
TEST_START_DATE = "2022-01-01"
TEST_END_DATE = "2024-01-01"


def compute_stats(portfolio_values, initial_amount):
    portfolio_values = np.array(portfolio_values, dtype=np.float64)

    if len(portfolio_values) == 0:
        raise ValueError("No portfolio values available for stats.")

    returns = portfolio_values[1:] / (portfolio_values[:-1] + 1e-8) - 1
    sharpe = (
        (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252)
        if len(returns) > 1 else 0.0
    )

    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / (peak + 1e-8)

    return {
        "final_portfolio": portfolio_values[-1],
        "total_return_pct": (portfolio_values[-1] / initial_amount - 1) * 100,
        "sharpe": sharpe,
        "max_drawdown_pct": np.max(drawdown) * 100,
        "portfolio_values": portfolio_values,
    }


def print_stats(name, stats):
    print(f"\n===== {name} =====")
    print(f"Final Portfolio: ${stats['final_portfolio']:,.2f}")
    print(f"Total Return: {stats['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {stats['sharpe']:.3f}")
    print(f"Max Drawdown: {stats['max_drawdown_pct']:.2f}%")


def build_price_matrix(test_df, tickers):
    dates = sorted(test_df["date"].unique())
    pivot = test_df.pivot(index="date", columns="tic", values="close")
    pivot = pivot.reindex(index=dates, columns=tickers)
    pivot = pivot.sort_index()
    pivot = pivot.ffill().bfill()
    pivot = pivot.replace([np.inf, -np.inf], np.nan)
    pivot.index = pd.to_datetime(pivot.index).normalize()
    return pivot


def run_spy_benchmark(test_df, initial_amount):
    spy_df = test_df[test_df["tic"] == "SPY"].sort_values("date").copy()
    spy_df["close"] = pd.to_numeric(spy_df["close"], errors="coerce")
    spy_df = spy_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])

    if spy_df.empty:
        raise ValueError("SPY benchmark failed: no valid close prices.")

    prices = spy_df["close"].values.astype(np.float64)
    shares = initial_amount / prices[0]
    portfolio_values = shares * prices

    return compute_stats(portfolio_values, initial_amount)


def run_equal_weight_benchmark(test_df, tickers, initial_amount):
    prices = build_price_matrix(test_df, tickers)

    start_idx = None
    for i in range(len(prices)):
        px = prices.iloc[i].astype(np.float64).values
        valid = np.isfinite(px) & (px > 0)
        if np.any(valid):
            start_idx = i
            break

    if start_idx is None:
        raise ValueError("Equal-weight benchmark failed: no valid starting prices.")

    start_prices = prices.iloc[start_idx].astype(np.float64).values
    valid_start = np.isfinite(start_prices) & (start_prices > 0)
    valid_indices = np.where(valid_start)[0]

    shares = np.zeros(len(tickers), dtype=np.float64)
    equal_weight = 1.0 / len(valid_indices)

    for idx in valid_indices:
        shares[idx] = (initial_amount * equal_weight) / start_prices[idx]

    portfolio_values = []

    for i in range(start_idx, len(prices)):
        px = prices.iloc[i].astype(np.float64).values
        value = 0.0
        for j in range(len(tickers)):
            if np.isfinite(px[j]) and px[j] > 0:
                value += shares[j] * px[j]
        portfolio_values.append(value)

    return compute_stats(portfolio_values, initial_amount)


def get_monthly_rebalance_dates(dates):
    dts = pd.to_datetime(pd.Series(dates)).dt.normalize()
    monthly = dts.groupby(dts.dt.to_period("M")).min()
    return set(pd.to_datetime(monthly))


def prepare_full_df():
    df = prepare_data()
    df = df[df["tic"].isin(TICKER_LIST)].copy()
    df = df.sort_values(["tic", "date"]).reset_index(drop=True)

    for horizon in [30, 60]:
        df[f"future_return_{horizon}"] = (
            df.groupby("tic")["close"].shift(-horizon) / df["close"] - 1.0
        )

    spy_df = df[df["tic"] == "SPY"][["date", "close"]].copy().sort_values("date")
    spy_df["spy_sma_200"] = spy_df["close"].rolling(200).mean()
    spy_df["spy_above_200sma"] = (spy_df["close"] > spy_df["spy_sma_200"]).astype(float)

    df = df.merge(
        spy_df[["date", "spy_sma_200", "spy_above_200sma"]],
        on="date",
        how="left"
    )

    needed_cols = FEATURES + [f"future_return_{TARGET_HORIZON}", "date", "tic", "close"]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=needed_cols).copy()

    return df


def run_xgb_topk_monthly_strategy(
    test_df,
    full_df,
    tickers,
    features,
    initial_amount,
    top_k=4,
    transaction_cost=0.001,
    target_horizon=30,
    cash_buffer=0.15,
    spy_trend_filter=False,
):
    train_df = full_df[full_df["date"] < TRAIN_END_DATE].copy()
    X_train = train_df[features]
    y_train = train_df[f"future_return_{target_horizon}"]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_train, y_train)

    scored = test_df.copy()
    scored["date"] = pd.to_datetime(scored["date"]).dt.normalize()
    scored["score"] = model.predict(scored[features])

    price_matrix = build_price_matrix(test_df, tickers)
    dates = list(price_matrix.index.unique())
    rebalance_dates = get_monthly_rebalance_dates(dates)

    spy_trend_map = (
        test_df[["date", "spy_above_200sma"]]
        .drop_duplicates("date")
        .copy()
    )
    spy_trend_map["date"] = pd.to_datetime(spy_trend_map["date"]).dt.normalize()
    spy_trend_map = spy_trend_map.set_index("date")["spy_above_200sma"].to_dict()

    cash = initial_amount
    shares = np.zeros(len(tickers), dtype=np.float64)
    portfolio_values = []

    first_trade_done = False

    for dt in dates:
        px_series = price_matrix.loc[dt].astype(np.float64)
        px = px_series.values
        valid_px = np.isfinite(px) & (px > 0)

        current_values = np.zeros(len(tickers), dtype=np.float64)
        for j in range(len(tickers)):
            if valid_px[j]:
                current_values[j] = shares[j] * px[j]

        portfolio_value = cash + np.sum(current_values)
        rebalance_now = (not first_trade_done) or (dt in rebalance_dates)

        if rebalance_now:
            trend_ok = True
            if spy_trend_filter:
                trend_ok = bool(spy_trend_map.get(dt, 0.0) > 0.5)

            if trend_ok:
                day_df = scored[scored["date"] == dt].copy()

                if not day_df.empty:
                    day_df = day_df.sort_values("score", ascending=False)
                    selected = day_df.head(top_k)["tic"].tolist()

                    selected_indices = [
                        i for i, t in enumerate(tickers)
                        if t in selected and valid_px[i]
                    ]

                    if len(selected_indices) > 0:
                        target_weights = np.zeros(len(tickers), dtype=np.float64)

                        invested_fraction = max(0.0, 1.0 - cash_buffer)
                        ew = invested_fraction / len(selected_indices)

                        for idx in selected_indices:
                            target_weights[idx] = ew

                        current_weights = current_values / (portfolio_value + 1e-8)
                        current_cash_weight = cash / (portfolio_value + 1e-8)

                        turnover = np.sum(np.abs(target_weights - current_weights)) + abs(
                            cash_buffer - current_cash_weight
                        )

                        cost = transaction_cost * turnover * portfolio_value
                        investable_value = max(portfolio_value - cost, 0.0)

                        new_shares = np.zeros(len(tickers), dtype=np.float64)
                        target_cash = investable_value * cash_buffer

                        for idx in selected_indices:
                            new_shares[idx] = (investable_value * target_weights[idx]) / px[idx]

                        shares = new_shares
                        cash = target_cash
                        first_trade_done = True
            else:
                turnover = np.sum(current_values) / (portfolio_value + 1e-8)
                cost = transaction_cost * turnover * portfolio_value
                cash = max(portfolio_value - cost, 0.0)
                shares = np.zeros(len(tickers), dtype=np.float64)
                first_trade_done = True

        current_values = np.zeros(len(tickers), dtype=np.float64)
        for j in range(len(tickers)):
            if valid_px[j]:
                current_values[j] = shares[j] * px[j]

        portfolio_value = cash + np.sum(current_values)
        portfolio_values.append(portfolio_value)

    return compute_stats(portfolio_values, initial_amount)


# ----------------------------
# PREP DATA
# ----------------------------
df = prepare_full_df()
test_df = df[
    (df["date"] >= TEST_START_DATE) &
    (df["date"] < TEST_END_DATE)
].copy()

# ----------------------------
# RUN STRATEGIES
# ----------------------------
xgb_stats = run_xgb_topk_monthly_strategy(
    test_df=test_df,
    full_df=df,
    tickers=TICKER_LIST,
    features=FEATURES,
    initial_amount=INITIAL_AMOUNT,
    top_k=TOP_K,
    transaction_cost=TRANSACTION_COST,
    target_horizon=TARGET_HORIZON,
    cash_buffer=CASH_BUFFER,
    spy_trend_filter=SPY_TREND_FILTER,
)

equal_weight_stats = run_equal_weight_benchmark(
    test_df=test_df,
    tickers=TICKER_LIST,
    initial_amount=INITIAL_AMOUNT,
)

spy_stats = run_spy_benchmark(
    test_df=test_df,
    initial_amount=INITIAL_AMOUNT,
)

label = (
    f"XGBOOST TOP-{TOP_K} MONTHLY "
    f"(target={TARGET_HORIZON}d, cost={TRANSACTION_COST}, "
    f"cash_buffer={CASH_BUFFER}, spy_filter={SPY_TREND_FILTER})"
)

print_stats(label, xgb_stats)
print_stats("EQUAL-WEIGHT BENCHMARK", equal_weight_stats)
print_stats("SPY BUY-AND-HOLD BENCHMARK", spy_stats)

plt.figure(figsize=(12, 6))
plt.plot(xgb_stats["portfolio_values"], label="XGBoost Monthly")
plt.plot(equal_weight_stats["portfolio_values"], label="Equal Weight")
plt.plot(spy_stats["portfolio_values"], label="SPY Buy & Hold")
plt.title("Equity Curve Comparison")
plt.xlabel("Time Step")
plt.ylabel("Portfolio Value")
plt.legend()
plt.tight_layout()
plt.show()