import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBRegressor

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from trading_env import TradingEnv
from data_module import prepare_data, INDICATORS

# ----------------------------
# CONFIG
# ----------------------------
TICKER_LIST = [
    "AAPL", "MSFT", "SPY", "GOOGL", "AMZN",
    "META", "TSLA", "NVDA", "JPM", "JNJ"
]

TECH_FEATURES = INDICATORS + [
    "volatility_30",
    "return_5",
    "return_10",
    "price_vs_sma30",
    "spy_trend",
]

INITIAL_AMOUNT = 100000
TOP_K = 4
TARGET_HORIZON = 30
TRANSACTION_COST = 0.001

TRAIN_START = "2019-01-01"
TEST_START = "2022-01-01"
TEST_END = "2022-04-01"

# Smoke-test settings
PPO_TIMESTEPS_PER_MONTH = 32
PPO_N_STEPS = 32
PPO_BATCH_SIZE = 16
PPO_N_EPOCHS = 2

# Train PPO only on recent history for the selected 4
PPO_HISTORY_LOOKBACK_DAYS = 252  # about 1 trading year


# ----------------------------
# HELPERS
# ----------------------------
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


def normalize_dates(df):
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True).dt.normalize()
    return out


def get_monthly_rebalance_dates(dates):
    dts = pd.to_datetime(pd.Series(sorted(pd.Series(dates).unique()))).dt.normalize()
    monthly = dts.groupby(dts.dt.to_period("M")).min()
    return list(pd.to_datetime(monthly))


def run_spy_benchmark(test_df, initial_amount):
    spy_df = test_df[test_df["tic"] == "SPY"].sort_values("date").copy()
    spy_df["close"] = pd.to_numeric(spy_df["close"], errors="coerce")
    spy_df = spy_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])

    prices = spy_df["close"].values.astype(np.float64)
    shares = initial_amount / prices[0]
    portfolio_values = shares * prices

    return compute_stats(portfolio_values, initial_amount)


def build_training_frame(df, horizon):
    out = df.copy().sort_values(["tic", "date"]).reset_index(drop=True)
    out[f"future_return_{horizon}"] = (
        out.groupby("tic")["close"].shift(-horizon) / out["close"] - 1.0
    )
    return out


def select_top_k_for_date(trainable_df, asof_date, top_k):
    feature_train = trainable_df[trainable_df["date"] < asof_date].copy()
    target_col = f"future_return_{TARGET_HORIZON}"

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )

    model.fit(feature_train[TECH_FEATURES], feature_train[target_col])

    day_df = trainable_df[trainable_df["date"] == asof_date].copy()
    day_df["score"] = model.predict(day_df[TECH_FEATURES])
    day_df = day_df.sort_values("score", ascending=False)

    selected = day_df.head(top_k)["tic"].tolist()
    return selected, day_df


def build_ppo_env_for_selected(df, selected_tickers, asof_date):
    lookback_start = asof_date - pd.Timedelta(days=PPO_HISTORY_LOOKBACK_DAYS)

    hist_df = df[
        (df["tic"].isin(selected_tickers)) &
        (df["date"] < asof_date) &
        (df["date"] >= lookback_start)
    ].copy()

    return TradingEnv(
        df=hist_df,
        ticker_list=selected_tickers,
        tech_indicator_list=TECH_FEATURES,
        initial_amount=INITIAL_AMOUNT,
    ), hist_df


def train_monthly_ppo_allocator(df, selected_tickers, asof_date):
    env_template, hist_df = build_ppo_env_for_selected(df, selected_tickers, asof_date)

    unique_dates = hist_df["date"].nunique()
    row_count = len(hist_df)

    print(
        f"PPO history window: rows={row_count}, unique_dates={unique_dates}, "
        f"tickers={selected_tickers}",
        flush=True,
    )

    # If not enough history, skip PPO and fall back to equal weight
    if unique_dates < 40:
        print("Not enough PPO history. Falling back to equal-weight.", flush=True)
        return None

    def make_env():
        env, _ = build_ppo_env_for_selected(df, selected_tickers, asof_date)
        return env

    env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        n_epochs=PPO_N_EPOCHS,
        gamma=0.99,
        verbose=1,
        device="cpu",
    )

    model.learn(total_timesteps=PPO_TIMESTEPS_PER_MONTH)
    return model


def build_single_observation(day_df, selected_tickers, portfolio_value):
    sub = day_df[day_df["tic"].isin(selected_tickers)].copy()
    sub = sub.set_index("tic").reindex(selected_tickers).reset_index()

    prices = sub["close"].values.astype(np.float32)
    techs = []
    for _, row in sub.iterrows():
        for f in TECH_FEATURES:
            techs.append(float(row[f]))

    holdings = np.zeros(len(selected_tickers), dtype=np.float32)

    obs = np.concatenate([
        np.array([portfolio_value / INITIAL_AMOUNT], dtype=np.float32),
        holdings,
        prices / 1000.0,
        np.clip(np.array(techs, dtype=np.float32), -5, 5),
    ]).astype(np.float32)

    return obs


def get_next_month_return(df, selected_tickers, current_date, next_date, weights):
    cur = df[(df["date"] == current_date) & (df["tic"].isin(selected_tickers))].copy()
    nxt = df[(df["date"] == next_date) & (df["tic"].isin(selected_tickers))].copy()

    cur = cur.set_index("tic").reindex(selected_tickers)
    nxt = nxt.set_index("tic").reindex(selected_tickers)

    cur_prices = cur["close"].values.astype(np.float64)
    nxt_prices = nxt["close"].values.astype(np.float64)

    asset_returns = nxt_prices / (cur_prices + 1e-8) - 1.0
    gross_return = float(np.dot(weights, asset_returns))
    return gross_return


# ----------------------------
# DATA PREP
# ----------------------------
print("Loading data...", flush=True)
df = prepare_data()
df = df[df["tic"].isin(TICKER_LIST)].copy()
df = normalize_dates(df)
df = build_training_frame(df, TARGET_HORIZON)

needed_cols = ["date", "tic", "close"] + TECH_FEATURES + [f"future_return_{TARGET_HORIZON}"]
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=needed_cols).copy()

test_start = pd.Timestamp(TEST_START, tz="UTC")
test_end = pd.Timestamp(TEST_END, tz="UTC")

test_dates_all = df[(df["date"] >= test_start) & (df["date"] < test_end)]["date"]
rebalance_dates = get_monthly_rebalance_dates(test_dates_all)

portfolio_value = INITIAL_AMOUNT
portfolio_values = [portfolio_value]

print(f"Running walk-forward months: {len(rebalance_dates)-1}", flush=True)

for i in range(len(rebalance_dates) - 1):
    current_date = rebalance_dates[i]
    next_date = rebalance_dates[i + 1]

    print(f"\nMonth {i+1}/{len(rebalance_dates)-1}: {current_date.date()}", flush=True)

    selected, scored_day = select_top_k_for_date(df, current_date, TOP_K)
    print("Selected:", selected, flush=True)

    print("Skipping PPO for smoke test. Using equal-weight.", flush=True)
    weights = np.ones(TOP_K, dtype=np.float64) / TOP_K


    """
    print("Training PPO allocator for this month...", flush=True)
    model = train_monthly_ppo_allocator(df, selected, current_date)
    print("Finished PPO training for this month.", flush=True)

    if model is None:
        weights = np.ones(TOP_K, dtype=np.float64) / TOP_K
        print("Using equal-weight fallback.", flush=True)
    else:
        obs = build_single_observation(scored_day, selected, portfolio_value)
        print("Built observation.", flush=True)

        action, _ = model.predict(obs, deterministic=True)
        print("Predicted action.", flush=True)

        a = np.array(action).reshape(-1)
        exp_a = np.exp(a - np.max(a))
        weights = exp_a / (np.sum(exp_a) + 1e-8)
        """

    monthly_return = get_next_month_return(df, selected, current_date, next_date, weights)
    turnover_cost = TRANSACTION_COST * np.sum(np.abs(weights))
    net_return = monthly_return - turnover_cost

    print("Computed month return.", flush=True)

    portfolio_value = portfolio_value * (1.0 + net_return)
    portfolio_values.append(portfolio_value)

    print(f"Weights: {weights}", flush=True)
    print(f"Monthly return: {monthly_return:.4f}", flush=True)
    print(f"Net return: {net_return:.4f}", flush=True)
    print(f"Portfolio: ${portfolio_value:,.2f}", flush=True)

hybrid_stats = compute_stats(portfolio_values, INITIAL_AMOUNT)

test_df = df[(df["date"] >= test_start) & (df["date"] < test_end)].copy()
spy_stats = run_spy_benchmark(test_df, INITIAL_AMOUNT)

print_stats("WALK-FORWARD XGB + MONTHLY PPO", hybrid_stats)
print_stats("SPY BUY-AND-HOLD BENCHMARK", spy_stats)

Path("results").mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(12, 6))
plt.plot(hybrid_stats["portfolio_values"], label="Walk-Forward XGB + PPO")
plt.plot(spy_stats["portfolio_values"], label="SPY Buy & Hold")
plt.title("Walk-Forward Hybrid Comparison")
plt.xlabel("Month")
plt.ylabel("Portfolio Value")
plt.legend()
plt.tight_layout()
plt.savefig("results/walkforward_xgb_ppo_monthly.png")
print("Saved chart to results/walkforward_xgb_ppo_monthly.png")