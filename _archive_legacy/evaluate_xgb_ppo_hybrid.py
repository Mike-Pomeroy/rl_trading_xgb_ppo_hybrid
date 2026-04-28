import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


from config import (
    TICKER_LIST,
    INDICATOR_COLUMNS,
    TOP_K,
    TARGET_HORIZON,
    TEST_START_DATE,
    INITIAL_AMOUNT,
    TRANSACTION_COST,
    XGB_PARAMS,
    TEST_BASKET_DATA_PATH,
    PPO_MODEL_PATH,
    PPO_VECNORM_PATH,
)
from data_module import prepare_data
from trading_env_xgb_top4 import XGBTop4TradingEnv


print("Preloading PPO model", flush=True)
ppo_model = PPO.load(PPO_MODEL_PATH, device="cpu")
print("Preloaded PPO model", flush=True)


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


def run_spy_benchmark(test_df, initial_amount):
    spy_df = test_df[test_df["tic"] == "SPY"].sort_values("date").copy()
    spy_df["close"] = pd.to_numeric(spy_df["close"], errors="coerce")
    spy_df = spy_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])

    prices = spy_df["close"].values.astype(np.float64)
    shares = initial_amount / prices[0]
    portfolio_values = shares * prices

    return compute_stats(portfolio_values, initial_amount)


def run_xgb_equal_weight_from_baskets(basket_csv_path, initial_amount, transaction_cost):
    df = pd.read_csv(basket_csv_path)
    df["rebalance_date"] = pd.to_datetime(df["rebalance_date"], utc=True)
    df = df.sort_values(["rebalance_date", "slot"]).reset_index(drop=True)

    dates = sorted(df["rebalance_date"].unique())
    portfolio_value = initial_amount
    portfolio_values = []

    prev_weights = np.zeros(5, dtype=np.float64)
    prev_weights[4] = 1.0  # start fully cash

    for i in range(len(dates) - 1):
        dt = dates[i]
        next_dt = dates[i + 1]

        cur = df[df["rebalance_date"] == dt].sort_values("slot")
        nxt = df[df["rebalance_date"] == next_dt].sort_values("slot")

        cur_prices = cur["close"].values.astype(np.float64)
        next_prices = nxt["close"].values.astype(np.float64)

        asset_returns = next_prices / (cur_prices + 1e-8) - 1.0

        weights = np.array([0.25, 0.25, 0.25, 0.25, 0.0], dtype=np.float64)
        turnover = np.sum(np.abs(weights - prev_weights))
        cost = transaction_cost * turnover
        gross_return = float(np.mean(asset_returns))
        net_return = gross_return - cost

        portfolio_value = portfolio_value * (1.0 + net_return)
        portfolio_values.append(portfolio_value)
        prev_weights = weights.copy()

    return compute_stats(portfolio_values, initial_amount)


def run_xgb_cash_buffer_from_baskets(basket_csv_path, initial_amount, transaction_cost, cash_buffer=0.15):
    df = pd.read_csv(basket_csv_path)
    df["rebalance_date"] = pd.to_datetime(df["rebalance_date"], utc=True)
    df = df.sort_values(["rebalance_date", "slot"]).reset_index(drop=True)

    dates = sorted(df["rebalance_date"].unique())
    portfolio_value = initial_amount
    portfolio_values = []

    invested = 1.0 - cash_buffer
    prev_weights = np.zeros(5, dtype=np.float64)
    prev_weights[4] = 1.0

    for i in range(len(dates) - 1):
        dt = dates[i]
        next_dt = dates[i + 1]

        cur = df[df["rebalance_date"] == dt].sort_values("slot")
        nxt = df[df["rebalance_date"] == next_dt].sort_values("slot")

        cur_prices = cur["close"].values.astype(np.float64)
        next_prices = nxt["close"].values.astype(np.float64)

        asset_returns = next_prices / (cur_prices + 1e-8) - 1.0

        weights = np.array(
            [invested / 4, invested / 4, invested / 4, invested / 4, cash_buffer],
            dtype=np.float64,
        )

        turnover = np.sum(np.abs(weights - prev_weights))
        cost = transaction_cost * turnover
        gross_return = float(np.dot(weights[:4], asset_returns))
        net_return = gross_return - cost

        portfolio_value = portfolio_value * (1.0 + net_return)
        portfolio_values.append(portfolio_value)
        prev_weights = weights.copy()

    return compute_stats(portfolio_values, initial_amount)

def run_xgb_ppo_hybrid(basket_csv_path, initial_amount, transaction_cost):
    print("hybrid: loading PPO model", flush=True)
    model = ppo_model
    print("hybrid: loaded PPO model", flush=True)

    def make_env():
        return XGBTop4TradingEnv(
            basket_csv_path=basket_csv_path,
            initial_amount=initial_amount,
            transaction_cost=transaction_cost,
            include_cash=True,
        )

    print("hybrid: creating DummyVecEnv", flush=True)
    env = DummyVecEnv([make_env])

    print("hybrid: loading VecNormalize", flush=True)
    env = VecNormalize.load(PPO_VECNORM_PATH, env)
    print("hybrid: loaded VecNormalize", flush=True)

    env.training = False
    env.norm_reward = False

    print("hybrid: resetting env", flush=True)
    obs = env.reset()
    print("hybrid: env reset complete", flush=True)

    portfolio_values = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        portfolio_values.append(info[0]["portfolio_value"])

    return compute_stats(portfolio_values, initial_amount)


# full source data for SPY benchmark

print("Starting prepare_data()", flush=True)
df = prepare_data()
print("Finished prepare_data()", flush=True)



df = df[df["tic"].isin(TICKER_LIST)].copy()
df = normalize_dates(df)

test_cutoff = pd.Timestamp(TEST_START_DATE, tz="UTC")
test_df = df[df["date"] >= test_cutoff].copy()

print("Running hybrid_stats", flush=True)
hybrid_stats = run_xgb_ppo_hybrid(
    basket_csv_path=TEST_BASKET_DATA_PATH,
    initial_amount=INITIAL_AMOUNT,
    transaction_cost=TRANSACTION_COST,
)
print("Finished hybrid_stats", flush=True)

print("Running xgb_eq_stats", flush=True)

xgb_eq_stats = run_xgb_equal_weight_from_baskets(
    basket_csv_path=TEST_BASKET_DATA_PATH,
    initial_amount=INITIAL_AMOUNT,
    transaction_cost=TRANSACTION_COST,
)
print("Finished xgb_eq_stats", flush=True)

print("Running xgb_cash_stats", flush=True)

xgb_cash_stats = run_xgb_cash_buffer_from_baskets(
    basket_csv_path=TEST_BASKET_DATA_PATH,
    initial_amount=INITIAL_AMOUNT,
    transaction_cost=TRANSACTION_COST,
    cash_buffer=0.15,
)
print("Finished xgb_cash_stats", flush=True)

print("Running spy_stats", flush=True)
spy_stats = run_spy_benchmark(test_df, INITIAL_AMOUNT)
print("Finished spy_stats", flush=True)

print_stats("XGB + PPO HYBRID", hybrid_stats)
print_stats("XGB TOP-4 EQUAL WEIGHT", xgb_eq_stats)
print_stats("XGB TOP-4 + 15% CASH", xgb_cash_stats)
print_stats("SPY BUY-AND-HOLD BENCHMARK", spy_stats)

"""
plt.figure(figsize=(12, 6))
plt.plot(hybrid_stats["portfolio_values"], label="XGB + PPO Hybrid")
plt.plot(xgb_eq_stats["portfolio_values"], label="XGB Top-4 Equal Weight")
plt.plot(xgb_cash_stats["portfolio_values"], label="XGB Top-4 + 15% Cash")
plt.plot(spy_stats["portfolio_values"], label="SPY Buy & Hold")
plt.title("Hybrid Strategy Comparison")
plt.xlabel("Rebalance Step / Time")
plt.ylabel("Portfolio Value")
plt.legend()
plt.tight_layout()
plt.savefig("results/hybrid_strategy_comparison.png")
print("Saved chart to results/hybrid_strategy_comparison.png")
"""