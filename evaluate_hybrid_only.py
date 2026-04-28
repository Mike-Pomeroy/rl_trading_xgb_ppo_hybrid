import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config import (
    TEST_BASKET_DATA_PATH,
    INITIAL_AMOUNT,
    TRANSACTION_COST,
    PPO_MODEL_PATH,
    PPO_VECNORM_PATH,
)
from trading_env_xgb_top4 import XGBTop4TradingEnv


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


def make_env():
    return XGBTop4TradingEnv(
        basket_csv_path=TEST_BASKET_DATA_PATH,
        initial_amount=INITIAL_AMOUNT,
        transaction_cost=TRANSACTION_COST,
        include_cash=True,
    )


print("Creating env...", flush=True)
env = DummyVecEnv([make_env])

print("Loading VecNormalize...", flush=True)
env = VecNormalize.load(PPO_VECNORM_PATH, env)
env.training = False
env.norm_reward = False

print("Loading PPO model...", flush=True)
model = PPO.load(PPO_MODEL_PATH, device="cpu")

print("Running hybrid backtest...", flush=True)
obs = env.reset()
portfolio_values = []
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    portfolio_values.append(info[0]["portfolio_value"])

hybrid_stats = compute_stats(portfolio_values, INITIAL_AMOUNT)
print_stats("XGB + PPO HYBRID", hybrid_stats)

plt.figure(figsize=(12, 6))
plt.plot(hybrid_stats["portfolio_values"], label="XGB + PPO Hybrid")
plt.title("XGB + PPO Hybrid Equity Curve")
plt.xlabel("Rebalance Step")
plt.ylabel("Portfolio Value")
plt.legend()
plt.tight_layout()

from pathlib import Path
Path("results").mkdir(parents=True, exist_ok=True)


plt.savefig("results/hybrid_only_equity_curve.png")
print("Saved chart to results/hybrid_only_equity_curve.png")