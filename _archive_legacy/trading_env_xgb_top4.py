import json
from pathlib import Path
from typing import List, Dict, Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class XGBTop4TradingEnv(gym.Env):
    """
    PPO environment for dynamic XGBoost-selected top-4 baskets.

    Each step corresponds to one rebalance date.
    At each step, the environment exposes 4 selected stock slots.
    The action is 5 logits:
        - slot_1
        - slot_2
        - slot_3
        - slot_4
        - cash

    The logits are softmaxed into portfolio weights.
    The environment then computes next-period portfolio return from the
    current rebalance date to the next rebalance date.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        basket_csv_path: str,
        initial_amount: float = 100000.0,
        transaction_cost: float = 0.001,
        include_cash: bool = True,
    ):
        super().__init__()

        self.basket_csv_path = basket_csv_path
        self.initial_amount = float(initial_amount)
        self.transaction_cost = float(transaction_cost)
        self.include_cash = include_cash

        self.df = pd.read_csv(basket_csv_path)
        self.df["rebalance_date"] = pd.to_datetime(self.df["rebalance_date"], utc=True)

        required_cols = {
            "rebalance_date",
            "slot",
            "tic",
            "score",
            "close",
            "macd",
            "rsi_30",
            "cci_30",
            "dx_30",
            "close_30_sma",
            "close_60_sma",
            "volatility_30",
            "return_5",
            "return_10",
            "price_vs_sma30",
            "spy_trend",
        }
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns in basket dataset: {missing}")

        self.feature_cols = [
            "score",
            "close",
            "macd",
            "rsi_30",
            "cci_30",
            "dx_30",
            "close_30_sma",
            "close_60_sma",
            "volatility_30",
            "return_5",
            "return_10",
            "price_vs_sma30",
            "spy_trend",
        ]

        self.rebalance_dates = sorted(self.df["rebalance_date"].unique())
        if len(self.rebalance_dates) < 2:
            raise ValueError("Need at least 2 rebalance dates in basket dataset.")

        # State:
        # [portfolio_value_norm]
        # [prev_weights (5)]
        # [4 slots * len(feature_cols)]
        self.n_slots = 4
        self.n_assets = 5 if include_cash else 4
        self.slot_feature_dim = len(self.feature_cols)

        self.obs_dim = 1 + self.n_assets + (self.n_slots * self.slot_feature_dim)

        self.action_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(self.n_assets,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.reset()

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        e = np.exp(x)
        return e / (np.sum(e) + 1e-8)

    def _get_day_basket(self, dt) -> pd.DataFrame:
        day_df = self.df[self.df["rebalance_date"] == dt].copy()
        day_df = day_df.sort_values("slot").reset_index(drop=True)

        if len(day_df) != self.n_slots:
            raise ValueError(f"Expected {self.n_slots} rows for {dt}, got {len(day_df)}")

        return day_df

    def _build_observation(self, day_df: pd.DataFrame) -> np.ndarray:
        features = []

        # normalize close-related columns lightly to keep magnitudes reasonable
        for _, row in day_df.iterrows():
            row_features = [
                float(row["score"]),
                float(row["close"]) / 1000.0,
                float(row["macd"]),
                float(row["rsi_30"]) / 100.0,
                float(row["cci_30"]) / 100.0,
                float(row["dx_30"]) / 100.0,
                float(row["close_30_sma"]) / 1000.0,
                float(row["close_60_sma"]) / 1000.0,
                float(row["volatility_30"]),
                float(row["return_5"]),
                float(row["return_10"]),
                float(row["price_vs_sma30"]),
                float(row["spy_trend"]),
            ]
            features.extend(row_features)

        obs = np.concatenate([
            np.array([self.portfolio_value / self.initial_amount], dtype=np.float32),
            self.prev_weights.astype(np.float32),
            np.array(features, dtype=np.float32),
        ]).astype(np.float32)

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_idx = 0
        self.portfolio_value = self.initial_amount
        self.prev_weights = np.zeros(self.n_assets, dtype=np.float32)

        if self.include_cash:
            self.prev_weights[-1] = 1.0  # start fully in cash
        else:
            self.prev_weights[:] = 1.0 / self.n_assets

        self.current_day_df = self._get_day_basket(self.rebalance_dates[self.step_idx])
        obs = self._build_observation(self.current_day_df)
        info = {"portfolio_value": self.portfolio_value}
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        weights = self._softmax(action)

        current_df = self.current_day_df
        next_idx = self.step_idx + 1
        terminated = next_idx >= len(self.rebalance_dates) - 1

        next_day_df = self._get_day_basket(self.rebalance_dates[next_idx])

        current_prices = current_df["close"].values.astype(np.float64)
        next_prices = next_day_df["close"].values.astype(np.float64)

        # slot-based return from current basket to next basket row-by-row
        # assumes slot ordering is stable within rebalance dataset generation
        asset_returns = (next_prices / (current_prices + 1e-8)) - 1.0

        if self.include_cash:
            stock_weights = weights[:4]
            cash_weight = weights[4]
        else:
            stock_weights = weights
            cash_weight = 0.0

        gross_return = float(np.dot(stock_weights, asset_returns))
        current_weights = self.prev_weights.copy()

        turnover = float(np.sum(np.abs(weights - current_weights)))
        cost_penalty = self.transaction_cost * turnover

        net_return = gross_return - cost_penalty

        prev_value = self.portfolio_value
        self.portfolio_value = self.portfolio_value * (1.0 + net_return)

        reward = net_return

        self.prev_weights = weights.astype(np.float32)
        self.step_idx = next_idx
        self.current_day_df = next_day_df

        obs = self._build_observation(next_day_df)

        info = {
            "portfolio_value": self.portfolio_value,
            "prev_portfolio_value": prev_value,
            "gross_return": gross_return,
            "net_return": net_return,
            "turnover": turnover,
            "cost_penalty": cost_penalty,
            "weights": weights,
            "selected_tickers": list(current_df["tic"].values),
        }

        return obs, reward, terminated, False, info