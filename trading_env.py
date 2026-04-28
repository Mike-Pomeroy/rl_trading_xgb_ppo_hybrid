import gymnasium as gym
import numpy as np
from gymnasium import spaces

from feature_engine import FeatureEngine


class TradingEnv(gym.Env):
    def __init__(
        self,
        df,
        ticker_list,
        tech_indicator_list,
        initial_amount=100000,
        transaction_cost=0.002,
    ):
        super().__init__()

        self.df = df.copy().reset_index(drop=True)
        self.df = self.df.sort_values(["date", "tic"]).reset_index(drop=True)

        self.ticker_list = ticker_list
        self.tech_indicator_list = tech_indicator_list

        self.stock_dim = len(ticker_list)
        self.initial_amount = initial_amount
        self.transaction_cost = transaction_cost

        self.feature_engine = FeatureEngine(
            ticker_list=ticker_list,
            tech_indicator_list=tech_indicator_list,
            initial_amount=initial_amount,
        )

        # portfolio state
        self.cash = initial_amount
        self.holdings = np.zeros(self.stock_dim, dtype=np.float32)

        # tracking
        self.peak_value = initial_amount
        self.prev_portfolio_value = initial_amount
        self.returns_buffer = []
        self.prev_target_weights = np.zeros(self.stock_dim, dtype=np.float32)

        # PPO emits raw logits; env converts to portfolio weights
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.stock_dim,), dtype=np.float32
        )

        self.state_dim = (
            1
            + self.stock_dim
            + self.stock_dim
            + self.stock_dim * len(self.tech_indicator_list)
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        self.dates = sorted(self.df["date"].unique())
        self.reset()

    # -----------------------------------------------------
    # UTIL
    # -----------------------------------------------------
    def softmax(self, x):
        x = x - np.max(x)
        e = np.exp(x)
        return e / (np.sum(e) + 1e-8)

    def _get_day_data(self, date):
        day_df = self.df[self.df["date"] == date]
        return {row["tic"]: row for _, row in day_df.iterrows()}

    def _get_prices(self, data):
        return np.array(
            [data[tic]["close"] if tic in data else 0.0 for tic in self.ticker_list],
            dtype=np.float32,
        )

    def _get_portfolio_value(self, prices):
        return float(self.cash + np.sum(self.holdings * prices))

    # -----------------------------------------------------
    # STATE
    # -----------------------------------------------------
    def _build_state(self, data):
        return self.feature_engine.build_state(
            data=data,
            cash=self.cash,
            holdings=self.holdings,
        )

    # -----------------------------------------------------
    # RESET
    # -----------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.day = 0
        self.cash = self.initial_amount
        self.holdings = np.zeros(self.stock_dim, dtype=np.float32)

        self.peak_value = self.initial_amount
        self.prev_portfolio_value = self.initial_amount
        self.returns_buffer = []
        self.prev_target_weights = np.zeros(self.stock_dim, dtype=np.float32)

        data = self._get_day_data(self.dates[self.day])
        return self._build_state(data), {}

    # -----------------------------------------------------
    # STEP (PORTFOLIO WEIGHT VERSION)
    # -----------------------------------------------------
    def step(self, actions):
        data = self._get_day_data(self.dates[self.day])
        prices = self._get_prices(data)

        # =========================
        # PORTFOLIO WEIGHTS
        # =========================
        target_weights = self.softmax(actions)

        # hard cap any single position
        max_weight = 0.10
        target_weights = np.minimum(target_weights, max_weight)

        # renormalize
        target_weights = target_weights / (np.sum(target_weights) + 1e-8)

        portfolio_value = self.cash + np.sum(self.holdings * prices)
        target_values = target_weights * portfolio_value

        # turnover penalty input
        turnover = np.sum(np.abs(target_weights - self.prev_target_weights))

        # =========================
        # REBALANCE PORTFOLIO
        # =========================
        for i in range(self.stock_dim):
            price = prices[i]
            if price <= 0:
                continue

            current_asset_value = self.holdings[i] * price
            diff_value = target_values[i] - current_asset_value

            # BUY
            if diff_value > 0:
                total_cost = diff_value * (1 + self.transaction_cost)
                qty = diff_value / price

                if total_cost <= self.cash:
                    self.cash -= total_cost
                    self.holdings[i] += qty

            # SELL
            elif diff_value < 0:
                qty = min(abs(diff_value) / price, self.holdings[i])
                proceeds = qty * price * (1 - self.transaction_cost)

                self.cash += proceeds
                self.holdings[i] -= qty

        # =========================
        # UPDATE
        # =========================
        self.day += 1
        done = self.day >= len(self.dates) - 1

        next_data = self._get_day_data(self.dates[self.day]) if not done else data
        obs = self._build_state(next_data)

        next_prices = self._get_prices(next_data)
        new_value = self._get_portfolio_value(next_prices)

        # =========================
        # REWARD
        # =========================
        ret = (new_value - self.prev_portfolio_value) / (
            self.prev_portfolio_value + 1e-8
        )
        self.prev_portfolio_value = new_value

        # drawdown
        self.peak_value = max(self.peak_value, new_value)
        drawdown = (self.peak_value - new_value) / (self.peak_value + 1e-8)

        # rolling volatility
        self.returns_buffer.append(ret)
        if len(self.returns_buffer) > 30:
            self.returns_buffer.pop(0)

        vol = np.std(self.returns_buffer) if len(self.returns_buffer) > 5 else 0.0

        # concentration based on post-trade holdings and next prices
        realized_weights = (self.holdings * next_prices) / (new_value + 1e-8)
        concentration = np.sum(realized_weights ** 2)

        # final reward with lighter turnover penalty
        reward = (
            ret
            - 2.0 * drawdown
            - 1.0 * vol
            - 1.0 * concentration
            - 0.02 * turnover
        )

        # store current target weights for next step
        self.prev_target_weights = target_weights.copy()

        info = {
            "portfolio_value": new_value,
            "return": ret,
            "drawdown": drawdown,
            "volatility": vol,
            "concentration": concentration,
            "turnover": turnover,
        }

        return obs, reward, done, False, info