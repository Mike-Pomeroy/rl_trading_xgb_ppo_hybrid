import numpy as np


class FeatureEngine:
    def __init__(self, ticker_list, tech_indicator_list, initial_amount):
        self.tickers = ticker_list
        self.techs = tech_indicator_list
        self.initial_amount = initial_amount

    def build_state(self, data, cash, holdings):
        prices = []
        tech_features = []

        for tic in self.tickers:
            row = data.get(tic)

            if row is None:
                prices.append(0.0)
                tech_features.extend([0.0] * len(self.techs))
                continue

            prices.append(row["close"])

            for t in self.techs:
                tech_features.append(row[t])

        prices = np.array(prices, dtype=np.float32)

        cash_norm = cash / self.initial_amount
        holdings_norm = holdings / (self.initial_amount + 1e-8)

        state = np.concatenate([
            np.array([cash_norm], dtype=np.float32),
            holdings_norm.astype(np.float32),
            prices / (np.max(prices) + 1e-8),
            np.clip(np.array(tech_features, dtype=np.float32), -5, 5),
        ])

        return state.astype(np.float32)