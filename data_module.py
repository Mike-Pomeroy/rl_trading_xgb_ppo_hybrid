"""
Data preparation for the XGBoost top-k stock strategy.

Design goals:
- Build features using only same-day/past information.
- Keep target creation out of this module so live prediction rows do not need future data.
- Avoid blanket fillna(0) in the final output; the strategy file decides which rows are usable.
"""

import os
from datetime import datetime
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import pytz
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from finrl.meta.preprocessor.preprocessors import FeatureEngineer

load_dotenv()

# =========================
# CONFIG
# =========================
TICKER_LIST = [
    "AAPL", "MSFT", "SPY", "GOOGL", "AMZN",
    "META", "TSLA", "NVDA", "JPM", "JNJ"
]

# Only FinRL-supported indicators here.
INDICATORS = [
    "macd", "rsi_30", "cci_30",
    "dx_30", "close_30_sma", "close_60_sma"
]

START_DATE = "2019-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

REQUIRED_OHLCV_COLUMNS = ["date", "tic", "open", "high", "low", "close", "volume"]


def _get_alpaca_client() -> StockHistoricalDataClient:
    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")

    if not key or not secret:
        raise RuntimeError(
            "Missing Alpaca credentials. Set APCA_API_KEY_ID and "
            "APCA_API_SECRET_KEY in your environment or .env file."
        )

    return StockHistoricalDataClient(key, secret)


def fetch_alpaca_daily_bars(
    tickers: Optional[Iterable[str]] = None,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    adjustment: str = "split",
    feed: str = "sip",
) -> pd.DataFrame:
    """Fetch daily OHLCV data from Alpaca and return a FinRL-friendly dataframe."""
    tickers = list(tickers or TICKER_LIST)
    client = _get_alpaca_client()

    tz = pytz.timezone("US/Eastern")
    start = tz.localize(datetime.strptime(start_date, "%Y-%m-%d"))
    end = tz.localize(datetime.strptime(end_date, "%Y-%m-%d"))

    request = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment=adjustment,
        feed=feed,
    )

    bars = client.get_stock_bars(request)

    df = bars.df.reset_index()
    df = df.rename(columns={"timestamp": "date", "symbol": "tic"})

    # Normalize to date-only timestamps so merges/pivots are stable.
    df["date"] = (
        pd.to_datetime(df["date"], utc=True)
        .dt.tz_convert("US/Eastern")
        .dt.tz_localize(None)
        .dt.normalize()
    )

    df["tic"] = df["tic"].astype(str)
    df = df.sort_values(["tic", "date"]).reset_index(drop=True)

    missing = [c for c in REQUIRED_OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Alpaca response is missing required columns: {missing}")

    return df


def _add_finrl_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add FinRL technical indicators.

    FinRL indicators are trailing/same-day technical indicators.
    This function does not create any future-return target.
    """
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_turbulence=False,
        user_defined_feature=False,
    )

    out = fe.preprocess_data(df.copy())
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()

    return out.sort_values(["tic", "date"]).reset_index(drop=True)


def _add_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add custom features using only current and past values."""
    df = df.sort_values(["tic", "date"]).copy()

    grouped_close = df.groupby("tic", group_keys=False)["close"]

    df["return_5"] = grouped_close.pct_change(5)
    df["return_10"] = grouped_close.pct_change(10)

    # Safe division; missing SMA remains NaN and is handled by the strategy file.
    df["price_vs_sma30"] = np.where(
        df["close_30_sma"].abs() > 1e-12,
        df["close"] / df["close_30_sma"] - 1.0,
        np.nan,
    )

    daily_ret = grouped_close.pct_change()

    # Important: rolling volatility is grouped by ticker, so it cannot roll across tickers.
    df["volatility_30"] = (
        daily_ret.groupby(df["tic"], group_keys=False)
        .rolling(30, min_periods=20)
        .std()
        .reset_index(level=0, drop=True)
    )

    spy = df.loc[df["tic"] == "SPY", ["date", "close"]].copy().sort_values("date")
    spy["spy_trend"] = spy["close"].pct_change(50)
    spy["spy_sma_200"] = spy["close"].rolling(200, min_periods=200).mean()
    spy["spy_above_200sma"] = (spy["close"] > spy["spy_sma_200"]).astype(float)

    df = df.merge(
        spy[["date", "spy_trend", "spy_sma_200", "spy_above_200sma"]],
        on="date",
        how="left",
    )

    df = df.replace([np.inf, -np.inf], np.nan)

    return df.sort_values(["tic", "date"]).reset_index(drop=True)


def prepare_data(
    tickers: Optional[Iterable[str]] = None,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    adjustment: str = "split",
    feed: str = "sip",
    raw_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Prepare feature dataframe.

    Parameters
    ----------
    tickers:
        Tickers to fetch/use. Defaults to TICKER_LIST.
    start_date, end_date:
        Alpaca fetch window. Ignored when raw_df is supplied.
    adjustment, feed:
        Alpaca data settings.
    raw_df:
        Optional pre-fetched OHLCV dataframe. Useful for tests/research without
        hitting Alpaca.

    Returns
    -------
    pd.DataFrame
        Dataframe with OHLCV, FinRL indicators, and custom trailing features.
        No future-return target is created here.
    """
    tickers = list(tickers or TICKER_LIST)

    if raw_df is None:
        df = fetch_alpaca_daily_bars(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            adjustment=adjustment,
            feed=feed,
        )
    else:
        df = raw_df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["tic"] = df["tic"].astype(str)
        df = (
            df[df["tic"].isin(tickers)]
            .sort_values(["tic", "date"])
            .reset_index(drop=True)
        )

    df = df[df["tic"].isin(tickers)].copy()
    df = _add_finrl_features(df)
    df = _add_custom_features(df)

    # Do not fill all NaNs with 0.
    # Warmup rows should be dropped by the strategy only when a selected feature
    # or training target is unavailable.
    return df