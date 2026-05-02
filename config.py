# config.py
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ----------------------------
# ALPACA PAPER TRADING
API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")

# ----------------------------
"""
ALPACA_API_KEY = os.getenv("APCA_API_KEY_ID", "").strip()
ALPACA_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "").strip()
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").strip()

if not ALPACA_API_KEY:
    raise ValueError("Missing ALPACA_API_KEY in environment or .env file")

if not ALPACA_SECRET_KEY:
    raise ValueError("Missing ALPACA_SECRET_KEY in environment or .env file")
"""

# ----------------------------
# MONTHLY XGBOOST PAPER TRADING
# ----------------------------
XGB_SYMBOLS = [
   "AAPL", "MSFT", "SPY", "GOOGL", "AMZN",
            "META", "TSLA", "NVDA", "JPM", "JNJ",
            "AVGO", "LLY", "UNH", "COST", "V",
            "MA", "HD", "PG", "XOM", "AMD"
]

XGB_TOP_K = 3
XGB_CASH_BUFFER = 0.20
XGB_TARGET_HORIZON = 30
XGB_STATE_FILE = "state/xgb_monthly_state.json"
XGB_DRY_RUN = True


# ----------------------------
# RESEARCH UNIVERSE
# ----------------------------
TICKER_LIST = [
    "AAPL", "MSFT", "SPY", "GOOGL", "AMZN",
    "META", "TSLA", "NVDA", "JPM", "JNJ"
]

# ----------------------------
# FEATURES
# ----------------------------
INDICATOR_COLUMNS = [
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

# ----------------------------
# XGBOOST SELECTOR SETTINGS
# ----------------------------
#TOP_K = 3
#TARGET_HORIZON = 21
REBALANCE_FREQUENCY = "M"

# ----------------------------
# TRAIN / TEST SPLIT
# ----------------------------
TRAIN_END_DATE = "2022-01-01"
TEST_START_DATE = "2022-01-01"

# ----------------------------
# CAPITAL / COSTS
# ----------------------------
INITIAL_AMOUNT = 3000
TRANSACTION_COST = 0.005

INITIAL_AMOUNT = 3000.0
TOP_K = 3

MIN_DOLLARS_PER_POSITION = 500.0
ALLOW_FRACTIONAL_SHARES = True

TARGET_HORIZON = 45

CASH_BUFFER = 0.20
SPY_TREND_FILTER = False

# ----------------------------
# XGBOOST PARAMS
# ----------------------------
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 42,
}

# ----------------------------
# PATHS
# ----------------------------
XGB_MODEL_PATH = "models/xgb_selector.json"

TRAIN_BASKET_DATA_PATH = "data_cache/xgb_top4_train.csv"
TEST_BASKET_DATA_PATH = "data_cache/xgb_top4_test.csv"
BASKET_METADATA_PATH = "data_cache/xgb_top4_metadata.json"

PPO_MODEL_PATH = "models/ppo_xgb_top4_allocator.zip"
PPO_VECNORM_PATH = "models/vecnormalize_xgb_top4.pkl"

RESULTS_REPORT_PATH = "results/backtest_report.txt"
RESULTS_PLOT_PATH = "results/hybrid_equity_curve.png"