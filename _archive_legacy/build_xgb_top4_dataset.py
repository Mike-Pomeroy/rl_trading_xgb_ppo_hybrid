import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from config import (
    TICKER_LIST,
    INDICATOR_COLUMNS,
    TOP_K,
    TARGET_HORIZON,
    TRAIN_END_DATE,
    TEST_START_DATE,
    XGB_PARAMS,
    XGB_MODEL_PATH,
    TRAIN_BASKET_DATA_PATH,
    TEST_BASKET_DATA_PATH,
    BASKET_METADATA_PATH,
)
from data_module import prepare_data


def ensure_parent_dir(path_str: str) -> None:
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def add_forward_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["tic", "date"]).reset_index(drop=True)
    out[f"future_return_{horizon}"] = (
        out.groupby("tic")["close"].shift(-horizon) / out["close"] - 1.0
    )
    return out


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True).dt.normalize()
    return out


def train_selector(train_df: pd.DataFrame, feature_cols: list[str], target_col: str) -> XGBRegressor:
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(train_df[feature_cols], train_df[target_col])
    return model


def get_monthly_rebalance_dates(dates: pd.Series) -> list[pd.Timestamp]:
    dts = pd.to_datetime(pd.Series(sorted(dates.unique()))).dt.normalize()
    monthly = dts.groupby(dts.dt.to_period("M")).min()
    return list(pd.to_datetime(monthly))


def build_selected_basket_rows(
    scored_df: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    feature_cols: list[str],
    top_k: int,
) -> pd.DataFrame:
    basket_rows = []

    for dt in rebalance_dates:
        day_df = scored_df[scored_df["date"] == dt].copy()
        if day_df.empty:
            continue

        day_df = day_df.sort_values("score", ascending=False).head(top_k).copy()
        if len(day_df) < top_k:
            continue

        day_df = day_df.sort_values("score", ascending=False).reset_index(drop=True)
        day_df["basket_rank"] = np.arange(1, len(day_df) + 1)

        # stable slot naming for PPO dataset
        for slot_idx, (_, row) in enumerate(day_df.iterrows(), start=1):
            record = {
                "rebalance_date": dt,
                "slot": slot_idx,
                "tic": row["tic"],
                "score": float(row["score"]),
                "close": float(row["close"]),
            }

            for col in feature_cols:
                record[col] = float(row[col])

            basket_rows.append(record)

    return pd.DataFrame(basket_rows)


def main() -> None:
    print("Loading source data...")
    df = prepare_data()
    df = df[df["tic"].isin(TICKER_LIST)].copy()
    df = normalize_dates(df)
    df = add_forward_target(df, TARGET_HORIZON)

    target_col = f"future_return_{TARGET_HORIZON}"

    needed_cols = ["date", "tic", "close"] + INDICATOR_COLUMNS + [target_col]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=needed_cols).copy()

    print(f"Usable rows: {len(df)}")

    train_cutoff = pd.Timestamp(TRAIN_END_DATE, tz="UTC")
    test_cutoff = pd.Timestamp(TEST_START_DATE, tz="UTC")

    train_df = df[df["date"] < train_cutoff].copy()
    test_df = df[df["date"] >= test_cutoff].copy()

    if train_df.empty:
        raise ValueError("Training dataframe is empty.")
    if test_df.empty:
        raise ValueError("Test dataframe is empty.")

    print("Training XGBoost selector...")
    model = train_selector(train_df, INDICATOR_COLUMNS, target_col)

    ensure_parent_dir(XGB_MODEL_PATH)
    model.save_model(XGB_MODEL_PATH)
    print(f"Saved selector model to {XGB_MODEL_PATH}")

    # score both train and test so PPO can later train on train baskets and evaluate on test baskets
    print("Scoring train/test universes...")
    train_scored = train_df.copy()
    train_scored["score"] = model.predict(train_scored[INDICATOR_COLUMNS])

    test_scored = test_df.copy()
    test_scored["score"] = model.predict(test_scored[INDICATOR_COLUMNS])

    train_rebalance_dates = get_monthly_rebalance_dates(train_scored["date"])
    test_rebalance_dates = get_monthly_rebalance_dates(test_scored["date"])

    print("Building monthly top-4 basket datasets...")
    train_baskets = build_selected_basket_rows(
        scored_df=train_scored,
        rebalance_dates=train_rebalance_dates,
        feature_cols=INDICATOR_COLUMNS,
        top_k=TOP_K,
    )

    test_baskets = build_selected_basket_rows(
        scored_df=test_scored,
        rebalance_dates=test_rebalance_dates,
        feature_cols=INDICATOR_COLUMNS,
        top_k=TOP_K,
    )

    if train_baskets.empty:
        raise ValueError("Train basket dataset is empty.")
    if test_baskets.empty:
        raise ValueError("Test basket dataset is empty.")

    ensure_parent_dir(TRAIN_BASKET_DATA_PATH)
    ensure_parent_dir(TEST_BASKET_DATA_PATH)
    ensure_parent_dir(BASKET_METADATA_PATH)

    train_baskets.to_csv(TRAIN_BASKET_DATA_PATH, index=False)
    test_baskets.to_csv(TEST_BASKET_DATA_PATH, index=False)

    metadata = {
        "tickers": TICKER_LIST,
        "feature_columns": INDICATOR_COLUMNS,
        "top_k": TOP_K,
        "target_horizon": TARGET_HORIZON,
        "train_end_date": TRAIN_END_DATE,
        "test_start_date": TEST_START_DATE,
        "train_rebalance_dates": len(train_rebalance_dates),
        "test_rebalance_dates": len(test_rebalance_dates),
        "train_basket_rows": int(len(train_baskets)),
        "test_basket_rows": int(len(test_baskets)),
    }

    Path(BASKET_METADATA_PATH).write_text(json.dumps(metadata, indent=2))

    print(f"Saved train baskets to {TRAIN_BASKET_DATA_PATH}")
    print(f"Saved test baskets to {TEST_BASKET_DATA_PATH}")
    print(f"Saved metadata to {BASKET_METADATA_PATH}")

    print("\nSample train basket rows:")
    print(train_baskets.head(8).to_string(index=False))

    print("\nSample test basket rows:")
    print(test_baskets.head(8).to_string(index=False))


if __name__ == "__main__":
    main()