"""
Target snapshot utilities for the hybrid rebalance workflow.

Purpose:
- Freeze the official monthly target after a rebalance submission.
- Let reconciliation compare Alpaca positions against the frozen submitted target,
  instead of chasing later preview changes.
"""

from pathlib import Path

import pandas as pd


TARGET_SNAPSHOT_DIR = Path("monthly_target_snapshots")


def get_snapshot_path(
    strategy_name: str,
    rebalance_period: str,
    mode: str,
) -> Path:
    safe_strategy = str(strategy_name).replace("/", "_").replace(" ", "_")
    safe_period = str(rebalance_period).replace("/", "_").replace(" ", "_")
    safe_mode = str(mode).replace("/", "_").replace(" ", "_")

    return TARGET_SNAPSHOT_DIR / f"{safe_strategy}_{safe_period}_{safe_mode}_target.csv"


def save_target_snapshot(
    proposed_orders_path: str | Path,
    strategy_name: str,
    rebalance_period: str,
    mode: str,
) -> Path:
    """
    Save a frozen monthly target snapshot from proposed_orders.csv.

    This snapshot becomes the official target for post-submit reconciliation.
    """
    proposed_orders_path = Path(proposed_orders_path)

    if not proposed_orders_path.exists():
        raise FileNotFoundError(f"Proposed orders file not found: {proposed_orders_path}")

    df = pd.read_csv(proposed_orders_path)

    if df.empty:
        raise RuntimeError(f"Proposed orders file is empty: {proposed_orders_path}")

    required_cols = {"symbol", "target_value"}

    missing = required_cols - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Cannot create target snapshot. Missing columns: {sorted(missing)}"
        )

    snapshot_df = df.copy()

    snapshot_df["snapshot_strategy_name"] = strategy_name
    snapshot_df["snapshot_rebalance_period"] = rebalance_period
    snapshot_df["snapshot_mode"] = mode

    TARGET_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_path = get_snapshot_path(
        strategy_name=strategy_name,
        rebalance_period=rebalance_period,
        mode=mode,
    )

    snapshot_df.to_csv(snapshot_path, index=False)

    return snapshot_path


def load_target_snapshot_if_exists(
    strategy_name: str,
    rebalance_period: str,
    mode: str,
) -> tuple[pd.DataFrame, Path | None]:
    """
    Load frozen target snapshot if it exists.

    Returns:
        (df, path)
    """
    snapshot_path = get_snapshot_path(
        strategy_name=strategy_name,
        rebalance_period=rebalance_period,
        mode=mode,
    )

    if not snapshot_path.exists() or snapshot_path.stat().st_size == 0:
        return pd.DataFrame(), None

    return pd.read_csv(snapshot_path), snapshot_path