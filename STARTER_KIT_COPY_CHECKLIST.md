# Starter Kit Copy Checklist

Use this checklist before giving a family member or friend their own private copy of the trading starter kit.

---

## Safe to copy

These files are generally safe to copy:

```text
Python source files
requirements.txt
runtime.txt
README files
Markdown setup guides
.streamlit/secrets.example.toml
streamlit_starter_app.py
check_starter_safety.py
models/
data_cache/xgb_top4_metadata.json
```

---

## Do not copy

Do not copy personal or generated files:

```text
.env
.streamlit/secrets.toml
alpaca_preview_orders/
alpaca_preview_orders_hybrid/
alpaca_submitted_orders/
alpaca_submitted_orders_hybrid/
logs/
ticker_ranking_results/
trade_log_results/
hybrid_trade_log_results/
rebalance_guard_logs/
account_reconciliation_reports/
results/
robustness_results/
universe_overlap_results/
walkforward_universe_results/
```

---

## Before sharing

Run:

```bash
python3 check_starter_safety.py
```

Confirm:

```text
.env is not tracked
.streamlit/secrets.toml is not tracked
required starter files are present
model files are present
generated output folders are not tracked
```

---

## Recipient setup

Each recipient should:

```text
Create their own private GitHub repository
Create their own Alpaca paper account
Use their own API keys
Create their own .streamlit/secrets.toml
Run paper trading first
Review all proposed orders manually
```

---

## Live trading rule

Live trading should not be enabled by default.

Recommended starter settings:

```toml
TRADING_MODE = "paper"
ENABLE_LIVE_TRADING = false
REQUIRE_MANUAL_APPROVAL = true
```

