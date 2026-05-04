# Family/Friend Trading Starter Setup

This guide explains how a family member or friend can create their own private copy of the trading starter kit.

Each person should use:

- Their own private GitHub repository
- Their own Alpaca account
- Their own Alpaca paper API keys
- Their own Streamlit secrets
- Their own generated preview/output files

No one should share API keys.

---

## 1. Create a private GitHub repository

Create a new private GitHub repository under your own GitHub account.

Recommended name:

```text
my-trading-starter
```

Keep it private.

---

## 2. Copy the starter kit files

Copy the approved starter-kit repository files into your private repository.

Do not copy:

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
```

Those files are personal, generated, or sensitive.

---

## 3. Install Python requirements

From the repo folder, run:

```bash
pip install -r requirements.txt
```

If Streamlit is not installed:

```bash
pip install streamlit
```

---

## 4. Create Alpaca paper account keys

Log in to Alpaca and create paper trading API keys.

Use paper trading first.

Do not start with live trading.

---

## 5. Create local Streamlit secrets

From the repo folder, run:

```bash
mkdir -p .streamlit
cp .streamlit/secrets.example.toml .streamlit/secrets.toml
```

Then edit:

```text
.streamlit/secrets.toml
```

Add your own Alpaca paper keys.

Never commit `.streamlit/secrets.toml`.

---

## 6. Run the safety check

Run:

```bash
python3 check_starter_safety.py
```

Confirm:

```text
.env is not tracked
.streamlit/secrets.toml is not tracked
starter files are present
```

Warnings about missing preview CSV files are normal before the preview workflow has been run.

---

## 7. Run the normal preview workflow

Run the existing trading preview process before opening the dashboard.

The dashboard expects these local files:

```text
alpaca_preview_orders_hybrid/proposed_orders.csv
alpaca_preview_orders_hybrid/current_positions.csv
alpaca_preview_orders_hybrid/model_scores.csv
alpaca_preview_orders_hybrid/open_orders.csv
```

The dashboard only reads these files.

---

## 8. Open the read-only dashboard

Run:

```bash
streamlit run streamlit_starter_app.py
```

The dashboard shows:

```text
Trading mode
Live trading enabled flag
Manual approval flag
Cash buffer
Proposed orders
Current positions
Model scores
Open orders
Safety checklist
```

---

## 9. Important safety rules

This starter dashboard is read-only.

It does not:

```text
Submit orders
Cancel orders
Close positions
Replace orders
Connect directly to Alpaca
Modify trading files
```

Actual trade submission should only be done through the tested trading workflow.

---

## 10. Live trading warning

Use paper trading first.

Live trading uses real money.

Do not enable live trading unless you understand the risk and have reviewed every proposed order manually.

Recommended live trading safeguards:

```text
Manual approval required
Small position sizes
Cash buffer enabled
No automatic live trading
Review proposed orders before submitting
Keep API keys private
```

