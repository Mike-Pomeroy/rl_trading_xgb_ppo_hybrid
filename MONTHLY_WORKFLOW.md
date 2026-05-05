# Monthly Trading Workflow

This guide explains the recommended monthly workflow for using the trading starter kit.

The goal is to review proposed trades safely before submitting anything.

---

## 1. Open the repo folder

```bash
cd /Users/pomeroy/projects/rl_trading_xgb_ppo_hybrid
```

For another user, replace the path with their own repo folder.

---

## 2. Confirm the correct branch

```bash
git branch --show-current
```

For the starter kit, the branch should be:

```text
streamlit-starter-kit
```

---

## 3. Pull the latest code

```bash
git pull
```

If there are local changes, stop and review them before continuing.

---

## 4. Activate the Python environment

If using a virtual environment:

```bash
source .venv/bin/activate
```

If no virtual environment exists yet:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 5. Confirm secrets are local and private

Run:

```bash
python3 check_starter_safety.py
```

Confirm that:

```text
.env is not tracked
.streamlit/secrets.toml is not tracked
required starter files are present
```

Do not continue if real API keys are tracked by Git.

---

## 6. Run the normal preview workflow

Run the existing trading preview script.

For the current hybrid workflow, this is usually:

```bash
python3 alpaca_order_preview_hybrid.py
```

This should create or update:

```text
alpaca_preview_orders_hybrid/proposed_orders.csv
alpaca_preview_orders_hybrid/current_positions.csv
alpaca_preview_orders_hybrid/model_scores.csv
alpaca_preview_orders_hybrid/open_orders.csv
```

---

## 7. Open the read-only dashboard

```bash
streamlit run streamlit_starter_app.py
```

Review:

```text
Preview file freshness
Proposed orders
Current positions
Model scores
Open orders
Account reconciliation summary
Latest ticker rankings
Safety checklist
```

---

## 8. Check file freshness

In the dashboard, confirm preview files are fresh.

If the dashboard says files are stale or missing, rerun the preview workflow before relying on any proposed orders.

---

## 9. Review proposed orders manually

Before submitting anything, review:

```text
Symbols
Buy/sell direction
Estimated order size
Cash remaining
Existing positions
Open orders
Account reconciliation status
```

Do not submit orders if anything looks unexpected.

---

## 10. Submit orders only after review

Order submission should be done only through the tested trading workflow.

For paper trading, the current hybrid submit script is usually:

```bash
python3 alpaca_order_submit_paper_hybrid.py
```

Do not use live trading unless it has been separately reviewed and enabled.

---

## 11. Re-run the dashboard after submission

After submitting orders, rerun the dashboard:

```bash
streamlit run streamlit_starter_app.py
```

Confirm:

```text
Submitted orders look correct
Open orders are expected
Positions are expected
Cash/buying power are expected
```

---

## 12. Commit only code or documentation changes

Do not commit:

```text
.env
.streamlit/secrets.toml
preview CSVs
submitted order CSVs
logs
local account data
```

Safe files to commit usually include:

```text
Python source files
Markdown documentation
example config files
README updates
```

---

## 13. Monthly safety rules

```text
Use paper trading first
Review all proposed orders manually
Keep API keys private
Do not share .env or secrets.toml
Do not commit generated trading outputs
Do not enable live trading casually
Stop if the dashboard shows stale preview files
Stop if account reconciliation looks wrong
```

