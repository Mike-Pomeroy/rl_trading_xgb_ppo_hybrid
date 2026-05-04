# Trading Starter Dashboard

This is a read-only Streamlit dashboard for viewing local trading preview output files.

It does not submit orders.
It does not cancel orders.
It does not close positions.
It does not connect directly to Alpaca.
It only reads CSV files created by the existing trading workflow.

## 1. Install requirements

```bash
pip install -r requirements.txt
```

If Streamlit is not installed:

```bash
pip install streamlit
```

## 2. Configure secrets

Copy the example secrets file:

```bash
mkdir -p .streamlit
cp .streamlit/secrets.example.toml .streamlit/secrets.toml
```

Then edit:

```bash
.streamlit/secrets.toml
```

Do not commit real secrets.

## 3. Run the normal trading preview workflow

Before opening the dashboard, run the existing order preview process so these files are created:

```text
alpaca_preview_orders_hybrid/proposed_orders.csv
alpaca_preview_orders_hybrid/current_positions.csv
alpaca_preview_orders_hybrid/model_scores.csv
alpaca_preview_orders_hybrid/open_orders.csv
```

## 4. Start the dashboard

```bash
streamlit run streamlit_starter_app.py
```

## 5. Safety notes

This dashboard is read-only.

Actual order submission should continue through the existing tested trading workflow.

Live trading should require manual review and approval.
