#!/bin/bash

cd /Users/pomeroy/projects/rl_trading_xgb_ppo_hybrid || exit 1

# Use the project Python environment if it exists.
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

streamlit run dashboard.py