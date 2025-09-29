# Execution-Aware Intraday Alpha Lab — Crypto Edition

End-to-end research-to-execution scaffold for intraday crypto alphas:
- **Signals:** order-flow imbalance, microprice tilt, short-horizon vol, intraday seasonality
- **Policies:** Market, TWAP, POV, queue-aware Limit, tiny RL agent (market/join/improve)
- **Backtest:** event-driven, top-of-book fill simulation, implementation shortfall attribution
- **Risk:** position, turnover, EWMA VaR caps; volatility-scaled sizing
- **Repro:** config-driven, tests, CI, Docker, MLflow logging

## Quickstart
```bash
# 1) create venv and install
python -m venv .venv && source .venv/bin/activate
pip install -e .  # or: pip install -r requirements.txt

# 2) run a tiny demo on BTCUSDT 1s bars (synthetic book from trades/klines)
python examples/quickstart_demo.py

# 3) generate a tear sheet HTML in ./reports
python examples/generate_tearsheet.py
