from dataclasses import dataclass
import pandas as pd

@dataclass
class BacktestEngine:
    costs: dict
    latency_ms: int = 50
    slippage_ticks_when_market: float = 0.5

    def run(self, df: pd.DataFrame, signal: pd.Series, policy, sizing_fn):
        # TODO: backtest; placeholder returns empty pnl frame
        return pd.DataFrame(index=df.index, data={"pnl": 0.0})
