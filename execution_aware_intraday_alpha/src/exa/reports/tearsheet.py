from dataclasses import dataclass
import pandas as pd

@dataclass
class Tearsheet:
    out_dir: str = "./reports"

    def make(self, prices: pd.DataFrame, bt: pd.DataFrame, signal: pd.Series, name: str):
        # TODO: write plots/summary; placeholder prints
        print(f"[tearsheet] {name} — rows: {len(bt)}")
