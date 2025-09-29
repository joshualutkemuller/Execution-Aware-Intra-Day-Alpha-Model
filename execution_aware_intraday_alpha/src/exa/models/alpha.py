from dataclasses import dataclass
import pandas as pd

@dataclass
class RidgeAlpha:
    alpha: float = 1.0
    cols: list[str] | None = None

    def fit(self, df: pd.DataFrame):
        # TODO: fit model; placeholder remembers feature columns
        self.cols = [c for c in df.columns if c not in ("y","mid","bid","ask","close")]
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        # TODO: predict; placeholder returns zeros
        return pd.Series(0.0, index=df.index, name="yhat")
