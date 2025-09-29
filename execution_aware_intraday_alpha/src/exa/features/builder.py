# src/exa/features/builder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Dict, Optional

import numpy as np
import pandas as pd


# ----------- feature primitives -----------

def compute_returns(mid: pd.Series) -> pd.Series:
    """Log return at 1-second step."""
    r = np.log(mid).diff()
    return r


def realized_vol(log_ret: pd.Series, window: int) -> pd.Series:
    """Rolling std of log returns as a short-horizon volatility proxy."""
    vol = log_ret.rolling(window, min_periods=max(5, window // 3)).std()
    return vol


def ofi_proxy(bid: pd.Series, ask: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """
    Order-Flow Imbalance (proxy):
      sign(delta mid) * volume, aggregated over a rolling window.
    This is a simple L1 proxy when L2 queues are unavailable.
    """
    mid = (bid + ask) / 2.0
    signed_flow = np.sign(mid.diff().fillna(0.0)) * volume.fillna(0.0)
    return signed_flow.rolling(window, min_periods=1).sum()


def microprice(bid: pd.Series, ask: pd.Series) -> pd.Series:
    """
    Microprice (placeholder without queue sizes):
      With no depth data, we use the midpoint; you can upgrade this when L2 is available.
    """
    return (bid + ask) / 2.0


def pct_trend(series: pd.Series, window: int) -> pd.Series:
    """Percent change over a window (non-annualized)."""
    return series.pct_change(window)


def add_lags(df: pd.DataFrame, cols: Iterable[str], lags: Iterable[int]) -> pd.DataFrame:
    """Create lagged copies of the specified columns."""
    out = df.copy()
    for c in cols:
        for L in lags:
            out[f"{c}_lag{L}"] = out[c].shift(L)
    return out


def cyclical_time_encodings(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Cyclical encodings for minute-of-hour and hour-of-day:
      sin/cos(2π * t / T) for T in {60, 24}.
    """
    # Ensure tz-aware for repeatability (not strictly required)
    minutes = idx.minute.astype(float)
    hours = idx.hour.astype(float)

    two_pi = 2.0 * np.pi
    enc = pd.DataFrame(index=idx)
    # Minute-of-hour (0..59)
    enc["min_sin"] = np.sin(two_pi * minutes / 60.0)
    enc["min_cos"] = np.cos(two_pi * minutes / 60.0)
    # Hour-of-day (0..23)
    enc["hour_sin"] = np.sin(two_pi * hours / 24.0)
    enc["hour_cos"] = np.cos(two_pi * hours / 24.0)
    return enc


# ----------- main builder -----------

@dataclass
class FeatureBuilder:
    """
    Leak-safe feature builder for intraday crypto alphas.

    Pipeline:
      1) Base signals from top-of-book approximations (OFI proxy, microprice tilt, short vol, trend)
      2) Intraday seasonality (cyclical hour/minute)
      3) Optional lags for selected cols
      4) Target y = log(mid[t + H]) - log(mid[t])
      5) Lag all *explanatory* features by 1 step to prevent look-ahead

    Expected input columns: ['mid','bid','ask','close','volume'] with a UTC DateTimeIndex at 1s freq.
    """

    # windows (in seconds)
    ofi_window: int = 10
    vol_window: int = 60
    trend_window: int = 30

    # target horizon (seconds ahead)
    target_horizon: int = 60

    # optional lags to add on selected columns (applied before the universal +1 lag)
    lags: Optional[List[int]] = None
    lag_columns: Optional[List[str]] = None  # which base columns to lag; None => sensible defaults

    # drop rows with any NaN after building
    dropna: bool = True

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(df)

        out = pd.DataFrame(index=df.index.copy())
        out["mid"] = df["mid"].astype(float)
        out["bid"] = df["bid"].astype(float)
        out["ask"] = df["ask"].astype(float)
        out["close"] = df["close"].astype(float)
        out["volume"] = df["volume"].astype(float)

        # 1) Base derived series
        ret1 = compute_returns(out["mid"])
        out["vol"] = realized_vol(ret1, self.vol_window)
        out["ofi"] = ofi_proxy(out["bid"], out["ask"], out["volume"], self.ofi_window)
        out["microprice"] = microprice(out["bid"], out["ask"])
        out["trend"] = pct_trend(out["mid"], self.trend_window)

        # 2) Intraday seasonality (sin/cos encodings)
        season = cyclical_time_encodings(out.index)
        out = out.join(season)

        # 3) Optional explicit lags on select columns (before universal +1 lag)
        if self.lags and len(self.lags) > 0:
            base_cols = self.lag_columns or ["ofi", "vol", "trend"]
            out = add_lags(out, base_cols, self.lags)

        # 4) Target — future log-return over horizon
        out["y"] = np.log(out["mid"]).shift(-self.target_horizon) - np.log(out["mid"])

        # 5) Universal +1 lag on *explanatory* features to avoid look-ahead
        feature_cols = [c for c in out.columns if c not in ("y",)]
        out[feature_cols] = out[feature_cols].shift(1)

        # Final cleanup
        if self.dropna:
            out = out.dropna()

        return out

    # ----------- utilities -----------

    @staticmethod
    def _validate_input(df: pd.DataFrame) -> None:
        required = {"mid", "bid", "ask", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Input dataframe is missing required columns: {sorted(missing)}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Input dataframe must have a DatetimeIndex.")
        if df.index.tz is None:
            # keep going, but strongly recommended to be tz-aware (UTC)
            pass


# Convenience for quick ad-hoc builds (keeps a stable public API)
def build_features(
    raw: pd.DataFrame,
    ofi_window: int = 10,
    vol_window: int = 60,
    trend_window: int = 30,
    target_horizon: int = 60,
    lags: Optional[List[int]] = None,
    lag_columns: Optional[List[str]] = None,
    dropna: bool = True,
) -> pd.DataFrame:
    fb = FeatureBuilder(
        ofi_window=ofi_window,
        vol_window=vol_window,
        trend_window=trend_window,
        target_horizon=target_horizon,
        lags=lags,
        lag_columns=lag_columns,
        dropna=dropna,
    )
    return fb.build(raw)
