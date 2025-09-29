# src/exa/data/loader.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import requests
import yaml

DEFAULT_BASE_URLS: List[str] = [
    "https://api.binance.com",           # global
    "https://api.binance.us",            # US fallback
    "https://data-api.binance.vision",   # public data mirror
]

def _retry_get(url: str, params: dict, headers: dict, retries: int = 3, timeout: int = 20) -> requests.Response:
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            # Some cloudflare/proxy setups return 200 with error payloads; still raise_for_status first.
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            time.sleep(min(10, 1.5 ** attempt))
    assert last_err is not None
    raise last_err

@dataclass
class DataLoader:
    """
    Resilient loader: tries multiple Binance endpoints and falls back to local CSV or synthetic.
    Outputs 1-second bars with columns ['mid','bid','ask','close','volume'] indexed by UTC seconds.
    """
    store_dir: str = "./data"
    bar_seconds: int = 1
    session_tz: str = "UTC"
    user_agent: str = "exa-crypto-loader/0.1"
    base_urls: List[str] = field(default_factory=lambda: DEFAULT_BASE_URLS)

    @property
    def _headers(self) -> Dict[str, str]:
        return {"User-Agent": self.user_agent}

    # ---------------- Public API ----------------

    def ensure_local(
    self,
    symbol: str,
    *,
    lookback_minutes: int = 1000,
    force_refresh: bool = False,
    path_override: Optional[str] = None,
    # NEW flags:
    allow_cache: bool = True,
    allow_local_csv: bool = True,
    allow_synthetic: bool = True,
    require_remote: bool = False,   # if True: remote-only; no cache/CSV/synth; raise on failure
) -> pd.DataFrame:
        os.makedirs(self.store_dir, exist_ok=True)
        path = path_override or os.path.join(self.store_dir, f"{symbol}_1s.parquet")

        # Remote-only strict mode: no cache, no fallbacks
        if require_remote:
            kl = self.fetch_binance_klines(symbol=symbol, interval="1m",
                                        limit=min(lookback_minutes, 1000))
            sec = self.build_second_bars(kl)
            if allow_cache:  # still let you persist for later runs, if you want
                sec.to_parquet(path)
            return sec

        # Normal mode with optional cache
        if allow_cache and (not force_refresh) and os.path.exists(path):
            df = pd.read_parquet(path)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            return df

        # Try remote, then (optionally) CSV, then (optionally) synthetic
        try:
            kl = self.fetch_binance_klines(symbol=symbol, interval="1m",
                                        limit=min(lookback_minutes, 1000))
            sec = self.build_second_bars(kl)
        except Exception:
            if allow_local_csv:
                csv_path = os.path.join(self.store_dir, f"{symbol}_1m.csv")
                if os.path.exists(csv_path):
                    kl = self._read_local_klines_csv(csv_path)
                    sec = self.build_second_bars(kl)
                else:
                    sec = None
            else:
                sec = None

            if sec is None:
                if allow_synthetic:
                    sec = self._make_synthetic_1s_series(n_seconds=min(lookback_minutes*60, 6*60*60))
                else:
                    # hard fail — exactly what you want while testing Binance only
                    raise

        if allow_cache:
            sec.to_parquet(path)
        return sec

    # ---------------- Remote fetchers ----------------

    def fetch_binance_klines(self, symbol: str, interval: str = "1m", limit: int = 1000) -> pd.DataFrame:
        """
        Try multiple base URLs to avoid 451/geo blocks.
        """
        last_err: Optional[Exception] = None
        for base in self.base_urls:
            try:
                url = f"{base}/api/v3/klines"
                params = {"symbol": symbol, "interval": interval, "limit": int(limit)}
                r = _retry_get(url, params, self._headers, timeout=20)
                arr = r.json()
                # If we accidentally hit a page that returned HTML/str, json() may fail silently elsewhere
                if not isinstance(arr, list) or not arr:
                    raise RuntimeError(f"Unexpected response from {base}")
                df = self._klines_to_df(arr)
                return df
            except Exception as e:
                last_err = e
                continue
        assert last_err is not None
        raise last_err

    # ---------------- Builders & helpers ----------------

    def _klines_to_df(self, arr) -> pd.DataFrame:
        cols = [
            "open_time","open","high","low","close","volume",
            "close_time","qav","trades","tb_base","tb_quote","ignore"
        ]
        df = pd.DataFrame(arr, columns=cols)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        for c in ("open","high","low","close","volume","qav","tb_base","tb_quote"):
            df[c] = df[c].astype(float)
        df["trades"] = df["trades"].astype(int)
        return df.drop(columns=["ignore"])

    def _read_local_klines_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        # Be tolerant of epoch ms or ISO
        if np.issubdtype(df["open_time"].dtype, np.number):
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        else:
            df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
            df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
        for c in ("open","high","low","close","volume"):
            df[c] = df[c].astype(float)
        if "trades" in df.columns:
            df["trades"] = df["trades"].astype(int)
        else:
            df["trades"] = 0
        if "qav" not in df.columns: df["qav"]=0.0
        if "tb_base" not in df.columns: df["tb_base"]=0.0
        if "tb_quote" not in df.columns: df["tb_quote"]=0.0
        return df

    def build_second_bars(self, klines: pd.DataFrame) -> pd.DataFrame:
        if klines.empty:
            raise ValueError("`klines` is empty. Provide data or lower lookback.")
        start = klines["open_time"].min()
        end = klines["close_time"].max()
        idx = pd.date_range(start, end, freq=f"{self.bar_seconds}s", tz="UTC")

        s = klines.set_index("open_time")[["close","volume"]].copy()
        s = s.resample("1s").ffill()
        s = s.reindex(idx).ffill()

        ret = np.log(s["close"]).diff().fillna(0.0)
        rv = ret.rolling(60, min_periods=5).std().bfill()
        spread = (s["close"].abs() * (rv.clip(lower=1e-6) * 0.25)).clip(lower=1e-4)

        mid = s["close"].astype(float)
        bid = (mid - spread/2).astype(float)
        ask = (mid + spread/2).astype(float)

        out = pd.DataFrame(
            {"mid": mid, "bid": bid, "ask": ask, "close": s["close"].astype(float), "volume": s["volume"].fillna(0.0).astype(float)},
            index=idx,
        )
        out.index.name = "ts"
        return out

    def _make_synthetic_1s_series(self, n_seconds: int = 3600, s0: float = 30000.0, mu: float = 0.0, sigma: float = 0.50):
        """
        Simple GBM-style synthetic series for offline runs. Annualized sigma maps to per-second.
        """
        # Approx per-second vol from annualized sigma (rough heuristic)
        dt = 1.0 / (365*24*60*60)
        z = np.random.standard_normal(n_seconds)
        log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        price = s0 * np.exp(np.cumsum(log_ret))
        idx = pd.date_range(pd.Timestamp.utcnow().round("S") - pd.Timedelta(seconds=n_seconds-1),
                            periods=n_seconds, freq="1s", tz="UTC")
        close = pd.Series(price, index=idx)
        rv = close.pct_change().rolling(60, min_periods=5).std().bfill()
        spread = (close.abs() * (rv.clip(lower=1e-6) * 0.25)).clip(lower=1e-4)
        mid = close
        bid = mid - spread/2
        ask = mid + spread/2
        vol = pd.Series(0.5, index=idx)  # flat tiny volume
        out = pd.DataFrame({"mid": mid, "bid": bid, "ask": ask, "close": close, "volume": vol})
        out.index.name = "ts"
        return out

    # -------- Optional YAML universe helpers (unchanged) --------
    @staticmethod
    def load_universe_yaml(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg or {}

    def ensure_from_universe(self, universe_yaml: str) -> Dict[str, pd.DataFrame]:
        cfg = self.load_universe_yaml(universe_yaml)
        bar_seconds = int(cfg.get("bar_seconds", self.bar_seconds))
        store_dir = cfg.get("store_dir", self.store_dir)
        lookback_days = int(cfg.get("lookback_days", 3))
        results: Dict[str, pd.DataFrame] = {}
        for s in cfg.get("symbols", []):
            sym = s["symbol"]
            df = DataLoader(store_dir=store_dir, bar_seconds=bar_seconds).ensure_local(
                sym, lookback_minutes=lookback_days * 24 * 60   , force_refresh=True,
                allow_cache=False,
                allow_local_csv=False,
                allow_synthetic=False,
                require_remote=True,
            )
            results[sym] = df
        return results

__all__ = ["DataLoader"]
