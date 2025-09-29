from __future__ import annotations
from pathlib import Path
import textwrap, json, os
import sys

base = Path(__file__).resolve().parents[1]
print(base)
examples_path = base / "examples" / "quickstart_demo.py"
sys.path.insert(0, str(base / "src"))


import logging
import yaml
import numpy as np
import pandas as pd
from exa.data.loader import DataLoader
# from exa.features.builder import FeatureBuilder
# from exa.models.alpha import RidgeAlpha, LightGBMAlpha
# from exa.exec.policies import MarketAtSignal, LimitQueueAware
# from exa.backtest.engine import BacktestEngine
# from exa.reports.tearsheet import Tearsheet


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("exa.quickstart")


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def make_feature_builder(cfg: dict) -> FeatureBuilder:
    return FeatureBuilder(
        ofi_window=int(cfg.get("windows", {}).get("ofi", 10)),
        vol_window=int(cfg.get("windows", {}).get("vol", 60)),
        trend_window=int(cfg.get("windows", {}).get("trend", 30)),
        target_horizon=int(cfg.get("target", {}).get("horizon_seconds", 60)),
    )


def choose_policy(policies: list[str]):
    if "limit_queue_aware" in policies:
        return LimitQueueAware(improve_ticks=1, queue_fill_intensity=0.7)
    return MarketAtSignal()


def vol_scaled_sizer(row: pd.Series, sig: float, vol_floor: float = 1e-6, base_qty: float = 0.001) -> float:
    scale = min(0.02, max(base_qty, abs(sig) / (float(row.get("vol", 0.0)) + vol_floor)))
    return float(scale)


if __name__ == "__main__":
    log = setup_logger()

    repo_root = Path(__file__).resolve().parents[1]
    cfg_dir = repo_root / "config"

    uni_cfg = load_yaml(cfg_dir / "universe.yaml")
    feat_cfg = load_yaml(cfg_dir / "features.yaml")
    bt_cfg = load_yaml(cfg_dir / "backtest.yaml")

    symbol = uni_cfg.get("symbols", [{"symbol": "BTCUSDT"}])[0]["symbol"]
    store_dir = uni_cfg.get("store_dir", "./data")
    bar_seconds = int(uni_cfg.get("bar_seconds", 1))
    lookback_days = int(uni_cfg.get("lookback_days", 2))

    log.info(f"Symbol={symbol} | store_dir={store_dir} | bar={bar_seconds}s | lookback_days={lookback_days}")

    # 1) Data
    dl = DataLoader(store_dir=store_dir, bar_seconds=bar_seconds)
    df = dl.ensure_local(symbol, lookback_minutes=lookback_days * 24 * 60,     force_refresh=True,
        allow_cache=False,
        allow_local_csv=False,
        allow_synthetic=False,
        require_remote=True,)
    df.reset_index(drop=False,inplace=True)
    df.rename(columns={'ts':'timestamp'},inplace=True)
    df.to_csv(repo_root/ "data"/ f"{symbol}.csv")
    log.info(f"Loaded data rows={len(df):,}, range=[{df.index.min()} .. {df.index.max()}]")

    # 2) Features
    fb = make_feature_builder(feat_cfg)
    feat = fb.build(df)
    log.info(f"Features built rows={len(feat):,}, cols={list(feat.columns)}")

    # 3) Model
    model = RidgeAlpha(alpha=10.0).fit(feat)
    signal = model.predict(feat)
    log.info("Model fit complete; sample signal stats: "
             f"mean={signal.mean():.3e}, std={signal.std():.3e}, corr(y)={signal.corr(feat['y']):.3f}")

    # 4) Policy & Backtest
    policies_list = bt_cfg.get("policies", ["market_at_signal"])
    policy = choose_policy(policies_list)

    costs = bt_cfg.get("costs", {"maker_bps": 0.5, "taker_bps": 1.0})
    sim = bt_cfg.get("simulation", {})
    engine = BacktestEngine(
        costs=costs,
        latency_ms=int(sim.get("latency_ms", 50)),
        slippage_ticks_when_market=float(sim.get("slippage_ticks_when_market", 0.5)),
    )

    bt = engine.run(feat, signal, policy, sizing_fn=lambda r, s: vol_scaled_sizer(r, s))

    # 5) Tearsheet
    reports_dir = repo_root / "reports"
    Tearsheet(out_dir=str(reports_dir)).make(feat, bt, signal, name=f"{symbol.lower()}_demo")

    log.info(f"Done. Reports saved to {reports_dir.resolve()}")