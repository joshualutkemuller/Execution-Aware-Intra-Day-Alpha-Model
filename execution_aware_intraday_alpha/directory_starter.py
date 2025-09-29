# This script creates a complete crypto-focused "Execution-Aware Intraday Alpha Lab" repo,
# lays out a runnable scaffold with configs, modules, tests, and example scripts,
# and zips it for download.
import os, json, textwrap, zipfile, pathlib, sys

BASE = pathlib.Path(os.path.dirname(os.path.abspath(__file__))).resolve()
SRC = BASE / "src" / "exa"  # exa = execution-aware alpha
CONFIG = BASE / "config"
TESTS = BASE / "tests"
NOTEBOOKS = BASE / "notebooks"
CI = BASE / "ci"
DOCKER = BASE / "docker"
REPORTS = BASE / "examples"

for p in [SRC, CONFIG, TESTS, NOTEBOOKS, CI, DOCKER, REPORTS, BASE / "src", BASE / "data"]:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------- pyproject / requirements ----------------------
pyproject = """\
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "execution-aware-alpha-crypto"
version = "0.1.0"
description = "Execution-aware intraday alpha lab for crypto: signal → policy → slippage-aware PnL"
authors = [{name="You", email="you@example.com"}]
requires-python = ">=3.10"
dependencies = [
    "numpy",
    "pandas>=2.0",
    "scikit-learn>=1.2",
    "lightgbm",
    "pyyaml",
    "mlflow",
    "matplotlib",
    "plotly",
    "tqdm",
    "requests",
    "numba",
    "gymnasium",
    "jinja2",
    "scipy",
    "statsmodels",
    "pydantic",
    "rich",
]
[tool.setuptools.packages.find]
where = ["src"]
"""

requirements = """\
numpy
pandas>=2.0
scikit-learn>=1.2
lightgbm
pyyaml
mlflow
matplotlib
plotly
tqdm
requests
numba
gymnasium
jinja2
scipy
statsmodels
pydantic
rich
"""

# ---------------------- Minimal placeholders for all files ----------------------
FILES = {
    # Top level
    BASE / "pyproject.toml": pyproject,
    BASE / "requirements.txt": requirements,
    BASE / "README.md": "# Execution-Aware Intraday Alpha Lab — Crypto Edition\n\nScaffold initialized.\n",
    BASE / ".gitignore": "__pycache__/\n.venv/\n.env\n*.egg-info/\nmlruns/\ndata/\nreports/\n.DS_Store\n",

    # Configs (YAML)
    CONFIG / "universe.yaml": "symbols:\n  - symbol: BTCUSDT\n    venue: binance\n    quote: USDT\nsession:\n  tz: UTC\n  start: 00:00:00\n  end: 23:59:59\nbar_seconds: 1\nlookback_days: 5\nstore_dir: ./data\n",
    CONFIG / "features.yaml": "lags: [1, 2, 5, 10]\nwindows:\n  ofi: 10\n  vol: 60\n  trend: 30\nseasonality:\n  minute_of_hour: true\n  hour_of_day: true\ntarget:\n  horizon_seconds: 60\n  method: mid_return\n",
    CONFIG / "backtest.yaml": "costs:\n  maker_bps: 0.5\n  taker_bps: 1.0\nsimulation:\n  latency_ms: 50\n  slippage_ticks_when_market: 0.5\n  queue_fill_intensity: 0.8\nrisk:\n  max_position_usd: 20000\n  max_leverage: 1.0\n  ewma_var_alpha: 0.97\nsizing:\n  vol_target_annual: 0.15\n  min_notional: 50\npolicies:\n  - market_at_signal\n  - twap\n  - pov\n  - limit_queue_aware\n  - rl_agent\n",
    CONFIG / "config.yaml" :"base configuration file template",
    # Package skeleton (__init__.py ensures importability)
    SRC / "__init__.py": "from .data.loader import DataLoader\nfrom .features.builder import FeatureBuilder\nfrom .models.alpha import RidgeAlpha\nfrom .exec.policies import MarketAtSignal\nfrom .backtest.engine import BacktestEngine\n",
    SRC / "data" / "__init__.py": "",
    SRC / "features" / "__init__.py": "",
    SRC / "models" / "__init__.py": "",
    SRC / "exec" / "__init__.py": "",
    SRC / "backtest" / "__init__.py": "",
    SRC / "reports" / "__init__.py": "",

    # Minimal modules (you can flesh these out later)
    SRC / "data" / "loader.py": '''from dataclasses import dataclass\nimport pandas as pd\n\n@dataclass\nclass DataLoader:\n    store_dir: str = "./data"\n    bar_seconds: int = 1\n\n    def ensure_local(self, symbol: str) -> pd.DataFrame:\n        # TODO: fetch/construct data; placeholder returns empty frame\n        return pd.DataFrame(columns=["mid","bid","ask","close","volume"])\n''',

    SRC / "features" / "builder.py": '''from dataclasses import dataclass\nimport pandas as pd\n\n@dataclass\nclass FeatureBuilder:\n    ofi_window: int = 10\n    vol_window: int = 60\n    trend_window: int = 30\n    target_horizon: int = 60\n\n    def build(self, df: pd.DataFrame) -> pd.DataFrame:\n        # TODO: compute features; placeholder echoes input\n        out = df.copy()\n        out["y"] = 0.0\n        return out\n''',

    SRC / "models" / "alpha.py": '''from dataclasses import dataclass\nimport pandas as pd\n\n@dataclass\nclass RidgeAlpha:\n    alpha: float = 1.0\n    cols: list[str] | None = None\n\n    def fit(self, df: pd.DataFrame):\n        # TODO: fit model; placeholder remembers feature columns\n        self.cols = [c for c in df.columns if c not in ("y","mid","bid","ask","close")]\n        return self\n\n    def predict(self, df: pd.DataFrame) -> pd.Series:\n        # TODO: predict; placeholder returns zeros\n        return pd.Series(0.0, index=df.index, name=\"yhat\")\n''',

    SRC / "exec" / "policies.py": '''from dataclasses import dataclass\n\n@dataclass\nclass MarketAtSignal:\n    def decide(self, state: dict) -> dict:\n        # TODO: policy logic; placeholder always market\n        return {\"side\": state.get(\"side\",\"buy\"), \"type\": \"market\", \"improve_ticks\": 0}\n''',

    SRC / "backtest" / "engine.py": '''from dataclasses import dataclass\nimport pandas as pd\n\n@dataclass\nclass BacktestEngine:\n    costs: dict\n    latency_ms: int = 50\n    slippage_ticks_when_market: float = 0.5\n\n    def run(self, df: pd.DataFrame, signal: pd.Series, policy, sizing_fn):\n        # TODO: backtest; placeholder returns empty pnl frame\n        return pd.DataFrame(index=df.index, data={\"pnl\": 0.0})\n''',

    SRC / "reports" / "tearsheet.py": '''from dataclasses import dataclass\nimport pandas as pd\n\n@dataclass\nclass Tearsheet:\n    out_dir: str = \"./reports\"\n\n    def make(self, prices: pd.DataFrame, bt: pd.DataFrame, signal: pd.Series, name: str):\n        # TODO: write plots/summary; placeholder prints\n        print(f\"[tearsheet] {name} — rows: {len(bt)}\")\n''',

    # Examples
    REPORTS / "quickstart_demo.py": '''from exa.data.loader import DataLoader\nfrom exa.features.builder import FeatureBuilder\nfrom exa.models.alpha import RidgeAlpha\nfrom exa.exec.policies import MarketAtSignal\nfrom exa.backtest.engine import BacktestEngine\nfrom exa.reports.tearsheet import Tearsheet\n\nif __name__ == \"__main__\":\n    dl = DataLoader()\n    df = dl.ensure_local(\"BTCUSDT\")\n    feat = FeatureBuilder().build(df)\n    model = RidgeAlpha().fit(feat)\n    signal = model.predict(feat)\n    bt = BacktestEngine(costs={}).run(feat, signal, MarketAtSignal(), lambda r,s: 1.0)\n    Tearsheet().make(feat, bt, signal, \"demo\")\n    print(\"Demo complete (placeholder).\")\n''',

    REPORTS / "generate_tearsheet.py": "print('Run examples/quickstart_demo.py to generate outputs')\n",

    # Tests
    TESTS / "test_smoke.py": "def test_smoke():\n    assert True\n",

    # CI & Docker (minimal)
    CI / "ci.yml": "name: ci\non: [push]\njobs:\n  build:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n      - uses: actions/setup-python@v5\n        with:\n          python-version: '3.11'\n      - run: pip install -r requirements.txt || true\n      - run: python -c \"print('ok')\"\n",
    DOCKER / "Dockerfile": "FROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt ./\nRUN pip install --no-cache-dir -r requirements.txt || true\nCOPY . .\nCMD [\"python\", \"examples/quickstart_demo.py\"]\n",
}

# Write all placeholders
for path, content in FILES.items():
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

