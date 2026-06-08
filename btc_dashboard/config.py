from dataclasses import dataclass
from pathlib import Path


SCHEMA_VERSION = 2
REQUIRED_MARKET_COLUMNS = ["BTC", "NASDAQ", "DXY", "GOLD", "OIL", "REAL_YIELD"]
YFINANCE_TICKERS = [
    ("QQQ", "NASDAQ"),
    ("DX-Y.NYB", "DXY"),
    ("GC=F", "GOLD"),
    ("CL=F", "OIL"),
]
NASDAQ_FALLBACK_TICKERS = [
    ("QQQ", "NASDAQ"),
    ("UUP", "DXY"),
    ("GLD", "GOLD"),
    ("USO", "OIL"),
]
COMMON_LOCAL_PROXY_PORTS = (7890, 7897, 10809, 1080)


@dataclass(frozen=True)
class DataPaths:
    base_dir: Path

    @property
    def cache_file(self):
        return self.base_dir / "backup_data.csv"

    @property
    def cache_metadata_file(self):
        return self.base_dir / "backup_data.meta.json"

    @property
    def real_yield_cache_file(self):
        return self.base_dir / "real_yield_cache.csv"

    @property
    def real_yield_seed_file(self):
        return self.base_dir / "data" / "real_yield_seed.csv"


@dataclass(frozen=True)
class ModelConfig:
    min_train_days: int = 365
    train_window_days: int = 730
    holdout_days: int = 180
    tune_every_days: int = 30
    inner_splits: int = 4
    validation_gap_days: int = 1
    bootstrap_samples: int = 400
    bootstrap_block_days: int = 7
    random_seed: int = 42
    trading_days_per_year: int = 365
