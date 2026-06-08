import shutil
from pathlib import Path

import pandas as pd

from btc_dashboard.config import DataPaths
from btc_dashboard.data import read_real_yield_fallback


def test_bundled_seed_survives_cold_start(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source = DataPaths(Path(__file__).resolve().parents[1]).real_yield_seed_file
    shutil.copy2(source, data_dir / "real_yield_seed.csv")

    series, label, errors = read_real_yield_fallback(DataPaths(tmp_path))

    assert label == "bundled official seed"
    assert series.name == "REAL_YIELD"
    assert series.index[-1] == pd.Timestamp("2026-06-05")
    assert float(series.iloc[-1]) == 2.19
    assert not errors


def test_newer_complete_cache_beats_older_seed(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {"REAL_YIELD": [2.0]},
        index=[pd.Timestamp("2026-06-05")],
    ).to_csv(data_dir / "real_yield_seed.csv")
    pd.DataFrame(
        {"REAL_YIELD": [2.1]},
        index=[pd.Timestamp("2026-06-08")],
    ).to_csv(tmp_path / "backup_data.csv")

    series, label, _ = read_real_yield_fallback(DataPaths(tmp_path))

    assert label == "complete-data cache"
    assert series.index[-1] == pd.Timestamp("2026-06-08")
