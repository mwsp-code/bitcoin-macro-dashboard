from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_synthetic_cache(path):
    rng = np.random.default_rng(42)
    index = pd.date_range(
        end=pd.Timestamp.now().normalize(),
        periods=1_200,
        freq="D",
        name="time",
    )

    def random_walk(start, volatility):
        shocks = rng.normal(0, volatility, len(index))
        return start * np.exp(np.cumsum(shocks))

    data = pd.DataFrame(
        {
            "BTC": random_walk(30_000, 0.02),
            "NASDAQ": random_walk(350, 0.008),
            "DXY": random_walk(100, 0.003),
            "GOLD": random_walk(1_800, 0.006),
            "OIL": random_walk(75, 0.015),
            "REAL_YIELD": 1.5 + np.cumsum(rng.normal(0, 0.02, len(index))),
        },
        index=index,
    )
    data.to_csv(path)


def test_dashboard_starts_from_complete_cache(tmp_path):
    app_path = tmp_path / "app.py"
    shutil.copy2(PROJECT_ROOT / "app.py", app_path)
    build_synthetic_cache(tmp_path / "backup_data.csv")

    app = AppTest.from_file(str(app_path), default_timeout=120)
    app.run(timeout=120)

    assert not app.exception
    assert not app.error
