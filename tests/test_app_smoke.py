from pathlib import Path
import json
import shutil

from streamlit.testing.v1 import AppTest

from btc_dashboard.config import SCHEMA_VERSION
from tests.helpers import synthetic_market_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_starts_from_complete_cache(tmp_path):
    shutil.copy2(PROJECT_ROOT / "app.py", tmp_path / "app.py")
    shutil.copytree(PROJECT_ROOT / "btc_dashboard", tmp_path / "btc_dashboard")
    data = synthetic_market_data(periods=900)
    data.index = __import__("pandas").date_range(
        end=__import__("pandas").Timestamp.now().normalize()
        - __import__("pandas").Timedelta(days=1),
        periods=len(data),
        freq="D",
    )
    data.to_csv(tmp_path / "backup_data.csv")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": str(__import__("pandas").Timestamp.utcnow()),
        "btc_source": "synthetic cache",
        "status": {
            "BTC": "cache",
            "NASDAQ": "cache",
            "DXY": "cache",
            "GOLD": "cache",
            "OIL": "cache",
            "REAL_YIELD": "cache",
        },
        "timestamps": {
            name: str(data.index[-1])
            for name in ("BTC", "NASDAQ", "DXY", "GOLD", "OIL", "REAL_YIELD")
        },
    }
    (tmp_path / "backup_data.meta.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    app = AppTest.from_file(str(tmp_path / "app.py"), default_timeout=180)
    app.run(timeout=180)

    assert not app.exception
    assert not app.error
