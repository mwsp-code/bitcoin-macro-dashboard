from pathlib import Path

import pandas as pd

from real_yield_data import load_real_yield


class FailedSession:
    def get(self, *args, **kwargs):
        raise TimeoutError("simulated source outage")


class Response:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        return None


class FredSession:
    def get(self, url, **kwargs):
        assert "fredgraph.csv" in url
        return Response(
            "DATE,DFII10\n"
            "2026-06-05,2.19\n"
            "2026-06-06,.\n"
        )


def write_seed(path):
    pd.DataFrame(
        {"REAL_YIELD": [2.07, 2.19]},
        index=pd.to_datetime(["2026-06-01", "2026-06-05"]),
    ).to_csv(path)


def test_uses_bundled_seed_when_live_sources_fail(tmp_path):
    seed_file = tmp_path / "seed.csv"
    cache_file = tmp_path / "runtime.csv"
    write_seed(seed_file)

    series, status, timestamp, errors = load_real_yield(
        FailedSession(),
        cache_file,
        seed_file,
        now="2026-06-08",
    )

    assert status == "stale bundled/cache fallback"
    assert timestamp == pd.Timestamp("2026-06-05")
    assert series.iloc[-1] == 2.19
    assert len(errors) == 2


def test_fred_updates_seed_and_ignores_missing_values(tmp_path):
    seed_file = tmp_path / "seed.csv"
    cache_file = tmp_path / "runtime.csv"
    write_seed(seed_file)

    series, status, timestamp, errors = load_real_yield(
        FredSession(),
        cache_file,
        seed_file,
        now="2026-06-08",
    )

    assert status == "live (FRED)"
    assert timestamp == pd.Timestamp("2026-06-05")
    assert series.iloc[-1] == 2.19
    assert not errors
    assert Path(cache_file).exists()
