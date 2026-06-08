import pandas as pd

import btc_dashboard.data as market_data
from btc_dashboard.data import parse_binance_klines


def _row(open_time, close_time, close):
    return [
        open_time,
        "100",
        "110",
        "90",
        str(close),
        "1000",
        close_time,
        "100000",
        500,
        "550",
        "55000",
        "0",
    ]


def test_binance_parser_excludes_unfinished_daily_bar():
    completed_open = int(pd.Timestamp("2026-06-07", tz="UTC").timestamp() * 1000)
    partial_open = int(pd.Timestamp("2026-06-08", tz="UTC").timestamp() * 1000)
    completed_close = completed_open + 86_400_000 - 1
    partial_close = partial_open + 86_400_000 - 1

    frame = parse_binance_klines(
        [
            _row(completed_open, completed_close, 105),
            _row(partial_open, partial_close, 106),
        ],
        now_utc=pd.Timestamp("2026-06-08 12:00:00"),
    )

    assert list(frame.index) == [pd.Timestamp("2026-06-07")]
    assert frame.iloc[-1]["BTC"] == 105


def test_btc_source_priority_starts_with_binance(monkeypatch):
    expected = pd.DataFrame(
        {"BTC": range(500)},
        index=pd.date_range("2025-01-01", periods=500, freq="D"),
    )
    for column in market_data.BTC_COLUMNS:
        if column not in expected:
            expected[column] = 1.0

    monkeypatch.setattr(
        market_data,
        "load_btc_binance",
        lambda session: expected[market_data.BTC_COLUMNS],
    )

    def unexpected_call(session):
        raise AssertionError("fallback should not run after Binance succeeds")

    monkeypatch.setattr(market_data, "load_btc_huobi", unexpected_call)
    frame, source, errors = market_data.load_btc(object())

    assert source == "binance"
    assert len(frame) == 500
    assert not errors
