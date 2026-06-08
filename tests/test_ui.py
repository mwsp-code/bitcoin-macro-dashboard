import numpy as np
import pandas as pd

from btc_dashboard.ui import build_holdout_wealth, plot_holdout_wealth


def test_holdout_wealth_uses_equal_starting_capital_and_tracks_drawdown():
    index = pd.date_range("2026-01-01", periods=4, freq="D")
    strategy = pd.Series([0.01, -0.02, 0.03, 0.0], index=index)
    bitcoin = pd.Series([0.02, -0.04, 0.01, 0.01], index=index)

    wealth, drawdown = build_holdout_wealth(strategy, bitcoin)

    assert list(wealth.columns) == ["Model Strategy", "BTC Buy & Hold"]
    assert np.isclose(wealth.iloc[-1, 0], 10_000 * np.exp(strategy.sum()))
    assert np.isclose(wealth.iloc[-1, 1], 10_000 * np.exp(bitcoin.sum()))
    assert (drawdown <= 0).all().all()
    figure = plot_holdout_wealth(wealth, drawdown)
    assert len(figure.axes) == 2
