import numpy as np
import pandas as pd

from btc_dashboard.backtest import build_evaluation_report
from btc_dashboard.config import ModelConfig
from btc_dashboard.features import build_feature_set
from btc_dashboard.models import build_forecast


def synthetic_market_data(periods=420):
    rng = np.random.default_rng(7)
    index = pd.date_range("2025-01-01", periods=periods, freq="D")
    btc_returns = rng.normal(0.0005, 0.018, periods)
    close = 50_000 * np.exp(np.cumsum(btc_returns))
    data = pd.DataFrame(index=index)
    data["BTC"] = close
    data["BTC_OPEN"] = close * np.exp(rng.normal(0, 0.003, periods))
    data["BTC_HIGH"] = np.maximum(data["BTC"], data["BTC_OPEN"]) * 1.01
    data["BTC_LOW"] = np.minimum(data["BTC"], data["BTC_OPEN"]) * 0.99
    data["BTC_VOLUME"] = rng.lognormal(10, 0.25, periods)
    data["BTC_QUOTE_VOLUME"] = data["BTC_VOLUME"] * data["BTC"]
    data["BTC_TRADES"] = rng.lognormal(12, 0.15, periods)
    data["BTC_TAKER_BUY_VOLUME"] = data["BTC_VOLUME"] * rng.uniform(
        0.42, 0.58, periods
    )

    for name, start, volatility in (
        ("NASDAQ", 400, 0.008),
        ("DXY", 100, 0.003),
        ("GOLD", 2000, 0.006),
        ("OIL", 75, 0.014),
    ):
        observed = index.dayofweek < 5
        native_values = start * np.exp(
            np.cumsum(rng.normal(0, volatility, observed.sum()))
        )
        series = pd.Series(np.nan, index=index)
        series.loc[observed] = native_values
        data[name] = series.ffill().bfill()
        data[f"{name}_OBSERVED"] = observed
        observation_dates = pd.Series(pd.NaT, index=index, dtype="datetime64[ns]")
        observation_dates.loc[observed] = index[observed]
        data[f"{name}_AGE_DAYS"] = (
            index.to_series(index=index) - observation_dates.ffill().bfill()
        ).dt.days

    observed = index.dayofweek < 5
    real_yield = pd.Series(np.nan, index=index)
    real_yield.loc[observed] = 1.5 + np.cumsum(
        rng.normal(0, 0.02, observed.sum())
    )
    data["REAL_YIELD"] = real_yield.ffill().bfill()
    data["REAL_YIELD_OBSERVED"] = observed
    observation_dates = pd.Series(pd.NaT, index=index, dtype="datetime64[ns]")
    observation_dates.loc[observed] = index[observed]
    data["REAL_YIELD_AGE_DAYS"] = (
        index.to_series(index=index) - observation_dates.ffill().bfill()
    ).dt.days
    return data


def test_forecast_has_live_row_and_frozen_holdout():
    features = build_feature_set(synthetic_market_data())
    config = ModelConfig(
        min_train_days=120,
        train_window_days=240,
        holdout_days=40,
        tune_every_days=40,
        inner_splits=3,
        bootstrap_samples=20,
    )
    forecast = build_forecast(features, config)

    assert forecast.live_feature_date == features.features.index[-1]
    assert forecast.live_feature_date not in features.labeled_target.index[-1:]
    assert forecast.live_target_date == forecast.live_feature_date + pd.Timedelta(
        days=1
    )
    holdout = forecast.predictions.query("segment == 'Frozen Holdout'")
    assert len(holdout) == config.holdout_days
    assert holdout["model_spec"].nunique() == 1
    assert holdout["model_spec"].iloc[0] == forecast.frozen_spec.label

    report, _ = build_evaluation_report(
        forecast.predictions,
        bootstrap_samples=20,
        block_size=5,
    )
    assert ("Frozen Holdout", "Regularized Model") in report.index
    assert ("Frozen Holdout", "Zero Return") in report.index
