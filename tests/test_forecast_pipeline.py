import pandas as pd

from btc_dashboard.backtest import build_evaluation_report
from btc_dashboard.config import ModelConfig
from btc_dashboard.features import build_feature_set
from btc_dashboard.models import build_forecast, candidate_specs
from tests.helpers import synthetic_market_data


def test_forecast_has_live_row_and_frozen_holdout():
    features = build_feature_set(synthetic_market_data())
    config = ModelConfig(
        min_train_days=120,
        train_window_days=240,
        holdout_days=40,
        tune_every_days=40,
        refit_every_days=7,
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
    assert len(forecast.standardized_coefficients) == len(
        forecast.coefficients
    )
    assert "Zero-return baseline" in forecast.validation_comparison.index
    assert forecast.frozen_spec.label in forecast.validation_comparison.index
    assert "Status" in forecast.validation_comparison.columns
    assert (
        forecast.validation_comparison.loc[
            forecast.frozen_spec.label, "Selected"
        ]
        == True
    )

    report, _ = build_evaluation_report(
        forecast.predictions,
        bootstrap_samples=20,
        block_size=5,
    )
    assert ("Frozen Holdout", "Regularized Model") in report.index
    assert ("Frozen Holdout", "Zero Return") in report.index


def test_candidate_grid_includes_small_elastic_net_penalties():
    penalties = {
        spec.alpha
        for spec in candidate_specs()
        if spec.family == "elastic_net"
    }

    assert 1e-6 in penalties
    assert 1e-5 in penalties
