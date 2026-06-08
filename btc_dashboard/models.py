from dataclasses import dataclass
from math import erf, sqrt
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ModelConfig


@dataclass(frozen=True)
class ModelSpec:
    family: str
    alpha: float
    l1_ratio: float = 0.0

    @property
    def label(self):
        if self.family == "elastic_net":
            return (
                f"ElasticNet(alpha={self.alpha:g}, "
                f"l1_ratio={self.l1_ratio:g})"
            )
        return f"Ridge(alpha={self.alpha:g})"


@dataclass
class ForecastResult:
    predictions: pd.DataFrame
    live_feature_date: pd.Timestamp
    live_target_date: pd.Timestamp
    live_prediction: float
    live_probability_up: float
    frozen_spec: ModelSpec
    coefficients: pd.Series
    standardized_coefficients: pd.Series
    intercept: float
    residual_std: float
    holdout_start: pd.Timestamp
    tuning_history: pd.DataFrame
    validation_comparison: pd.DataFrame


@dataclass
class AttributionResult:
    date: pd.Timestamp
    coefficients: pd.Series
    contributions: pd.Series
    intercept: float
    fitted_return: float
    actual_return: float
    residual: float


def candidate_specs():
    specs = [ModelSpec("ridge", alpha) for alpha in (0.1, 1.0, 10.0, 100.0)]
    specs.extend(
        ModelSpec("elastic_net", alpha, l1_ratio)
        for alpha in (0.000001, 0.00001, 0.0001, 0.001, 0.01)
        for l1_ratio in (0.1, 0.5, 0.9)
    )
    return specs


def make_estimator(spec):
    if spec.family == "elastic_net":
        model = ElasticNet(
            alpha=spec.alpha,
            l1_ratio=spec.l1_ratio,
            max_iter=5_000,
            tol=5e-3,
            selection="cyclic",
        )
    else:
        model = Ridge(alpha=spec.alpha)
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def _inner_cv_score(x, y, spec, splits, gap):
    if len(x) < max(60, splits * 20):
        return np.inf
    splitter = TimeSeriesSplit(n_splits=splits, gap=gap)
    fold_scores = []
    for train_idx, validation_idx in splitter.split(x):
        estimator = make_estimator(spec)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                estimator.fit(x.iloc[train_idx], y.iloc[train_idx])
            except ConvergenceWarning:
                return np.inf
        prediction = estimator.predict(x.iloc[validation_idx])
        fold_scores.append(mean_absolute_error(y.iloc[validation_idx], prediction))
    return float(np.mean(fold_scores))


def _zero_return_cv_score(x, y, splits, gap):
    if len(x) < max(60, splits * 20):
        return np.inf
    splitter = TimeSeriesSplit(n_splits=splits, gap=gap)
    fold_scores = []
    for _, validation_idx in splitter.split(x):
        prediction = np.zeros(len(validation_idx))
        fold_scores.append(mean_absolute_error(y.iloc[validation_idx], prediction))
    return float(np.mean(fold_scores))


def select_spec(x, y, config):
    scores = {
        spec: _inner_cv_score(
            x,
            y,
            spec,
            splits=config.inner_splits,
            gap=config.validation_gap_days,
        )
        for spec in candidate_specs()
    }
    return min(scores, key=scores.get), scores


def _coefficient_views(estimator, columns):
    scaler = estimator.named_steps["scale"]
    model = estimator.named_steps["model"]
    standardized = pd.Series(
        model.coef_,
        index=columns,
        name="Standardized Coefficient",
    )
    raw = model.coef_ / scaler.scale_
    intercept = float(model.intercept_ - np.dot(raw, scaler.mean_))
    return (
        pd.Series(raw, index=columns, name="Raw Coefficient"),
        standardized,
        intercept,
    )


def _validation_comparison(scores, zero_mae, selected_spec):
    rows = []
    for spec, score in scores.items():
        converged = np.isfinite(score)
        rows.append(
            {
                "Candidate": spec.label,
                "Validation MAE": score if converged else np.nan,
                "MAE vs Zero": score - zero_mae if converged else np.nan,
                "Improvement vs Zero": (
                    (zero_mae - score) / zero_mae
                    if converged and zero_mae > 0
                    else np.nan
                ),
                "Status": "Converged" if converged else "Did not converge",
                "Selected": spec == selected_spec,
            }
        )
    rows.append(
        {
            "Candidate": "Zero-return baseline",
            "Validation MAE": zero_mae,
            "MAE vs Zero": 0.0,
            "Improvement vs Zero": 0.0,
            "Status": "Baseline",
            "Selected": False,
        }
    )
    return (
        pd.DataFrame(rows)
        .set_index("Candidate")
        .sort_values("Validation MAE")
    )


def _historical_baselines(y_train, x_row):
    historical_mean = float(y_train.tail(90).mean())
    momentum = float(x_row.get("BTC_RETURN_7D", 0.0) / 7)
    return historical_mean, momentum


def build_forecast(feature_set, config=None):
    config = config or ModelConfig()
    x = feature_set.labeled_features.copy()
    y = feature_set.labeled_target.copy()
    if len(x) < config.min_train_days + config.holdout_days + 30:
        raise ValueError(
            "Need at least "
            f"{config.min_train_days + config.holdout_days + 30} labeled days; "
            f"found {len(x)}."
        )

    holdout_position = len(x) - config.holdout_days
    holdout_start = x.index[holdout_position]
    rows = []
    tuning_rows = []
    active_spec = None
    last_tuned = None
    development_estimator = None
    last_development_refit = None

    for position in range(config.min_train_days, holdout_position):
        start = max(0, position - config.train_window_days)
        x_train = x.iloc[start:position]
        y_train = y.iloc[start:position]
        specification_changed = False
        if active_spec is None or position - last_tuned >= config.tune_every_days:
            selected_spec, scores = select_spec(x_train, y_train, config)
            specification_changed = selected_spec != active_spec
            active_spec = selected_spec
            tuning_rows.append(
                {
                    "date": x.index[position],
                    "selected": active_spec.label,
                    "validation_mae": scores[active_spec],
                }
            )
            last_tuned = position
        if (
            development_estimator is None
            or specification_changed
            or position - last_development_refit >= config.refit_every_days
        ):
            development_estimator = make_estimator(active_spec).fit(
                x_train,
                y_train,
            )
            last_development_refit = position
        prediction = float(
            development_estimator.predict(x.iloc[[position]])[0]
        )
        mean_baseline, momentum_baseline = _historical_baselines(
            y_train, x.iloc[position]
        )
        rows.append(
            {
                "date": x.index[position],
                "segment": "Development",
                "actual": y.iloc[position],
                "model": prediction,
                "zero_return": 0.0,
                "historical_mean": mean_baseline,
                "momentum_7d": momentum_baseline,
                "always_up": 1e-8,
                "model_spec": active_spec.label,
            }
        )

    final_train_start = max(0, holdout_position - config.train_window_days)
    frozen_spec, final_scores = select_spec(
        x.iloc[final_train_start:holdout_position],
        y.iloc[final_train_start:holdout_position],
        config,
    )
    zero_validation_mae = _zero_return_cv_score(
        x.iloc[final_train_start:holdout_position],
        y.iloc[final_train_start:holdout_position],
        splits=config.inner_splits,
        gap=config.validation_gap_days,
    )
    validation_comparison = _validation_comparison(
        final_scores,
        zero_validation_mae,
        frozen_spec,
    )
    tuning_rows.append(
        {
            "date": holdout_start,
            "selected": frozen_spec.label,
            "validation_mae": final_scores[frozen_spec],
            "frozen_for_holdout": True,
        }
    )

    holdout_estimator = None
    last_holdout_refit = None
    for position in range(holdout_position, len(x)):
        start = max(0, position - config.train_window_days)
        x_train = x.iloc[start:position]
        y_train = y.iloc[start:position]
        if (
            holdout_estimator is None
            or position - last_holdout_refit >= config.refit_every_days
        ):
            holdout_estimator = make_estimator(frozen_spec).fit(
                x_train,
                y_train,
            )
            last_holdout_refit = position
        prediction = float(holdout_estimator.predict(x.iloc[[position]])[0])
        mean_baseline, momentum_baseline = _historical_baselines(
            y_train, x.iloc[position]
        )
        rows.append(
            {
                "date": x.index[position],
                "segment": "Frozen Holdout",
                "actual": y.iloc[position],
                "model": prediction,
                "zero_return": 0.0,
                "historical_mean": mean_baseline,
                "momentum_7d": momentum_baseline,
                "always_up": 1e-8,
                "model_spec": frozen_spec.label,
            }
        )

    prediction_frame = pd.DataFrame(rows).set_index("date")
    live_start = max(0, len(x) - config.train_window_days)
    live_estimator = make_estimator(frozen_spec).fit(
        x.iloc[live_start:], y.iloc[live_start:]
    )
    live_row = feature_set.inference_features.iloc[-1]
    live_prediction = float(
        live_estimator.predict(live_row.to_frame().T)[0]
    )
    development_predictions = prediction_frame.loc[
        prediction_frame["segment"] == "Development"
    ]
    residual_std = float(
        np.std(
            development_predictions["actual"]
            - development_predictions["model"],
            ddof=1,
        )
    )
    if not np.isfinite(residual_std) or residual_std <= 0:
        fitted_train = live_estimator.predict(x.iloc[live_start:])
        residual_std = float(
            np.std(y.iloc[live_start:].to_numpy() - fitted_train, ddof=1)
        )
    if not np.isfinite(residual_std) or residual_std <= 0:
        probability_up = 0.5
    else:
        z_value = live_prediction / residual_std
        probability_up = 0.5 * (1 + erf(z_value / sqrt(2)))
    coefficients, standardized_coefficients, intercept = _coefficient_views(
        live_estimator, feature_set.features.columns
    )
    live_feature_date = feature_set.inference_features.index[-1]
    return ForecastResult(
        predictions=prediction_frame,
        live_feature_date=live_feature_date,
        live_target_date=live_feature_date + pd.Timedelta(days=1),
        live_prediction=live_prediction,
        live_probability_up=float(probability_up),
        frozen_spec=frozen_spec,
        coefficients=coefficients,
        standardized_coefficients=standardized_coefficients,
        intercept=intercept,
        residual_std=residual_std,
        holdout_start=holdout_start,
        tuning_history=pd.DataFrame(tuning_rows).set_index("date"),
        validation_comparison=validation_comparison,
    )


def fit_same_day_attribution(feature_set, alpha=1.0):
    frame = feature_set.attribution_frame.dropna()
    feature_columns = [
        column for column in frame if column != "BTC_LOG_RETURN"
    ]
    if len(frame) < 60 or not feature_columns:
        raise ValueError("Insufficient active macro sessions for attribution.")
    x = frame[feature_columns]
    y = frame["BTC_LOG_RETURN"]
    estimator = make_estimator(ModelSpec("ridge", alpha)).fit(x, y)
    coefficients, _, intercept = _coefficient_views(estimator, feature_columns)
    latest = x.iloc[-1]
    fitted = float(estimator.predict(latest.to_frame().T)[0])
    actual = float(y.iloc[-1])
    contributions = (coefficients * latest).rename("Contribution")
    return AttributionResult(
        date=latest.name,
        coefficients=coefficients,
        contributions=contributions,
        intercept=intercept,
        fitted_return=fitted,
        actual_return=actual,
        residual=actual - fitted,
    )
