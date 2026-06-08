import numpy as np
import pandas as pd


MODEL_COLUMNS = {
    "model": "Regularized Model",
    "zero_return": "Zero Return",
    "historical_mean": "Rolling Mean",
    "momentum_7d": "7D Momentum",
    "always_up": "Always Up",
}


def annualized_sharpe(returns, periods=365):
    returns = pd.Series(returns).dropna()
    if len(returns) < 2 or returns.std() == 0:
        return np.nan
    return float(np.sqrt(periods) * returns.mean() / returns.std())


def max_drawdown(equity):
    equity = pd.Series(equity).dropna()
    if equity.empty:
        return np.nan
    return float((equity / equity.cummax() - 1).min())


def strategy_returns(prediction, actual, cost_bps=10.0):
    cost = cost_bps / 10_000
    prediction = pd.Series(prediction, index=actual.index)
    position = pd.Series(
        np.where(
            prediction > cost,
            1.0,
            np.where(prediction < -cost, -1.0, 0.0),
        ),
        index=actual.index,
    )
    turnover = position.diff().abs().fillna(position.abs())
    returns = position * actual - turnover * cost
    return returns, position


def _metric_values(actual, prediction, cost_bps):
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = actual - prediction
    direction = np.mean(np.sign(prediction) == np.sign(actual))
    correlation = (
        np.corrcoef(actual, prediction)[0, 1]
        if np.std(actual) > 0 and np.std(prediction) > 0
        else np.nan
    )
    zero_sse = np.sum(actual**2)
    predictive_r2 = 1 - np.sum(error**2) / zero_sse if zero_sse > 0 else np.nan
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "Directional Accuracy": float(direction),
        "Information Coefficient": float(correlation),
        "Predictive R2 vs Zero": float(predictive_r2),
    }


def _block_bootstrap_intervals(
    actual,
    prediction,
    cost_bps,
    samples=400,
    block_size=7,
    seed=42,
):
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    n_obs = len(actual)
    if n_obs < max(20, block_size * 2) or samples <= 0:
        return {}
    rng = np.random.default_rng(seed)
    metric_samples = {
        "MAE": [],
        "Directional Accuracy": [],
        "Information Coefficient": [],
    }
    block_starts = np.arange(max(1, n_obs - block_size + 1))
    blocks_needed = int(np.ceil(n_obs / block_size))
    for _ in range(samples):
        starts = rng.choice(block_starts, size=blocks_needed, replace=True)
        indices = np.concatenate(
            [np.arange(start, min(start + block_size, n_obs)) for start in starts]
        )[:n_obs]
        values = _metric_values(actual[indices], prediction[indices], cost_bps)
        for metric in metric_samples:
            metric_samples[metric].append(values[metric])
    intervals = {}
    for metric, values in metric_samples.items():
        finite = np.asarray(values)[np.isfinite(values)]
        if finite.size:
            intervals[metric] = tuple(np.quantile(finite, [0.025, 0.975]))
    return intervals


def build_evaluation_report(
    predictions,
    cost_bps=10.0,
    bootstrap_samples=400,
    block_size=7,
    seed=42,
):
    report_rows = []
    strategy_series = {}
    for segment, segment_frame in predictions.groupby("segment", sort=False):
        actual = segment_frame["actual"]
        for column, label in MODEL_COLUMNS.items():
            prediction = segment_frame[column]
            metrics = _metric_values(actual, prediction, cost_bps)
            intervals = _block_bootstrap_intervals(
                actual,
                prediction,
                cost_bps,
                samples=bootstrap_samples,
                block_size=block_size,
                seed=seed,
            )
            strategy, position = strategy_returns(prediction, actual, cost_bps)
            equity = np.exp(strategy.cumsum())
            key = f"{segment}: {label}"
            strategy_series[key] = strategy
            report_rows.append(
                {
                    "Segment": segment,
                    "Forecast": label,
                    "Observations": len(segment_frame),
                    **metrics,
                    "MAE 95% CI": _format_interval(intervals.get("MAE")),
                    "Direction 95% CI": _format_interval(
                        intervals.get("Directional Accuracy"), percent=True
                    ),
                    "IC 95% CI": _format_interval(
                        intervals.get("Information Coefficient")
                    ),
                    "Strategy Sharpe": annualized_sharpe(strategy),
                    "Strategy Return": float(equity.iloc[-1] - 1),
                    "Strategy Max Drawdown": max_drawdown(equity),
                    "Average Exposure": float(position.abs().mean()),
                }
            )
    report = pd.DataFrame(report_rows).set_index(["Segment", "Forecast"])
    return report, pd.DataFrame(strategy_series)


def _format_interval(interval, percent=False):
    if not interval:
        return "n/a"
    if percent:
        return f"{interval[0]:.1%} to {interval[1]:.1%}"
    return f"{interval[0]:.4f} to {interval[1]:.4f}"
