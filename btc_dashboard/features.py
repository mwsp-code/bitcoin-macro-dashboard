from dataclasses import dataclass

import numpy as np
import pandas as pd


MACRO_COLUMNS = ["NASDAQ", "DXY", "GOLD", "OIL", "REAL_YIELD"]


@dataclass
class FeatureSet:
    features: pd.DataFrame
    target: pd.Series
    labeled_features: pd.DataFrame
    labeled_target: pd.Series
    inference_features: pd.DataFrame
    btc_returns: pd.Series
    attribution_frame: pd.DataFrame
    feature_groups: dict


def _rolling_z(series, window=30, min_periods=15):
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan)


def _native_macro_changes(data, column):
    observed_name = f"{column}_OBSERVED"
    if observed_name in data:
        observed = data[observed_name].fillna(False).astype(bool)
    else:
        observed = data.index.to_series().dt.dayofweek.lt(5)
    native = data.loc[observed, column].dropna()
    if column == "REAL_YIELD":
        changes = native.diff()
    else:
        changes = np.log(native).diff()
    current = changes.reindex(data.index).ffill()
    previous = changes.shift(1).reindex(data.index).ffill()
    return current, previous, observed


def _safe_log_ratio(numerator, denominator):
    ratio = numerator.astype(float) / denominator.astype(float).replace(0, np.nan)
    return np.log(ratio.where(ratio > 0))


def build_feature_set(data):
    data = data.sort_index().copy()
    close = data["BTC"].astype(float)
    btc_return = np.log(close).diff().rename("BTC_LOG_RETURN")
    features = pd.DataFrame(index=data.index)

    for horizon in (1, 2, 3, 7, 14, 30):
        features[f"BTC_RETURN_{horizon}D"] = np.log(close).diff(horizon)
    for window in (7, 14, 30):
        features[f"BTC_REALIZED_VOL_{window}D"] = (
            btc_return.rolling(window).std() * np.sqrt(365)
        )

    if {"BTC_HIGH", "BTC_LOW"}.issubset(data.columns):
        features["BTC_INTRADAY_RANGE"] = _safe_log_ratio(
            data["BTC_HIGH"], data["BTC_LOW"]
        )
    if {"BTC", "BTC_OPEN"}.issubset(data.columns):
        features["BTC_BODY_RETURN"] = _safe_log_ratio(
            data["BTC"], data["BTC_OPEN"]
        )

    if "BTC_VOLUME" in data:
        log_volume = np.log1p(data["BTC_VOLUME"].where(data["BTC_VOLUME"] >= 0))
        if log_volume.notna().mean() >= 0.7:
            features["BTC_VOLUME_Z_30D"] = _rolling_z(log_volume)
            features["BTC_VOLUME_CHANGE"] = log_volume.diff()
    if {"BTC_TAKER_BUY_VOLUME", "BTC_VOLUME"}.issubset(data.columns):
        taker_ratio = data["BTC_TAKER_BUY_VOLUME"] / data["BTC_VOLUME"].replace(
            0, np.nan
        )
        if taker_ratio.notna().mean() >= 0.7:
            features["BTC_TAKER_BUY_RATIO"] = taker_ratio.clip(0, 1)
            features["BTC_TAKER_IMBALANCE"] = 2 * taker_ratio.clip(0, 1) - 1
    if "BTC_TRADES" in data:
        log_trades = np.log1p(data["BTC_TRADES"].where(data["BTC_TRADES"] >= 0))
        if log_trades.notna().mean() >= 0.7:
            features["BTC_TRADES_Z_30D"] = _rolling_z(log_trades)

    attribution = pd.DataFrame(index=data.index)
    macro_feature_names = []
    for column in MACRO_COLUMNS:
        current, previous, observed = _native_macro_changes(data, column)
        current_name = f"{column}_SESSION_CHANGE"
        lag_name = f"{column}_PREVIOUS_SESSION_CHANGE"
        features[current_name] = current
        features[lag_name] = previous
        attribution[current_name] = current.where(observed)
        macro_feature_names.extend([current_name, lag_name])
        age_name = f"{column}_AGE_DAYS"
        if age_name in data:
            features[age_name] = data[age_name].astype(float).clip(0, 14)

    features["IS_WEEKEND"] = (features.index.dayofweek >= 5).astype(float)
    features["DAY_OF_WEEK_SIN"] = np.sin(2 * np.pi * features.index.dayofweek / 7)
    features["DAY_OF_WEEK_COS"] = np.cos(2 * np.pi * features.index.dayofweek / 7)

    features = features.replace([np.inf, -np.inf], np.nan)
    minimum_coverage = max(60, int(len(features) * 0.6))
    usable_columns = [
        column
        for column in features
        if features[column].notna().sum() >= minimum_coverage
        and features[column].nunique(dropna=True) > 1
    ]
    features = features[usable_columns].dropna()
    target = btc_return.shift(-1).rename("NEXT_DAY_BTC_LOG_RETURN")
    labeled = features.join(target).dropna()
    if labeled.empty or features.empty:
        raise ValueError("Insufficient complete observations for feature engineering.")

    latest_date = features.index[-1]
    inference_features = features.loc[[latest_date]]
    labeled_features = labeled[features.columns]
    labeled_target = labeled[target.name]

    attribution["BTC_LOG_RETURN"] = btc_return
    attribution = attribution.dropna(subset=["BTC_LOG_RETURN"])
    attribution = attribution.loc[
        attribution.drop(columns=["BTC_LOG_RETURN"]).notna().any(axis=1)
    ]

    crypto_names = [
        column for column in features if column.startswith("BTC_")
    ]
    calendar_names = [
        column
        for column in ("IS_WEEKEND", "DAY_OF_WEEK_SIN", "DAY_OF_WEEK_COS")
        if column in features
    ]
    age_names = [column for column in features if column.endswith("_AGE_DAYS")]
    return FeatureSet(
        features=features,
        target=target,
        labeled_features=labeled_features,
        labeled_target=labeled_target,
        inference_features=inference_features,
        btc_returns=btc_return,
        attribution_frame=attribution,
        feature_groups={
            "crypto": crypto_names,
            "macro": [name for name in macro_feature_names if name in features],
            "freshness": age_names,
            "calendar": calendar_names,
        },
    )
