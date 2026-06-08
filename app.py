from pathlib import Path
import time

import streamlit as st

from btc_dashboard.backtest import build_evaluation_report
from btc_dashboard.config import ModelConfig
from btc_dashboard.data import load_market_data, resolve_proxy
from btc_dashboard.features import build_feature_set
from btc_dashboard.models import build_forecast, fit_same_day_attribution
from btc_dashboard.ui import render_dashboard


BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="BTC Macro Research Terminal",
    layout="wide",
)

manual_refresh = st.sidebar.button("Refresh data", type="primary")
transaction_cost_bps = st.sidebar.number_input(
    "Round-trip cost assumption (bps)",
    min_value=0.0,
    max_value=100.0,
    value=10.0,
    step=1.0,
)
show_timing = st.sidebar.checkbox("Show timing", value=False)

secret_settings = {}
for key in ("BTC_PROXY_URL", "BTC_PROXY_PORT", "BTC_PROXY_AUTO_DETECT"):
    try:
        if key in st.secrets:
            secret_settings[key] = st.secrets[key]
    except Exception:
        pass
proxy_url, proxy_source = resolve_proxy(secret_settings)

if manual_refresh:
    st.cache_data.clear()


@st.cache_data(ttl=900, show_spinner="Loading completed market data...")
def cached_market_data(force_refresh, route):
    del route
    return load_market_data(
        BASE_DIR,
        proxy_url=proxy_url,
        force_refresh=force_refresh,
    )


@st.cache_data(show_spinner="Building session-aware features...")
def cached_features(data):
    return build_feature_set(data)


@st.cache_data(show_spinner="Running nested walk-forward model...")
def cached_forecast(feature_set, config):
    return build_forecast(feature_set, config)


started = time.perf_counter()
try:
    bundle = cached_market_data(manual_refresh, proxy_url or "direct")
    data_loaded = time.perf_counter()
    feature_set = cached_features(bundle.data)
    features_loaded = time.perf_counter()
    config = ModelConfig()
    forecast = cached_forecast(feature_set, config)
    attribution = fit_same_day_attribution(feature_set)
    evaluation, strategy_returns = build_evaluation_report(
        forecast.predictions,
        cost_bps=float(transaction_cost_bps),
        bootstrap_samples=config.bootstrap_samples,
        block_size=config.bootstrap_block_days,
        seed=config.random_seed,
    )
except (RuntimeError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

render_dashboard(
    bundle=bundle,
    feature_set=feature_set,
    forecast=forecast,
    attribution=attribution,
    evaluation=evaluation,
    strategy_returns=strategy_returns,
    transaction_cost_bps=float(transaction_cost_bps),
    proxy_source=proxy_source,
)

if show_timing:
    st.sidebar.write(f"Data: {data_loaded - started:.2f}s")
    st.sidebar.write(f"Features: {features_loaded - data_loaded:.2f}s")
    st.sidebar.write(f"Model + UI: {time.perf_counter() - features_loaded:.2f}s")
