# Changelog

All notable changes to this project will be documented here.

## Unreleased

### Added

- Modular data, feature, model, backtest, and Streamlit UI packages.
- Binance-first completed UTC OHLCV data with taker-flow and activity features.
- Native-session macro lags, explicit data-age features, and a true live
  unknown-target inference row.
- Nested time-series tuning for Ridge and ElasticNet models with a frozen
  180-day holdout.
- Zero-return, rolling-mean, momentum, and always-up baselines with
  moving-block bootstrap confidence intervals.
- Standardized coefficient reporting, explicit intercept-only model status,
  and a full candidate validation-MAE comparison against zero return.
- A concise on-dashboard research summary and an intuitive $10,000 frozen
  holdout comparison with ending values and drawdown.
- Automatic detection of common local proxy ports.
- Nasdaq ETF macro fallback using QQQ, UUP, GLD, and USO.
- U.S. Treasury fallback for the 10-year real yield.
- Dedicated real-yield cache and cache source metadata.
- Data freshness, stale-cache, and historical-mode reporting.
- Offline Streamlit smoke test and GitHub Actions CI.

### Changed

- The top-level `app.py` now orchestrates package APIs instead of mixing data,
  model, backtest, and rendering logic.
- BTC source priority is Binance, HTX, OKX, then CoinGecko.
- FRED requests now use the proxy-aware HTTP session.
- Cached data is accepted only when both the file and observations are recent.
- Macro proxy fallbacks use a consistent full history instead of splicing instruments.
- The same-day fitted return includes the regression intercept.
- The main application entry point is `app.py`.

### Fixed

- Incomplete daily BTC candles no longer enter model training or evaluation.
- Weekend and holiday macro lags now refer to the prior observed market
  session rather than a forward-filled calendar-day zero.
- Weekend launches no longer fail only because FRED has no new observation.
- Old complete caches are no longer presented as current live signals.
- Streamlit cold starts retain `REAL_YIELD` through a bundled official seed
  when FRED and Treasury are temporarily unavailable.
- Same-day driver attribution excludes forward-filled weekends and market
  holidays, and reports the latest active macro-session date.
