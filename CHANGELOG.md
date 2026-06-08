# Changelog

All notable changes to this project will be documented here.

## Unreleased

### Added

- Automatic detection of common local proxy ports.
- Nasdaq ETF macro fallback using QQQ, UUP, GLD, and USO.
- U.S. Treasury fallback for the 10-year real yield.
- Dedicated real-yield cache and cache source metadata.
- Data freshness, stale-cache, and historical-mode reporting.
- Offline Streamlit smoke test and GitHub Actions CI.

### Changed

- FRED requests now use the proxy-aware HTTP session.
- Cached data is accepted only when both the file and observations are recent.
- Macro proxy fallbacks use a consistent full history instead of splicing instruments.
- The same-day fitted return includes the regression intercept.
- The main application entry point is `app.py`.

### Fixed

- Weekend launches no longer fail only because FRED has no new observation.
- Old complete caches are no longer presented as current live signals.
- Streamlit Cloud cold starts now retain `REAL_YIELD` when FRED and Treasury
  are temporarily unavailable by using a bundled official seed history.
