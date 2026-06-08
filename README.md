# BTC Macro Signal Dashboard

A Streamlit research dashboard for Bitcoin macro attribution, regime detection,
walk-forward signal testing, and data-source monitoring.

Repository: [mwsp-code/bitcoin-macro-dashboard](https://github.com/mwsp-code/bitcoin-macro-dashboard)

## Current Capabilities

- Binance BTC/USDT completed UTC daily OHLCV candles, with HTX, OKX, and
  CoinGecko fallbacks in that order.
- Preferred macro instruments from Yahoo:
  - `QQQ`
  - `DX-Y.NYB`
  - `GC=F`
  - `CL=F`
- Consistent Nasdaq ETF fallback set when Yahoo is unavailable:
  - `QQQ`
  - `UUP`
  - `GLD`
  - `USO`
- 10-year real yield from FRED `DFII10`, with an official U.S. Treasury fallback.
- Weekend operation using the last published traditional-market observations.
- Source timestamps, cache metadata, stale-data warnings, and historical mode.
- Session-aware macro features, crypto-native price/volume features, and a
  genuine live next-day inference row.
- Nested walk-forward Ridge/ElasticNet selection with a frozen final holdout.
- Baseline comparisons, block-bootstrap confidence intervals, transaction
  costs, and out-of-sample evaluation statistics.

## Important Data Note

`UUP`, `GLD`, and `USO` are fallback ETF proxies. They are not identical to
DXY, COMEX gold futures, or WTI futures. When the fallback activates, the app
uses the proxy set for the complete macro history instead of splicing two
different instrument definitions.

See [Data Sources](docs/DATA_SOURCES.md) for details.

## Installation

Python 3.11 is recommended.

```powershell
git clone https://github.com/mwsp-code/bitcoin-macro-dashboard.git
cd bitcoin-macro-dashboard
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For development:

```powershell
pip install -r requirements-dev.txt
```

## Proxy Configuration

The app checks settings in this order:

1. `BTC_PROXY_URL`
2. `BTC_PROXY_PORT`
3. standard `HTTPS_PROXY`, `ALL_PROXY`, or `HTTP_PROXY` variables
4. automatic detection of common local ports such as `7890`

PowerShell example:

```powershell
$env:BTC_PROXY_URL="http://127.0.0.1:7890"
streamlit run app.py
```

You may instead copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml`. The real secrets file is ignored by Git.

To use Yahoo's exact DXY, gold-futures, and oil-futures symbols, select a
non-mainland VPN exit, enable global or TUN routing, and confirm that Yahoo does
not return an empty response or HTTP 403. The application always prefers Yahoo
when all four configured macro instruments are available.

## Run

```powershell
streamlit run app.py
```

## Validate

```powershell
python -m py_compile app.py
pytest -q
```

CI runs the same compilation and offline smoke test for every pull request into
`main`.

## Architecture

- `btc_dashboard/data.py`: completed-bar market data, source fallback, caching,
  observation timestamps, and freshness.
- `btc_dashboard/features.py`: native-session macro lags and crypto features.
- `btc_dashboard/models.py`: nested tuning, frozen holdout, live inference,
  and same-day attribution.
- `btc_dashboard/backtest.py`: baselines, confidence intervals, and strategy
  evaluation.
- `btc_dashboard/ui.py`: Streamlit presentation only.

## Model Outline

The dashboard provides two related views:

1. Same-day attribution uses regularized macro coefficients on active market
   sessions and is explicitly descriptive rather than causal.
2. Next-day prediction uses completed BTC candles, native-session macro
   changes, market-data age, momentum, volatility, range, volume, trade-count,
   and taker-flow features when Binance supplies them.
3. Development predictions use nested time-series validation. Model
   specification is frozen before the final holdout, then refitted only on
   information available before each prediction.
4. The latest completed candle produces a separate live forecast whose target
   is still unknown.

This remains an experimental research model. Walk-forward evaluation reduces,
but does not eliminate, overfitting and data-leakage risk.

## Repository Workflow

Use a branch and pull request for each iteration. See:

- [Contributing](CONTRIBUTING.md)
- [GitHub Workflow](docs/GITHUB_WORKFLOW.md)
- [Roadmap](ROADMAP.md)
- [Changelog](CHANGELOG.md)

Generated cache files and local proxy secrets must not be committed.

## Disclaimer

This project is for research and educational purposes only. It is not financial
advice or a recommendation to buy or sell any asset.
