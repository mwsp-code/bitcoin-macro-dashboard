# BTC Macro Signal Dashboard

A Streamlit research dashboard for Bitcoin macro attribution, regime detection,
walk-forward signal testing, and data-source monitoring.

Repository: [mwsp-code/bitcoin-macro-dashboard](https://github.com/mwsp-code/bitcoin-macro-dashboard)

## Current Capabilities

- BTC daily prices from HTX with OKX, Binance, and CoinGecko fallbacks.
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
- Same-day attribution, random-forest feature importance, walk-forward
  next-day signals, transaction costs, and backtest statistics.

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

## Model Outline

The dashboard provides two related views:

1. Same-day attribution fits BTC returns against contemporaneous and lagged
   macro features.
2. Walk-forward testing estimates next-day BTC returns using only prior
   training observations for each evaluated date.

The alpha layer combines model prediction, macro regime prediction, and
seven-day BTC momentum. Signal weights and thresholds are periodically
re-optimized on trailing data with transaction costs included.

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
