# BTC Macro Signal Dashboard

A Streamlit dashboard for analyzing Bitcoin price movement through macro market drivers, risk-on/risk-off regime detection, walk-forward alpha scoring, trading signals, and Sharpe-based backtesting.

With a VPN or local proxy enabled, users in Mainland China can use this app to call Bitcoin and market data APIs directly.

## Features

- Live BTC price data with multiple fallback sources: HTX/Huobi, OKX, Binance, and CoinGecko.
- Macro market data from yfinance, including NASDAQ proxy, DXY, gold, and crude oil.
- Real yield data from FRED using the 10-year TIPS real yield series.
- Risk-on / risk-off regime detection.
- BTC alpha score from 0 to 100.
- Trading signal: `BUY`, `SELL`, or `NEUTRAL`.
- Walk-forward next-day signal backtest.
- Sharpe ratio, total return, max drawdown, and win-rate panels.
- Transaction cost input in basis points per trade.
- Cache support to avoid unnecessary API calls.

## What The Dashboard Does

The app pulls BTC and macro market data, aligns all series by date, then estimates how macro conditions may be influencing Bitcoin.

It separates the analysis into two layers:

1. **Same-day attribution**
   - Explains recent BTC movement using macro returns and changes.
   - Shows model coefficients, feature importance, and driver attribution.

2. **Next-day signal testing**
   - Uses today's features to predict next-day BTC return.
   - Trains models using walk-forward logic, so each signal only uses historical data available at that time.
   - Optimizes alpha weights and buy/sell thresholds using prior data.
   - Tests whether the resulting signal beats buy-and-hold after transaction costs.

## Data Sources

- BTC:
  - HTX/Huobi
  - OKX
  - Binance
  - CoinGecko
- Macro:
  - `QQQ` as a NASDAQ-100 proxy
  - `DX-Y.NYB` for DXY
  - `GC=F` for gold futures
  - `CL=F` for WTI crude oil futures
- Real yield:
  - FRED `DFII10`

## Mainland China VPN / Proxy Setup

Crypto APIs and some market data sources may be blocked or unreliable from Mainland China without a VPN or proxy. The app keeps proxy usage off by default so it can run on Streamlit Cloud, where `127.0.0.1:7890` would not point to your local VPN.

For local use in Mainland China, enable your VPN/proxy and set an environment variable before running Streamlit:

```bash
set BTC_PROXY_PORT=7890
streamlit run btc_macro_htx.py
```

On PowerShell:

```powershell
$env:BTC_PROXY_PORT="7890"
streamlit run btc_macro_htx.py
```

Common proxy ports:

- Clash: `7890`
- V2Ray: `10809`
- Shadowsocks: `1080`

Set `BTC_PROXY_PORT` to match your local VPN/proxy tool. If you are outside Mainland China, deploying on Streamlit Cloud, or do not need a proxy, leave `BTC_PROXY_PORT` unset.

For Streamlit Cloud deployment, do not set `BTC_PROXY_PORT` unless your cloud environment provides a reachable proxy.

## Installation

Create and activate a Python environment, then install the required packages:

```bash
pip install streamlit pandas numpy matplotlib scikit-learn requests yfinance
```

## Run The App

From the project folder:

```bash
streamlit run btc_macro_htx.py
```

Then open the local Streamlit URL shown in your terminal.

## Signal Methodology

### Regime Score

The regime score estimates whether the macro environment is more risk-on or risk-off for BTC. It uses walk-forward calibrated relationships between next-day BTC returns and macro features such as NASDAQ, DXY, real yields, and gold.

General interpretation:

- `58-100`: Risk-on
- `42-58`: Mixed
- `0-42`: Risk-off

### BTC Alpha Score

The alpha score is a 0 to 100 tactical score built from:

- Walk-forward model prediction
- Calibrated macro regime component
- BTC short-term momentum

General interpretation:

- `65-100`: BUY
- `35-65`: NEUTRAL
- `0-35`: SELL

The app optimizes alpha weights and buy/sell thresholds using only historical data available before each test period.

## Backtest Notes

The backtest is designed to be more realistic than a full-sample fit:

- Uses next-day BTC returns as the prediction target.
- Uses walk-forward training.
- Includes transaction cost assumptions.
- Compares signal strategy performance against BTC buy-and-hold.
- Reports Sharpe ratio, total return, max drawdown, and signal win rate.

The backtest is still experimental and should be treated as research, not a production trading system.

## Limitations

- Data sources can fail, rate-limit, or change their API behavior.
- Daily data may not capture intraday BTC volatility.
- Backtest results depend on transaction cost assumptions.
- Walk-forward calibration reduces lookahead bias but does not eliminate overfitting risk.
- Historical relationships between BTC and macro assets can break down.

## Disclaimer

This project is for research and educational purposes only. It is not financial advice, investment advice, or a recommendation to buy or sell Bitcoin or any other asset. Always do your own research and consider the risks before trading.
