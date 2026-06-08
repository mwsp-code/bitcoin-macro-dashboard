# Roadmap

## Phase 1: Reliability

- Add source-contract tests for HTX, Nasdaq, Yahoo, FRED, and Treasury parsers.
- Store cache metadata with instrument identity and schema version.
- Add structured logging and a data-health summary.

## Phase 2: Prediction Quality

- Add point-in-time ETF-flow, funding, basis, open-interest, options, and
  stablecoin features after source licensing and timestamp validation.
- Add formal probability calibration and Brier-score evaluation.
- Add model-stability and feature-drift monitoring across market regimes.

## Phase 3: Trader Interface

- Add Monitor, Drivers, Backtest, and Data Health tabs.
- Replace static plots with interactive price, drawdown, exposure, and
  contribution charts.
- Show signal confidence, feature timestamp, model version, and active data
  source in a persistent status strip.
- Add configurable horizons and saved research presets.

## Phase 4: Deployment

- Add scheduled data refresh outside the Streamlit request cycle.
- Persist approved market data in managed object storage or a database.
- Add release tags, model-version records, and deployment rollback instructions.
- Deploy the public dashboard with cloud-accessible sources rather than a local
  `127.0.0.1` proxy.
