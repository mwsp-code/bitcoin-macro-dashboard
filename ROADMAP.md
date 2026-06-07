# Roadmap

## Phase 1: Reliability

- Separate data acquisition, feature engineering, modeling, and UI modules.
- Add source-contract tests for HTX, Nasdaq, Yahoo, FRED, and Treasury parsers.
- Store cache metadata with instrument identity and schema version.
- Add structured logging and a data-health summary.

## Phase 2: Prediction Quality

- Create a true live inference row that does not require a known next-day target.
- Align features by market close and publication timestamp in UTC.
- Compare ordinary least squares with Ridge and ElasticNet.
- Add nested walk-forward validation and an untouched final holdout period.
- Report directional accuracy, information coefficient, calibration, turnover,
  drawdown, and confidence intervals.
- Test volatility, liquidity, ETF-flow, funding, basis, open-interest, and
  stablecoin features only after publication-time alignment.

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
