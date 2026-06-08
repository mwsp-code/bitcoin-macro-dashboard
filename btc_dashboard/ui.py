from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


def _format_percent(value):
    return "n/a" if pd.isna(value) else f"{value:.2%}"


def _signal_label(prediction, cost_bps):
    threshold = cost_bps / 10_000
    if prediction > threshold:
        return "LONG"
    if prediction < -threshold:
        return "SHORT"
    return "FLAT"


def render_dashboard(
    bundle,
    feature_set,
    forecast,
    attribution,
    evaluation,
    strategy_returns,
    transaction_cost_bps,
    proxy_source,
):
    data = bundle.data
    holdout = forecast.predictions.loc[
        forecast.predictions["segment"] == "Frozen Holdout"
    ]
    model_report = evaluation.loc[("Frozen Holdout", "Regularized Model")]
    baseline_mae = evaluation.loc["Frozen Holdout"].drop(
        index="Regularized Model"
    )["MAE"].min()
    validated_edge = (
        model_report["MAE"] < baseline_mae
        and model_report["Directional Accuracy"] > 0.5
        and model_report["Information Coefficient"] > 0
    )
    latest_price = float(data["BTC"].iloc[-1])
    signal = (
        _signal_label(forecast.live_prediction, transaction_cost_bps)
        if validated_edge
        else "NO VALIDATED EDGE"
    )

    st.title("BTC Macro Research Terminal")
    st.caption(
        f"Completed UTC bars through {forecast.live_feature_date:%Y-%m-%d} | "
        f"Forecast target {forecast.live_target_date:%Y-%m-%d} | "
        f"Model {forecast.frozen_spec.label}"
    )

    top = st.columns(6)
    top[0].metric("BTC Close", f"${latest_price:,.0f}")
    top[1].metric("Next-Day Forecast", _format_percent(forecast.live_prediction))
    top[2].metric("Probability Up", f"{forecast.live_probability_up:.1%}")
    top[3].metric("Position", signal)
    top[4].metric(
        "Holdout Direction",
        f"{model_report['Directional Accuracy']:.1%}",
    )
    top[5].metric("Holdout IC", f"{model_report['Information Coefficient']:.3f}")

    if bundle.historical_mode:
        st.warning("Historical mode: at least one displayed dataset came from cache.")
    if not validated_edge:
        st.warning(
            "The regularized model does not currently beat the holdout "
            "baselines on direction, information coefficient, and MAE. "
            "The forecast is shown for research, but no trade is endorsed."
        )
    if forecast.live_feature_date < pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=2):
        st.error("The latest completed BTC bar is more than two days old.")

    overview, evaluation_tab, drivers, data_health = st.tabs(
        ["Monitor", "Evaluation", "Drivers", "Data Health"]
    )

    with overview:
        chart_frame = pd.DataFrame(
            {
                "Actual next-day return": forecast.predictions["actual"],
                "Model forecast": forecast.predictions["model"],
            }
        ).tail(180)
        st.subheader("Forecast Monitor")
        st.line_chart(chart_frame)

        left, right = st.columns(2)
        with left:
            st.subheader("Frozen Holdout Equity")
            holdout_strategy = strategy_returns[
                "Frozen Holdout: Regularized Model"
            ]
            equity = pd.DataFrame(
                {
                    "Model Strategy": holdout_strategy.cumsum().apply(np.exp),
                    "BTC Buy & Hold": holdout["actual"].cumsum().apply(np.exp),
                }
            )
            st.line_chart(equity)
        with right:
            st.subheader("Recent Forecasts")
            recent = holdout[
                ["actual", "model", "historical_mean", "momentum_7d"]
            ].tail(20)
            recent.columns = [
                "Actual",
                "Model",
                "Rolling Mean",
                "7D Momentum",
            ]
            st.dataframe(recent, width="stretch")

    with evaluation_tab:
        st.subheader("Out-of-Sample Evaluation")
        display = evaluation.copy()
        for column in (
            "MAE",
            "RMSE",
            "Information Coefficient",
            "Predictive R2 vs Zero",
            "Strategy Sharpe",
            "Average Exposure",
        ):
            display[column] = display[column].map(
                lambda value: "n/a" if pd.isna(value) else f"{value:.4f}"
            )
        for column in (
            "Directional Accuracy",
            "Strategy Return",
            "Strategy Max Drawdown",
        ):
            display[column] = display[column].map(
                lambda value: "n/a" if pd.isna(value) else f"{value:.2%}"
            )
        st.dataframe(display, width="stretch")
        st.caption(
            "The final holdout model specification is selected before the "
            "holdout starts. Confidence intervals use moving-block bootstrap."
        )
        with st.expander("Nested tuning history"):
            st.dataframe(forecast.tuning_history, width="stretch")

    with drivers:
        st.subheader("Live Regularized Forecast Coefficients")
        coefficient_table = forecast.coefficients.reindex(
            forecast.coefficients.abs().sort_values(ascending=False).index
        ).to_frame()
        st.dataframe(coefficient_table.head(25), width="stretch")

        st.subheader("Latest Same-Day Macro Attribution")
        st.caption(
            f"Active macro session {attribution.date:%Y-%m-%d}; descriptive "
            "association, not a causal claim."
        )
        contribution_table = attribution.contributions.reindex(
            attribution.contributions.abs().sort_values(ascending=False).index
        ).to_frame()
        st.dataframe(contribution_table, width="stretch")
        metrics = st.columns(3)
        metrics[0].metric("Fitted Return", _format_percent(attribution.fitted_return))
        metrics[1].metric("Actual Return", _format_percent(attribution.actual_return))
        metrics[2].metric("Residual", _format_percent(attribution.residual))

    with data_health:
        left, right = st.columns(2)
        with left:
            st.subheader("Source Status")
            rows = []
            for name, value in bundle.status.items():
                if name.startswith("_"):
                    continue
                rows.append({"Dataset": name, "Status": str(value)})
            st.dataframe(pd.DataFrame(rows).set_index("Dataset"))
        with right:
            st.subheader("Source Timestamps")
            timestamps = pd.Series(bundle.timestamps, name="As of").sort_index()
            st.dataframe(timestamps)

        st.caption(
            f"BTC source priority: Binance, HTX, OKX, CoinGecko | "
            f"Active source: {bundle.btc_source} | Route: {proxy_source} | "
            f"Loaded {datetime.now():%Y-%m-%d %H:%M:%S}"
        )
        with st.expander("Feature groups"):
            for name, columns in feature_set.feature_groups.items():
                st.write(f"{name.title()}: {', '.join(columns)}")
        with st.expander("Recent aligned market data"):
            st.dataframe(data.tail(20), width="stretch")
        with st.expander("Source diagnostics"):
            for name, value in bundle.status.items():
                if name.startswith("_"):
                    st.write(name, value)
