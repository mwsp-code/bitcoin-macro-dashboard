from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, StrMethodFormatter
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


def build_holdout_wealth(holdout_strategy, btc_returns, initial_capital=10_000):
    aligned = pd.concat(
        [
            pd.Series(holdout_strategy, name="Model Strategy"),
            pd.Series(btc_returns, name="BTC Buy & Hold"),
        ],
        axis=1,
    ).dropna()
    wealth = initial_capital * np.exp(aligned.cumsum())
    starting_date = wealth.index[0] - pd.Timedelta(days=1)
    starting_row = pd.DataFrame(
        initial_capital,
        index=[starting_date],
        columns=wealth.columns,
    )
    wealth = pd.concat([starting_row, wealth])
    drawdown = wealth.div(wealth.cummax()).sub(1)
    return wealth, drawdown


def plot_holdout_wealth(wealth, drawdown):
    colors = {
        "Model Strategy": "#087E8B",
        "BTC Buy & Hold": "#F0A202",
    }
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1]},
    )
    wealth_axis, drawdown_axis = axes
    for column in wealth:
        wealth_axis.plot(
            wealth.index,
            wealth[column],
            label=column,
            color=colors[column],
            linewidth=2,
        )
        wealth_axis.annotate(
            f"${wealth[column].iloc[-1]:,.0f}",
            xy=(wealth.index[-1], wealth[column].iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=colors[column],
            fontsize=9,
        )
    wealth_axis.axhline(
        wealth.iloc[0].mean(),
        color="#808080",
        linewidth=0.8,
        linestyle="--",
        alpha=0.7,
    )
    wealth_axis.set_ylabel("Portfolio value")
    wealth_axis.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    wealth_axis.legend(loc="upper left", frameon=False)
    wealth_axis.grid(axis="y", alpha=0.2)

    for column in drawdown:
        drawdown_axis.plot(
            drawdown.index,
            drawdown[column],
            label=column,
            color=colors[column],
            linewidth=1.5,
        )
    drawdown_axis.axhline(0, color="#808080", linewidth=0.8)
    drawdown_axis.set_ylabel("Drawdown")
    drawdown_axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:.0%}")
    )
    drawdown_axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    return figure


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
    st.markdown(
        "**Purpose.** Test whether completed Bitcoin market activity and "
        "lagged macro signals contain robust information about next-day BTC "
        "returns.  \n"
        "**Method.** Binance-first completed UTC candles, session-aware macro "
        "features, nested Ridge/ElasticNet tuning, transaction costs, and a "
        "frozen holdout period that is not used for model selection.  \n"
        "**Interpretation.** Forecasts are shown as research evidence, and the "
        "dashboard withholds a trade signal when the model lacks validated "
        "out-of-sample edge."
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

        st.subheader("Growth of $10,000 on Unseen Holdout Data")
        st.caption(
            f"Frozen evaluation period: {holdout.index.min():%Y-%m-%d} to "
            f"{holdout.index.max():%Y-%m-%d}. The model specification was "
            "locked before this period. The model path includes the selected "
            f"{transaction_cost_bps:.0f} bps trading-cost assumption; BTC is "
            "shown as passive buy-and-hold."
        )
        holdout_strategy = strategy_returns[
            "Frozen Holdout: Regularized Model"
        ]
        wealth, drawdown = build_holdout_wealth(
            holdout_strategy,
            holdout["actual"],
        )
        model_end = float(wealth["Model Strategy"].iloc[-1])
        btc_end = float(wealth["BTC Buy & Hold"].iloc[-1])
        model_drawdown = float(drawdown["Model Strategy"].min())
        btc_drawdown = float(drawdown["BTC Buy & Hold"].min())
        holdout_metrics = st.columns(4)
        holdout_metrics[0].metric("Starting Capital", "$10,000")
        holdout_metrics[1].metric(
            "Model Ending Value",
            f"${model_end:,.0f}",
            delta=f"{model_end / 10_000 - 1:+.1%}",
        )
        holdout_metrics[2].metric(
            "BTC Ending Value",
            f"${btc_end:,.0f}",
            delta=f"{btc_end / 10_000 - 1:+.1%}",
        )
        holdout_metrics[3].metric(
            "Model vs BTC",
            f"${model_end - btc_end:+,.0f}",
            delta=f"DD {model_drawdown:.1%} vs {btc_drawdown:.1%}",
        )
        relative_result = (
            f"outperformed passive BTC by ${model_end - btc_end:,.0f}"
            if model_end >= btc_end
            else f"underperformed passive BTC by ${btc_end - model_end:,.0f}"
        )
        absolute_result = (
            "Both portfolios finished below the starting capital."
            if model_end < 10_000 and btc_end < 10_000
            else "At least one portfolio finished above the starting capital."
        )
        st.markdown(
            f"**Holdout takeaway.** The model {relative_result}. "
            f"{absolute_result}"
        )
        holdout_figure = plot_holdout_wealth(wealth, drawdown)
        st.pyplot(holdout_figure)
        plt.close(holdout_figure)
        st.caption(
            "Top: portfolio value from the same $10,000 starting capital. "
            "Bottom: percentage decline from each strategy's previous peak; "
            "shallower drawdowns indicate lower realized downside."
        )

        st.subheader("Recent Holdout Forecasts")
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

        st.subheader("Candidate Validation MAE")
        validation_display = forecast.validation_comparison.copy()
        validation_display["Validation MAE"] = validation_display[
            "Validation MAE"
        ].map(lambda value: "n/a" if pd.isna(value) else f"{value:.6f}")
        validation_display["MAE vs Zero"] = validation_display[
            "MAE vs Zero"
        ].map(lambda value: "n/a" if pd.isna(value) else f"{value:+.6f}")
        validation_display["Improvement vs Zero"] = validation_display[
            "Improvement vs Zero"
        ].map(lambda value: "n/a" if pd.isna(value) else f"{value:+.2%}")
        st.dataframe(validation_display, width="stretch")
        st.caption(
            "Negative MAE vs Zero and positive improvement indicate a candidate "
            "beat the zero-return baseline on the same inner time-series folds."
        )

    with drivers:
        st.subheader("Model Coefficients")
        nonzero_count = int(
            (forecast.standardized_coefficients.abs() > 1e-12).sum()
        )
        st.caption(f"Selected specification: `{forecast.frozen_spec.label}`")
        coefficient_metrics = st.columns(2)
        coefficient_metrics[0].metric(
            "Intercept",
            f"{forecast.intercept:+.6f}",
        )
        coefficient_metrics[1].metric(
            "Non-Zero Features",
            f"{nonzero_count}/{len(forecast.standardized_coefficients)}",
        )
        if nonzero_count == 0:
            st.info(
                "Intercept-only model selected. Validation preferred shrinking "
                "every feature coefficient to zero; no artificial driver is shown."
            )
        coefficient_table = forecast.standardized_coefficients.reindex(
            forecast.standardized_coefficients.abs().sort_values(
                ascending=False
            ).index
        ).to_frame()
        st.dataframe(coefficient_table.head(25), width="stretch")
        st.caption(
            "Standardized coefficients show the expected target-return change "
            "for a one-standard-deviation feature move."
        )
        with st.expander("Raw coefficients by original feature units"):
            raw_table = forecast.coefficients.reindex(
                forecast.coefficients.abs().sort_values(ascending=False).index
            ).to_frame()
            st.dataframe(raw_table, width="stretch")

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
