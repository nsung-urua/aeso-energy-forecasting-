import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="AESO Load & Pool Price Forecasting", page_icon="⚡", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "aeso_final_outputs"
if not DATA_DIR.exists():
    DATA_DIR = BASE_DIR

@st.cache_data
def load_data():
    files = {
        "dashboard": "dashboard_data.csv",
        "load_metrics": "load_model_metrics.csv",
        "price_metrics": "price_model_metrics.csv",
        "spike_metrics": "high_price_classifier_metrics.csv",
        "load_importance": "load_feature_importance.csv",
        "price_importance": "price_feature_importance.csv",
        "price_regime": "price_error_by_regime.csv",
    }
    data, missing = {}, []
    for key, filename in files.items():
        path = DATA_DIR / filename
        if path.exists(): data[key] = pd.read_csv(path)
        else: missing.append(filename)
    if missing:
        st.error("Missing dashboard files: " + ", ".join(missing)); st.stop()
    data["dashboard"]["local_datetime_naive"] = pd.to_datetime(data["dashboard"]["local_datetime_naive"], errors="coerce")
    return data

data = load_data()
dashboard = data["dashboard"].sort_values("local_datetime_naive").copy()
load_metrics, price_metrics = data["load_metrics"], data["price_metrics"]
spike_metrics, price_regime = data["spike_metrics"], data["price_regime"]
load_importance, price_importance = data["load_importance"], data["price_importance"]

st.title("⚡ AESO Short-Term Load & Pool Price Forecasting")
st.caption("Alberta hourly market data | leakage-safe time-series features | Linear Regression + XGBoost")
st.markdown("""
This project asks two related questions: **how predictable is Alberta electricity demand, and how predictable is the hourly pool price using historical market information?** The results show a clear contrast: load follows strong recurring patterns, while price becomes much harder to forecast during volatile market conditions.
""")

with st.expander("📖 How to read this dashboard"):
    st.markdown("""
**AIL (Alberta Internal Load)** — electricity demand within Alberta, measured in megawatts (MW).

**Pool Price ($/MWh)** — Alberta's hourly wholesale electricity price in Canadian dollars per megawatt-hour.

**MAE (Mean Absolute Error)** — the average size of a forecast error. Lower is better.

**RMSE (Root Mean Squared Error)** — similar to MAE, but gives larger errors more weight. Lower is better.

**R²** — measures how much variation in observed values is explained by the model. Values closer to 1 indicate a stronger fit on the test data.

**ROC-AUC** — measures how well the high-price classifier ranks high-price hours above normal-price hours across probability thresholds. Values closer to 1 indicate stronger discrimination.

**High-Price Probability** — the estimated probability that pool price is at least **$100/MWh**, the threshold used in this project.

**Lag Feature** — a past observation used to predict a future observation. *Load — Same Hour Yesterday* uses load 24 hours earlier.

**Feature Importance** — an XGBoost measure of how much each input contributed to model decisions. It is not proof of causation.
""")

load_xgb = load_metrics.loc[load_metrics["Model"] == "XGBoost"].iloc[0]
price_xgb = price_metrics.loc[price_metrics["Model"] == "XGBoost"].iloc[0]
price_naive = price_metrics.loc[price_metrics["Model"] == "Naive: previous hour"].iloc[0]
spike = spike_metrics.iloc[0]
auc_col = "ROC_AUC" if "ROC_AUC" in spike.index else "ROC-AUC"

days_available = max(1, int((dashboard["local_datetime_naive"].max() - dashboard["local_datetime_naive"].min()).days) + 1)
window_days = st.sidebar.selectbox("Detailed chart window", [7, 14, 30, 60], index=1, help="Most recent days of the held-out test period.")
window_days = min(window_days, days_available)
window = dashboard.tail(min(24 * window_days, len(dashboard))).copy()

overview_tab, load_tab, price_tab, risk_tab, insights_tab = st.tabs(["Overview", "Load Forecast", "Pool Price", "High-Price Risk", "Model Insights"])

with overview_tab:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Load MAE", f"{load_xgb['MAE']:.1f} MW")
    c2.metric("Load R²", f"{load_xgb['R2']:.3f}")
    c3.metric("Price MAE", f"${price_xgb['MAE']:.2f}/MWh")
    c4.metric("High-Price ROC-AUC", f"{spike[auc_col]:.3f}")
    st.subheader("What stood out")
    st.markdown(f"""
- **Load was highly predictable:** XGBoost achieved an R² of **{load_xgb['R2']:.3f}** with an MAE of **{load_xgb['MAE']:.1f} MW**.
- **Price was more difficult:** XGBoost explained more price variation than the previous-hour baseline, but its typical absolute error was **${price_xgb['MAE']:.2f}/MWh** versus **${price_naive['MAE']:.2f}/MWh** for the simple previous-hour forecast.
- **Extreme prices were the hardest problem:** overall price metrics hide much larger errors during high-price periods.
- **Reframing helped:** the high-price classifier provides a probability signal even when exact spike magnitude is difficult to predict.
""")
    st.info("Held-out historical test performance — not a live AESO price or demand forecast.")

with load_tab:
    st.subheader(f"Actual vs forecast Alberta Internal Load — last {window_days} days shown")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=window["local_datetime_naive"], y=window["load"], name="Actual Load", mode="lines"))
    fig.add_trace(go.Scatter(x=window["local_datetime_naive"], y=window["pred_load_xgb"], name="XGBoost Forecast", mode="lines"))
    fig.update_layout(xaxis_title="Alberta local time", yaxis_title="Load (MW)", hovermode="x unified", legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**How to interpret it:** load follows strong hourly and daily patterns, so the forecast usually tracks the actual series closely. Larger misses can point to missing drivers such as unusual weather, holidays, or operational changes.")
    st.dataframe(load_metrics, use_container_width=True, hide_index=True)

with price_tab:
    st.subheader(f"Actual vs forecast pool price — last {window_days} days shown")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=window["local_datetime_naive"], y=window["pool_price"], name="Actual Pool Price", mode="lines"))
    fig.add_trace(go.Scatter(x=window["local_datetime_naive"], y=window["pred_price_xgb"], name="XGBoost Forecast", mode="lines"))
    fig.update_layout(xaxis_title="Alberta local time", yaxis_title="Pool Price (CAD/MWh)", hovermode="x unified", legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**How to interpret it:** the model follows many normal price movements but is less reliable at predicting the size of sharp price spikes, where omitted supply-side information matters more.")
    st.subheader("Forecast error by actual price regime")
    mae_candidates = [c for c in price_regime.columns if "mae" in c.lower()]
    regime_col = "price_regime" if "price_regime" in price_regime.columns else price_regime.columns[0]
    if mae_candidates:
        mae_col = mae_candidates[0]
        fig_regime = px.bar(price_regime, x=regime_col, y=mae_col, labels={regime_col:"Actual Pool Price Range", mae_col:"MAE (CAD/MWh)"})
        st.plotly_chart(fig_regime, use_container_width=True)
    else:
        st.dataframe(price_regime, use_container_width=True, hide_index=True)
    st.dataframe(price_metrics, use_container_width=True, hide_index=True)

with risk_tab:
    st.subheader("High-price probability")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=window["local_datetime_naive"], y=window["pool_price"], name="Actual Pool Price", mode="lines", yaxis="y1"))
    fig.add_trace(go.Scatter(x=window["local_datetime_naive"], y=window["prob_high_price"], name="High-Price Probability", mode="lines", yaxis="y2"))
    fig.update_layout(xaxis_title="Alberta local time", yaxis=dict(title="Pool Price (CAD/MWh)"), yaxis2=dict(title="Probability", overlaying="y", side="right", range=[0,1]), hovermode="x unified", legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("The exact price forecast may miss spike magnitude while the classifier can still recognize conditions resembling previous high-price periods. Price forecasting and high-price risk detection are therefore treated as related but different questions.")
    st.dataframe(spike_metrics, use_container_width=True, hide_index=True)

with insights_tab:
    st.subheader("What drives the load model?")
    top_load = load_importance.nlargest(12, "importance").sort_values("importance")
    label_col = "display_feature" if "display_feature" in top_load.columns else "feature"
    st.plotly_chart(px.bar(top_load, x="importance", y=label_col, orientation="h", labels={"importance":"XGBoost Importance", label_col:"Feature"}), use_container_width=True)
    st.caption("Feature importance shows what the model relied on, not necessarily a causal relationship.")
    st.subheader("What drives the price model?")
    top_price = price_importance.nlargest(12, "importance").sort_values("importance")
    label_col = "display_feature" if "display_feature" in top_price.columns else "feature"
    st.plotly_chart(px.bar(top_price, x="importance", y=label_col, orientation="h", labels={"importance":"XGBoost Importance", label_col:"Feature"}), use_container_width=True)
    with st.expander("Methodology and project evolution"):
        st.markdown("""
**Forecast design:** chronological train/test split; only information available before the forecast hour; rolling features shifted by one hour; realized same-hour target variables excluded from the opposite model.

**Why this differs from the original load-only project:** this final version revisits the feature engineering, makes information timing more explicit, expands into pool-price forecasting, adds simple baselines, and adds high-price risk classification.

**What is still missing:** historical load, price, and calendar information cannot fully describe market conditions. Weather, generation mix, wind/solar output and forecasts, outages, imports/exports, and available capability are logical additions for a production-oriented price model.
""")

st.divider()
st.caption("Portfolio project using historical AESO hourly market data. Outputs shown are held-out historical test results, not operational forecasts.")
