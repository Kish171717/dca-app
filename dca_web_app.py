import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.optimize import curve_fit
import io

st.set_page_config(page_title="DCA Forecast - Auto vs Manual", layout="wide")
st.title("📊 Decline Curve Analysis: Auto-Fit vs Manual Forecast")

# Upload Excel file
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
if not uploaded_file:
    st.stop()

# Read data
try:
    df = pd.read_excel(uploaded_file)
    df = df[["Month", "Oil Production (m3/d)", "Oil m3"]].copy()
    df["Month"] = pd.to_datetime(df["Month"])
    df = df.sort_values("Month")
    df["Days"] = (df["Month"] - df["Month"].iloc[0]).dt.days
    df["Qo"] = df["Oil Production (m3/d)"]
    df["CumOil"] = df["Oil m3"].cumsum()
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

st.success("✅ File loaded successfully!")

# Forecast Settings
st.sidebar.header("📌 Forecast Settings")

eur_input = st.sidebar.number_input("EUR (Million m³)", value=86.0)
cutoff = st.sidebar.number_input("Cutoff Qo (m³/d)", value=0.5)
forecast_years = st.sidebar.slider("Forecast Horizon (years)", 1, 100, 50)

# Auto-fit Initial Guesses
st.sidebar.subheader("Auto-Fit Initial Guesses")
init_decline_pct = st.sidebar.slider("Decline % per year", 1.0, 100.0, 14.0)
init_b = st.sidebar.slider("b (hyperbolic exponent)", 0.1, 1.0, 0.5, step=0.01)

# Manual Forecast Inputs
st.sidebar.subheader("Manual Forecast Settings")
manual_qi = st.sidebar.number_input("Manual qi (initial rate)", value=100.0)
manual_decline_pct = st.sidebar.slider("Manual Decline % per year", 1.0, 100.0, 10.0)
manual_b = st.sidebar.slider("Manual b (hyperbolic)", 0.1, 1.0, 0.5, step=0.01)

run_btn = st.sidebar.button("🔍 Run Forecast")

# Forecast formulas
def hyperbolic(t, qi, D, b):
    return qi / (1 + b * D * t)**(1/b)

def exponential(t, qi, D):
    return qi * np.exp(-D * t)

# Forecasting block
if run_btn:
    forecast_days = int(forecast_years * 365.25)
    horizon = np.arange(forecast_days)
    years = horizon / 365.25
    t0 = df["Days"].iloc[0]

    # Prepare data for curve fitting
    fit_df = df[df["Qo"] > 0].copy()
    t_fit = (fit_df["Days"] - fit_df["Days"].iloc[0]) / 365.25
    q_fit = fit_df["Qo"].values
    qi_guess = q_fit[0]
    D_guess = init_decline_pct / 100

    # Fit curve
    try:
        popt, _ = curve_fit(hyperbolic, t_fit, q_fit, p0=[qi_guess, D_guess, init_b])
        qi_fit, D_fit, b_fit = popt
        qo_auto = hyperbolic(years, qi_fit, D_fit, b_fit)
    except:
        qo_auto = hyperbolic(years, qi_guess, D_guess, init_b)

    # Apply cutoff & EUR
    cum_auto = np.cumsum(qo_auto) / 1e6
    stop_auto = (qo_auto < cutoff) | (cum_auto > eur_input)
    stop_idx_auto = np.argmax(stop_auto) if stop_auto.any() else len(qo_auto)

    # Manual forecast
    D_manual = manual_decline_pct / 100
    qo_manual = hyperbolic(years, manual_qi, D_manual, manual_b)
    cum_manual = np.cumsum(qo_manual) / 1e6
    stop_manual = (qo_manual < cutoff) | (cum_manual > eur_input)
    stop_idx_manual = np.argmax(stop_manual) if stop_manual.any() else len(qo_manual)

    # Graph 1 – Auto
    st.subheader("📈 Forecast Graph 1: Auto-Fit")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df["Days"], y=df["Qo"], mode="lines+markers", name="Actual"))
    fig1.add_trace(go.Scatter(x=horizon[:stop_idx_auto], y=qo_auto[:stop_idx_auto],
                              mode="lines", name="Auto Forecast", line=dict(color="orange", dash="dash")))
    fig1.add_shape(type="line", x0=horizon[stop_idx_auto], x1=horizon[stop_idx_auto],
                   y0=0, y1=max(qo_auto), line=dict(color="orange", dash="dot"))
    fig1.update_layout(xaxis_title="Days", yaxis_title="Qo (m³/d)")
    st.plotly_chart(fig1, use_container_width=True)

    # Graph 2 – Manual
    st.subheader("📘 Forecast Graph 2: Manual Forecast")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["Days"], y=df["Qo"], mode="lines+markers", name="Actual"))
    fig2.add_trace(go.Scatter(x=horizon[:stop_idx_manual], y=qo_manual[:stop_idx_manual],
                              mode="lines", name="Manual Forecast", line=dict(color="blue", dash="dot")))
    fig2.add_shape(type="line", x0=horizon[stop_idx_manual], x1=horizon[stop_idx_manual],
                   y0=0, y1=max(qo_manual), line=dict(color="blue", dash="dot"))
    fig2.update_layout(xaxis_title="Days", yaxis_title="Qo (m³/d)")
    st.plotly_chart(fig2, use_container_width=True)

    # Excel output
    result_df = pd.DataFrame({
        "Days": horizon[:max(stop_idx_auto, stop_idx_manual)],
        "Auto Qo": qo_auto[:max(stop_idx_auto, stop_idx_manual)],
        "Cum Auto (M m³)": cum_auto[:max(stop_idx_auto, stop_idx_manual)],
        "Manual Qo": qo_manual[:max(stop_idx_auto, stop_idx_manual)],
        "Cum Manual (M m³)": cum_manual[:max(stop_idx_auto, stop_idx_manual)],
    })

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Historical", index=False)
        result_df.to_excel(writer, sheet_name="Forecasts", index=False)

    st.download_button("📥 Download Forecasts", buffer.getvalue(), file_name="Forecast_Output.xlsx")

