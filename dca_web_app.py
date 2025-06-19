import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.optimize import curve_fit
import io

st.set_page_config(page_title="DCA Forecast Tool", layout="centered")
st.title("📉 Decline Curve Analysis — Auto vs Manual Forecast")

file = st.file_uploader("Upload Excel (Month, Oil Production (m3/d), Oil m3)", ["xlsx"])
if not file:
    st.info("Upload an Excel file to begin.")
    st.stop()

try:
    df = pd.read_excel(file)
except Exception as e:
    st.error(f"Excel read failed: {e}")
    st.stop()

required = ["Month", "Oil Production (m3/d)", "Oil m3"]
if any(c not in df.columns for c in required):
    st.error(f"Missing required columns: {list(df.columns)}")
    st.stop()

df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").reset_index(drop=True)
df["Days"] = (df["Month"] - df["Month"].iloc[0]).dt.days
df["Qo"] = df["Oil Production (m3/d)"]
df["CumOil"] = df["Oil m3"].cumsum()

st.plotly_chart(
    go.Figure(go.Scatter(x=df["Days"], y=df["Qo"], mode="lines+markers", name="Actual Qo"))
    .update_layout(title="Historical Production", xaxis_title="Days", yaxis_title="Qo (m³/d)"),
    use_container_width=True,
)

st.header("⚙️ Forecast Settings")
start_day = st.number_input("Start Day", 0, value=0)
ignore_txt = st.text_input("Ignore Days (e.g. 200-220, 300)")
eur_mcm = st.number_input("EUR (million m³)", 1.0, 1e4, value=86.0)
use_cut = st.checkbox("Apply Cut-off?", value=True)
cutoff_qo = st.number_input("Cut-off Qo (m³/d)", 0.1, 100.0, value=0.5, disabled=not use_cut)
forecast_yrs = st.slider("Forecast horizon (years)", 1, 100, 50)

st.subheader("🟠 Auto-fit initial guesses")
decl_pct_guess = st.slider("Initial Decline guess (%/yr)", 0.1, 100.0, 14.0, 0.1)
b_guess = st.slider("Initial b guess (hyperbolic)", 0.05, 1.0, 0.5, 0.01)
model_type = st.radio("Model", ["Hyperbolic", "Exponential"])

st.subheader("🔵 Manual forecast parameters")
manual_qi = st.number_input("Manual Initial Qo (m³/d)", 0.1, 10000.0, step=0.1)
manual_decl_pct = st.slider("Manual Decline %/yr", 0.1, 100.0, 20.0, 0.1)
manual_b = st.slider("Manual b (hyperbolic)", 0.05, 1.0, 0.5, 0.01)

run_btn = st.button("🔍 Run Forecast")

def parse_ignore(txt):
    out = set()
    for part in txt.split(","):
        part = part.strip()
        if "-" in part:
            a, b = map(int, part.split("-"))
            out.update(range(a, b + 1))
        elif part.isdigit():
            out.add(int(part))
    return out

def hyperbolic(t, qi, D, b):
    return qi / ((1 + b * D * t) ** (1 / b))

def exponential(t, qi, D):
    return qi * np.exp(-D * t)

if run_btn:
    ignore = parse_ignore(ignore_txt)
    hist = df[df["Days"] >= start_day].copy()
    if ignore:
        hist = hist[~hist["Days"].isin(ignore)]

    if len(hist) < 3:
        st.warning("Too few data points after filtering.")
        st.stop()

    t0 = hist["Days"].iloc[0]
    t_hist_yrs = (hist["Days"] - t0) / 365.25
    q_hist = hist["Qo"].values
    mask = (q_hist > 0) & ~np.isnan(q_hist)
    t_hist_yrs, q_hist = t_hist_yrs[mask], q_hist[mask]

    if len(q_hist) < 3:
        st.warning("Filtered data too small.")
        st.stop()

    qi_guess = q_hist[0]
    D_guess = max(decl_pct_guess / 100, 0.01)

    try:
        if model_type == "Hyperbolic":
            popt, _ = curve_fit(
                hyperbolic, t_hist_yrs, q_hist,
                p0=[qi_guess, D_guess, b_guess],
                bounds=([0.01, 1e-5, 0.05],
                        [q_hist.max()*10, 5.0, 1.0]),
                maxfev=60000
            )
            qi_fit, D_fit, b_fit = popt
            auto_fun = lambda yrs: hyperbolic(yrs, qi_fit, D_fit, b_fit)
        else:
            popt, _ = curve_fit(
                exponential, t_hist_yrs, q_hist,
                p0=[qi_guess, D_guess],
                bounds=([0.01, 1e-5],
                        [q_hist.max()*10, 5.0]),
                maxfev=60000
            )
            qi_fit, D_fit = popt
            b_fit = None
            auto_fun = lambda yrs: exponential(yrs, qi_fit, D_fit)
    except Exception as e:
        st.warning(f"Auto-fit failed: {e} — using guesses.")
        auto_fun = lambda yrs: hyperbolic(yrs, qi_guess, D_guess, b_guess) \
            if model_type == "Hyperbolic" else \
            lambda yrs: exponential(yrs, qi_guess, D_guess)

    # Fitted Decline Info
    fitted_decl_pct = round(D_fit * 100, 2)
    fitted_b = round(b_fit, 3) if b_fit is not None else None
    st.subheader(f"🟠 Graph 1 – Auto-Fit Forecast (D = {fitted_decl_pct}%/yr"
                 + (f", b = {fitted_b})" if model_type == "Hyperbolic" else ")"))

    D_manual = manual_decl_pct / 100
    manual_fun = lambda yrs: hyperbolic(yrs, manual_qi, D_manual, manual_b) \
        if model_type == "Hyperbolic" else \
        lambda yrs: exponential(yrs, manual_qi, D_manual)

    horizon = np.arange(0, int(forecast_yrs * 365.25))
    yrs = horizon / 365.25

    qo_auto = auto_fun(yrs)
    qo_man  = manual_fun(yrs)

    def apply_stop(qo):
        cum_m3  = np.cumsum(qo)
        cum_mcm = cum_m3 / 1e6
        stop = cum_mcm > eur_mcm
        if use_cut:
            stop |= (qo < cutoff_qo)
        stop_idx = np.argmax(stop) if stop.any() else len(qo)
        return stop_idx, cum_m3

    idx_auto, cum_auto = apply_stop(qo_auto)
    idx_man,  cum_man  = apply_stop(qo_man)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Days"], y=df["Qo"], mode="lines+markers", name="Actual Qo"))
    fig.add_trace(go.Scatter(x=horizon[:idx_auto] + t0, y=qo_auto[:idx_auto],
                             mode="lines", name="Auto Forecast", line=dict(color="orange", dash="dash")))
    fig.add_vline(x=horizon[idx_auto-1] + t0, line_dash="dash", line_color="orange")
    fig.add_trace(go.Scatter(x=horizon[:idx_man] + t0, y=qo_man[:idx_man],
                             mode="lines", name="Manual Forecast", line=dict(color="blue", dash="dot")))
    fig.add_vline(x=horizon[idx_man-1] + t0, line_dash="dot", line_color="blue")
    if use_cut:
        fig.add_trace(go.Scatter(x=[0, max(horizon[idx_auto-1], horizon[idx_man-1]) + t0],
                                 y=[cutoff_qo]*2, mode="lines", name="Cut-off",
                                 line=dict(color="red", dash="dot")))
    fig.update_layout(title="Auto vs Manual Forecast", xaxis_title="Days", yaxis_title="Qo (m³/d)")
    st.plotly_chart(fig, use_container_width=True)

    out_auto = pd.DataFrame({
        "Days": horizon[:idx_auto] + t0,
        "Auto Forecast Qo": qo_auto[:idx_auto],
        "Cum Forecast Auto (m³)": cum_auto[:idx_auto]
    })
    out_man = pd.DataFrame({
        "Days": horizon[:idx_man] + t0,
        "Manual Forecast Qo": qo_man[:idx_man],
        "Cum Forecast Manual (m³)": cum_man[:idx_man]
    })

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        hist.to_excel(writer, sheet_name="Historical", index=False)
        out_auto.to_excel(writer, sheet_name="Auto Forecast", index=False)
        out_man.to_excel(writer, sheet_name="Manual Forecast", index=False)

    st.download_button("📥 Download Excel", buf.getvalue(), "DCA_Forecast.xlsx",
                       mime="application/vnd.ms-excel")
