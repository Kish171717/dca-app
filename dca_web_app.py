import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.optimize import curve_fit
import io

# ─── page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="DCA Forecast Tool", layout="centered")
st.title("📉 Decline Curve Analysis — Auto vs Manual Forecast")

# ─── file upload ───────────────────────────────────────────────────────────
file = st.file_uploader("Upload Excel (Month, Oil Production (m3/d), Oil m3)", ["xlsx"])
if not file:
    st.stop()

try:
    df = pd.read_excel(file)
except Exception as e:
    st.error(f"Excel read failed: {e}")
    st.stop()

need = ["Month", "Oil Production (m3/d)", "Oil m3"]
if any(c not in df.columns for c in need):
    st.error(f"Missing required columns: {list(df.columns)}")
    st.stop()

# ─── prep dataframe ────────────────────────────────────────────────────────
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").reset_index(drop=True)
df["Days"] = (df["Month"] - df["Month"].iloc[0]).dt.days
df["Qo"] = df["Oil Production (m3/d)"]
df["CumOil"] = df["Oil m3"].cumsum()

# ─── historical graph ──────────────────────────────────────────────────────
st.plotly_chart(
    go.Figure(
        go.Scatter(x=df["Days"], y=df["Qo"], mode="lines+markers", name="Actual Qo")
    ).update_layout(title="Historical Production", xaxis_title="Days", yaxis_title="Qo (m³/d)"),
    use_container_width=True,
)

# ─── forecast settings ─────────────────────────────────────────────────────
st.header("⚙️ Forecast Settings")
start_day   = st.number_input("Start Day", 0, value=0)
ignore_txt  = st.text_input("Ignore Days (e.g. 200-220, 300)")
eur_mcm     = st.number_input("EUR (million m³)", 1.0, 1e4, value=86.0)
use_cut     = st.checkbox("Apply Cut-off?", value=True)
cutoff_qo   = st.number_input("Cut-off Qo (m³/d)", 0.1, 100.0, value=0.5, disabled=not use_cut)
forecast_yrs= st.slider("Forecast horizon (years)", 1, 100, 50)

st.subheader("🟠 Auto-Fit Initial Guesses")
decl_pct_guess = st.slider("Initial Decline guess (%/yr)", 0.1, 100.0, 14.0, 0.1)
b_guess        = st.slider("Initial b guess (hyperbolic)", 0.05, 1.0, 0.5, 0.01)
model_type     = st.radio("Auto-Fit Model", ["Hyperbolic", "Exponential"])

st.subheader("🔵 Manual Forecast (always Hyperbolic)")
default_qi = float(df["Qo"].iloc[0])
manual_qi        = st.number_input("Manual qi (m³/d)", 0.01, 10000.0, value=default_qi, step=0.1)
manual_decl_pct  = st.slider("Manual Decline %/yr", 0.1, 100.0, 20.0, 0.1)
manual_b         = st.slider("Manual b", 0.05, 1.0, 0.5, 0.01)

run_btn = st.button("🔍 Run Forecast")

# ─── helpers ───────────────────────────────────────────────────────────────
def parse_ignore(txt):
    out=set()
    for p in txt.split(","):
        p=p.strip()
        if "-" in p:
            a,b=map(int,p.split("-")); out.update(range(a,b+1))
        elif p.isdigit():
            out.add(int(p))
    return out

def hyperbolic(t, qi, D, b): return qi / ((1 + b*D*t)**(1/b))
def exponential(t, qi, D):   return qi * np.exp(-D*t)

# ─── run forecast ──────────────────────────────────────────────────────────
if run_btn:

    ignore = parse_ignore(ignore_txt)
    hist = df[df["Days"] >= start_day].copy()
    if ignore:
        hist = hist[~hist["Days"].isin(ignore)]

    if len(hist) < 3:
        st.warning("Too few data after filtering."); st.stop()

    # arrays for fit
    t0  = hist["Days"].iloc[0]
    yrs_hist = (hist["Days"] - t0) / 365.25
    q_hist   = hist["Qo"].values
    mask = (q_hist > 0) & ~np.isnan(q_hist)
    yrs_hist, q_hist = yrs_hist[mask], q_hist[mask]

    # ─── Auto-fit model ───────────────────────────────────────────────────
    qi_guess = q_hist[0]
    D_guess  = decl_pct_guess / 100

    try:
        if model_type == "Hyperbolic":
            popt,_ = curve_fit(hyperbolic, yrs_hist, q_hist,
                               p0=[qi_guess, D_guess, b_guess],
                               bounds=([0.01,1e-5,0.05],[q_hist.max()*10,5.0,1.0]),
                               maxfev=60000)
            qi_fit,D_fit,b_fit = popt
            auto_fun = lambda y: hyperbolic(y, qi_fit, D_fit, b_fit)
        else:
            popt,_ = curve_fit(exponential, yrs_hist, q_hist,
                               p0=[qi_guess, D_guess],
                               bounds=([0.01,1e-5],[q_hist.max()*10,5.0]),
                               maxfev=60000)
            qi_fit,D_fit = popt
            auto_fun = lambda y: exponential(y, qi_fit, D_fit)
    except Exception as e:
        st.warning(f"Auto-fit failed → using guesses ({e})")
        auto_fun = lambda y: hyperbolic(y, qi_guess, D_guess, b_guess) \
            if model_type=="Hyperbolic" else lambda y: exponential(y, qi_guess, D_guess)

    # ─── Manual hyperbolic model ──────────────────────────────────────────
    D_manual = manual_decl_pct / 100
    manual_fun = lambda y: hyperbolic(y, manual_qi, D_manual, manual_b)

    # horizon
    horizon = np.arange(0, int(forecast_yrs*365.25))
    yrs = horizon/365.25

    qo_auto = auto_fun(yrs)
    qo_man  = manual_fun(yrs)

    def trim(qo):
        cum_mcm = np.cumsum(qo)/1e6
        stop = (cum_mcm > eur_mcm) | ((qo < cutoff_qo) if use_cut else False)
        idx  = np.argmax(stop) if stop.any() else len(qo)
        return idx, cum_mcm

    idx_auto, cum_auto = trim(qo_auto)
    idx_man , cum_man  = trim(qo_man)

    # ─── Graph 1 – Auto ───────────────────────────────────────────────────
    st.subheader("🟠 Graph 1 – Auto-Fit Forecast")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df["Days"], y=df["Qo"], mode="lines+markers", name="Actual"))
    fig1.add_trace(go.Scatter(
        x=horizon[:idx_auto]+t0, y=qo_auto[:idx_auto],
        mode="lines", name="Auto Forecast", line=dict(color="orange", dash="dash")))
    fig1.add_vline(x=horizon[idx_auto-1]+t0, line_color="orange", line_dash="dash")
    if use_cut:
        fig1.add_trace(go.Scatter(
            x=[0,horizon[idx_auto-1]+t0], y=[cutoff_qo]*2,
            mode="lines", name="Cut-off", line=dict(color="red", dash="dot")))
    st.plotly_chart(fig1, use_container_width=True)

    # ─── Graph 2 – Manual hyperbolic ──────────────────────────────────────
    st.subheader("🔵 Graph 2 – Manual Hyperbolic Forecast")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["Days"], y=df["Qo"], mode="lines+markers", name="Actual"))
    fig2.add_trace(go.Scatter(
        x=horizon[:idx_man]+t0, y=qo_man[:idx_man],
        mode="lines", name="Manual Forecast", line=dict(color="blue", dash="dot")))
    fig2.add_vline(x=horizon[idx_man-1]+t0, line_color="blue", line_dash="dot")
    if use_cut:
        fig2.add_trace(go.Scatter(
            x=[0,horizon[idx_man-1]+t0], y=[cutoff_qo]*2,
            mode="lines", name="Cut-off", line=dict(color="red", dash="dot")))
    st.plotly_chart(fig2, use_container_width=True)

    # ─── Excel export ────────────────────────────────────────────────────
    out_auto = pd.DataFrame({"Days": horizon[:idx_auto]+t0, "Qo Auto": qo_auto[:idx_auto], "Cum Auto (m³)": cum_auto[:idx_auto]*1e6})
    out_man  = pd.DataFrame({"Days": horizon[:idx_man]+t0,  "Qo Manual": qo_man[:idx_man], "Cum Manual (m³)": cum_man[:idx_man]*1e6})

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        df.to_excel(w, sheet_name="Historical", index=False)
        out_auto.to_excel(w, sheet_name="Auto Forecast", index=False)
        out_man.to_excel(w,  sheet_name="Manual Forecast", index=False)

    st.download_button("📥 Download Excel", buf.getvalue(), "DCA_Forecast.xlsx",
                       mime="application/vnd.ms-excel")
