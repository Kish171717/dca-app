import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.optimize import curve_fit
import io

# ─── page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="DCA Forecast Tool", layout="centered")
st.title("📉 Decline Curve Analysis (Hyperbolic / Exponential)")

# ─── file upload ───────────────────────────────────────────────────────────
file = st.file_uploader(
    "Upload well data (columns: Month, Oil Production (m3/d), Oil m3)", ["xlsx"]
)

if not file:
    st.info("Upload an Excel file to begin.")
    st.stop()

# ─── read & validate ───────────────────────────────────────────────────────
try:
    df = pd.read_excel(file)
except Exception as e:
    st.error(f"Could not read Excel: {e}")
    st.stop()

required_cols = ["Month", "Oil Production (m3/d)", "Oil m3"]
if any(c not in df.columns for c in required_cols):
    st.error(f"Missing required columns. Found: {list(df.columns)}")
    st.stop()

# ─── normalize dataframe ───────────────────────────────────────────────────
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").reset_index(drop=True)
df["Days"]   = (df["Month"] - df["Month"].iloc[0]).dt.days
df["Qo"]     = df["Oil Production (m3/d)"]
df["CumOil"] = df["Oil m3"].cumsum()

st.plotly_chart(
    go.Figure(
        go.Scatter(x=df["Days"], y=df["Qo"], mode="lines+markers", name="Actual Qo")
    ).update_layout(
        title="Historical Production",
        xaxis_title="Days",
        yaxis_title="Qo (m³/d)"
    ),
    use_container_width=True
)

# ─── forecast settings ─────────────────────────────────────────────────────
st.header("⚙️ Forecast Settings")

start_day   = st.number_input("Start Day", min_value=0, value=0)
ignore_txt  = st.text_input("Ignore Days (e.g. 200-220, 300)")
eur_mcm     = st.number_input("EUR (million m³)", 1.0, 1e4, value=86.0)
use_cutoff  = st.checkbox("Apply Cut-off Qo?", value=True)
cutoff_qo   = st.number_input("Cut-off Qo (m³/d)",
                              0.1, 100.0, value=0.5, step=0.1,
                              disabled=not use_cutoff)
decl_pct    = st.slider("Initial Decline Guess  (%/yr)", 0.1, 100.0, 14.0, 0.1)
b_val       = st.slider("Initial b Guess (hyperbolic)", 0.05, 1.0, 0.5, 0.01)
model_type  = st.radio("Model Type", ["Hyperbolic", "Exponential"])
forecast_yrs= st.slider("Forecast Horizon (years)", 1, 100, 50)
run_btn     = st.button("🔍 Run Forecast")

# ─── helper functions ──────────────────────────────────────────────────────
def parse_ignore(text: str) -> set[int]:
    out = set()
    for part in text.split(","):
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

# ─── forecast computation ──────────────────────────────────────────────────
if run_btn:

    ignore_days = parse_ignore(ignore_txt)
    hist = df[df["Days"] >= start_day].copy()
    if ignore_days:
        hist = hist[~hist["Days"].isin(ignore_days)]

    if len(hist) < 3:
        st.warning("Too few data points after filtering.")
        st.stop()

    # arrays for fitting
    t0 = hist["Days"].iloc[0]
    t_hist_yrs = (hist["Days"] - t0) / 365.25
    q_hist = hist["Qo"].values
    mask = (q_hist > 0) & ~np.isnan(q_hist)
    t_hist_yrs, q_hist = t_hist_yrs[mask], q_hist[mask]

    if len(q_hist) < 3:
        st.warning("Filtered data has insufficient positive points.")
        st.stop()

    qi_guess = np.nanmax(q_hist[:5])
    D_guess  = max(decl_pct / 100, 0.01)

    # robust auto-fit (qi, D, b or qi, D)
    try:
        if model_type == "Hyperbolic":
            popt,_ = curve_fit(
                hyperbolic, t_hist_yrs, q_hist,
                p0=[qi_guess, D_guess, b_val],
                bounds=([0.01, 1e-5, 0.05],
                        [q_hist.max()*10, 5.0, 1.0]),
                maxfev=60000
            )
            qi_fit, D_fit, b_fit = popt
            forecast_fun = lambda yrs: hyperbolic(yrs, qi_fit, D_fit, b_fit)
        else:
            popt,_ = curve_fit(
                exponential, t_hist_yrs, q_hist,
                p0=[qi_guess, D_guess],
                bounds=([0.01, 1e-5],
                        [q_hist.max()*10, 5.0]),
                maxfev=60000
            )
            qi_fit, D_fit = popt
            forecast_fun = lambda yrs: exponential(yrs, qi_fit, D_fit)
    except Exception as e:
        st.warning(f"Fit warning, falling back to initial guess ({e})")
        if model_type == "Hyperbolic":
            forecast_fun = lambda yrs: hyperbolic(yrs, qi_guess, D_guess, b_val)
        else:
            forecast_fun = lambda yrs: exponential(yrs, qi_guess, D_guess)

    # forecast horizon
    horizon_days = np.arange(0, int(forecast_yrs * 365.25))
    yrs = horizon_days / 365.25
    qo_fore = forecast_fun(yrs)

    cum_fore_m3  = np.cumsum(qo_fore)
    cum_fore_mcm = cum_fore_m3 / 1e6
    EUR_limit    = eur_mcm

    # stopping mask
    stop_mask = cum_fore_mcm > EUR_limit
    if use_cutoff:
        stop_mask |= qo_fore < cutoff_qo
    stop_mask &= yrs > t_hist_yrs.iloc[-1]

    stop_idx = np.argmax(stop_mask) if stop_mask.any() else len(yrs)
    f_days   = horizon_days[:stop_idx]
    f_qo     = qo_fore[:stop_idx]
    hit_reason = "Cut-off" if use_cutoff and f_qo[-1] < cutoff_qo else "EUR limit"

    # ─── plot ──────────────────────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Days"], y=df["Qo"],
        mode="lines+markers", name="Actual Qo"
    ))
    fig.add_trace(go.Scatter(
        x=f_days + t0, y=f_qo,
        mode="lines", name=f"{model_type} Forecast",
        line=dict(color="orange", dash="dash")
    ))
    fig.add_vline(x=f_days[-1] + t0, line_dash="dash", line_color="green")
    fig.add_annotation(
        x=f_days[-1] + t0, y=f_qo[-1],
        text=f"Stopped by {hit_reason}",
        showarrow=True, arrowhead=1, ay=-40
    )
    if use_cutoff:
        fig.add_trace(go.Scatter(
            x=[0, f_days[-1] + t0], y=[cutoff_qo]*2,
            mode="lines", name="Cut-off",
            line=dict(color="red", dash="dot")
        ))
    fig.update_layout(
        title="Forecast Result",
        xaxis_title="Days",
        yaxis_title="Qo (m³/d)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ─── excel export ──────────────────────────────────────────────────────
    forecast_df = pd.DataFrame({
        "Days": f_days + t0,
        "Forecast Qo": f_qo,
        "Cum Forecast (m³)": cum_fore_m3[:stop_idx]
    })
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        hist.to_excel(writer, sheet_name="Historical", index=False)
        forecast_df.to_excel(writer, sheet_name="Forecast", index=False)

    st.download_button(
        "📥 Download Forecast Excel",
        data=buffer.getvalue(),
        file_name="DCA_Forecast.xlsx",
        mime="application/vnd.ms-excel"
    )
