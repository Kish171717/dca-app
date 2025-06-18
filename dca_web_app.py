import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import io

# ─── page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="DCA Manual Decline Tool", layout="centered")
st.title("📉 Decline Curve Analysis – Manual Decline Model")

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

# ─── show historical plot ──────────────────────────────────────────────────
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
use_cut_off = st.checkbox("Apply Cut-off Qo?", value=True)
cut_off_qo  = st.number_input("Cut-off Qo (m³/d)",
                              0.1, 100.0, value=0.5, step=0.1,
                              disabled=not use_cut_off)
decl_pct    = st.slider("Decline Rate (% per year)", 0.1, 100.0, 14.0, 0.1)
b_val       = st.slider("Hyperbolic exponent b", 0.05, 1.0, 0.5, 0.01)
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

def hyperbolic(t_yrs, qi, D, b):
    """t_yrs in years, qi in Qo units, D fraction/yr"""
    return qi / ((1 + b * D * t_yrs) ** (1 / b))

def exponential(t_yrs, qi, D):
    return qi * np.exp(-D * t_yrs)

# ─── forecast computation ──────────────────────────────────────────────────
if run_btn:

    ignore_days = parse_ignore(ignore_txt)
    hist = df[df["Days"] >= start_day].copy()
    if ignore_days:
        hist = hist[~hist["Days"].isin(ignore_days)]

    if len(hist) < 3:
        st.warning("Too few data points after filtering.")
        st.stop()

    # time & rate arrays
    t0_days = hist["Days"].iloc[0]
    t_hist_yrs = (hist["Days"] - t0_days) / 365.25
    q_hist = hist["Qo"].values

    # initial parameters
    qi = q_hist[0]            # initial rate from cleaned data
    D  = decl_pct / 100       # decline fraction per year

    # forecast horizon
    horizon_days = np.arange(0, int(forecast_yrs * 365.25))
    yrs = horizon_days / 365.25

    # calculate forecast Qo
    if model_type == "Hyperbolic":
        qo_fore = hyperbolic(yrs, qi, D, b_val)
    else:
        qo_fore = exponential(yrs, qi, D)

    # cumulative forecast (m³) & in million m³
    cum_fore_m3  = np.cumsum(qo_fore)
    cum_fore_mcm = cum_fore_m3 / 1e6

    # stopping condition
    stop_mask = cum_fore_mcm > eur_mcm
    if use_cut_off:
        stop_mask |= qo_fore < cut_off_qo
    stop_mask &= yrs > t_hist_yrs.iloc[-1]

    stop_idx = np.argmax(stop_mask) if stop_mask.any() else len(yrs)
    f_days   = horizon_days[:stop_idx]
    f_qo     = qo_fore[:stop_idx]
    hit_msg  = "Cut-off" if use_cut_off and f_qo[-1] < cut_off_qo else "EUR limit"

    # ─── plot forecast ─────────────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Days"], y=df["Qo"], mode="lines+markers", name="Actual Qo"))
    fig.add_trace(go.Scatter(x=f_days + t0_days, y=f_qo,
                             mode="lines", name=f"{model_type} Forecast",
                             line=dict(color="orange", dash="dash")))
    fig.add_vline(x=f_days[-1] + t0_days, line_dash="dash", line_color="green")
    fig.add_annotation(
        x=f_days[-1] + t0_days, y=f_qo[-1],
        text=f"Stopped by {hit_msg}",
        showarrow=True, arrowhead=1, ay=-40
    )
    if use_cut_off:
        fig.add_trace(go.Scatter(
            x=[0, f_days[-1] + t0_days], y=[cut_off_qo] * 2,
            mode="lines", name="Cut-off",
            line=dict(color="red", dash="dot")
        ))
    fig.update_layout(title="Forecast Result", xaxis_title="Days", yaxis_title="Qo (m³/d)")
    st.plotly_chart(fig, use_container_width=True)

    # ─── excel export ──────────────────────────────────────────────────────
    forecast_df = pd.DataFrame({
        "Days": f_days + t0_days,
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
        mime="application/vnd.ms-excel",
    )
