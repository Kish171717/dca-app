import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.optimize import curve_fit
import io

st.set_page_config(page_title="DCA Forecast Tool", layout="centered")
st.title("📉 Decline Curve Analysis (DCA) Forecast Tool")

# File uploader
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

# Forecast settings
start_day   = st.number_input("Start Forecast From Day", min_value=0, value=0)
ignore_txt  = st.text_input("Ignore Days (e.g. 200-220, 300)")
eur         = st.number_input("EUR (Million m³)", value=86.0, step=1.0)
cutoff      = st.number_input("Cutoff Rate (m³/d)", value=0.5, step=0.1)
decline_pct = st.slider("Annual Decline Rate (%)", 0.1, 100.0, 14.0, 0.1)
b_val       = st.slider("Hyperbolic exponent (b)", 0.1, 1.0, 0.5, 0.01)
model_type  = st.radio("Forecast Type", ["Hyperbolic", "Exponential"])
run_btn     = st.button("📈 Analyze & Forecast")

def parse_ignore(s):
    out = set()
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            a, b = map(int, part.split('-'))
            out.update(range(a, b+1))
        elif part.isdigit():
            out.add(int(part))
    return list(out)

def hyperbolic(t, qi, D, b):
    return qi / ((1 + b * D * t) ** (1 / b))

def exponential(t, qi, D):
    return qi * np.exp(-D * t)

if uploaded_file:
    # Load and validate
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read Excel: {e}")
        st.stop()

    required = ['Month', 'Oil Production (m3/d)', 'Oil m3']
    if not all(c in df.columns for c in required):
        st.error(f"Missing columns. Found: {list(df.columns)}")
        st.stop()

    # Prepare data
    df['Month'] = pd.to_datetime(df['Month'])
    df = df.sort_values('Month').reset_index(drop=True)
    df['Days']   = (df['Month'] - df['Month'].iloc[0]).dt.days
    df['Qo']     = df['Oil Production (m3/d)']
    df['CumOil'] = df['Oil m3'].cumsum()

    # Show historical
    st.success("✅ File loaded")
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Scatter(
        x=df['Days'], y=df['Qo'],
        mode='markers+lines', name='Actual Qo'
    ))
    hist_fig.update_layout(
        title="Actual Production",
        xaxis_title="Days",
        yaxis_title="Qo (m³/d)"
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    if run_btn:
        ignores = parse_ignore(ignore_txt)
        hist = df[df['Days'] >= start_day].copy()
        if ignores:
            hist = hist[~hist['Days'].isin(ignores)]

        # Time & rate arrays (in years)
        t = (hist['Days'] - hist['Days'].iloc[0]) / 365.25
        q = hist['Qo'].values

        # Filter zeros/NaNs
        mask = (q > 0) & ~np.isnan(q)
        t = t[mask]; q = q[mask]
        if len(q) < 5:
            st.warning("Too few valid points after filtering.")
            st.stop()

        # Initial guesses
        qi     = np.nanmax(q[:5])
        D_init = max(decline_pct/100, 0.01)

        # Fit
        try:
            if model_type == "Hyperbolic":
                popt, _ = curve_fit(
                    lambda tt, D: hyperbolic(tt, qi, D, b_val),
                    t, q, p0=[D_init], bounds=([1e-5],[1.0]), maxfev=20000
                )
                D_fit = popt[0]
                forecast_func = lambda tt: hyperbolic(tt, qi, D_fit, b_val)
            else:
                popt, _ = curve_fit(
                    lambda tt, D: exponential(tt, qi, D),
                    t, q, p0=[D_init], bounds=([1e-5],[1.0]), maxfev=20000
                )
                D_fit = popt[0]
                forecast_func = lambda tt: exponential(tt, qi, D_fit)
        except Exception as e:
            st.error(f"Curve fit failed: {e}")
            st.stop()

        # Generate forecast until cutoff/EUR
        full_days   = np.arange(0, int(100*365.25))
        full_years  = full_days / 365.25
        forecast_q  = forecast_func(full_years)
        cum_forecast = np.cumsum(forecast_q)
        EUR_lim     = eur * 1e6

        stop_mask = ((forecast_q < cutoff) | (cum_forecast > EUR_lim)) & (full_years > t[-1])
        cutoff_idx = np.argmax(stop_mask) if stop_mask.any() else len(full_days)

        fd = full_days[:cutoff_idx]
        fq = forecast_q[:cutoff_idx]

        # Plot forecast
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Days'], y=df['Qo'],
            mode='markers+lines', name='Actual Qo'
        ))
        fig.add_trace(go.Scatter(
            x=fd + hist['Days'].iloc[0], y=fq,
            mode='lines', name=f'{model_type} Forecast',
            line=dict(color='orange', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=[0, fd[-1]+hist['Days'].iloc[0]],
            y=[cutoff, cutoff],
            mode='lines', name='Cutoff',
            line=dict(color='red', dash='dot')
        ))
        fig.update_layout(
            title="Forecast",
            xaxis_title="Days",
            yaxis_title="Qo (m³/d)"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Excel export
        df_out = pd.DataFrame({
            'Days': fd + hist['Days'].iloc[0],
            'Forecast Qo': fq,
            'Cum Forecast': cum_forecast[:cutoff_idx]
        })
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
            hist.to_excel(writer, sheet_name='Historical', index=False)
            df_out.to_excel(writer, sheet_name='Forecast', index=False)
        st.download_button(
            "📥 Download Excel",
            data=buf.getvalue(),
            file_name="DCA_Forecast.xlsx"
        )
