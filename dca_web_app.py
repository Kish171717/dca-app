import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import io

st.set_page_config(page_title="DCA – Manual Decline Forecast", layout="centered")
st.title("📉 Decline Curve Analysis (with Custom Decline %)")

# ───────── Upload File ─────────
file = st.file_uploader("Upload Excel (must contain: Month, Oil Production (m3/d), Oil m3)", ["xlsx"])

# ───────── Load and Plot Actual ─────────
if file:
    try:
        df = pd.read_excel(file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    need = ['Month', 'Oil Production (m3/d)', 'Oil m3']
    if any(col not in df.columns for col in need):
        st.error(f"Missing columns in Excel: {list(df.columns)}")
        st.stop()

    df['Month'] = pd.to_datetime(df['Month'])
    df = df.sort_values('Month').reset_index(drop=True)
    df['Days'] = (df['Month'] - df['Month'].iloc[0]).dt.days
    df['Qo'] = df['Oil Production (m3/d)']
    df['CumOil'] = df['Oil m3'].cumsum()

    st.plotly_chart(
        go.Figure(go.Scatter(x=df['Days'], y=df['Qo'],
                             mode='lines+markers', name='Actual Qo'))
        .update_layout(title="Actual Production", xaxis_title="Days", yaxis_title="Qo (m³/d)"),
        use_container_width=True)

    # ───────── Forecast Settings ─────────
    st.header("🔧 Forecast Settings")
    start_day = st.number_input("Start Day (from actual data)", min_value=0, value=0)
    ignore_text = st.text_input("Ignore Days (e.g. 200-220, 300)")
    eur_mcm = st.number_input("EUR (million m³)", min_value=1.0, max_value=1e4, value=86.0)
    use_cutoff = st.checkbox("Apply Cut-off?", value=True)
    cutoff = st.number_input("Cut-off Qo (m³/d)", min_value=0.1, max_value=100.0,
                             value=0.5, step=0.1, disabled=not use_cutoff)
    decline_pct = st.slider("Decline Rate (%/year)", 0.1, 100.0, 14.0, 0.1)
    b_val = st.slider("Hyperbolic Exponent (b)", 0.05, 1.0, 0.5, 0.01)
    model = st.radio("Forecast Type", ["Hyperbolic", "Exponential"])
    run_btn = st.button("🔍 Run Forecast")

    # ───────── Forecast Logic ─────────
    if run_btn:
        def parse_ignore(text):
            ignore_days = set()
            for part in text.split(','):
                if '-' in part:
                    a, b = map(int, part.strip().split('-'))
                    ignore_days.update(range(a, b + 1))
                elif part.strip().isdigit():
                    ignore_days.add(int(part.strip()))
            return ignore_days

        ignore = parse_ignore(ignore_text)
        hist = df[df['Days'] >= start_day].copy()
        if ignore:
            hist = hist[~hist['Days'].isin(ignore)]

        if len(hist) < 5:
            st.warning("Too few data points after filtering.")
            st.stop()

        t0 = hist['Days'].iloc[0]
        t_hist = (hist['Days'] - t0) / 365.25
        q_hist = hist['Qo'].values
        qi = q_hist[0]  # starting rate from cleaned data
        D = decline_pct / 100  # decline rate per year

        # Forecast curve (manual model)
        def hyperbolic(t, qi, D, b): return qi / ((1 + b * D * t) ** (1 / b))
        def exponential(t, qi, D): return qi * np.exp(-D * t)

        horizon = np.arange(0, int(100 * 365.25))  # forecast range in days
        yrs = horizon / 365.25
        qo = hyperbolic(yrs, qi, D, b_val) if model == "Hyperbolic" else exponential(yrs, qi, D)
        cum = np.cumsum(qo)
        cum_mcm = cum / 1e6  # convert to million m³

        # Stopping logic
        stop_mask = cum_mcm > eur_mcm
        if use_cutoff:
            stop_mask |= (qo < cutoff)
        stop_mask &= (yrs > t_hist.iloc[-1])

        stop_idx = np.argmax(stop_mask) if stop_mask.any() else len(yrs)

        forecast_days = horizon[:stop_idx]
        forecast_qo = qo[:stop_idx]
        forecast_cum = cum[:stop_idx]
        x_shift = int(t0)

        # Determine stopping reason
        hit_reason = "Cut-off" if use_cutoff and forecast_qo[-1] < cutoff else "EUR"

        # ───────── Plot Forecast ─────────
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Days'], y=df['Qo'],
                                 mode='lines+markers', name='Actual Qo'))
        fig.add_trace(go.Scatter(x=forecast_days + x_shift, y=forecast_qo,
                                 mode='lines', name=f'{model} Forecast',
                                 line=dict(color='orange', dash='dash')))
        fig.add_vline(x=forecast_days[-1] + x_shift, line_dash="dash", line_color="green")
        fig.add_annotation(x=forecast_days[-1] + x_shift, y=forecast_qo[-1],
                           text=f"Stopped by {hit_reason}",
                           showarrow=True, arrowhead=1, ay=-40)
        if use_cutoff:
            fig.add_trace(go.Scatter(x=[0, forecast_days[-1] + x_shift],
                                     y=[cutoff] * 2, mode='lines', name='Cut-off',
                                     line=dict(color='red', dash='dot')))
        fig.update_layout(title='Forecast', xaxis_title='Days', yaxis_title='Qo (m³/d)')
        st.plotly_chart(fig, use_container_width=True)

        # ───────── Excel Export ─────────
        forecast_df = pd.DataFrame({
            'Days': forecast_days + x_shift,
            'Forecast Qo': forecast_qo,
            'Cumulative Forecast (m³)': forecast_cum
        })

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
            hist.to_excel(writer, sheet_name="Historical", index=False)
            forecast_df.to_excel(writer, sheet_name="Forecast", index=False)

        st.download_button("📥 Download Forecast Excel", data=buf.getvalue(),
                           file_name="DCA_Forecast_Output.xlsx", mime="application/vnd.ms-excel")
