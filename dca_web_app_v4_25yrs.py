
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.optimize import curve_fit
import io

st.set_page_config(page_title="DCA Forecast Tool", layout="centered")
st.title("📉 Decline Curve Analysis (DCA) Forecast Tool")

st.markdown("Upload an Excel file with the following columns:")
st.markdown("- **Month** (Date format)")
st.markdown("- **Oil Production (m3/d)**")
st.markdown("- **Oil m3** (Cumulative Oil)")

uploaded_file = st.file_uploader("Upload your well production Excel file", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        df['Month'] = pd.to_datetime(df['Month'])
        df = df.sort_values('Month').reset_index(drop=True)
        df['Days'] = (df['Month'] - df['Month'].iloc[0]).dt.days
        df['Qo (m3/day)'] = df['Oil Production (m3/d)']
        df['CumOil (m3)'] = df['Oil m3'].cumsum()
        st.success("✅ File loaded successfully!")

        # Plot preview
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Days'], y=df['Qo (m3/day)'],
                                 mode='markers+lines', name='Actual Qo',
                                 marker=dict(color='blue')))
        fig.update_layout(title='Preview: Actual Oil Rate',
                          xaxis_title='Days',
                          yaxis_title='Qo (m³/day)',
                          hovermode='closest')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📌 Forecast Settings")
        start_day = st.number_input("Start Day", min_value=0, value=0)
        ignore_days_input = st.text_input("Ignore Days (e.g. 200-220, 250)")
        eur = st.number_input("Estimated Ultimate Recovery (EUR) in million m³", value=86.0)
        decline_pct = st.slider("Annual Decline Rate (%)", min_value=0.1, max_value=100.0, step=0.1, value=14.0)
        cutoff = st.number_input("Cutoff Rate (m³/day)", min_value=0.1, max_value=100.0, step=0.1, value=0.5)
        b_val = st.slider("Hyperbolic Exponent (b)", min_value=0.1, max_value=1.0, step=0.01, value=0.5)
        model_type = st.radio("Forecast Type", ['Hyperbolic', 'Exponential'])

        def parse_ignore_input(text):
            ignore = set()
            for part in text.split(','):
                part = part.strip()
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    ignore.update(range(start, end + 1))
                elif part.isdigit():
                    ignore.add(int(part))
            return list(ignore)

        if st.button("🔍 Analyze Forecast"):
            ignore_days = parse_ignore_input(ignore_days_input)
            df_filtered = df[df['Days'] >= start_day].copy()
            if ignore_days:
                df_filtered = df_filtered[~df_filtered['Days'].isin(ignore_days)]

            # Convert days to years for proper decline rate handling
            t = (df_filtered['Days'].values - df_filtered['Days'].values[0]) / 365.25
            q = df_filtered['Qo (m3/day)'].values
            D_year = decline_pct / 100  # convert to fractional annual rate

            def hyperbolic(t, qi):
                return qi / ((1 + b_val * D_year * t) ** (1 / b_val))

            def exponential(t, qi):
                return qi * np.exp(-D_year * t)

            try:
                if model_type == 'Hyperbolic':
                    popt_D, _ = curve_fit(lambda t, D: hyperbolic(t, q[0], D, b_val), t, q, p0=[D_year], bounds=([1e-5], [1.0]), maxfev=10000)
                    forecast_func = lambda x: hyperbolic(x, q[0], popt_D[0], b_val)
                    forecast_func = lambda x: hyperbolic(x, *popt)
                else:
                    popt_D, _ = curve_fit(lambda t, D: exponential(t, q[0], D), t, q, p0=[D_year], bounds=([1e-5], [1.0]), maxfev=10000)
                    forecast_func = lambda x: exponential(x, q[0], popt_D[0])
                    forecast_func = lambda x: exponential(x, *popt)

                # Forecast for 15 years
                full_days = np.arange(0, int(25 * 365.25))
                full_years = full_days / 365.25
                forecast_values = forecast_func(full_years)
                cum_forecast = np.cumsum(forecast_values)
                EUR_limit = eur * 1e6
                stop_mask = (forecast_values < cutoff) | (cum_forecast > EUR_limit)
                stop_mask &= (full_years > t[-1])
                if not stop_mask.any():
                cutoff_idx = len(full_days)
                else:
                    cutoff_idx = np.argmax(stop_mask)
                if cutoff_idx == 0:
                    cutoff_idx = len(full_days)

                forecast_df = pd.DataFrame({
                    'Days': df_filtered['Days'].values[0] + full_days[:cutoff_idx],
                    f'{model_type} Forecast': forecast_values[:cutoff_idx]
                })

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=df['Days'], y=df['Qo (m3/day)'],
                                          mode='lines+markers', name='Actual Qo'))
                fig2.add_trace(go.Scatter(x=forecast_df['Days'], y=forecast_df[f'{model_type} Forecast'],
                                          mode='lines', name=f'{model_type} Forecast',
                                          line=dict(color='orange', dash='dash')))
                fig2.add_trace(go.Scatter(x=[0, forecast_df['Days'].max()], y=[cutoff, cutoff],
                                          mode='lines', name=f'{cutoff} m³/day Cutoff',
                                          line=dict(color='red', dash='dot')))
                fig2.update_layout(title=f'{model_type} Forecast Result',
                                   xaxis_title='Days',
                                   yaxis_title='Qo (m³/day)',
                                   hovermode='closest')
                st.plotly_chart(fig2, use_container_width=True)

                df_export = pd.merge_asof(df[['Days', 'Qo (m3/day)', 'CumOil (m3)', 'Month']],
                                          forecast_df, on='Days', direction='nearest')
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_export.to_excel(writer, index=False, sheet_name='Forecast')
                st.download_button("📥 Download Forecast Excel", data=output.getvalue(),
                                   file_name="Well_Forecast_Output.xlsx")
            except Exception as e:
                st.error(f"Forecasting failed: {e}")
    except Exception as e:
        st.error(f"File processing failed: {e}")
