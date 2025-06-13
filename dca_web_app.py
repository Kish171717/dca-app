
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.optimize import curve_fit
import io

st.set_page_config(page_title='DCA Forecast Tool', layout='centered')
st.title('📉 Decline Curve Analysis (DCA) Forecast Tool')

st.markdown("""### Expected columns
• **Month** – date column  
• **Oil Production (m3/d)** – daily rate  
• **Oil m3** – cumulative production
""")

# ---------------- Widgets ----------------
uploaded_file = st.file_uploader('Upload Excel file', type=['xlsx'])

st.sidebar.header('Forecast parameters')
start_day   = st.sidebar.number_input('Start Day (from X‑axis)', 0, value=0)
ignore_txt  = st.sidebar.text_input('Ignore Days (e.g. 200-220, 300)', '')
eur_mcm     = st.sidebar.number_input('EUR (million m³)', 1.0, value=86.0, step=1.0)
cutoff_rate = st.sidebar.number_input('Cut‑off Qo (m³/d)', 0.1, value=0.5, step=0.1)
decline_pct = st.sidebar.slider('Annual Decline Rate %', 0.1, 100.0, 14.0, 0.1)
b_val       = st.sidebar.slider('Hyperbolic exponent b', 0.1, 1.0, 0.5, 0.01)
model_type  = st.sidebar.radio('Model type', ['Hyperbolic', 'Exponential'])
run_btn     = st.sidebar.button('🔍 Run Forecast')

# --------------- Helper funcs ---------------
def parse_ignore(text: str):
    out = set()
    for part in text.split(','):
        part = part.strip()
        if '-' in part:
            a, b = map(int, part.split('-'))
            out.update(range(a, b + 1))
        elif part.isdigit():
            out.add(int(part))
    return out

def hyperbolic(t, qi, D, b):
    return qi / ((1 + b * D * t) ** (1 / b))

def exponential(t, qi, D):
    return qi * np.exp(-D * t)

# --------------- Workflow -------------------
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f'Failed to read Excel: {e}')
        st.stop()

    # Minimal column validation
    required = ['Month', 'Oil Production (m3/d)', 'Oil m3']
    if not all(col in df.columns for col in required):
        st.error(f'Missing required columns. Found columns: {list(df.columns)}')
        st.stop()

    # Pre‑process
    df['Month'] = pd.to_datetime(df['Month'])
    df = df.sort_values('Month').reset_index(drop=True)
    df['Days'] = (df['Month'] - df['Month'].iloc[0]).dt.days
    df['Qo']   = df['Oil Production (m3/d)']
    df['CumOil'] = df['Oil m3'].cumsum()

    st.success('File loaded!')
    st.plotly_chart(
        go.Figure(
            data=go.Scatter(x=df['Days'], y=df['Qo'], mode='markers+lines', name='Actual Qo')
        ).update_layout(title='Actual Production', xaxis_title='Days', yaxis_title='Qo (m³/d)'),
        use_container_width=True
    )

    if run_btn:
        # Filter by start / ignore
        ignore_set = parse_ignore(ignore_txt)
        hist = df[df['Days'] >= start_day].copy()
        if ignore_set:
            hist = hist[~hist['Days'].isin(ignore_set)]
        if hist.empty:
            st.error('No data left after filtering.')
            st.stop()

        t_years = (hist['Days'] - hist['Days'].iloc[0]) / 365.25
        q_data  = hist['Qo'].values
        qi      = q_data[0]
        D_year  = decline_pct / 100

        # Fit
        try:
            if model_type == 'Hyperbolic':
                popt, _ = curve_fit(lambda t, D: hyperbolic(t, qi, D, b_val),
                                    t_years, q_data,
                                    p0=[D_year],
                                    bounds=([1e-5], [1.0]), maxfev=10000)
                D_fit = popt[0]
                forecast_func = lambda yrs: hyperbolic(yrs, qi, D_fit, b_val)
            else:
                popt, _ = curve_fit(lambda t, D: exponential(t, qi, D),
                                    t_years, q_data,
                                    p0=[D_year],
                                    bounds=([1e-5], [1.0]), maxfev=10000)
                D_fit = popt[0]
                forecast_func = lambda yrs: exponential(yrs, qi, D_fit)
        except Exception as e:
            st.error(f'Curve fitting failed: {e}')
            st.stop()

        # Forecast horizon: 100 years (effectively uncapped)
        full_days  = np.arange(0, int(100 * 365.25))
        full_years = full_days / 365.25
        forecast_q = forecast_func(full_years)
        cum_q      = np.cumsum(forecast_q)

        EUR_limit = eur_mcm * 1e6
        valid_mask = full_years > t_years.iloc[-1]  # enforce after history
        stop_mask = ((forecast_q < cutoff_rate) | (cum_q > EUR_limit)) & valid_mask
        cutoff_idx = np.argmax(stop_mask) if stop_mask.any() else len(full_days)

        f_days = full_days[:cutoff_idx]
        f_q    = forecast_q[:cutoff_idx]
        f_cum  = cum_q[:cutoff_idx]

        # Plot
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['Days'], y=df['Qo'], mode='markers+lines', name='Actual Qo'))
        fig2.add_trace(go.Scatter(x=f_days + hist['Days'].iloc[0], y=f_q,
                                  mode='lines', name=f'{model_type} Forecast',
                                  line=dict(color='orange', dash='dash')))
        fig2.add_trace(go.Scatter(x=[0, f_days.max() + hist['Days'].iloc[0]],
                                  y=[cutoff_rate, cutoff_rate],
                                  mode='lines', name='Cut‑off',
                                  line=dict(color='red', dash='dot')))
        fig2.update_layout(title='Forecast', xaxis_title='Days', yaxis_title='Qo (m³/d)')
        st.plotly_chart(fig2, use_container_width=True)

        # Export
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
            hist.to_excel(writer, index=False, sheet_name='Historical')
            pd.DataFrame({
                'Days': f_days + hist['Days'].iloc[0],
                'Forecast Qo': f_q,
                'Cum Forecast': f_cum
            }).to_excel(writer, index=False, sheet_name='Forecast')
        st.download_button('Download Excel', out.getvalue(), file_name='forecast_output.xlsx')
