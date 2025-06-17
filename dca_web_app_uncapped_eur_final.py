
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.optimize import curve_fit
import io

st.set_page_config(page_title="DCA Forecast Tool", layout="centered")
st.title("📉 Decline Curve Analysis (Hyperbolic + Exponential with EUR)")

uploaded = st.file_uploader("Upload Excel (Month, Oil Production (m3/d), Oil m3)", type=["xlsx"])
start_day = st.number_input("Start Day", 0, value=0)
ignore_txt = st.text_input("Ignore Days (e.g. 200-220, 300)")
eur_mcm = st.number_input("EUR (million m³)", 1.0, 1e4, value=86.0)
cutoff = st.number_input("Cut-off Qo (m³/d)", 0.1, 100.0, value=0.5, step=0.1)
decl_pct = st.slider("Initial Decline %/yr", 0.1, 100.0, 14.0, 0.1)
b_user = st.slider("Hyperbolic b", 0.05, 1.0, 0.5, 0.01)
model = st.radio("Model", ["Hyperbolic", "Exponential"])
go_btn = st.button("🔍 Forecast")

def parse_ignore(txt):
    out = set()
    for part in txt.split(','):
        part = part.strip()
        if '-' in part:
            a, b = map(int, part.split('-'))
            out.update(range(a, b + 1))
        elif part.isdigit():
            out.add(int(part))
    return list(out)

def hyperbolic(t, qi, D, b): return qi / ((1 + b * D * t) ** (1 / b))
def exponential(t, qi, D): return qi * np.exp(-D * t)

if uploaded:
    try:
        df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Read error: {e}")
        st.stop()

    need = ['Month', 'Oil Production (m3/d)', 'Oil m3']
    if any(c not in df.columns for c in need):
        st.error(f"Missing columns → found {list(df.columns)}")
        st.stop()

    df['Month'] = pd.to_datetime(df['Month'])
    df = df.sort_values('Month').reset_index(drop=True)
    df['Days'] = (df['Month'] - df['Month'].iloc[0]).dt.days
    df['Qo'] = df['Oil Production (m3/d)']
    df['CumOil'] = df['Oil m3'].cumsum()

    st.plotly_chart(
        go.Figure(go.Scatter(x=df['Days'], y=df['Qo'], mode='markers+lines'))
        .update_layout(title="Actual Production", xaxis_title='Days', yaxis_title='Qo (m³/d)'),
        use_container_width=True)

    if go_btn:
        ig = parse_ignore(ignore_txt)
        hist = df[df['Days'] >= start_day].copy()
        if ig:
            hist = hist[~hist['Days'].isin(ig)]

        t_yrs = (hist['Days'] - hist['Days'].iloc[0]) / 365.25
        q = hist['Qo'].values
        mask = (q > 0) & ~np.isnan(q)
        t_yrs, q = t_yrs[mask], q[mask]
        if len(q) < 5:
            st.warning("Too few valid points")
            st.stop()

        qi_guess = np.nanmax(q[:5])
        D_guess = max(decl_pct / 100, 0.01)

        try:
            if model == "Hyperbolic":
                popt, _ = curve_fit(
                    hyperbolic, t_yrs, q,
                    p0=[qi_guess, D_guess, b_user],
                    bounds=([0.01, 1e-5, 0.05], [q.max() * 10, 5.0, 1.0]),
                    maxfev=60000)
                qi_fit, D_fit, b_fit = popt
                forecast_func = lambda yrs: hyperbolic(yrs, qi_fit, D_fit, b_fit)
            else:
                popt, _ = curve_fit(
                    exponential, t_yrs, q,
                    p0=[qi_guess, D_guess],
                    bounds=([0.01, 1e-5], [q.max() * 10, 5.0]),
                    maxfev=60000)
                qi_fit, D_fit = popt
                forecast_func = lambda yrs: exponential(yrs, qi_fit, D_fit)
        except Exception as e:
            st.warning(f"Fit warning → using initial guess ({e})")
            if model == "Hyperbolic":
                forecast_func = lambda yrs: hyperbolic(yrs, qi_guess, D_guess, b_user)
            else:
                forecast_func = lambda yrs: exponential(yrs, qi_guess, D_guess)

        horizon_days = np.arange(0, int(100 * 365.25))
        yrs = horizon_days / 365.25
        qo = forecast_func(yrs)
        cum = np.cumsum(qo)
        EUR = eur_mcm * 1e6

        t_np = t_yrs.to_numpy() if hasattr(t_yrs, 'to_numpy') else np.array(t_yrs)
        stop = ((qo < cutoff) | (cum > EUR)) & (yrs > t_np[-1])
        end = np.argmax(stop) if stop.any() else len(horizon_days)

        fd = horizon_days[:end]
        fq = qo[:end]
        cum = cum[:end]
        x0 = hist['Days'].iloc[0]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Days'], y=df['Qo'], mode='markers+lines', name='Actual Qo'))
        fig.add_trace(go.Scatter(x=fd + x0, y=fq, mode='lines',
                                 name=f'{model} Forecast',
                                 line=dict(color='orange', dash='dash')))
        fig.add_trace(go.Scatter(x=[0, fd[-1] + x0], y=[cutoff] * 2,
                                 mode='lines', name='Cut-off',
                                 line=dict(color='red', dash='dot')))
        fig.update_layout(title='Forecast', xaxis_title='Days', yaxis_title='Qo (m³/d)')
        st.plotly_chart(fig, use_container_width=True)

        out = pd.DataFrame({'Days': fd + x0, 'Forecast Qo': fq, 'Cum Forecast': cum})
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='xlsxwriter') as w:
            hist.to_excel(w, 'Historical', index=False)
            out.to_excel(w, 'Forecast', index=False)
        st.download_button("📥 Download Excel", data=buf.getvalue(), file_name='DCA_Forecast.xlsx')
