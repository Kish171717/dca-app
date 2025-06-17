import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.optimize import curve_fit
import io

st.set_page_config(page_title="DCA – EUR / Cut-off Control", layout="centered")
st.title("📉 Decline Curve Analysis (Hyperbolic & Exponential)")

# ───────── File upload ─────────
file = st.file_uploader("Upload Excel (must contain: Month, Oil Production (m3/d), Oil m3)", ["xlsx"])

# ───────── Load & show historical first ─────────
if file:
    try:
        df = pd.read_excel(file)
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
        go.Figure(go.Scatter(x=df['Days'], y=df['Qo'],
                             mode='markers+lines', name='Actual Qo'))
        .update_layout(title="Actual Production",
                       xaxis_title='Days', yaxis_title='Qo (m³/d)'),
        use_container_width=True)

    # ───────── Forecast settings ─────────
    st.header("⚙️ Forecast settings")
    start_day = st.number_input("Start Day", 0, value=0)
    ignore_tx = st.text_input("Ignore Days (e.g. 200-220, 300)")
    eur_mcm   = st.number_input("EUR (million m³)", 1.0, 1e4, value=86.0)
    use_cut   = st.checkbox("Apply cut-off?", value=True)
    cutoff    = st.number_input("Cut-off Qo (m³/d)", 0.1, 100.0,
                                value=0.5, step=0.1, disabled=not use_cut)
    decl_pct  = st.slider("Initial Decline %/yr", 0.1, 100.0, 14.0, 0.1)
    b_user    = st.slider("Hyperbolic b", 0.05, 1.0, 0.5, 0.01)
    model     = st.radio("Model", ["Hyperbolic", "Exponential"])
    run_btn   = st.button("🔍 Forecast")

    # ───────── helpers ─────────
    def parse_ignore(txt):
        out=set()
        for part in txt.split(','):
            part=part.strip()
            if '-' in part:
                a,b=map(int,part.split('-')); out.update(range(a,b+1))
            elif part.isdigit():
                out.add(int(part))
        return list(out)

    def hyperbolic(t, qi, D, b): return qi/((1+b*D*t)**(1/b))
    def exponential(t, qi, D):   return qi*np.exp(-D*t)

    # ───────── Forecast on click ─────────
    if run_btn:
        ignore = parse_ignore(ignore_tx)
        hist = df[df['Days'] >= start_day].copy()
        if ignore:
            hist = hist[~hist['Days'].isin(ignore)]

        t = (hist['Days'] - hist['Days'].iloc[0]) / 365.25  # yrs
        q = hist['Qo'].values
        m = (q > 0) & ~np.isnan(q)
        t, q = t[m], q[m]
        if len(q) < 5:
            st.warning("Too few valid points")
            st.stop()

        qi_guess = np.nanmax(q[:5])
        D_guess  = max(decl_pct / 100, 0.01)

        # robust fit
        try:
            if model == "Hyperbolic":
                popt,_ = curve_fit(hyperbolic, t, q,
                                   p0=[qi_guess, D_guess, b_user],
                                   bounds=([0.01, 1e-5, 0.05],
                                           [q.max()*10, 5.0, 1.0]),
                                   maxfev=60000)
                qi_fit, D_fit, b_fit = popt
                f = lambda yrs: hyperbolic(yrs, qi_fit, D_fit, b_fit)
            else:
                popt,_ = curve_fit(exponential, t, q,
                                   p0=[qi_guess, D_guess],
                                   bounds=([0.01, 1e-5],
                                           [q.max()*10, 5.0]),
                                   maxfev=60000)
                qi_fit, D_fit = popt
                f = lambda yrs: exponential(yrs, qi_fit, D_fit)
        except Exception as e:
            st.warning(f"Fit warning → using initial guess ({e})")
            if model == "Hyperbolic":
                f = lambda yrs: hyperbolic(yrs, qi_guess, D_guess, b_user)
            else:
                f = lambda yrs: exponential(yrs, qi_guess, D_guess)

        # make forecast
        horizon = np.arange(0, int(100 * 365.25))
        yrs = horizon / 365.25
        qo = f(yrs)
        cum = np.cumsum(qo)        # m³
        cum_mcm = cum / 1e6        # million m³
        EUR = eur_mcm              # slider already in million m³

        last_hist = float(t.iloc[-1])
        stop_mask = (cum_mcm > EUR)
        if use_cut:
            stop_mask |= (qo < cutoff)
        stop_mask &= (yrs > last_hist)

        end = np.argmax(stop_mask) if stop_mask.any() else len(horizon)

        fd = horizon[:end]
        fq = qo[:end]
        cum_seg = cum[:end]
        x0 = hist['Days'].iloc[0]

        # Determine reason for stop
        hit_type = "Cut-off" if use_cut and fq[-1] < cutoff else "EUR limit"

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Days'], y=df['Qo'],
                                 mode='markers+lines', name='Actual Qo'))
        fig.add_trace(go.Scatter(x=fd + x0, y=fq,
                                 mode='lines', name=f'{model} Forecast',
                                 line=dict(color='orange', dash='dash')))
        fig.add_vline(x=fd[-1] + x0, line_dash="dash", line_color="green")
        fig.add_annotation(x=fd[-1] + x0, y=cutoff,
                           text=f"Stopped by {hit_type}",
                           showarrow=True, arrowhead=1, ay=-40)
        if use_cut:
            fig.add_trace(go.Scatter(x=[0, fd[-1] + x0], y=[cutoff]*2,
                                     mode='lines', name='Cut-off',
                                     line=dict(color='red', dash='dot')))
        fig.update_layout(title='Forecast', xaxis_title='Days',
                          yaxis_title='Qo (m³/d)')
        st.plotly_chart(fig, use_container_width=True)

        # Excel export
        out = pd.DataFrame({'Days': fd + x0,
                            'Forecast Qo': fq,
                            'Cum Forecast (m³)': cum_seg})
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='xlsxwriter') as w:
            hist.to_excel(w, 'Historical', index=False)
            out.to_excel(w, 'Forecast', index=False)
        st.download_button("📥 Download Excel",
                           data=buf.getvalue(),
                           file_name='DCA_Forecast.xlsx')
