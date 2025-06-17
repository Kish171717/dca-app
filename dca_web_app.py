import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.optimize import curve_fit
import io

st.set_page_config(page_title="DCA Forecast Tool", layout="centered")
st.title("📉 Decline Curve Analysis (DCA – EUR-Sensitive)")

# ────────────────────────── widgets ──────────────────────────
uploaded = st.file_uploader("Upload Excel (Month, Oil Production (m3/d), Oil m3)", type=["xlsx"])
start_day   = st.number_input("Start Day", 0, value=0)
ignore_txt  = st.text_input("Ignore Days (e.g. 200-220, 300)")
eur_mcm     = st.number_input("EUR (million m³)", 1.0, 1e4, value=86.0, step=1.0)
cutoff      = st.number_input("Cut-off Qo (m³/d)", 0.1, 100.0, value=0.5, step=0.1)
decline_pct = st.slider("Annual Decline % (initial D)", 0.1, 100.0, 14.0, 0.1)
b_val       = st.slider("Hyperbolic b", 0.1, 1.0, 0.5, 0.01)
model_type  = st.radio("Model", ["Hyperbolic", "Exponential"])
run_btn     = st.button("🔍 Forecast")

# ───────────────────── helper functions ──────────────────────
def parse_ignore(s):
    out=set()
    for part in s.split(','):
        part=part.strip()
        if '-' in part:
            a,b=map(int,part.split('-')); out.update(range(a,b+1))
        elif part.isdigit():
            out.add(int(part))
    return list(out)

def hyperbolic(t, qi, D, b): return qi/((1+b*D*t)**(1/b))
def exponential(t, qi, D):   return qi*np.exp(-D*t)

# ─────────────────────── main pipeline ───────────────────────
if uploaded:
    try:
        df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Excel load error: {e}"); st.stop()

    must=['Month','Oil Production (m3/d)','Oil m3']
    if not all(c in df.columns for c in must):
        st.error(f"Missing columns → found {list(df.columns)}"); st.stop()

    # prep
    df['Month']=pd.to_datetime(df['Month'])
    df=df.sort_values('Month').reset_index(drop=True)
    df['Days']   =(df['Month']-df['Month'].iloc[0]).dt.days
    df['Qo']     = df['Oil Production (m3/d)']
    df['CumOil'] = df['Oil m3'].cumsum()

    st.success("✅ File loaded")
    st.plotly_chart(
        go.Figure(go.Scatter(x=df['Days'],y=df['Qo'],mode='lines+markers',name='Actual Qo'))
        .update_layout(title="Historical Production",xaxis_title='Days',yaxis_title='Qo (m³/d)'),
        use_container_width=True)

    if run_btn:
        ignore=parse_ignore(ignore_txt)
        hist=df[df['Days']>=start_day].copy()
        if ignore: hist=hist[~hist['Days'].isin(ignore)]

        # fit arrays (years)
        t=(hist['Days']-hist['Days'].iloc[0])/365.25
        q=hist['Qo'].values
        valid=(q>0)&(~np.isnan(q)); t,q=t[valid],q[valid]
        if len(q)<5: st.warning("Too few valid points"); st.stop()

        qi=np.nanmax(q[:5]); D0=max(decline_pct/100,0.01)
        try:
            if model_type=="Hyperbolic":
                popt,_=curve_fit(lambda tt,D:hyperbolic(tt,qi,D,b_val),t,q,p0=[D0],
                                 bounds=([1e-5],[1.0]),maxfev=20000)
                D=popt[0]; f=lambda yrs:hyperbolic(yrs,qi,D,b_val)
            else:
                popt,_=curve_fit(lambda tt,D:exponential(tt,qi,D),t,q,p0=[D0],
                                 bounds=([1e-5],[1.0]),maxfev=20000)
                D=popt[0]; f=lambda yrs:exponential(yrs,qi,D)
        except Exception as e:
            st.error(f"Fit failed: {e}"); st.stop()

        full_days = np.arange(0,int(100*365.25))            # 100 yr horizon
        yrs       = full_days/365.25
        fq        = f(yrs)
        cum       = np.cumsum(fq)
        EUR_lim   = eur_mcm*1e6

        stop = ((fq<cutoff)|(cum>EUR_lim)) & (yrs>t.iloc[-1])
        end  = np.argmax(stop) if stop.any() else len(full_days)

        fd = full_days[:end]; fq=fq[:end]; cum=cum[:end]
        x0 = hist['Days'].iloc[0]

        # Plot forecast
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=df['Days'],y=df['Qo'],mode='lines+markers',name='Actual Qo'))
        fig.add_trace(go.Scatter(x=fd+x0,y=fq,mode='lines',
                                 name=f'{model_type} Forecast',
                                 line=dict(color='orange',dash='dash')))
        fig.add_trace(go.Scatter(x=[0,fd[-1]+x0],y=[cutoff]*2,mode='lines',
                                 name='Cut-off',line=dict(color='red',dash='dot')))
        fig.update_layout(title='Forecast',
                          xaxis_title='Days',yaxis_title='Qo (m³/d)')
        st.plotly_chart(fig,use_container_width=True)

        # Excel out
        out_df=pd.DataFrame({'Days':fd+x0,'Forecast Qo':fq,'Cum Forecast':cum})
        buf=io.BytesIO()
        with pd.ExcelWriter(buf,engine='xlsxwriter') as w:
            hist.to_excel(w,'Historical',index=False)
            out_df.to_excel(w,'Forecast',index=False)
        st.download_button("📥 Download Excel",data=buf.getvalue(),
                           file_name='DCA_Forecast.xlsx')
