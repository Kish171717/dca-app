
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.optimize import curve_fit
import io

st.set_page_config(page_title="DCA Forecast Tool", layout="centered")
st.title("📉 Decline Curve Analysis (DCA) Forecast Tool – EUR‑Sensitive")

st.markdown("Upload an Excel file with these columns:")
st.markdown("• **Month** (date)  
• **Oil Production (m3/d)**  
• **Oil m3** (cumulative)")

uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])

# --- Widgets for forecast parameters ---
start_day = st.number_input("Start Day", min_value=0, value=0)
ignore_text = st.text_input("Ignore Days (e.g. 200-220, 300)")
eur_mcm = st.number_input("EUR (million m³)", value=86.0, step=1.0)
cutoff = st.number_input("Cut‑off Rate (m³/d)", value=0.5, step=0.1)
decline_pct = st.slider("Annual Decline Rate %", 0.1, 100.0, 14.0, 0.1)
b_val = st.slider("Hyperbolic exponent b", 0.1, 1.0, 0.5, 0.01)
model_type = st.radio("Model", ["Hyperbolic", "Exponential"])
forecast_button = st.button("🔍 Forecast")

def parse_ignore(s):
    out=set()
    for part in s.split(','):
        part=part.strip()
        if '-' in part:
            a,b=map(int,part.split('-'))
            out.update(range(a,b+1))
        elif part.isdigit():
            out.add(int(part))
    return list(out)

def hyperbolic(t, qi, D, b):
    return qi/((1+b*D*t)**(1/b))

def exponential(t, qi, D):
    return qi*np.exp(-D*t)

if uploaded_file:
    df=pd.read_excel(uploaded_file)
    df['Month']=pd.to_datetime(df['Month'])
    df=df.sort_values('Month').reset_index(drop=True)
    df['Days']=(df['Month']-df['Month'].iloc[0]).dt.days
    df['Qo (m3/d)']=df['Oil Production (m3/d)']
    df['CumOil']=df['Oil m3'].cumsum()

    st.success("File loaded!")
    st.plotly_chart(go.Figure(data=go.Scatter(x=df['Days'],y=df['Qo (m3/d)'],mode='markers+lines',name='Actual Qo')),use_container_width=True)

    if forecast_button:
        ignores=parse_ignore(ignore_text)
        data=df[df['Days']>=start_day].copy()
        if ignores:
            data=data[~data['Days'].isin(ignores)]

        t_years=(data['Days']-data['Days'].iloc[0])/365.25
        q=data['Qo (m3/d)'].values
        qi=q[0]
        D_year=decline_pct/100

        try:
            if model_type=="Hyperbolic":
                popt,_=curve_fit(lambda t,D:hyperbolic(t,qi,D,b_val),t_years,q,p0=[D_year],bounds=([1e-5],[1.0]),maxfev=10000)
                f=lambda yrs: hyperbolic(yrs,qi,popt[0],b_val)
            else:
                popt,_=curve_fit(lambda t,D:exponential(t,qi,D),t_years,q,p0=[D_year],bounds=([1e-5],[1.0]),maxfev=10000)
                f=lambda yrs: exponential(yrs,qi,popt[0])
        except Exception as e:
            st.error(f"Curve fit failed: {e}")
            st.stop()

        # generate long horizon (100 years)
        full_days=np.arange(0,int(100*365.25))
        yrs=full_days/365.25
        forecast=f(yrs)
        cum=np.cumsum(forecast)
        eur_lim=eur_mcm*1e6

        stop_idx=len(full_days)
        mask=(forecast<cutoff)|(cum>eur_lim)
        if mask.any():
            stop_idx=np.argmax(mask)

        forecast_days=full_days[:stop_idx]
        forecast_vals=forecast[:stop_idx]
        cum_vals=cum[:stop_idx]

        fig=go.Figure()
        fig.add_trace(go.Scatter(x=df['Days'],y=df['Qo (m3/d)'],mode='markers+lines',name='Actual Qo'))
        fig.add_trace(go.Scatter(x=forecast_days+data['Days'].iloc[0],y=forecast_vals,mode='lines',name=f'{model_type} Forecast',line=dict(color='orange',dash='dash')))
        fig.add_trace(go.Scatter(x=[0,forecast_days[-1]+data['Days'].iloc[0]],y=[cutoff]*2,mode='lines',name='Cut‑off',line=dict(color='red',dash='dot')))
        fig.update_layout(title='Forecast',xaxis_title='Days',yaxis_title='Qo (m3/d)')
        st.plotly_chart(fig,use_container_width=True)

        # Excel export
        df_export=pd.DataFrame({
            'Days':forecast_days+data['Days'].iloc[0],
            'Forecast Qo':forecast_vals,
            'Cum Forecast':cum_vals
        })
        out=io.BytesIO()
        with pd.ExcelWriter(out,engine='xlsxwriter') as writer:
            data.to_excel(writer,index=False,sheet_name='Historical')
            df_export.to_excel(writer,index=False,sheet_name='Forecast')
        st.download_button("Download Excel",data=out.getvalue(),file_name="forecast_output.xlsx")
