import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import io

st.set_page_config(page_title="DCA – Manual Decline", layout="centered")
st.title("📉 Decline Curve Analysis (Manual Decline %)")

# ───────── Upload file ─────────
file = st.file_uploader(
    "Upload Excel (must contain: Month, Oil Production (m3/d), Oil m3)", ["xlsx"]
)

# ───────── If file provided, show historical ─────────
if file:
    try:
        df = pd.read_excel(file)
    except Exception as e:
        st.error(f"Failed to read Excel: {e}")
        st.stop()

    required = ["Month", "Oil Production (m3/d)", "Oil m3"]
    if any(c not in df.columns for c in required):
        st.error(f"Missing columns → found {list(df.columns)}")
        st.stop()

    # Prep dataframe
    df["Month"] = pd.to_datetime(df["Month"])
    df = df.sort_values("Month").reset_index(drop=True)
    df["Days"] = (df["Month"] - df["Month"].iloc[0]).dt.days
    df["Qo"] = df["Oil Production (m3/d)"]
    df["CumOil"] = df["Oil m3"].cumsum()

    st.plotly_chart(
        go.Figure(
            go.Scatter(x=df["Days"], y=df["Qo"], mode="lines+markers", name="Actual Qo")
        ).update_layout(title="Actual Production", xaxis_title="Days", yaxis_title="Qo (m³/d)"),
        use_container_width=True,
    )

    # ───────── Forecast settings ─────────
    st.header("⚙️ Forecast settings (all user-driven, no curve fitting)")
    start_day   = st.number_input("Start Day", min_value=0, value=0)
    ignore_txt  = st.text_input("Ignore Days (e.g. 200-220, 300)")
    eur_mcm     = st.number_input("EUR (million m³)", min_value=1.0, max_value=1e4, value=86.0)
    use_cutoff  = st.checkbox("Apply cut-off Qo?", value=True)
    cutoff_qo   = st.number_input("Cut-off Qo (m³/d)", 0.1, 100.0, value=0.5, step=0.1, disabled=not use_cutoff)
    decline_pct = st.slider("Decline Rate (% per year)", 0.1, 100.0, 14.0, 0.1)
    b_val       = st.slider("Hyperbolic exponent b", 0.05, 1.0, 0.5, 0.01)
    model       = st.radio("Model type", ["Hyperbolic", "Exponential"])
    run_btn     = st.button("🔍 Forecast")

    # ───────── Helper functions ─────────
    def parse_ignore(txt):
        out = set()
        for part in txt.split(","):
            part = part.strip()
            if "-" in part:
                a, b = map(int, part.split("-"))
                out.update(range(a, b + 1))
            elif part.isdigit():
                out.add(int(part))
        return out

    def hyperbolic(t, qi, D, b):  # t in years
        return qi / ((1 + b * D * t) ** (1 / b))

    def exponential(t, qi, D):
        return qi * np.exp(-D * t)

    # ───────── Run forecast ─────────
    if run_btn:
        # Filter history
        ignore_days = parse_ignore(ignore_txt)
        hist = df[df["Days"] >= start_day].copy()
        if ignore_days:
            hist = hist[~hist["Days"].isin(ignore_days)]

        if len(hist) < 3:
            st.warning("Not enough data after filtering.")
            st.stop()

        # Convert to years axis
        t_hist_yrs = (hist["Days"] - hist["Days"].iloc[0]) / 365.25
        qi = hist["Qo"].iloc[0]       # initial rate
        D  = decline_pct / 100        # user decline fraction per year

        # Forecast horizon (100 yrs max)
        horizon_days = np.arange(0, int(100 * 365.25))
        yrs = horizon_days / 365.25

        # Forecast Qo
        if model == "Hyperbolic":
            qo_fore = hyperbolic(yrs, qi, D, b_val)
        else:
            qo_fore = exponential(yrs, qi, D)

        # Cum forecast (m³) and convert to million m³
        cum_fore_m3  = np.cumsum(qo_fore)
        cum_fore_mcm = cum_fore_m3 / 1e6

        # Determine stop condition
        stop_mask = cum_fore_mcm > eur_mcm
        if use_cutoff:
            stop_mask |= (qo_fore < cutoff_qo)
        stop_mask &= (yrs > t_hist_yrs.iloc[-1])

        stop_idx = np.argmax(stop_mask) if stop_mask.any() else len(yrs)

        # Slice forecast to stop point
        f_days = horizon_days[:stop_idx]
        f_qo   = qo_fore[:stop_idx]
        hit_reason = "Cut-off" if use_cutoff and f_qo[-1] < cutoff_qo else "EUR limit"

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Days"], y=df["Qo"], mode="lines+markers", name="Actual Qo"))
        fig.add_trace(
            go.Scatter(
                x=f_days + hist["Days"].iloc[0],
                y=f_qo,
                mode="lines",
                name=f"{model} Forecast",
                line=dict(color="orange", dash="dash"),
            )
        )
        # Vertical stop line
        fig.add_vline(x=f_days[-1] + hist["Days"].iloc[0], line_dash="dash", line_color="green")
        fig.add_annotation(
            x=f_days[-1] + hist["Days"].iloc[0],
            y=f_qo[-1],
            text=f"Stopped by {hit_reason}",
            showarrow=True,
            arrowhead=1,
            ay=-40,
        )
        # Cut-off line if enabled
        if use_cutoff:
            fig.add_trace(
                go.Scatter(
                    x=[0, f_days[-1] + hist["Days"].iloc[0]],
                    y=[cutoff_qo] * 2,
                    mode="lines",
                    name="Cut-off",
                    line=dict(color="red", dash="dot"),
                )
            )
        fig.update_layout(title="Forecast", xaxis_title="Days", yaxis_title="Qo (m³/d)")
        st.plotly_chart(fig, use_container_width=True)

        # Export to Excel
        forecast_df = pd.DataFrame(
            {
                "Days": f_days + hist["Days"].iloc[0],
                "Forecast Qo": f_qo,
                "Cum Forecast (m³)": cum_fore_m3[:stop_idx],
            }
        )
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            hist.to_excel(writer, sheet_name="Historical", index=False)
            forecast_df.to_excel(writer, sheet_name="Forecast", index=False)

        st.download_button(
            "📥 Download Excel",
            data=buffer.getvalue(),
            file_name="DCA_Forecast.xlsx",
            mime="application/vnd.ms-excel",
        )
