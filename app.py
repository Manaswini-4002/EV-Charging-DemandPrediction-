import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="EV Adoption Forecaster", layout="wide")

# === LOAD MODEL ===
model = joblib.load('forecasting_ev_model.pkl')

# === SIDEBAR ===
st.sidebar.title("🔌 EV Forecasting App")
st.sidebar.markdown("""
This tool forecasts the growth of electric vehicle (EV) adoption for counties in Washington State using historical data and machine learning.
""")
st.sidebar.markdown("---")
st.sidebar.markdown("🔗 [GitHub Repository](https://github.com/RGS-AI/AICTE_Internships)")

# === MAIN TITLE ===
st.markdown("<h1 style='text-align: center; color: #1e90ff;'>🔮 EV Adoption Forecaster</h1>", unsafe_allow_html=True)

# === LOAD DATA ===
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()
county_list = sorted(df['County'].dropna().unique().tolist())

# === USER INPUT ===
county = st.selectbox("📍 Select a County", county_list)
if county not in df['County'].unique():
    st.warning(f"County '{county}' not found in dataset.")
    st.stop()

with st.spinner("⏳ Forecasting... Please wait..."):
    county_df = df[df['County'] == county].sort_values("Date")
    county_code = county_df['county_encoded'].iloc[0]
    historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
    cumulative_ev = list(np.cumsum(historical_ev))
    months_since_start = county_df['months_since_start'].max()
    latest_date = county_df['Date'].max()
    
    future_rows = []
    forecast_horizon = 36
    
    for i in range(1, forecast_horizon + 1):
        forecast_date = latest_date + pd.DateOffset(months=i)
        months_since_start += 1
        lag1, lag2, lag3 = historical_ev[-1], historical_ev[-2], historical_ev[-3]
        roll_mean = np.mean([lag1, lag2, lag3])
        pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
        pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
        ev_growth_slope = np.polyfit(range(6), cumulative_ev[-6:], 1)[0] if len(cumulative_ev) >= 6 else 0

        new_row = {
            'months_since_start': months_since_start,
            'county_encoded': county_code,
            'ev_total_lag1': lag1,
            'ev_total_lag2': lag2,
            'ev_total_lag3': lag3,
            'ev_total_roll_mean_3': roll_mean,
            'ev_total_pct_change_1': pct_change_1,
            'ev_total_pct_change_3': pct_change_3,
            'ev_growth_slope': ev_growth_slope
        }

        pred = model.predict(pd.DataFrame([new_row]))[0]
        future_rows.append({"Date": forecast_date, "Predicted EV Total": round(pred)})

        historical_ev.append(pred)
        if len(historical_ev) > 6:
            historical_ev.pop(0)
        cumulative_ev.append(cumulative_ev[-1] + pred)
        if len(cumulative_ev) > 6:
            cumulative_ev.pop(0)

# === CUMULATIVE GRAPH ===
historical_cum = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
historical_cum['Source'] = 'Historical'
historical_cum['Cumulative EV'] = historical_cum['Electric Vehicle (EV) Total'].cumsum()

forecast_df = pd.DataFrame(future_rows)
forecast_df['Source'] = 'Forecast'
forecast_df['Cumulative EV'] = forecast_df['Predicted EV Total'].cumsum() + historical_cum['Cumulative EV'].iloc[-1]

combined = pd.concat([
    historical_cum[['Date', 'Cumulative EV', 'Source']],
    forecast_df[['Date', 'Cumulative EV', 'Source']]
], ignore_index=True)

# === METRICS AT TOP ===
total_forecasted = int(forecast_df['Cumulative EV'].iloc[-1])
historical_total = int(historical_cum['Cumulative EV'].iloc[-1])
growth_pct = ((total_forecasted - historical_total) / historical_total) * 100 if historical_total > 0 else 0

col1, col2 = st.columns(2)
col1.metric("🔋 Total Forecasted EVs", f"{total_forecasted:,}")
col2.metric("📈 Growth Over 3 Years", f"{growth_pct:.2f} %")

# === PLOT CHART ===
st.subheader(f"📊 Cumulative EV Forecast – {county} County")
fig, ax = plt.subplots(figsize=(12, 6))
for label, data in combined.groupby('Source'):
    ax.plot(data['Date'], data['Cumulative EV'], label=label, marker='o')
ax.set_title("Cumulative EV Growth (Historical + Forecast)")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative EVs")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# === CONCLUSION MESSAGE ===
trend = "increase 📈" if growth_pct > 0 else "decrease 📉"
st.success(f"Forecast shows a **{trend} of {growth_pct:.2f}%** in EVs in **{county}** County over the next 3 years.")

# === FOOTER ===
st.markdown("---")
st.markdown("© 2025 Manaswini Ramavath | AICTE Shell–Edunet Internship")
