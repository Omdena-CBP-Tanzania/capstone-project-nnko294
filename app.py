import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load data and model
data = pd.read_csv(r"D:\TANZANIA_KIC\capstone-project-nnko294\data\tanzania_climate_data.csv")
data.columns = data.columns.str.strip().str.lower()
data.rename(columns={
    'average_temperature_c': 'temperature',
    'total_rainfall_mm': 'precipitation',
}, inplace=True)
model = joblib.load('random_forest_model.pkl')

# App Title
st.title("Tanzania Climate Forecast Dashboard")
st.write("This app predicts the average temperature and total rainfall in Tanzania based on historical data.")

# Section: Data Overview
st.subheader("Historical Climate Trends")

# Average temperature by year
avg_temp = data.groupby('year')['temperature'].mean()
fig_temp, ax1 = plt.subplots()
avg_temp.plot(ax=ax1, label='Avg Temperature (°C)', color='orange')
ax1.set_xlabel('Year')
ax1.set_ylabel('Temperature (°C)')
ax1.set_title('Average Temperature Over Years')
ax1.legend()
st.pyplot(fig_temp)

# Average rainfall by year
avg_rain = data.groupby('year')['precipitation'].mean()
fig_rain, ax2 = plt.subplots()
avg_rain.plot(ax=ax2, label='Avg Rainfall (mm)', color='blue')
ax2.set_xlabel('Year')
ax2.set_ylabel('Precipitation (mm)')
ax2.set_title('Average Precipitation Over Years')
ax2.legend()
st.pyplot(fig_rain)

# Section: Prediction
st.subheader("Predict Future Temperature")

# Input fields
precip = st.slider("Total Rainfall (mm)", 0, 500, 100)
year = st.slider("Year", min_value=2023, max_value=2100, value=2030)
month = st.selectbox("Month", list(range(1, 13)))
season = st.selectbox("Season", ['Dry', 'Cool', 'Hot', 'Wet'])

# Encode season (matching model input encoding)
season_encoded = [0, 0, 0]  # Order: Season_Cool, Season_Hot, Season_Wet
if season == 'Cool':
    season_encoded[0] = 1
elif season == 'Hot':
    season_encoded[1] = 1
elif season == 'Wet':
    season_encoded[2] = 1

# Input dataframe for prediction
input_df = pd.DataFrame([[year, month, precip] + season_encoded],
    columns=['year', 'month', 'precipitation', 'season_cool', 'season_hot', 'season_wet'])

# Prediction
prediction = model.predict(input_df)[0]
st.success(f"Predicted Average Temperature: **{prediction:.2f} °C**")

