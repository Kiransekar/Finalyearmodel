import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import datetime

# Backend API URL
BASE_URL = "http://localhost:8000"

# Streamlit page configuration
st.set_page_config(page_title="Shrimp Aquaculture Dashboard", layout="wide")

# Title and description
st.title("🐟 Shrimp Aquaculture Dashboard")
st.markdown("Monitor water quality, shrimp growth, and feeding schedules for your aquaculture pond.")

# Initialize session state for feedback messages
if "message" not in st.session_state:
    st.session_state.message = ""

# Function to display feedback messages
def set_message(message, type="success"):
    st.session_state.message = {"text": message, "type": type}

# Function to render feedback
def render_message():
    if st.session_state.message:
        if st.session_state.message["type"] == "success":
            st.success(st.session_state.message["text"])
        else:
            st.error(st.session_state.message["text"])

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select a page", ["Farmer Data", "Sensor Data", "Analysis", "Reset Data"])

# Farmer Data Submission
if page == "Farmer Data":
    st.header("Submit Farmer Data")
    with st.form("farmer_data_form"):
        pond_area = st.number_input("Pond Area (m²)", min_value=100.0, value=1000.0, step=100.0)
        pond_depth = st.number_input("Pond Depth (m)", min_value=0.5, value=1.5, step=0.1)
        stocking_density = st.number_input("Stocking Density (shrimp/m²)", min_value=10.0, value=50.0, step=5.0)
        culture_start_date = st.date_input("Culture Start Date", value=datetime.date(2025, 1, 1))
        lat = st.number_input("Latitude", value=10.0, step=0.1)
        lon = st.number_input("Longitude", value=105.0, step=0.1)
        submit_button = st.form_submit_button("Submit Farmer Data")

        if submit_button:
            farmer_data = {
                "pond_area": pond_area,
                "pond_depth": pond_depth,
                "stocking_density": stocking_density,
                "culture_start_date": culture_start_date.strftime("%Y-%m-%dT00:00:00"),
                "location": {"lat": lat, "lon": lon}
            }
            try:
                response = requests.post(f"{BASE_URL}/set_farmer_data", json=farmer_data)
                response.raise_for_status()
                set_message("Farmer data submitted successfully!")
            except requests.RequestException as e:
                set_message(f"Error submitting farmer data: {str(e)}", type="error")
    
    render_message()

# Sensor Data Submission
elif page == "Sensor Data":
    st.header("Submit Sensor Data")
    with st.form("sensor_data_form"):
        timestamp = st.date_input("Date", value=datetime.date(2025, 1, 1))
        time = st.time_input("Time", value=datetime.time(8, 0))
        pH = st.number_input("pH (0–14)", min_value=0.0, max_value=14.0, value=7.5, step=0.1)
        temperature = st.number_input("Temperature (°C, 10–40)", min_value=10.0, max_value=40.0, value=28.0, step=0.1)
        turbidity = st.number_input("Turbidity (NTU, ≥0)", min_value=0.0, value=10.0, step=1.0)
        tds = st.number_input("TDS (ppm, ≥0)", min_value=0.0, value=500.0, step=10.0)
        submit_button = st.form_submit_button("Submit Sensor Data")

        if submit_button:
            timestamp_str = datetime.datetime.combine(timestamp, time).strftime("%Y-%m-%dT%H:%M:%S")
            sensor_data = {
                "timestamp": timestamp_str,
                "pH": pH,
                "temperature": temperature,
                "turbidity": turbidity,
                "tds": tds
            }
            try:
                response = requests.post(f"{BASE_URL}/submit_sensor_data", json=sensor_data)
                response.raise_for_status()
                set_message("Sensor data submitted successfully!")
            except requests.RequestException as e:
                set_message(f"Error submitting sensor data: {str(e)}", type="error")
    
    render_message()

# Analysis Dashboard
elif page == "Analysis":
    st.header("Analysis Dashboard")
    try:
        response = requests.get(f"{BASE_URL}/analyze")
        response.raise_for_status()
        data = response.json()

        # Water Quality Metrics
        st.subheader("Water Quality Metrics")
        water_quality = pd.DataFrame([data["water_quality"]])
        st.dataframe(water_quality, use_container_width=True)

        # Anomalies
        st.subheader("Detected Anomalies")
        if data["anomalies"]:
            anomalies = pd.DataFrame(data["anomalies"])
            st.dataframe(anomalies, use_container_width=True)
        else:
            st.write("No anomalies detected.")

        # Shrimp Metrics
        st.subheader("Shrimp Metrics")
        metrics = {
            "Shrimp Weight (g)": data["shrimp_weight"],
            "Growth Rate (g/day)": data["growth_rate"],
            "Predicted Yield (kg)": data["predicted_yield_kg"],
            "FCR": data["fcr"]
        }
        metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
        st.dataframe(metrics_df, use_container_width=True)

        # Feed Schedule
        st.subheader("Feed Schedule (kg)")
        feed_schedule = pd.DataFrame([data["feed_schedule"]])
        st.dataframe(feed_schedule, use_container_width=True)

        # Time-Series Analysis
        st.subheader("Time-Series Analysis")
        try:
            response = requests.get(f"{BASE_URL}/get_sensor_data")
            response.raise_for_status()
            sensor_data = response.json()["sensor_data"]
            df = pd.DataFrame(sensor_data)
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                fig = px.line(df, x="timestamp", y=["pH", "temperature", "turbidity", "tds"],
                              title="Sensor Data Over Time")
                st.plotly_chart(fig, use_container_width=True)
                # Plot shrimp weight if available
                if "shrimp_weight" in df.columns:
                    fig_weight = px.line(df, x="timestamp", y="shrimp_weight", title="Shrimp Weight Over Time")
                    st.plotly_chart(fig_weight, use_container_width=True)
            else:
                st.write("No sensor data available for plotting.")
        except requests.RequestException as e:
            st.write(f"Error fetching sensor data: {str(e)}")

    except requests.RequestException as e:
        set_message(f"Error fetching analysis: {str(e)}", type="error")
    
    render_message()

# Reset Data
elif page == "Reset Data":
    st.header("Reset Data")
    if st.button("Reset All Data"):
        try:
            response = requests.post(f"{BASE_URL}/reset")
            response.raise_for_status()
            set_message("All data reset successfully!")
        except requests.RequestException as e:
            set_message(f"Error resetting data: {str(e)}", type="error")
    
    render_message()