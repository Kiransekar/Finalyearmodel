import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time
import json
from typing import Dict, Any, List, Union

# Backend API URL - configurable for different environments
BASE_URL = st.sidebar.text_input("API URL", value="http://localhost:8000")


# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
    .subheader {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 7px;
        padding: 10px;
        margin: 5px 0;
    }
    .success-text {
        color: #4CAF50;
        font-weight: bold;
    }
    .warning-text {
        color: #FF9800;
        font-weight: bold;
    }
    .error-text {
        color: #F44336;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "message" not in st.session_state:
    st.session_state.message = {"text": "", "type": ""}

if "sensor_data_history" not in st.session_state:
    st.session_state.sensor_data_history = []

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = None

if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False

if "chart_type" not in st.session_state:
    st.session_state.chart_type = "line"

# Helper Functions
def set_message(message: str, message_type: str = "success"):
    """Set a message to display to the user."""
    st.session_state.message = {"text": message, "type": message_type}

def render_message():
    """Render any feedback messages."""
    if st.session_state.message["text"]:
        if st.session_state.message["type"] == "success":
            st.success(st.session_state.message["text"])
        elif st.session_state.message["type"] == "warning":
            st.warning(st.session_state.message["text"])
        else:
            st.error(st.session_state.message["text"])

def api_request(method: str, endpoint: str, data: Dict = None) -> Dict:
    """Make a request to the API with error handling."""
    url = f"{BASE_URL}{endpoint}"
    try:
        if method.lower() == "get":
            response = requests.get(url, timeout=10)
        elif method.lower() == "post":
            response = requests.post(url, json=data, timeout=10)
        else:
            return {"error": f"Unsupported method: {method}"}
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        error_message = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json().get('detail', str(e))
                error_message = f"{error_detail}"
            except:
                pass
        return {"error": error_message}

def fetch_sensor_data() -> pd.DataFrame:
    """Fetch sensor data from the API and convert to DataFrame."""
    response = api_request("get", "/get_sensor_data")
    if "error" in response:
        set_message(f"Error fetching sensor data: {response['error']}", "error")
        return pd.DataFrame()
    
    if not response.get("sensor_data", []):
        return pd.DataFrame()
    
    df = pd.DataFrame(response["sensor_data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def fetch_and_process_data(refresh: bool = False) -> Dict[str, Any]:
    """Fetch and process all necessary data."""
    # Update last refresh time
    if refresh or st.session_state.last_refresh is None:
        st.session_state.last_refresh = datetime.datetime.now()
    
    data = {}
    
    # Fetch sensor data
    sensor_df = fetch_sensor_data()
    data["sensor_df"] = sensor_df
    
    # Fetch analysis if sensor data exists
    if not sensor_df.empty:
        analysis = api_request("get", "/analyze")
        if "error" in analysis:
            set_message(f"Error fetching analysis: {analysis['error']}", "error")
        else:
            data["analysis"] = analysis
    
    return data

def create_time_series_chart(df: pd.DataFrame, params: List[str], title: str = "Sensor Data Over Time"):
    """Create a time series chart for the specified parameters."""
    if df.empty or "timestamp" not in df.columns:
        return go.Figure().update_layout(title=f"{title} - No Data Available")
    
    available_params = [p for p in params if p in df.columns]
    if not available_params:
        return go.Figure().update_layout(title=f"{title} - No Parameters Available")
    
    if st.session_state.chart_type == "line":
        fig = px.line(df, x="timestamp", y=available_params, title=title)
    else:
        # Create a figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add traces for each parameter
        for i, param in enumerate(available_params):
            fig.add_trace(
                go.Scatter(x=df["timestamp"], y=df[param], name=param),
                secondary_y=(i > 0)  # First parameter on primary axis, others on secondary
            )
        
        # Update layout
        fig.update_layout(title=title)
        fig.update_xaxes(title_text="Timestamp")
        fig.update_yaxes(title_text=available_params[0], secondary_y=False)
        if len(available_params) > 1:
            fig.update_yaxes(title_text=" & ".join(available_params[1:]), secondary_y=True)
    
    # Improve layout
    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=True,
        ),
        yaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=40, b=40),
    )
    
    return fig

def evaluate_water_quality(water_quality: Dict[str, float]) -> Dict[str, str]:
    """Evaluate water quality parameters and return status."""
    status = {}
    
    # pH evaluation
    if 7.0 <= water_quality.get("pH", 0) <= 8.5:
        status["pH"] = "optimal"
    elif 6.5 <= water_quality.get("pH", 0) < 7.0 or 8.5 < water_quality.get("pH", 0) <= 9.0:
        status["pH"] = "acceptable"
    else:
        status["pH"] = "critical"
    
    # Dissolved oxygen evaluation
    if water_quality.get("do", 0) >= 5.0:
        status["do"] = "optimal"
    elif 4.0 <= water_quality.get("do", 0) < 5.0:
        status["do"] = "acceptable"
    else:
        status["do"] = "critical"
    
    # Temperature evaluation
    if 26.0 <= water_quality.get("temperature", 0) <= 30.0:
        status["temperature"] = "optimal"
    elif 22.0 <= water_quality.get("temperature", 0) < 26.0 or 30.0 < water_quality.get("temperature", 0) <= 32.0:
        status["temperature"] = "acceptable"
    else:
        status["temperature"] = "critical"
    
    # Ammonia evaluation
    if water_quality.get("ammonia", 0) <= 0.1:
        status["ammonia"] = "optimal"
    elif 0.1 < water_quality.get("ammonia", 0) <= 0.3:
        status["ammonia"] = "acceptable"
    else:
        status["ammonia"] = "critical"
    
    # Nitrite evaluation
    if water_quality.get("nitrite", 0) <= 0.1:
        status["nitrite"] = "optimal"
    elif 0.1 < water_quality.get("nitrite", 0) <= 0.5:
        status["nitrite"] = "acceptable"
    else:
        status["nitrite"] = "critical"
    
    return status

# Title and Description
st.markdown("<h1 class='main-header'>🦐 Shrimp Aquaculture Monitoring System</h1>", unsafe_allow_html=True)
st.markdown("""
This dashboard helps shrimp farmers monitor water quality, track shrimp growth, 
detect anomalies, and optimize feeding schedules for maximum yield.
""")

# Sidebar navigation
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Dashboard", "Farmer Data", "Sensor Data", "Analysis", "Settings", "Reset Data"]
)

# Refresh data options in sidebar
st.sidebar.markdown("## Data Refresh")
refresh_col1, refresh_col2 = st.sidebar.columns([3, 1])
with refresh_col1:
    st.session_state.auto_refresh = st.checkbox("Auto Refresh", value=st.session_state.auto_refresh)
with refresh_col2:
    if st.button("🔄️"):
        set_message("Data refreshed", "success")

# Auto refresh logic
if st.session_state.auto_refresh:
    if st.session_state.last_refresh is None or (datetime.datetime.now() - st.session_state.last_refresh).seconds > 30:
        st.session_state.last_refresh = datetime.datetime.now()
        set_message("Data auto refreshed", "info")

# Fetch and process data
data = fetch_and_process_data()
sensor_df = data.get("sensor_df", pd.DataFrame())
analysis = data.get("analysis", {})

# Display last refresh time
if st.session_state.last_refresh:
    st.sidebar.markdown(f"Last refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}")

# Render any messages
render_message()

# Dashboard Page
if page == "Dashboard":
    st.markdown("<h2 class='subheader'>Dashboard Overview</h2>", unsafe_allow_html=True)
    
    if sensor_df.empty:
        st.info("No sensor data available. Please submit sensor data to see the dashboard.")
    else:
        # Display water quality metrics in columns
        if analysis and "water_quality" in analysis:
            water_quality = analysis["water_quality"]
            status = evaluate_water_quality(water_quality)
            
            st.markdown("<h3>Current Water Quality</h3>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.metric("pH", f"{water_quality.get('pH', 'N/A')}")
                status_class = "success-text" if status.get("pH") == "optimal" else "warning-text" if status.get("pH") == "acceptable" else "error-text"
                st.markdown(f"<p class='{status_class}'>{status.get('pH', 'unknown').capitalize()}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col2:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.metric("Temperature (°C)", f"{water_quality.get('temperature', 'N/A')}")
                status_class = "success-text" if status.get("temperature") == "optimal" else "warning-text" if status.get("temperature") == "acceptable" else "error-text"
                st.markdown(f"<p class='{status_class}'>{status.get('temperature', 'unknown').capitalize()}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col3:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.metric("Dissolved Oxygen (mg/L)", f"{water_quality.get('do', 'N/A')}")
                status_class = "success-text" if status.get("do") == "optimal" else "warning-text" if status.get("do") == "acceptable" else "error-text"
                st.markdown(f"<p class='{status_class}'>{status.get('do', 'unknown').capitalize()}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col4:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.metric("Ammonia (mg/L)", f"{water_quality.get('ammonia', 'N/A')}")
                status_class = "success-text" if status.get("ammonia") == "optimal" else "warning-text" if status.get("ammonia") == "acceptable" else "error-text"
                st.markdown(f"<p class='{status_class}'>{status.get('ammonia', 'unknown').capitalize()}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Recent time series chart
            st.markdown("<h3>Recent Measurements</h3>", unsafe_allow_html=True)
            chart_type = st.radio("Chart Type", ["Line Chart", "Dual Y-Axis"], horizontal=True)
            st.session_state.chart_type = "line" if chart_type == "Line Chart" else "dual"
            
            # Show different parameter groups
            tab1, tab2, tab3 = st.tabs(["Primary Parameters", "Chemical Parameters", "Physical Parameters"])
            
            with tab1:
                fig = create_time_series_chart(
                    sensor_df, 
                    ["pH", "temperature", "do", "ammonia"], 
                    "Primary Water Quality Parameters"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                fig = create_time_series_chart(
                    sensor_df, 
                    ["ammonia", "nitrite", "alkalinity", "salinity"], 
                    "Chemical Parameters"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                fig = create_time_series_chart(
                    sensor_df, 
                    ["temperature", "turbidity", "tds"], 
                    "Physical Parameters"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Shrimp growth and predictions
            if "shrimp_weight" in analysis and "growth_rate" in analysis:
                st.markdown("<h3>Shrimp Growth & Predictions</h3>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    shrimp_weight = analysis.get("shrimp_weight", 0)
                    st.metric("Current Weight (g/shrimp)", f"{shrimp_weight}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    growth_rate = analysis.get("growth_rate", 0)
                    st.metric("Growth Rate (g/day)", f"{growth_rate}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    predicted_yield = analysis.get("predicted_yield_kg", 0)
                    st.metric("Predicted Yield (kg)", f"{predicted_yield}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Show growth trend if applicable
                if "shrimp_weight" in sensor_df.columns:
                    fig = px.line(
                        sensor_df, 
                        x="timestamp", 
                        y="shrimp_weight", 
                        title="Shrimp Weight Trend"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Feeding schedule
            if "feed_schedule" in analysis:
                st.markdown("<h3>Today's Feeding Schedule</h3>", unsafe_allow_html=True)
                feed = analysis["feed_schedule"]
                feed_df = pd.DataFrame({
                    "Time": ["Morning", "Afternoon", "Evening"],
                    "Amount (kg)": [feed["morning"], feed["afternoon"], feed["evening"]]
                })
                
                fig = px.bar(
                    feed_df, 
                    x="Time", 
                    y="Amount (kg)", 
                    title="Feed Distribution",
                    color="Time",
                    color_discrete_sequence=["#1E88E5", "#42A5F5", "#90CAF9"]
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Anomalies
            if "anomalies" in analysis and analysis["anomalies"]:
                st.markdown("<h3>⚠️ Detected Anomalies</h3>", unsafe_allow_html=True)
                anomalies_df = pd.DataFrame(analysis["anomalies"])
                st.dataframe(anomalies_df, use_container_width=True)
            
            # FCR
            if "fcr" in analysis:
                st.markdown("<h3>Feed Conversion Ratio (FCR)</h3>", unsafe_allow_html=True)
                fcr = analysis.get("fcr", 0)
                if fcr <= 1.3:
                    st.success(f"Current FCR: {fcr} (Excellent)")
                elif fcr <= 1.6:
                    st.info(f"Current FCR: {fcr} (Good)")
                elif fcr <= 1.9:
                    st.warning(f"Current FCR: {fcr} (Needs Improvement)")
                else:
                    st.error(f"Current FCR: {fcr} (Poor - Check Feeding Strategy)")

# Farmer Data Page
elif page == "Farmer Data":
    st.markdown("<h2 class='subheader'>Farmer Data Submission</h2>", unsafe_allow_html=True)
    st.markdown("Enter details about your shrimp pond to enable accurate analysis.")
    
    with st.form("farmer_data_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            pond_area = st.number_input("Pond Area (m²)", min_value=100.0, value=1000.0, step=100.0)
            pond_depth = st.number_input("Pond Depth (m)", min_value=0.5, value=1.5, step=0.1)
            stocking_density = st.number_input("Stocking Density (shrimp/m²)", min_value=10.0, value=50.0, step=5.0)
        
        with col2:
            culture_start_date = st.date_input("Culture Start Date", value=datetime.date(2025, 1, 1))
            lat = st.number_input("Latitude", value=10.0, step=0.01)
            lon = st.number_input("Longitude", value=105.0, step=0.01)
        
        submit_button = st.form_submit_button("Submit Farmer Data")
        
        if submit_button:
            farmer_data = {
                "pond_area": pond_area,
                "pond_depth": pond_depth,
                "stocking_density": stocking_density,
                "culture_start_date": f"{culture_start_date.strftime('%Y-%m-%d')}T00:00:00",
                "location": {
                    "lat": lat,
                    "lon": lon
                }
            }
            
            response = api_request("post", "/set_farmer_data", farmer_data)
            if "error" in response:
                set_message(f"Error setting farmer data: {response['error']}", "error")
            else:
                set_message("Farmer data submitted successfully!", "success")

# Sensor Data Page
elif page == "Sensor Data":
    st.markdown("<h2 class='subheader'>Sensor Data Submission</h2>", unsafe_allow_html=True)
    st.markdown("Submit water quality sensor data for analysis.")
    
    with st.form("sensor_data_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            date = st.date_input("Date", value=datetime.date.today())
            time_input = st.time_input("Time", value=datetime.time(8, 0))
            pH = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.5, step=0.1)
        
        with col2:
            temperature = st.number_input("Temperature (°C)", min_value=10.0, max_value=40.0, value=28.0, step=0.1)
            turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=15.0, step=1.0)
        
        with col3:
            tds = st.number_input("Total Dissolved Solids (ppm)", min_value=0.0, value=500.0, step=10.0)
        
        submit_button = st.form_submit_button("Submit Sensor Data")
        
        if submit_button:
            timestamp = datetime.datetime.combine(date, time_input)
            sensor_data = {
                "timestamp": timestamp.isoformat(),
                "pH": pH,
                "temperature": temperature,
                "turbidity": turbidity,
                "tds": tds
            }
            
            response = api_request("post", "/submit_sensor_data", sensor_data)
            if "error" in response:
                set_message(f"Error submitting sensor data: {response['error']}", "error")
            else:
                set_message("Sensor data submitted successfully!", "success")
    
    # Display existing sensor data
    st.markdown("<h2 class='subheader'>Existing Sensor Data</h2>", unsafe_allow_html=True)
    
    if not sensor_df.empty:
        st.dataframe(sensor_df, use_container_width=True)
    else:
        st.info("No sensor data available. Please submit data using the form above.")

# Analysis Page
elif page == "Analysis":
    st.markdown("<h2 class='subheader'>Detailed Analysis</h2>", unsafe_allow_html=True)
    
    if not sensor_df.empty and analysis:
        # Water Quality Analysis
        st.markdown("<h3>Water Quality Analysis</h3>", unsafe_allow_html=True)
        water_quality = analysis.get("water_quality", {})
        
        if water_quality:
            # Convert to DataFrame for better display
            wq_df = pd.DataFrame({
                "Parameter": list(water_quality.keys()),
                "Value": list(water_quality.values())
            })
            
            # Add optimal ranges
            optimal_ranges = {
                "pH": "7.0 - 8.5",
                "temperature": "26.0 - 30.0",
                "turbidity": "< 30",
                "tds": "10,000 - 20,000",
                "salinity": "10 - 20",
                "ammonia": "< 0.1",
                "do": "> 5.0",
                "nitrite": "< 0.1",
                "alkalinity": "100 - 150"
            }
            
            wq_df["Optimal Range"] = wq_df["Parameter"].map(optimal_ranges)
            st.dataframe(wq_df, use_container_width=True)
            
            # Radar chart for water quality
            categories = ['pH', 'temperature', 'do', 'ammonia', 'nitrite']
            
            # Normalize values for radar chart
            normalized = {}
            norms = {
                'pH': {'min': 6.5, 'max': 9.0, 'optimal': 7.8, 'better': 'middle'},
                'temperature': {'min': 22, 'max': 32, 'optimal': 28, 'better': 'middle'},
                'do': {'min': 3, 'max': 8, 'optimal': 6, 'better': 'high'},
                'ammonia': {'min': 0, 'max': 1, 'optimal': 0, 'better': 'low'},
                'nitrite': {'min': 0, 'max': 1, 'optimal': 0, 'better': 'low'}
            }
            
            for cat in categories:
                if cat in water_quality:
                    val = water_quality[cat]
                    norm = norms[cat]
                    
                    if norm['better'] == 'middle':
                        # For parameters where middle is best
                        dist_from_optimal = abs(val - norm['optimal'])
                        max_dist = max(norm['optimal'] - norm['min'], norm['max'] - norm['optimal'])
                        normalized[cat] = 1 - (dist_from_optimal / max_dist)
                    elif norm['better'] == 'high':
                        # For parameters where higher is better
                        normalized[cat] = (val - norm['min']) / (norm['max'] - norm['min'])
                    else:
                        # For parameters where lower is better
                        normalized[cat] = 1 - ((val - norm['min']) / (norm['max'] - norm['min']))
                    
                    # Clamp between 0 and 1
                    normalized[cat] = max(0, min(1, normalized[cat]))
            
            # Create radar chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=[normalized.get(cat, 0) for cat in categories],
                theta=categories,
                fill='toself',
                name='Current Values'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title="Water Quality Radar (Normalized)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Growth Analysis
        st.markdown("<h3>Growth Analysis</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Weight (g)", analysis.get("shrimp_weight", "N/A"))
        with col2:
            st.metric("Growth Rate (g/day)", analysis.get("growth_rate", "N/A"))
        with col3:
            st.metric("Feed Conversion Ratio", analysis.get("fcr", "N/A"))
        
        # Prediction Analysis
        st.markdown("<h3>Predictions & Recommendations</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Yield (kg)", analysis.get("predicted_yield_kg", "N/A"))
            
            if "feed_schedule" in analysis:
                feed = analysis["feed_schedule"]
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.markdown("**Recommended Feed Schedule (kg)**")
                st.markdown(f"Morning: {feed.get('morning', 'N/A')} kg")
                st.markdown(f"Afternoon: {feed.get('afternoon', 'N/A')} kg")
                st.markdown(f"Evening: {feed.get('evening', 'N/A')} kg")
                st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            # Generate recommendations based on water quality
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.markdown("**Recommendations**")
            
            issues = []
            water_quality = analysis.get("water_quality", {})
            
            if water_quality.get("pH", 7) < 6.5:
                issues.append("Low pH: Consider adding agricultural lime to increase pH")
            elif water_quality.get("pH", 7) > 9.0:
                issues.append("High pH: Reduce feeding and consider partial water exchange")
            
            if water_quality.get("do", 5) < 4.0:
                issues.append("Low dissolved oxygen: Increase aeration immediately")
            
            if water_quality.get("ammonia", 0) > 0.3:
                issues.append("High ammonia: Reduce feeding and increase water exchange")
            
            if water_quality.get("nitrite", 0) > 0.5:
                issues.append("High nitrite: Check biofilter function and increase water exchange")
            
            if not issues:
                st.markdown("✅ All parameters within acceptable ranges. Maintain current management practices.")
            else:
                for issue in issues:
                    st.markdown(f"⚠️ {issue}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Anomaly Detection
        st.markdown("<h3>Anomaly Detection</h3>", unsafe_allow_html=True)
        
        if "anomalies" in analysis and analysis["anomalies"]:
            anomalies_df = pd.DataFrame(analysis["anomalies"])
            if len(anomalies_df) > 0:
                st.warning(f"{len(anomalies_df)} anomalies detected in your data")
                st.dataframe(anomalies_df, use_container_width=True)
        else:
            st.success("No anomalies detected in your data")
    else:
        st.info("No data available for analysis. Please submit sensor data first.")

# Settings Page
elif page == "Settings":
    st.markdown("<h2 class='subheader'>Dashboard Settings</h2>", unsafe_allow_html=True)
    
    # Chart settings
    st.markdown("<h3>Chart Settings</h3>", unsafe_allow_html=True)
    
    chart_type = st.radio(
        "Default Chart Type",
        ["Line Chart", "Dual Y-Axis"],
        index=0 if st.session_state.chart_type == "line" else 1
    )
    st.session_state.chart_type = "line" if chart_type == "Line Chart" else "dual"
    
    # Notification settings
    st.markdown("<h3>Notification Settings</h3>", unsafe_allow_html=True)
    
    notify_critical = st.checkbox("Notify on Critical Parameters", value=True)
    notify_anomaly = st.checkbox("Notify on Detected Anomalies", value=True)
    
    notify_channels = st.multiselect(
        "Notification Channels",
        ["Dashboard", "Email", "SMS", "Mobile Push"],
        default=["Dashboard"]
    )
    
    if st.button("Save Settings"):
        set_message("Settings saved successfully!", "success")
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        st.markdown("<h4>Analysis Parameters</h4>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("pH Lower Threshold", min_value=6.0, max_value=7.5, value=6.5, step=0.1)
            st.number_input("pH Upper Threshold", min_value=8.0, max_value=10.0, value=9.0, step=0.1)
            st.number_input("Temperature Lower Threshold (°C)", min_value=15.0, max_value=25.0, value=22.0, step=0.5)
            st.number_input("Temperature Upper Threshold (°C)", min_value=30.0, max_value=35.0, value=32.0, step=0.5)
        
        with col2:
            st.number_input("DO Minimum Threshold (mg/L)", min_value=2.0, max_value=5.0, value=4.0, step=0.1)
            st.number_input("Ammonia Maximum Threshold (mg/L)", min_value=0.1, max_value=1.0, value=0.3, step=0.1)
            st.number_input("Nitrite Maximum Threshold (mg/L)", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
        
        if st.button("Reset to Defaults", key="reset_advanced"):
            set_message("Advanced settings reset to defaults", "success")
    
    # API settings
    with st.expander("API Settings"):
        st.text_input("API Key (if required)", type="password")
        st.number_input("Request Timeout (seconds)", min_value=5, max_value=60, value=10, step=1)
        
        if st.button("Test API Connection"):
            response = api_request("get", "/health")
            if "error" in response:
                set_message(f"API connection failed: {response['error']}", "error")
            else:
                set_message("API connection successful", "success")

# Reset Data Page
elif page == "Reset Data":
    st.markdown("<h2 class='subheader'>Reset Data</h2>", unsafe_allow_html=True)
    st.warning("⚠️ This action will delete all your submitted data. This cannot be undone.")
    
    st.markdown("Select what data you want to reset:")
    reset_sensor = st.checkbox("Reset Sensor Data", value=True)
    reset_farmer = st.checkbox("Reset Farmer Data", value=False)
    
    confirm_text = st.text_input("Type 'RESET' to confirm data deletion:")
    
    if st.button("Reset Selected Data"):
        if confirm_text == "RESET":
            response = api_request("post", "/reset")
            if "error" in response:
                set_message(f"Error resetting data: {response['error']}", "error")
            else:
                set_message("Data reset successfully!", "success")
                # Clear session state data
                st.session_state.sensor_data_history = []
        else:
            set_message("Confirmation text doesn't match. Please type 'RESET' to confirm.", "warning")

# Add footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center">
        <p>🦐 Shrimp Aquaculture Monitoring System | Developed with Streamlit</p>
        <p style="font-size: 0.8rem">v1.0.0 | © 2025</p>
    </div>
    """,
    unsafe_allow_html=True
)