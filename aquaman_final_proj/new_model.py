import asyncio
import datetime
import math
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA

app = FastAPI()

class SensorData(BaseModel):
    timestamp: datetime.datetime
    pH: float = Field(ge=0.0, le=14.0, description="pH must be between 0 and 14")
    temperature: float = Field(ge=10.0, le=40.0, description="Temperature must be between 10°C and 40°C")
    turbidity: float = Field(ge=0.0, description="Turbidity must be non-negative")
    tds: float = Field(ge=0.0, description="TDS must be non-negative")

class FarmerData(BaseModel):
    pond_area: float
    pond_depth: float
    stocking_density: float
    culture_start_date: datetime.datetime
    location: dict

sensor_data = []
farmer_data = None

WEATHER_API_KEY = "81c1469af9999e3a8f5a9624cfabaacd"
WEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"

def get_lunar_phase(date: datetime.datetime) -> str:
    synodic_period = 29.53058867
    known_new_moon = datetime.datetime(2000, 1, 6, 18, 14)
    days_since = (date - known_new_moon).days
    phase = (days_since % synodic_period) / synodic_period * 8
    phases = ["New Moon", "Waxing Crescent", "First Quarter", "Waxing Gibbous",
              "Full Moon", "Waning Gibbous", "Last Quarter", "Waning Crescent"]
    return phases[int(phase)]

async def fetch_weather_data(lat: float, lon: float) -> dict:
    if WEATHER_API_KEY == "your_openweathermap_api_key":
        return {"temperature": 28.0, "humidity": 80, "precipitation": 0}
    try:
        params = {"lat": lat, "lon": lon, "appid": WEATHER_API_KEY, "units": "metric"}
        response = requests.get(WEATHER_API_URL, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        return {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "precipitation": data.get("rain", {}).get("1h", 0)
        }
    except requests.RequestException as e:
        print(f"Weather API error: {str(e)}")
        return {"temperature": 28.0, "humidity": 80, "precipitation": 0}

def estimate_parameters(df: pd.DataFrame) -> pd.DataFrame:
    df["salinity"] = df["tds"] * 0.001 + 0.5 * df["temperature"]
    df["ammonia"] = 0.02 * df["pH"] + 0.01 * df["turbidity"]
    df["do"] = 8.0 - 0.1 * df["temperature"] + 0.05 * df["pH"]
    df["nitrite"] = 0.01 * df["ammonia"] + 0.005 * df["turbidity"]
    df["alkalinity"] = 100 + 10 * df["pH"]
    return df

def calculate_growth(day: int, water_quality: pd.Series, lunar_phase: str) -> float:
    K = 20
    r = 0.05
    t0 = 10
    if water_quality["do"] < 4.03 or water_quality["pH"] < 6.7 or water_quality["pH"] > 8.7:
        r *= 0.8
    if water_quality["ammonia"] > 0.5:
        r *= 0.7
    if lunar_phase in ["Full Moon", "New Moon"]:
        r *= 1.1
    weight = K / (1 + np.exp(-r * (day - t0)))
    return weight

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Detect anomalies using Isolation Forest and threshold-based checks."""
    params = ["pH", "temperature", "turbidity", "tds", "salinity", "ammonia", "do", "nitrite", "alkalinity"]
    
    # Statistical anomaly detection with dynamic contamination
    n_samples = len(df)
    contamination = min(0.3, 1.0 / n_samples) if n_samples > 1 else 0.1
    clf = IsolationForest(contamination=contamination, random_state=42)
    df["stat_anomaly"] = clf.fit_predict(df[params])
    
    # Threshold-based anomaly detection (based on Undermind report [22])
    df["threshold_anomaly"] = (
        (df["pH"] < 6.7) | (df["pH"] > 8.7) |
        (df["do"] < 4.03) |
        (df["ammonia"] > 0.5)
    ).astype(int)
    
    # Combine anomalies: mark as anomaly (-1) if either method flags it
    df["anomaly"] = np.where(
        (df["stat_anomaly"] == -1) | (df["threshold_anomaly"] == 1),
        -1,  # Anomaly
        1   # Normal
    )
    
    return df

def predict_yield(df: pd.DataFrame, stocking_density: float, pond_area: float) -> float:
    latest_weight = df["shrimp_weight"].iloc[-1]
    survival_rate = 0.9
    if df["ammonia"].mean() > 0.5 or df["do"].mean() < 4.03:
        survival_rate *= 0.8
    yield_kg = latest_weight * stocking_density * pond_area * survival_rate / 1000
    return yield_kg

def calculate_fcr_and_feed_schedule(df: pd.DataFrame, stocking_density: float, pond_area: float) -> tuple:
    total_biomass = df["shrimp_weight"].mean() * stocking_density * pond_area / 1000
    feed_per_day = total_biomass * 0.03
    fcr = 1.5
    if df["do"].mean() < 4.03:
        fcr += 0.2
    schedule = {
        "morning": feed_per_day * 0.4,
        "afternoon": feed_per_day * 0.3,
        "evening": feed_per_day * 0.3
    }
    return fcr, schedule

@app.post("/submit_sensor_data")
async def submit_sensor_data(data: SensorData):
    global sensor_data, farmer_data
    if farmer_data is None:
        raise HTTPException(status_code=400, detail="Farmer data not set")
    record = data.dict()
    record["lunar_phase"] = get_lunar_phase(data.timestamp)
    weather = await fetch_weather_data(farmer_data.location["lat"], farmer_data.location["lon"])
    record.update(weather)
    sensor_data.append(record)
    return {"status": "Data received"}

@app.post("/set_farmer_data")
async def set_farmer_data(data: FarmerData):
    global farmer_data
    farmer_data = data
    return {"status": "Farmer data set"}

@app.get("/analyze")
async def analyze_data():
    if not sensor_data or not farmer_data:
        raise HTTPException(status_code=400, detail="Missing sensor or farmer data")
    df = pd.DataFrame(sensor_data)
    df["day"] = (df["timestamp"] - farmer_data.culture_start_date).dt.days
    df = estimate_parameters(df)
    df["shrimp_weight"] = df.apply(
        lambda row: calculate_growth(row["day"], row, row["lunar_phase"]), axis=1
    )
    df = detect_anomalies(df)
    yield_kg = predict_yield(df, farmer_data.stocking_density, farmer_data.pond_area)
    fcr, feed_schedule = calculate_fcr_and_feed_schedule(
        df, farmer_data.stocking_density, farmer_data.pond_area
    )
    growth_series = df["shrimp_weight"]
    try:
        model = ARIMA(growth_series, order=(1, 1, 1))
        model_fit = model.fit()
        growth_rate = model_fit.params.get("ar.L1", 0)
    except:
        growth_rate = growth_series.diff().mean()
    insights = {
        "water_quality": df[["pH", "temperature", "turbidity", "tds", "salinity", "ammonia", "do", "nitrite", "alkalinity"]].mean().to_dict(),
        "anomalies": df[df["anomaly"] == -1][["timestamp"]].to_dict(orient="records"),
        "shrimp_weight": df["shrimp_weight"].iloc[-1],
        "growth_rate": growth_rate,
        "predicted_yield_kg": yield_kg,
        "fcr": fcr,
        "feed_schedule": feed_schedule
    }
    return insights

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)