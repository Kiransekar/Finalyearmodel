import asyncio
import datetime
import math
import logging
from typing import List, Optional, Dict, Any
import requests
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Shrimp Aquaculture API",
    description="API for monitoring and managing shrimp aquaculture ponds",
    version="1.0.0",
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom JSON encoder for FastAPI to handle NaN, Infinity values
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

class CustomJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        # Replace NaN, Infinity, -Infinity with valid JSON values before encoding
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(i) for i in obj]
            elif isinstance(obj, float):
                if math.isnan(obj):
                    return None
                elif math.isinf(obj):
                    return None if obj < 0 else 1e38  # Use large but valid number for +Infinity
            return obj
        
        cleaned_content = clean_for_json(content)
        return super().render(cleaned_content)

class SensorData(BaseModel):
    timestamp: datetime.datetime
    pH: float = Field(ge=0.0, le=14.0, description="pH must be between 0 and 14")
    temperature: float = Field(ge=10.0, le=40.0, description="Temperature must be between 10°C and 40°C")
    turbidity: float = Field(ge=0.0, description="Turbidity must be non-negative")
    tds: float = Field(ge=0.0, description="TDS must be non-negative")
    
    class Config:
        schema_extra = {
            "example": {
                "timestamp": "2025-01-01T08:00:00",
                "pH": 7.5,
                "temperature": 28.0,
                "turbidity": 10.0,
                "tds": 500.0
            }
        }

class Location(BaseModel):
    lat: float = Field(..., description="Latitude coordinate")
    lon: float = Field(..., description="Longitude coordinate")
    
    @validator('lat')
    def validate_latitude(cls, v):
        if v < -90 or v > 90:
            raise ValueError("Latitude must be between -90 and 90")
        return v
    
    @validator('lon')
    def validate_longitude(cls, v):
        if v < -180 or v > 180:
            raise ValueError("Longitude must be between -180 and 180")
        return v

class FarmerData(BaseModel):
    pond_area: float = Field(gt=0, description="Pond area in square meters")
    pond_depth: float = Field(gt=0, description="Pond depth in meters")
    stocking_density: float = Field(gt=0, description="Stocking density in shrimp per square meter")
    culture_start_date: datetime.datetime = Field(..., description="Date when culture started")
    location: Location
    
    class Config:
        schema_extra = {
            "example": {
                "pond_area": 1000.0,
                "pond_depth": 1.5,
                "stocking_density": 50.0,
                "culture_start_date": "2025-01-01T00:00:00",
                "location": {
                    "lat": 10.0,
                    "lon": 105.0
                }
            }
        }

class WaterQualityParameters(BaseModel):
    pH: float
    temperature: float
    turbidity: float
    tds: float
    salinity: float
    ammonia: float
    do: float
    nitrite: float
    alkalinity: float

class FeedSchedule(BaseModel):
    morning: float
    afternoon: float
    evening: float

class AnalysisResponse(BaseModel):
    water_quality: WaterQualityParameters
    anomalies: List[Dict[str, Any]]
    shrimp_weight: float
    growth_rate: float
    predicted_yield_kg: float
    fcr: float
    feed_schedule: FeedSchedule

# Global storage - in a production app, use a database instead
class DataStore:
    def __init__(self):
        self.sensor_data = []
        self.farmer_data = None

data_store = DataStore()

# Environment configuration - in production, use proper environment variables
WEATHER_API_KEY = "81c1469af9999e3a8f5a9624cfabaacd"
WEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"

def get_lunar_phase(date: datetime.datetime) -> str:
    """Calculate lunar phase for a given date."""
    synodic_period = 29.53058867
    known_new_moon = datetime.datetime(2000, 1, 6, 18, 14)
    days_since = (date - known_new_moon).total_seconds() / 86400
    phase = (days_since % synodic_period) / synodic_period * 8
    phases = ["New Moon", "Waxing Crescent", "First Quarter", "Waxing Gibbous",
              "Full Moon", "Waning Gibbous", "Last Quarter", "Waning Crescent"]
    return phases[int(phase)]

async def fetch_weather_data(lat: float, lon: float) -> dict:
    """Fetch weather data from OpenWeatherMap API."""
    if WEATHER_API_KEY == "your_openweathermap_api_key":
        logger.warning("Using default weather values because API key is not set")
        return {"temperature": 28.0, "humidity": 80, "precipitation": 0}
    
    try:
        params = {"lat": lat, "lon": lon, "appid": WEATHER_API_KEY, "units": "metric"}
        # Using asyncio.wait_for instead of asyncio.timeout for compatibility
        # Create a synchronous request since fetch_weather_data is already async
        response = requests.get(WEATHER_API_URL, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        return {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "precipitation": data.get("rain", {}).get("1h", 0)
        }
    except (requests.RequestException, asyncio.TimeoutError) as e:
        logger.error(f"Weather API error: {str(e)}")
        return {"temperature": 28.0, "humidity": 80, "precipitation": 0}

def estimate_parameters(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate derived water quality parameters."""
    # Calculate derived parameters
    df["salinity"] = df["tds"] * 0.001 + 0.5 * df["temperature"]
    df["ammonia"] = 0.02 * df["pH"] + 0.01 * df["turbidity"]
    df["do"] = 8.0 - 0.1 * df["temperature"] + 0.05 * df["pH"]
    df["nitrite"] = 0.01 * df["ammonia"] + 0.005 * df["turbidity"]
    df["alkalinity"] = 100 + 10 * df["pH"]
    
    # Ensure all values are finite
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

def calculate_growth(day: int, water_quality: pd.Series, lunar_phase: str) -> float:
    """Calculate shrimp growth using a logistic growth model with environmental factors."""
    try:
        # Logistic growth parameters
        K = 20  # Maximum weight in grams
        r = 0.05  # Growth rate
        t0 = 10  # Inflection point
        
        # Adjust growth rate based on water quality
        if water_quality["do"] < 4.0 or water_quality["pH"] < 6.5 or water_quality["pH"] > 9.0:
            r *= 0.8
        if water_quality["ammonia"] > 0.5:
            r *= 0.7
        if water_quality["nitrite"] > 0.5:
            r *= 0.9
        
        # Lunar phase effect
        if lunar_phase in ["Full Moon", "New Moon"]:
            r *= 1.1
        
        # Calculate weight using logistic growth equation
        exp_term = np.exp(-r * (day - t0))
        # Guard against overflow or underflow
        if np.isnan(exp_term) or np.isinf(exp_term):
            if -r * (day - t0) > 0:  # Large positive exponent
                exp_term = 1e30
            else:  # Large negative exponent
                exp_term = 0
                
        weight = K / (1 + exp_term)
        
        # Validate the result
        if np.isnan(weight) or np.isinf(weight):
            return 0.1  # Default to small weight if calculation fails
            
        return weight
    except Exception as e:
        logger.warning(f"Error in growth calculation: {str(e)}")
        return 0.1  # Default to small weight if calculation fails

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Detect anomalies in water quality parameters using Isolation Forest."""
    # Parameters to analyze
    params = ["pH", "temperature", "turbidity", "tds", "salinity", "ammonia", "do", "nitrite", "alkalinity"]
    
    # Ensure all values are finite before anomaly detection
    df[params] = df[params].replace([np.inf, -np.inf], np.nan).fillna(df[params].mean())
    
    # Handle case with few samples
    n_samples = len(df)
    if n_samples <= 1:
        df["stat_anomaly"] = 1  # No anomalies with just one sample
        df["threshold_anomaly"] = 0
        df["anomaly"] = 1
        return df
    
    # Set contamination parameter based on number of samples
    contamination = min(0.3, 1.0 / n_samples)
    
    # Statistical anomaly detection
    try:
        clf = IsolationForest(contamination=contamination, random_state=42)
        df["stat_anomaly"] = clf.fit_predict(df[params])
    except Exception as e:
        logger.warning(f"Error in anomaly detection: {str(e)}")
        df["stat_anomaly"] = 1  # Default to no anomalies
    
    # Threshold-based anomaly detection
    df["threshold_anomaly"] = (
        (df["pH"] < 6.5) | (df["pH"] > 9.0) |
        (df["do"] < 4.0) |
        (df["ammonia"] > 0.5) |
        (df["nitrite"] > 0.5) |
        (df["temperature"] < 15.0) | (df["temperature"] > 35.0)
    ).astype(int)
    
    # Combined anomaly flag
    df["anomaly"] = np.where(
        (df["stat_anomaly"] == -1) | (df["threshold_anomaly"] == 1),
        -1,
        1
    )
    
    return df

def predict_yield(df: pd.DataFrame, stocking_density: float, pond_area: float) -> float:
    """Predict harvest yield based on growth and environmental factors."""
    if df.empty:
        return 0.0
    
    latest_weight = df["shrimp_weight"].iloc[-1]
    
    # Base survival rate
    survival_rate = 0.85
    
    # Adjust survival based on water quality
    if df["ammonia"].mean() > 0.5:
        survival_rate *= 0.8
    if df["do"].mean() < 4.0:
        survival_rate *= 0.8
    if df["nitrite"].mean() > 0.5:
        survival_rate *= 0.9
    
    # Adjust for culture duration (longer cultures have lower survival)
    days = df["day"].max()
    if days > 60:
        survival_rate *= 0.95
    
    # Calculate yield
    yield_kg = latest_weight * stocking_density * pond_area * survival_rate / 1000
    
    # Ensure value is finite
    if np.isnan(yield_kg) or np.isinf(yield_kg):
        yield_kg = 0.0
        
    return yield_kg

def calculate_fcr_and_feed_schedule(df: pd.DataFrame, stocking_density: float, pond_area: float) -> tuple:
    """Calculate feed conversion ratio and optimal feeding schedule."""
    if df.empty:
        return 1.5, {"morning": 0, "afternoon": 0, "evening": 0}
    
    # Calculate current biomass in kg
    current_weight_g = df["shrimp_weight"].iloc[-1]
    survival_rate = 0.85
    
    # Adjust survival based on water quality
    if df["ammonia"].mean() > 0.5:
        survival_rate *= 0.8
    if df["do"].mean() < 4.0:
        survival_rate *= 0.8
    
    total_biomass_kg = current_weight_g * stocking_density * pond_area * survival_rate / 1000
    
    # Calculate feed amount (3% of biomass)
    feed_per_day = total_biomass_kg * 0.03
    
    # Calculate FCR
    fcr = 1.5  # Base FCR
    
    # Adjust for water quality
    if df["do"].mean() < 4.0:
        fcr += 0.2
    if df["temperature"].mean() < 25 or df["temperature"].mean() > 32:
        fcr += 0.1
    
    # Feeding schedule distribution
    schedule = {
        "morning": round(feed_per_day * 0.4, 2),
        "afternoon": round(feed_per_day * 0.3, 2),
        "evening": round(feed_per_day * 0.3, 2)
    }
    
    # Ensure values are finite
    if np.isnan(fcr) or np.isinf(fcr):
        fcr = 1.5
    
    for key in schedule:
        if np.isnan(schedule[key]) or np.isinf(schedule[key]):
            schedule[key] = 0.0
    
    return fcr, schedule

def get_farmer_data() -> FarmerData:
    """Dependency to get farmer data or raise exception if not set."""
    if data_store.farmer_data is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Farmer data not set. Please submit farmer data first."
        )
    return data_store.farmer_data

# New endpoint to simulate capturing data from sensors
@app.get("/capture_sensor_data", status_code=status.HTTP_200_OK)
async def capture_sensor_data(farmer_data: FarmerData = Depends(get_farmer_data)):
    """Simulate capturing real-time sensor data from physical sensors."""
    try:
        # Simulate reading from connected sensors with some randomization
        current_time = datetime.datetime.now()
        
        # Generate simulated readings with some natural variation
        base_ph = 7.5 + np.random.normal(0, 0.2)  # pH varies around 7.5
        base_temp = 28.0 + np.random.normal(0, 0.5)  # Temperature varies around 28°C
        base_turbidity = 10.0 + np.random.normal(0, 1.0)  # Turbidity varies around 10 NTU
        base_tds = 500.0 + np.random.normal(0, 20.0)  # TDS varies around 500 ppm
        
        # Ensure values are within valid ranges
        ph = max(min(base_ph, 14.0), 0.0)
        temperature = max(min(base_temp, 40.0), 10.0)
        turbidity = max(base_turbidity, 0.0)
        tds = max(base_tds, 0.0)
        
        # Create sensor data record
        record = {
            "timestamp": current_time,
            "pH": round(ph, 2),
            "temperature": round(temperature, 2),
            "turbidity": round(turbidity, 2),
            "tds": round(tds, 2),
            "lunar_phase": get_lunar_phase(current_time)
        }
        
        # Fetch weather data
        weather = await fetch_weather_data(
            farmer_data.location.lat, 
            farmer_data.location.lon
        )
        record.update(weather)
        
        # Store data
        data_store.sensor_data.append(record)
        logger.info(f"Sensor data captured at timestamp: {current_time}")
        
        # Return the captured data to display in the UI
        return {
            "status": "success", 
            "message": "Sensor data captured successfully",
            "sensor_data": {
                "timestamp": current_time.isoformat(),
                "pH": round(ph, 2),
                "temperature": round(temperature, 2),
                "turbidity": round(turbidity, 2),
                "tds": round(tds, 2)
            }
        }
    except Exception as e:
        logger.error(f"Error capturing sensor data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error capturing sensor data: {str(e)}"
        )

@app.post("/submit_sensor_data", status_code=status.HTTP_201_CREATED)
async def submit_sensor_data(data: SensorData, farmer_data: FarmerData = Depends(get_farmer_data)):
    """Submit sensor data measurements for analysis."""
    try:
        # Prepare record with additional data
        record = data.model_dump()
        record["lunar_phase"] = get_lunar_phase(data.timestamp)
        
        # Fetch weather data
        weather = await fetch_weather_data(
            farmer_data.location.lat, 
            farmer_data.location.lon
        )
        record.update(weather)
        
        # Store data
        data_store.sensor_data.append(record)
        logger.info(f"Sensor data received for timestamp: {data.timestamp}")
        
        return {"status": "success", "message": "Data received successfully"}
    except Exception as e:
        logger.error(f"Error processing sensor data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing sensor data: {str(e)}"
        )

@app.post("/set_farmer_data", status_code=status.HTTP_201_CREATED)
async def set_farmer_data(data: FarmerData):
    """Set or update farmer data."""
    try:
        data_store.farmer_data = data
        logger.info(f"Farmer data set for pond area: {data.pond_area}m²")
        return {"status": "success", "message": "Farmer data set successfully"}
    except Exception as e:
        logger.error(f"Error setting farmer data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error setting farmer data: {str(e)}"
        )

@app.get("/analyze", response_model=AnalysisResponse, response_class=CustomJSONResponse)
async def analyze_data(farmer_data: FarmerData = Depends(get_farmer_data)):
    """Analyze sensor data and provide insights."""
    if not data_store.sensor_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No sensor data available. Please submit sensor data first."
        )
    
    try:
        # Convert to DataFrame for analysis
        df = pd.DataFrame(data_store.sensor_data)
        
        # Calculate days since culture start
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        culture_start = pd.to_datetime(farmer_data.culture_start_date)
        df["day"] = (df["timestamp"] - culture_start).dt.days
        
        # Calculate derived parameters
        df = estimate_parameters(df)
        
        # Calculate shrimp weight for each day
        df["shrimp_weight"] = df.apply(
            lambda row: calculate_growth(row["day"], row, row["lunar_phase"]), 
            axis=1
        )
        
        # Detect anomalies
        df = detect_anomalies(df)
        
        # Calculate yield prediction
        yield_kg = predict_yield(df, farmer_data.stocking_density, farmer_data.pond_area)
        
        # Calculate FCR and feed schedule
        fcr, feed_schedule = calculate_fcr_and_feed_schedule(
            df, farmer_data.stocking_density, farmer_data.pond_area
        )
        
        # Calculate growth rate
        try:
            growth_series = df["shrimp_weight"]
            if len(growth_series) > 3:
                model = ARIMA(growth_series, order=(1, 1, 1))
                model_fit = model.fit()
                growth_rate = model_fit.params.get("ar.L1", 0)
            else:
                growth_rate = growth_series.diff().mean() or 0.05
        except Exception as e:
            logger.warning(f"Error in ARIMA calculation: {str(e)}")
            growth_rate = df["shrimp_weight"].diff().mean() or 0.05
        
        # Cap growth rate at reasonable limits and ensure it's a valid number
        if np.isnan(growth_rate) or np.isinf(growth_rate):
            growth_rate = 0.05
        else:
            growth_rate = max(min(growth_rate, 0.5), 0.01)  # Between 0.01 and 0.5 g/day
        
        # Prepare response - ensure all values are finite
        water_quality_data = {
            param: round(df[param].mean(), 2) 
            for param in ["pH", "temperature", "turbidity", "tds", "salinity", "ammonia", "do", "nitrite", "alkalinity"]
        }
        
        # Replace any NaN or infinite values
        for key, value in water_quality_data.items():
            if np.isnan(value) or np.isinf(value):
                if key == "pH":
                    water_quality_data[key] = 7.0
                elif key == "temperature":
                    water_quality_data[key] = 28.0
                else:
                    water_quality_data[key] = 0.0
        
        # Get anomalies and ensure timestamps are JSON serializable
        anomalies = []
        try:
            anomaly_rows = df[df["anomaly"] == -1]
            for _, row in anomaly_rows.iterrows():
                anomalies.append({"timestamp": row["timestamp"].isoformat()})
        except Exception as e:
            logger.warning(f"Error preparing anomalies data: {str(e)}")
        
        insights = {
            "water_quality": water_quality_data,
            "anomalies": anomalies,
            "shrimp_weight": round(df["shrimp_weight"].iloc[-1], 2) if not df.empty else 0.0,
            "growth_rate": round(growth_rate, 3),
            "predicted_yield_kg": round(yield_kg, 2),
            "fcr": round(fcr, 2),
            "feed_schedule": {k: round(v, 2) for k, v in feed_schedule.items()}
        }
        
        logger.info("Analysis completed successfully")
        return insights
    
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during analysis: {str(e)}"
        )

@app.post("/reset", status_code=status.HTTP_200_OK)
async def reset_data():
    """Reset all stored data."""
    try:
        data_store.sensor_data = []
        data_store.farmer_data = None
        logger.info("All data reset successfully")
        return {"status": "success", "message": "All data reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error resetting data: {str(e)}"
        )

@app.get("/get_sensor_data", response_class=CustomJSONResponse)
async def get_sensor_data():
    """Get all stored sensor data."""
    if not data_store.sensor_data:
        return {"sensor_data": []}
    
    try:
        df = pd.DataFrame(data_store.sensor_data)
        # Convert timestamp to string for JSON serialization
        if "timestamp" in df.columns:
            df["timestamp"] = df["timestamp"].astype(str)
        
        # Replace NaN or infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        records = df.fillna(0).to_dict(orient="records")
        
        return {"sensor_data": records}
    except Exception as e:
        logger.error(f"Error retrieving sensor data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving sensor data: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "uptime": "up",
        "timestamp": datetime.datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)