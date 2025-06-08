import requests
from datetime import datetime, timedelta
import json

# API base URL (assuming local development server)
BASE_URL = "http://localhost:8000"

# Example 1: Morning Pond Check (6 AM)
def morning_check():
    # Current pond details
    pond_data = {
        "pond_area": 5000,  # 5000 square meters
        "pond_depth": 1.5,  # 1.5 meters
        "stocking_density": 40,  # 40 shrimp per square meter
        "culture_start_date": "2024-01-15T00:00:00",  # Culture started 40 days ago
        "latitude": 8.5241,  # Example: Location in Kerala
        "longitude": 76.9366,
        "current_average_weight": 18.5,  # 18.5 grams per shrimp
        "current_survival_rate": 85.0,  # 85% survival
        "water_temperature": 28.5,  # 28.5°C
        "salinity": 18.0,  # 18 ppt
        "ph_level": 7.8,  # pH 7.8
        "dissolved_oxygen": 4.5  # 4.5 mg/L
    }

    # 1. Check Growth Status
    response = requests.post(f"{BASE_URL}/analyze/growth", json=pond_data)
    growth_analysis = response.json()
    print("\n=== Morning Growth Analysis ===")
    print(f"Current Growth Rate: {growth_analysis['current_growth_rate']:.2f} g/day")
    print(f"Days until optimal harvest: {growth_analysis['estimated_days_to_harvest']}")
    print(f"Current yield percentage: {growth_analysis['current_yield_percentage']:.1f}%")

    # 2. Check Health Status
    response = requests.post(f"{BASE_URL}/analyze/health", json=pond_data)
    health_analysis = response.json()
    print("\n=== Health Status ===")
    print("Disease Risks:")
    for disease, risk in health_analysis['disease_risks'].items():
        risk_level = "Low" if risk < 0.3 else "Medium" if risk < 0.6 else "High"
        print(f"- {disease}: {risk_level} (Risk score: {risk:.2f})")
    print("\nWater Quality:")
    for param, status in health_analysis['water_quality_status'].items():
        print(f"- {param}: {status}")

    # 3. Get Feeding Recommendations
    feeding_history = [
        {
            "date": "2024-02-22T06:00:00",
            "feed_amount": 25.0,
            "feed_type": "CP Commercial Feed",
            "consumption_rate": 95.0
        },
        {
            "date": "2024-02-22T14:00:00",
            "feed_amount": 28.0,
            "feed_type": "CP Commercial Feed",
            "consumption_rate": 92.0
        },
        {
            "date": "2024-02-22T18:00:00",
            "feed_amount": 27.0,
            "feed_type": "CP Commercial Feed",
            "consumption_rate": 88.0
        }
    ]

    response = requests.post(
        f"{BASE_URL}/calculate/feeding",
        json={
            "pond": pond_data,
            "feeding_history": feeding_history
        }
    )
    feeding_analysis = response.json()
    print("\n=== Feeding Recommendations ===")
    print(f"FCR: {feeding_analysis['fcr_analysis']['fcr']:.2f} ({feeding_analysis['fcr_analysis']['status']})")
    print(f"Recommended daily feed: {feeding_analysis['recommended_daily_feed']:.1f} kg")
    print("\nFeeding Schedule:")
    for timing in feeding_analysis['feeding_schedule']:
        feed_amount = feeding_analysis['recommended_daily_feed'] * timing['portion']
        print(f"- {timing['time']}: {feed_amount:.1f} kg")

# Example 2: Evening Update (6 PM)
def evening_update():
    # Updated pond measurements
    evening_data = {
        "pond_area": 5000,
        "pond_depth": 1.5,
        "stocking_density": 40,
        "culture_start_date": "2024-01-15T00:00:00",
        "latitude": 8.5241,
        "longitude": 76.9366,
        "current_average_weight": 18.7,  # Slight growth
        "current_survival_rate": 85.0,
        "water_temperature": 29.2,  # Temperature increased
        "salinity": 18.5,  # Slight increase
        "ph_level": 7.9,
        "dissolved_oxygen": 4.3  # Slight decrease
    }

    # Check if any parameters need attention
    response = requests.post(f"{BASE_URL}/analyze/health", json=evening_data)
    health_analysis = response.json()
    
    print("\n=== Evening Health Check ===")
    print("Water Quality Changes:")
    critical_params = []
    if evening_data['dissolved_oxygen'] < 4.5:
        critical_params.append("DO levels dropping - Consider emergency aeration")
    if evening_data['temperature'] > 29:
        critical_params.append("Temperature rising - Monitor closely")
    
    if critical_params:
        print("\nWarnings:")
        for param in critical_params:
            print(f"- {param}")
    else:
        print("All parameters within acceptable range")

# Run the examples
if __name__ == "__main__":
    print("=== Simulating a day in shrimp farm management ===")
    morning_check()
    evening_update()