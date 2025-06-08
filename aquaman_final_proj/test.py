import requests
import pytest
import datetime
import time

BASE_URL = "http://localhost:8000"

FARMER_DATA = {
    "pond_area": 1000,
    "pond_depth": 1.5,
    "stocking_density": 50,
    "culture_start_date": "2025-01-01T00:00:00",
    "location": {"lat": 10.0, "lon": 105.0}
}

SENSOR_DATA = [
    {
        "timestamp": "2025-01-01T08:00:00",
        "pH": 7.5,
        "temperature": 28.0,
        "turbidity": 10.0,
        "tds": 500.0
    },
    {
        "timestamp": "2025-01-02T08:00:00",
        "pH": 7.8,
        "temperature": 27.5,
        "turbidity": 12.0,
        "tds": 510.0
    },
    {
        "timestamp": "2025-01-03T08:00:00",
        "pH": 6.5,
        "temperature": 29.0,
        "turbidity": 15.0,
        "tds": 490.0
    },
    {
        "timestamp": "2025-01-04T08:00:00",
        "pH": 7.6,
        "temperature": 28.2,
        "turbidity": 11.0,
        "tds": 505.0
    },
    {
        "timestamp": "2025-01-05T08:00:00",
        "pH": 7.7,
        "temperature": 28.0,
        "turbidity": 10.5,
        "tds": 500.0
    }
]

def test_set_farmer_data():
    response = requests.post(f"{BASE_URL}/set_farmer_data", json=FARMER_DATA)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert response.json() == {"status": "Farmer data set"}, "Unexpected response content"
    print("Farmer data set successfully")

def test_submit_sensor_data():
    for data in SENSOR_DATA:
        response = requests.post(f"{BASE_URL}/submit_sensor_data", json=data)
        assert response.status_code == 200, f"Expected 200, got {response.status_code} for {data['timestamp']}"
        assert response.json() == {"status": "Data received"}, f"Unexpected response for {data['timestamp']}"
        print(f"Submitted sensor data for {data['timestamp']}")
        time.sleep(0.1)

def test_analyze_data():
    response = requests.get(f"{BASE_URL}/analyze")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    result = response.json()
    expected_keys = ["water_quality", "anomalies", "shrimp_weight", "growth_rate", 
                     "predicted_yield_kg", "fcr", "feed_schedule"]
    assert all(key in result for key in expected_keys), "Missing expected keys in analysis result"
    water_quality = result["water_quality"]
    expected_params = ["pH", "temperature", "turbidity", "tds", "salinity", "ammonia", "do", "nitrite", "alkalinity"]
    assert all(param in water_quality for param in expected_params), "Missing water quality parameters"
    assert 6.7 <= water_quality["pH"] <= 8.7, f"pH {water_quality['pH']} out of range"
    assert water_quality["do"] > 4.03, f"DO {water_quality['do']} below threshold"
    assert water_quality["ammonia"] < 0.5, f"Ammonia {water_quality['ammonia']} too high"
    assert len(result["anomalies"]) > 0, "Expected at least one anomaly"
    anomaly_timestamps = [a["timestamp"] for a in result["anomalies"]]
    assert "2025-01-03T08:00:00" in anomaly_timestamps, "Expected anomaly on 2025-01-03"
    assert result["shrimp_weight"] > 0, "Shrimp weight should be positive"
    assert 0 < result["growth_rate"] < 2, f"Growth rate {result['growth_rate']} out of range"
    assert 0 < result["predicted_yield_kg"] < 1000, "Yield out of expected range"
    assert 1.0 < result["fcr"] < 2.0, f"FCR {result['fcr']} out of expected range"
    feed_schedule = result["feed_schedule"]
    assert all(key in feed_schedule for key in ["morning", "afternoon", "evening"]), "Missing feed schedule keys"
    assert all(v > 0 for v in feed_schedule.values()), "Feed amounts should be positive"
    print("Analysis validated successfully")
    print("Analysis Results:", result)

def test_invalid_sensor_data():
    invalid_data = {
        "timestamp": "2025-01-06T08:00:00",
        "pH": -1.0,
        "temperature": 28.0,
        "turbidity": 10.0,
        "tds": 500.0
    }
    response = requests.post(f"{BASE_URL}/submit_sensor_data", json=invalid_data)
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"
    assert "pH" in response.text, "Expected pH validation error"
    print("Invalid sensor data (negative pH) test passed")

def test_invalid_turbidity():
    invalid_data = {
        "timestamp": "2025-01-06T08:00:00",
        "pH": 7.5,
        "temperature": 28.0,
        "turbidity": -10.0,
        "tds": 500.0
    }
    response = requests.post(f"{BASE_URL}/submit_sensor_data", json=invalid_data)
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"
    assert "turbidity" in response.text, "Expected turbidity validation error"
    print("Invalid sensor data (negative turbidity) test passed")

def test_suboptimal_but_valid_data():
    suboptimal_data = {
        "timestamp": "2025-01-06T08:00:00",
        "pH": 6.5,
        "temperature": 28.0,
        "turbidity": 10.0,
        "tds": 500.0
    }
    response = requests.post(f"{BASE_URL}/submit_sensor_data", json=suboptimal_data)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    print("Suboptimal but valid data test passed")
    response = requests.get(f"{BASE_URL}/analyze")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    result = response.json()
    anomaly_timestamps = [a["timestamp"] for a in result["anomalies"]]
    assert "2025-01-06T08:00:00" in anomaly_timestamps, "Expected anomaly for suboptimal pH"
    print("Suboptimal data anomaly detection passed")

if __name__ == "__main__":
    try:
        print("Starting tests...")
        test_set_farmer_data()
        test_submit_sensor_data()
        test_analyze_data()
        test_invalid_sensor_data()
        test_invalid_turbidity()
        test_suboptimal_but_valid_data()
        print("All tests completed successfully!")
    except AssertionError as e:
        print(f"Test failed: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")