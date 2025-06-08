import requests
import json
import datetime
import time

# Base URL for the API
BASE_URL = "http://localhost:8000"

# Helper function to print responses nicely
def print_response(response, message=""):
    print(f"\n{message}" if message else "")
    print(f"Status Code: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text)

# Test the health endpoint
def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print_response(response, "Health Check:")

# Reset all data to start fresh
def test_reset():
    response = requests.post(f"{BASE_URL}/reset")
    print_response(response, "Reset Data:")

# Set farmer data
def test_set_farmer_data():
    farmer_data = {
        "pond_area": 1000.0,
        "pond_depth": 1.5,
        "stocking_density": 50.0,
        "culture_start_date": (datetime.datetime.now() - datetime.timedelta(days=30)).isoformat(),
        "location": {
            "lat": 10.0,
            "lon": 105.0
        }
    }
    response = requests.post(f"{BASE_URL}/set_farmer_data", json=farmer_data)
    print_response(response, "Set Farmer Data:")

# Submit sensor data manually
def test_submit_sensor_data():
    sensor_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "pH": 7.5,
        "temperature": 28.0,
        "turbidity": 10.0,
        "tds": 500.0
    }
    response = requests.post(f"{BASE_URL}/submit_sensor_data", json=sensor_data)
    print_response(response, "Submit Sensor Data:")

# Capture sensor data automatically
def test_capture_sensor_data():
    response = requests.get(f"{BASE_URL}/capture_sensor_data")
    print_response(response, "Capture Sensor Data:")

# Get all sensor data
def test_get_sensor_data():
    response = requests.get(f"{BASE_URL}/get_sensor_data")
    print_response(response, "Get Sensor Data:")

# Analyze the data
def test_analyze_data():
    response = requests.get(f"{BASE_URL}/analyze")
    print_response(response, "Analyze Data:")

# Run all tests in sequence
def run_all_tests():
    print("\n===== STARTING API TESTS =====\n")
    
    # First, check if the API is running
    test_health()
    
    # Reset all data
    test_reset()
    
    # Set farmer data (required before submitting sensor data)
    test_set_farmer_data()
    
    # Submit some sensor data - both manually and automatically
    test_submit_sensor_data()
    
    # Capture some additional data points
    for _ in range(3):
        test_capture_sensor_data()
        time.sleep(1)  # Wait a bit between captures
    
    # Get all sensor data
    test_get_sensor_data()
    
    # Analyze the data
    test_analyze_data()
    
    print("\n===== API TESTS COMPLETED =====\n")

# Main execution
if __name__ == "__main__":
    try:
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to the API server.")
        print("Make sure the API is running with: python new_model4.py")
