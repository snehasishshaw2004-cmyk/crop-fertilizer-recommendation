import joblib
import pandas as pd


# Load the trained model
model = joblib.load("models/crop_model.pkl")

print("======================================")
print("       CROP RECOMMENDATION SYSTEM")
print("======================================")

print("\nEnter the following soil and weather values:\n")


# Get user input
N = float(input("Nitrogen (N): "))
P = float(input("Phosphorus (P): "))
K = float(input("Potassium (K): "))

temperature = float(input("Temperature (°C): "))
humidity = float(input("Humidity (%): "))
ph = float(input("Soil pH: "))
rainfall = float(input("Rainfall (mm): "))


# Create input dataframe
input_data = pd.DataFrame(
    [[
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
    ]],
    columns=[
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]
)


# Make prediction
prediction = model.predict(input_data)


# Display result
print("\n======================================")
print("        RECOMMENDATION RESULT")
print("======================================")

print(f"\nRecommended Crop: {prediction[0]}")

print("\n======================================")