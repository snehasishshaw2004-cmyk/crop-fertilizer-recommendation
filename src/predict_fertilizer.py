import joblib
import pandas as pd


# ============================================================
# 1. LOAD TRAINED FERTILIZER MODEL
# ============================================================

model_path = "models/fertilizer_model.pkl"

model = joblib.load(model_path)

print("==========================================")
print("     FERTILIZER RECOMMENDATION SYSTEM")
print("==========================================")


# ============================================================
# 2. GET USER INPUT
# ============================================================

print("\nEnter the following soil and environmental values:\n")

temperature = float(input("Temperature (°C): "))
humidity = float(input("Humidity (%): "))
moisture = float(input("Soil Moisture: "))

soil_type = input("Soil Type: ")
crop_type = input("Crop Type: ")

nitrogen = float(input("Nitrogen: "))
potassium = float(input("Potassium: "))
phosphorous = float(input("Phosphorous: "))


# ============================================================
# 3. CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame(
    [[
        temperature,
        humidity,
        moisture,
        soil_type,
        crop_type,
        nitrogen,
        potassium,
        phosphorous
    ]],
    columns=[
        "Temperature",
        "Humidity",
        "Moisture",
        "Soil Type",
        "Crop Type",
        "Nitrogen",
        "Potassium",
        "Phosphorous"
    ]
)


# ============================================================
# 4. MAKE FERTILIZER PREDICTION
# ============================================================

prediction = model.predict(input_data)


# ============================================================
# 5. DISPLAY RESULT
# ============================================================

print("\n==========================================")
print("        RECOMMENDATION RESULT")
print("==========================================")

print(f"\nRecommended Fertilizer: {prediction[0]}")

print("\n==========================================")