import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================
# 1. LOAD DATASET
# ============================================

data = pd.read_csv("data/crop_recommendation.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# ============================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================

X = data[
    [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]
]

y = data["label"]


# ============================================
# 3. SPLIT DATA INTO TRAINING AND TESTING
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================
# 4. CREATE MACHINE LEARNING MODEL
# ============================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# ============================================
# 5. TRAIN MODEL
# ============================================

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ============================================
# 6. MAKE PREDICTIONS
# ============================================

y_pred = model.predict(X_test)


# ============================================
# 7. CALCULATE ACCURACY
# ============================================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")


# ============================================
# 8. CLASSIFICATION REPORT
# ============================================

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================
# 9. SAVE MODEL
# ============================================

joblib.dump(model, "models/crop_model.pkl")

print("\nModel saved successfully!")
print("Location: models/crop_model.pkl")