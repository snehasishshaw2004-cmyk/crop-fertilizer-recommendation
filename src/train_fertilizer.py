import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. LOAD FERTILIZER DATASET
# ============================================================

file_path = "fertilizer_data/fertilizer_prediction.csv"

data = pd.read_csv(file_path)

print("Fertilizer dataset loaded successfully!")
print("Dataset shape:", data.shape)


# ============================================================
# 2. SET COLUMN NAMES
# ============================================================

# Your dataset contains exactly 9 columns.
# We explicitly set the names to avoid column-name problems.

data.columns = [
    "Temperature",
    "Humidity",
    "Moisture",
    "Soil Type",
    "Crop Type",
    "Nitrogen",
    "Potassium",
    "Phosphorous",
    "Fertilizer Name"
]

print("\nColumns:")
print(data.columns.tolist())


# ============================================================
# 3. CHECK FOR MISSING VALUES
# ============================================================

print("\nMissing values:")
print(data.isnull().sum())


# ============================================================
# 4. REMOVE ROWS WITH MISSING VALUES
# ============================================================

data = data.dropna()

print("\nDataset shape after removing missing values:")
print(data.shape)


# ============================================================
# 5. SELECT INPUT FEATURES
# ============================================================

feature_columns = [
    "Temperature",
    "Humidity",
    "Moisture",
    "Soil Type",
    "Crop Type",
    "Nitrogen",
    "Potassium",
    "Phosphorous"
]

X = data[feature_columns]


# ============================================================
# 6. SELECT TARGET
# ============================================================

y = data["Fertilizer Name"]


# ============================================================
# 7. DEFINE NUMERICAL FEATURES
# ============================================================

numeric_features = [
    "Temperature",
    "Humidity",
    "Moisture",
    "Nitrogen",
    "Potassium",
    "Phosphorous"
]


# ============================================================
# 8. DEFINE CATEGORICAL FEATURES
# ============================================================

categorical_features = [
    "Soil Type",
    "Crop Type"
]


# ============================================================
# 9. PREPROCESS CATEGORICAL DATA
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ============================================================
# 10. CREATE RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 11. CREATE MACHINE LEARNING PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ============================================================
# 12. SPLIT DATA INTO TRAINING AND TESTING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 13. TRAIN MODEL
# ============================================================

print("\n==========================================")
print("TRAINING FERTILIZER RECOMMENDATION MODEL")
print("==========================================")

print("Please wait...")

pipeline.fit(X_train, y_train)

print("Training completed successfully!")


# ============================================================
# 14. MAKE PREDICTIONS
# ============================================================

y_pred = pipeline.predict(X_test)


# ============================================================
# 15. CALCULATE ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n==========================================")
print("FERTILIZER MODEL ACCURACY")
print("==========================================")

print(f"Accuracy: {accuracy * 100:.2f}%")


# ============================================================
# 16. CLASSIFICATION REPORT
# ============================================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 17. SAVE TRAINED MODEL
# ============================================================

model_path = "models/fertilizer_model.pkl"

joblib.dump(pipeline, model_path)


# ============================================================
# 18. FINISHED
# ============================================================

print("\n==========================================")
print("MODEL SAVED SUCCESSFULLY!")
print("==========================================")

print(f"Model location: {model_path}")