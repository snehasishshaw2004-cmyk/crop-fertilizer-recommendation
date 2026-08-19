import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Crop & Fertilizer Recommendation",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    crop_model = joblib.load(
        "models/crop_model.pkl"
    )

    fertilizer_model = joblib.load(
        "models/fertilizer_model.pkl"
    )

    return crop_model, fertilizer_model


# ============================================================
# LOAD FERTILIZER DATASET
# ============================================================

@st.cache_data
def load_fertilizer_data():

    data = pd.read_csv(
        "fertilizer_data/fertilizer_prediction.csv"
    )

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

    return data


# ============================================================
# LOAD EVERYTHING
# ============================================================

crop_model, fertilizer_model = load_models()

fertilizer_data = load_fertilizer_data()


# ============================================================
# HEADER
# ============================================================

st.title("🌱 Crop & Fertilizer Recommendation System")

st.markdown(
    """
    ### Machine Learning Based Agricultural Recommendation

    This application uses Machine Learning to recommend:

    - 🌾 The most suitable crop
    - 🧪 The most suitable fertilizer

    based on soil and environmental conditions.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🌱 About the Project")

st.sidebar.write(
    """
    This system uses:

    **Crop Recommendation**
    - Random Forest Classifier

    **Fertilizer Recommendation**
    - One-Hot Encoding
    - Random Forest Classifier

    The models were trained using agricultural datasets.
    """
)


# ============================================================
# TABS
# ============================================================

crop_tab, fertilizer_tab = st.tabs(
    [
        "🌾 Crop Recommendation",
        "🧪 Fertilizer Recommendation"
    ]
)


# ============================================================
# CROP RECOMMENDATION
# ============================================================

with crop_tab:

    st.header("🌾 Crop Recommendation")

    st.write(
        "Enter the soil and weather conditions to find the most suitable crop."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Soil Nutrients")

        nitrogen = st.number_input(
            "Nitrogen (N)",
            min_value=0.0,
            max_value=200.0,
            value=90.0,
            step=1.0
        )

        phosphorus = st.number_input(
            "Phosphorus (P)",
            min_value=0.0,
            max_value=200.0,
            value=42.0,
            step=1.0
        )

        potassium = st.number_input(
            "Potassium (K)",
            min_value=0.0,
            max_value=200.0,
            value=43.0,
            step=1.0
        )

        ph = st.number_input(
            "Soil pH",
            min_value=0.0,
            max_value=14.0,
            value=6.5,
            step=0.1
        )

    with col2:

        st.subheader("Weather Conditions")

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-10.0,
            max_value=60.0,
            value=25.0,
            step=0.1
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=0.1
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=500.0,
            value=200.0,
            step=1.0
        )

    st.write("")

    crop_button = st.button(
        "🌾 Recommend Crop",
        type="primary",
        use_container_width=True
    )

    if crop_button:

        crop_input = pd.DataFrame(
            [[
                nitrogen,
                phosphorus,
                potassium,
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

        crop_prediction = crop_model.predict(
            crop_input
        )[0]

        st.success(
            f"🌾 Recommended Crop: **{crop_prediction.title()}**"
        )


# ============================================================
# FERTILIZER RECOMMENDATION
# ============================================================

with fertilizer_tab:

    st.header("🧪 Fertilizer Recommendation")

    st.write(
        "Enter the soil, crop and environmental conditions to recommend a fertilizer."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Environmental Conditions")

        fertilizer_temperature = st.number_input(
            "Temperature (°C)",
            min_value=0.0,
            max_value=60.0,
            value=32.0,
            step=0.1,
            key="fert_temperature"
        )

        fertilizer_humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=0.1,
            key="fert_humidity"
        )

        moisture = st.number_input(
            "Soil Moisture",
            min_value=0.0,
            max_value=100.0,
            value=40.0,
            step=1.0
        )

    with col2:

        st.subheader("Soil and Crop Information")

        soil_types = sorted(
            fertilizer_data["Soil Type"]
            .dropna()
            .unique()
            .tolist()
        )

        crop_types = sorted(
            fertilizer_data["Crop Type"]
            .dropna()
            .unique()
            .tolist()
        )

        soil_type = st.selectbox(
            "Soil Type",
            soil_types
        )

        crop_type = st.selectbox(
            "Crop Type",
            crop_types
        )

    st.subheader("Soil Nutrients")

    col1, col2, col3 = st.columns(3)

    with col1:

        fertilizer_nitrogen = st.number_input(
            "Nitrogen",
            min_value=0.0,
            max_value=200.0,
            value=20.0,
            step=1.0
        )

    with col2:

        fertilizer_potassium = st.number_input(
            "Potassium",
            min_value=0.0,
            max_value=200.0,
            value=10.0,
            step=1.0
        )

    with col3:

        fertilizer_phosphorous = st.number_input(
            "Phosphorous",
            min_value=0.0,
            max_value=200.0,
            value=15.0,
            step=1.0
        )

    st.write("")

    fertilizer_button = st.button(
        "🧪 Recommend Fertilizer",
        type="primary",
        use_container_width=True
    )

    if fertilizer_button:

        fertilizer_input = pd.DataFrame(
            [[
                fertilizer_temperature,
                fertilizer_humidity,
                moisture,
                soil_type,
                crop_type,
                fertilizer_nitrogen,
                fertilizer_potassium,
                fertilizer_phosphorous
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

        fertilizer_prediction = fertilizer_model.predict(
            fertilizer_input
        )[0]

        st.success(
            f"🧪 Recommended Fertilizer: **{fertilizer_prediction}**"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Machine Learning Based Crop & Fertilizer Recommendation System"
)