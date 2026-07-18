import streamlit as st
import pandas as pd
import joblib

# Load saved objects
model = joblib.load("model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("🌾 Crop Yield Prediction")

# Crop dropdown
crop_columns = [
    col for col in feature_names
    if col.startswith("Crop Name_")
]

crop_names = [
    col.replace("Crop Name_", "")
    for col in crop_columns
]

crop = st.selectbox("Crop Name", sorted(crop_names))

# Numerical inputs
field_size = st.number_input("Field Size", min_value=0.0)
ph = st.number_input("Soil pH", min_value=0.0)
organic_carbon = st.number_input("Organic Carbon", min_value=0.0)
nitrogen = st.number_input("Total Nitrogen", min_value=0.0)
phosphorus = st.number_input("Phosphorus (M3)", min_value=0.0)
potassium = st.number_input("Potassium (exch.)", min_value=0.0)
soil_moisture = st.number_input("Soil Moisture", min_value=0.0)

# Prediction
if st.button("Predict Yield"):

    # Create an input row with all features initialized to 0
    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=feature_names
    )

    # Fill numerical features
    input_data.loc[0, "Field Size"] = field_size
    input_data.loc[0, "pH (water)"] = ph
    input_data.loc[0, "Organic Carbon"] = organic_carbon
    input_data.loc[0, "Total Nitrogen"] = nitrogen
    input_data.loc[0, "Phosphorus (M3)"] = phosphorus
    input_data.loc[0, "Potassium (exch.)"] = potassium
    input_data.loc[0, "Soil moisture"] = soil_moisture

    # One-hot encode the selected crop
    crop_column = f"Crop Name_{crop}"

    if crop_column in input_data.columns:
        input_data.loc[0, crop_column] = 1

    # Predict
    prediction = model.predict(input_data)

    st.success(f"Predicted Crop Yield: {prediction[0]:.2f}")