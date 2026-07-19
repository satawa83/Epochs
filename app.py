import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==========================
# Load trained model
# ==========================
model = joblib.load("model.pkl")
feature_names = joblib.load("feature_names.pkl")

# ==========================
# App Title
# ==========================
st.title("🌾 Crop Yield Prediction and Recommendation System")

st.write(
    """
    Enter the field and soil properties below.
    The model predicts the expected yield for your selected crop
    and recommends alternative crops that may perform better
    under the same conditions.
    """
)

# ==========================
# Crop Names
# ==========================
crop_columns = [
    col for col in feature_names
    if col.startswith("Crop Name_")
]

crop_names = sorted([
    col.replace("Crop Name_", "")
    for col in crop_columns
])

# ==========================
# User Inputs
# ==========================
crop = st.selectbox("Crop Name", crop_names)

field_size = st.number_input(
    "Field Size",
    min_value=0.0,
    value=1.0
)

ph = st.number_input(
    "Soil pH",
    min_value=0.0,
    value=6.5
)

organic_carbon = st.number_input(
    "Organic Carbon",
    min_value=0.0,
    value=2.0
)

nitrogen = st.number_input(
    "Total Nitrogen",
    min_value=0.0,
    value=0.2
)

phosphorus = st.number_input(
    "Phosphorus (M3)",
    min_value=0.0,
    value=15.0
)

potassium = st.number_input(
    "Potassium (exch.)",
    min_value=0.0,
    value=0.5
)

soil_moisture = st.number_input(
    "Soil Moisture",
    min_value=0.0,
    value=30.0
)

# =====================================================
# Prepare input for the trained model
# =====================================================
def prepare_crop_input(
    crop_name,
    field_size,
    ph,
    organic_carbon,
    total_nitrogen,
    phosphorus,
    potassium,
    soil_moisture
):

    input_data = pd.DataFrame(
        np.zeros((1, len(feature_names))),
        columns=feature_names
    )

    input_data.loc[0, "Field Size"] = field_size
    input_data.loc[0, "pH (water)"] = ph
    input_data.loc[0, "Organic Carbon"] = organic_carbon
    input_data.loc[0, "Total Nitrogen"] = total_nitrogen
    input_data.loc[0, "Phosphorus (M3)"] = phosphorus
    input_data.loc[0, "Potassium (exch.)"] = potassium
    input_data.loc[0, "Soil moisture"] = soil_moisture

    crop_column = f"Crop Name_{crop_name}"

    if crop_column in input_data.columns:
        input_data.loc[0, crop_column] = 1
    else:
        return None

    return input_data

# =====================================================
# Predict crop yield
# =====================================================
def predict_crop_yield(
    crop_name,
    field_size,
    ph,
    organic_carbon,
    total_nitrogen,
    phosphorus,
    potassium,
    soil_moisture
):

    input_data = prepare_crop_input(
        crop_name,
        field_size,
        ph,
        organic_carbon,
        total_nitrogen,
        phosphorus,
        potassium,
        soil_moisture
    )

    if input_data is None:
        return None

    # Model predicts log(Target Yield + 1)
    predicted_log = model.predict(input_data)[0]

    # Convert back to original yield
    predicted_yield = np.expm1(predicted_log)

    predicted_yield = max(0, predicted_yield)

    return predicted_yield

# =====================================================
# Predict selected crop and recommend alternatives
# =====================================================
def predict_and_recommend():

    selected_crop_yield = predict_crop_yield(
        crop,
        field_size,
        ph,
        organic_carbon,
        nitrogen,
        phosphorus,
        potassium,
        soil_moisture
    )

    results = []

    for alternative_crop in crop_names:

        predicted = predict_crop_yield(
            alternative_crop,
            field_size,
            ph,
            organic_carbon,
            nitrogen,
            phosphorus,
            potassium,
            soil_moisture
        )

        if predicted is None:
            continue

        results.append({

            "Crop Name": alternative_crop,

            "Predicted Yield (kg)": predicted

        })

    recommendations = pd.DataFrame(results)

    recommendations = recommendations.sort_values(
        by="Predicted Yield (kg)",
        ascending=False
    ).reset_index(drop=True)

    better_alternatives = recommendations[
        (recommendations["Crop Name"] != crop) &
        (recommendations["Predicted Yield (kg)"] > selected_crop_yield)
    ].copy()

    better_alternatives["Expected Increase (kg)"] = (

        better_alternatives["Predicted Yield (kg)"]

        - selected_crop_yield

    )

    better_alternatives["Expected Increase (%)"] = (

        better_alternatives["Expected Increase (kg)"]

        / selected_crop_yield

    ) * 100

    return selected_crop_yield, recommendations, better_alternatives

# =====================================================
# Prediction Button
# =====================================================
if st.button("Predict Yield"):

    selected_yield, recommendations, better_alternatives = predict_and_recommend()

    st.success(
        f"Predicted Yield for {crop}: "
        f"{selected_yield:.2f} kg"
    )

    st.divider()

    if better_alternatives.empty:

        st.info(
            f"{crop} is already the highest-yielding crop "
            "for these soil conditions."
        )

    else:

        st.subheader("🌱 Better Alternative Crops")

        st.dataframe(
            better_alternatives.round(2),
            width="stretch"
        )

    st.divider()

    st.subheader("📊 Predicted Yield for All Crops")

    st.dataframe(
        recommendations.round(2),
        width="stretch"
    )

    best_crop = recommendations.iloc[0]

    st.divider()

    st.success(

        f"🏆 Best Recommendation: **{best_crop['Crop Name']}**\n\n"

        f"Expected Yield: **{best_crop['Predicted Yield (kg)']:.2f} kg**"

    )