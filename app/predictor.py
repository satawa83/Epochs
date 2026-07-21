# Importing required libraries
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Importing the input preparation function
from preprocess import prepare_crop_input


# Defining the location of the saved model files
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"

MODEL_PATH = MODELS_DIR / "random_forest.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.pkl"
CROP_LIST_PATH = MODELS_DIR / "crop_list.pkl"


# Confirming that a required model file exists
def confirm_file_exists(file_path: Path) -> None:

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file was not found: {file_path}"
        )


# Loading the trained Random Forest model, feature columns, and crop list
@lru_cache(maxsize=1)
def load_model_files():

    required_files = [
        MODEL_PATH,
        FEATURE_COLUMNS_PATH,
        CROP_LIST_PATH
    ]

    for file_path in required_files:
        confirm_file_exists(file_path)

    random_forest_model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    crop_list = joblib.load(CROP_LIST_PATH)

    feature_columns = list(feature_columns)
    crop_list = sorted(list(crop_list))

    if len(feature_columns) == 0:
        raise ValueError(
            "The saved feature column list is empty."
        )

    if len(crop_list) == 0:
        raise ValueError(
            "The saved crop list is empty."
        )

    return random_forest_model, feature_columns, crop_list


# Returning the crops supported by the trained model
def get_available_crops() -> list[str]:

    _, _, available_crops = load_model_files()

    return available_crops.copy()


# Predicting crop yield for one selected crop
def predict_crop_yield(
    crop_name: str,
    field_size: float,
    soil_ph: float,
    organic_carbon: float,
    total_nitrogen: float,
    phosphorus: float,
    potassium: float,
    soil_moisture: float
) -> float:

    random_forest_model, feature_columns, available_crops = (
        load_model_files()
    )

    if crop_name not in available_crops:
        raise ValueError(
            f"{crop_name} is not supported by the trained model."
        )

    # Preparing the farmer's input in the same format used during training
    input_data = prepare_crop_input(
        crop_name=crop_name,
        field_size=field_size,
        soil_ph=soil_ph,
        organic_carbon=organic_carbon,
        total_nitrogen=total_nitrogen,
        phosphorus=phosphorus,
        potassium=potassium,
        soil_moisture=soil_moisture,
        feature_columns=feature_columns
    )

    # Predicting the log-transformed crop yield
    predicted_log_yield = random_forest_model.predict(
        input_data
    )[0]

    # Converting the prediction back to kilograms
    predicted_yield = np.expm1(predicted_log_yield)

    # Preventing negative crop yield predictions
    predicted_yield = max(0.0, float(predicted_yield))

    return predicted_yield


# Predicting the selected crop and comparing alternative crops
def predict_and_recommend(
    crop_name: str,
    field_size: float,
    soil_ph: float,
    organic_carbon: float,
    total_nitrogen: float,
    phosphorus: float,
    potassium: float,
    soil_moisture: float
) -> dict:

    _, _, available_crops = load_model_files()

    # Predicting the yield of the crop selected by the farmer
    selected_yield = predict_crop_yield(
        crop_name=crop_name,
        field_size=field_size,
        soil_ph=soil_ph,
        organic_carbon=organic_carbon,
        total_nitrogen=total_nitrogen,
        phosphorus=phosphorus,
        potassium=potassium,
        soil_moisture=soil_moisture
    )

    recommendation_rows = []

    # Predicting every other crop under the same conditions
    for alternative_crop in available_crops:

        if alternative_crop == crop_name:
            continue

        alternative_yield = predict_crop_yield(
            crop_name=alternative_crop,
            field_size=field_size,
            soil_ph=soil_ph,
            organic_carbon=organic_carbon,
            total_nitrogen=total_nitrogen,
            phosphorus=phosphorus,
            potassium=potassium,
            soil_moisture=soil_moisture
        )

        expected_increase = (
            alternative_yield - selected_yield
        )

        if selected_yield > 0:
            expected_percentage = (
                expected_increase / selected_yield
            ) * 100
        else:
            expected_percentage = 0.0

        recommendation_rows.append({
            "Crop Name": alternative_crop,
            "Predicted Yield (kg)": alternative_yield,
            "Expected Increase (kg)": expected_increase,
            "Expected Increase (%)": expected_percentage
        })

    # Converting the alternative predictions into a DataFrame
    recommendations = pd.DataFrame(recommendation_rows)

    if recommendations.empty:
        better_alternatives = pd.DataFrame(
            columns=[
                "Crop Name",
                "Predicted Yield (kg)",
                "Expected Increase (kg)",
                "Expected Increase (%)"
            ]
        )

    else:
        # Sorting crops from the highest predicted yield to the lowest
        recommendations = recommendations.sort_values(
            by="Predicted Yield (kg)",
            ascending=False
        ).reset_index(drop=True)

        # Keeping only crops predicted to yield more than the selected crop
        better_alternatives = recommendations[
            recommendations["Predicted Yield (kg)"]
            > selected_yield
        ].copy()

        better_alternatives = (
            better_alternatives.reset_index(drop=True)
        )

    # Returning the results to the Streamlit application
    return {
        "selected_crop": crop_name,
        "selected_yield": selected_yield,
        "recommendations": recommendations,
        "better_alternatives": better_alternatives
    }