# Importing required libraries
import math

import numpy as np
import pandas as pd


# Converting typed text values into numbers
def convert_to_float(
    value: str,
    field_name: str
) -> float:

    if value is None:
        raise ValueError(
            f"{field_name} is required."
        )

    cleaned_value = str(value).strip()

    if cleaned_value == "":
        raise ValueError(
            f"{field_name} is required."
        )

    try:
        numeric_value = float(cleaned_value)

    except ValueError as error:
        raise ValueError(
            f"{field_name} must contain a valid number."
        ) from error

    if not math.isfinite(numeric_value):
        raise ValueError(
            f"{field_name} must be a finite number."
        )

    return numeric_value


# Validating the crop name entered by the user
def validate_crop_name(
    crop_name: str | None,
    available_crops: list[str]
) -> str:

    if crop_name is None:
        raise ValueError(
            "Please search for and select a crop."
        )

    cleaned_crop_name = str(crop_name).strip()

    if cleaned_crop_name == "":
        raise ValueError(
            "Please search for and select a crop."
        )

    if cleaned_crop_name not in available_crops:
        raise ValueError(
            "The selected crop is not supported by the trained model."
        )

    return cleaned_crop_name


# Validating the field size and soil measurements
def validate_numeric_ranges(
    field_size: float,
    soil_ph: float,
    organic_carbon: float,
    total_nitrogen: float,
    phosphorus: float,
    potassium: float,
    soil_moisture: float
) -> None:

    errors = []

    if field_size <= 0:
        errors.append(
            "Field Size must be greater than 0 acres."
        )

    if not 3 <= soil_ph <= 10:
        errors.append(
            "Soil pH must be between 3 and 10."
        )

    if organic_carbon < 0:
        errors.append(
            "Organic Carbon cannot be negative."
        )

    if not 0 <= total_nitrogen <= 5:
        errors.append(
            "Nitrogen must be between 0 and 5%."
        )

    if not 0 <= phosphorus <= 200:
        errors.append(
            "Phosphorus must be between 0 and 200 ppm."
        )

    if not 0 <= potassium <= 10:
        errors.append(
            "Potassium must be between 0 and 10 meq%."
        )

    if not 0 <= soil_moisture <= 100:
        errors.append(
            "Soil Moisture must be between 0 and 100%."
        )

    if errors:
        error_message = (
            "Please correct the following input errors:\n\n"
            + "\n".join(
                f"- {error}"
                for error in errors
            )
        )

        raise ValueError(error_message)


# Converting and validating all user inputs
def parse_and_validate_inputs(
    crop_name: str | None,
    field_size: str,
    soil_ph: str,
    organic_carbon: str,
    total_nitrogen: str,
    phosphorus: str,
    potassium: str,
    soil_moisture: str,
    available_crops: list[str]
) -> dict:

    validated_crop_name = validate_crop_name(
        crop_name=crop_name,
        available_crops=available_crops
    )

    validated_field_size = convert_to_float(
        value=field_size,
        field_name="Field Size"
    )

    validated_soil_ph = convert_to_float(
        value=soil_ph,
        field_name="Soil pH"
    )

    validated_organic_carbon = convert_to_float(
        value=organic_carbon,
        field_name="Organic Carbon"
    )

    validated_total_nitrogen = convert_to_float(
        value=total_nitrogen,
        field_name="Nitrogen"
    )

    validated_phosphorus = convert_to_float(
        value=phosphorus,
        field_name="Phosphorus"
    )

    validated_potassium = convert_to_float(
        value=potassium,
        field_name="Potassium"
    )

    validated_soil_moisture = convert_to_float(
        value=soil_moisture,
        field_name="Soil Moisture"
    )

    validate_numeric_ranges(
        field_size=validated_field_size,
        soil_ph=validated_soil_ph,
        organic_carbon=validated_organic_carbon,
        total_nitrogen=validated_total_nitrogen,
        phosphorus=validated_phosphorus,
        potassium=validated_potassium,
        soil_moisture=validated_soil_moisture
    )

    return {
        "crop_name": validated_crop_name,
        "field_size": validated_field_size,
        "soil_ph": validated_soil_ph,
        "organic_carbon": validated_organic_carbon,
        "total_nitrogen": validated_total_nitrogen,
        "phosphorus": validated_phosphorus,
        "potassium": validated_potassium,
        "soil_moisture": validated_soil_moisture
    }


# Preparing one crop input in the same format used during training
def prepare_crop_input(
    crop_name: str,
    field_size: float,
    soil_ph: float,
    organic_carbon: float,
    total_nitrogen: float,
    phosphorus: float,
    potassium: float,
    soil_moisture: float,
    feature_columns: list[str]
) -> pd.DataFrame:

    if len(feature_columns) == 0:
        raise ValueError(
            "The saved feature column list is empty."
        )

    # Creating one row with zero for every training feature
    input_data = pd.DataFrame(
        np.zeros((1, len(feature_columns))),
        columns=feature_columns
    )

    numerical_values = {
        "Field Size": field_size,
        "pH (water)": soil_ph,
        "Organic Carbon": organic_carbon,
        "Total Nitrogen": total_nitrogen,
        "Phosphorus (M3)": phosphorus,
        "Potassium (exch.)": potassium,
        "Soil moisture": soil_moisture
    }

    # Checking that all required numerical features are available
    missing_numerical_columns = [
        column_name
        for column_name in numerical_values
        if column_name not in feature_columns
    ]

    if missing_numerical_columns:
        raise ValueError(
            "The following required feature columns are missing: "
            + ", ".join(missing_numerical_columns)
        )

    # Adding the farmer's numerical values
    for column_name, value in numerical_values.items():
        input_data.loc[0, column_name] = value

    # Activating the selected crop's one-hot encoded column
    crop_column = f"Crop Name_{crop_name}"

    if crop_column in feature_columns:
        input_data.loc[0, crop_column] = 1

    # Leaving all crop columns at zero for the reference crop that was removed using drop_first=True during encoding

    return input_data[feature_columns]