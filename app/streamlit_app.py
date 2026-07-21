# Importing required libraries
import pandas as pd
import streamlit as st

# Importing prediction and validation functions
from predictor import get_available_crops, predict_and_recommend
from preprocess import parse_and_validate_inputs


# Configuring the Streamlit page
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Adding custom styling
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .main-title {
            font-size: 2.7rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            font-size: 1.1rem;
            color: #667067;
            margin-bottom: 1.8rem;
        }

        .hero-card {
            padding: 1.5rem;
            border: 1px solid #dce7dc;
            border-radius: 14px;
            background-color: #f7faf7;
            margin-bottom: 1.5rem;
        }

        .prediction-card {
            padding: 1.4rem;
            border-radius: 14px;
            border: 1px solid #d9e5d9;
            background-color: #f6faf6;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }

        .recommendation-card {
            padding: 1.4rem;
            border-radius: 14px;
            border: 1px solid #add0ad;
            background-color: #ecf7ec;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }

        .warning-card {
            padding: 1.2rem;
            border-radius: 12px;
            border: 1px solid #ead9a5;
            background-color: #fff9e8;
            margin-top: 1rem;
        }

        div[data-testid="stFormSubmitButton"] > button {
            width: 100%;
            min-height: 3rem;
            font-size: 1rem;
            font-weight: 600;
        }

        div[data-testid="stMetric"] {
            border: 1px solid #e1e8e1;
            border-radius: 12px;
            padding: 1rem;
            background-color: #ffffff;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Loading the supported crop list
try:
    available_crops = get_available_crops()

except Exception as error:
    st.error("The crop list could not be loaded.")

    with st.expander("View technical details"):
        st.code(str(error))

    st.stop()


# Displaying the page title and introduction
st.markdown(
    '<div class="main-title">🌱 Crop Yield Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Predict crop yield and discover higher-yielding crop
        alternatives using field size and soil information.
    </div>
    """,
    unsafe_allow_html=True
)


# Explaining how the system works
st.markdown(
    """
    <div class="hero-card">
        <strong>How it works</strong><br><br>
        1. Select the crop you intend to grow.<br>
        2. Enter your field size and soil-test results.<br>
        3. Receive the predicted crop yield.<br>
        4. View higher-yielding crop alternatives under the same conditions.
    </div>
    """,
    unsafe_allow_html=True
)


# Displaying the input guide in the sidebar
with st.sidebar:
    st.header("Input Guide")

    st.markdown(
        """
        Enter values using the following units and ranges:

        - **Field Size:** Greater than 0 acres
        - **Soil pH:** 3–10
        - **Organic Carbon:** g/kg
        - **Nitrogen:** 0–5%
        - **Phosphorus:** 0–200 ppm
        - **Potassium:** 0–10 meq%
        - **Soil Moisture:** 0–100%
        """
    )

    st.divider()

    st.subheader("Model Information")

    st.markdown(
        """
        - **Model:** Random Forest Regressor
        - **R² Score:** Approximately 0.67
        - **Prediction Unit:** Kilograms
        """
    )



# Creating the crop and soil input form
st.subheader("Enter Farm and Soil Information")

with st.form("crop_prediction_form"):
    crop_name = st.selectbox(
        label="Crop to be grown",
        options=available_crops,
        index=None,
        placeholder="Search and select a crop..."
    )

    column_one, column_two = st.columns(2)

    with column_one:
        field_size_input = st.text_input(
            label="Field Size (> 0 acres)",
            placeholder="Example: 2.5"
        )

        soil_ph_input = st.text_input(
            label="Soil pH (3–10)",
            placeholder="Example: 6.5"
        )

        organic_carbon_input = st.text_input(
            label="Organic Carbon (g/kg)",
            placeholder="Example: 25"
        )

        nitrogen_input = st.text_input(
            label="Nitrogen (0–5%)",
            placeholder="Example: 2.4"
        )

    with column_two:
        phosphorus_input = st.text_input(
            label="Phosphorus (0–200 ppm)",
            placeholder="Example: 30"
        )

        potassium_input = st.text_input(
            label="Potassium (0–10 meq%)",
            placeholder="Example: 4.2"
        )

        soil_moisture_input = st.text_input(
            label="Soil Moisture (0–100%)",
            placeholder="Example: 20"
        )

    submitted = st.form_submit_button(
        label="🌾 Predict Yield and Recommend",
        type="primary"
    )


# Running the prediction after the form is submitted
if submitted:
    try:
        validated_inputs = parse_and_validate_inputs(
            crop_name=crop_name,
            field_size=field_size_input,
            soil_ph=soil_ph_input,
            organic_carbon=organic_carbon_input,
            total_nitrogen=nitrogen_input,
            phosphorus=phosphorus_input,
            potassium=potassium_input,
            soil_moisture=soil_moisture_input,
            available_crops=available_crops
        )

        with st.spinner(
            "Predicting crop yield and comparing alternatives..."
        ):
            prediction_result = predict_and_recommend(
                crop_name=validated_inputs["crop_name"],
                field_size=validated_inputs["field_size"],
                soil_ph=validated_inputs["soil_ph"],
                organic_carbon=validated_inputs["organic_carbon"],
                total_nitrogen=validated_inputs["total_nitrogen"],
                phosphorus=validated_inputs["phosphorus"],
                potassium=validated_inputs["potassium"],
                soil_moisture=validated_inputs["soil_moisture"]
            )

        selected_yield = prediction_result["selected_yield"]
        better_alternatives = prediction_result["better_alternatives"]

        st.success("Prediction completed successfully.")
        st.divider()


        # Displaying the selected crop prediction
        st.subheader("Selected Crop Prediction")

        result_column_one, result_column_two, result_column_three = (
            st.columns(3)
        )

        result_column_one.metric(
            label="🌱 Selected Crop",
            value=validated_inputs["crop_name"].title()
        )

        result_column_two.metric(
            label="📏 Field Size",
            value=f'{validated_inputs["field_size"]:,.2f} acres'
        )

        result_column_three.metric(
            label="🌾 Predicted Yield",
            value=f"{selected_yield:,.2f} kg"
        )

        st.markdown(
            f"""
            <div class="prediction-card">
                <strong>{validated_inputs["crop_name"].title()}</strong>
                is predicted to produce approximately
                <strong>{selected_yield:,.2f} kg</strong>
                under the provided field and soil conditions.
            </div>
            """,
            unsafe_allow_html=True
        )


        # Displaying the soil information used
        st.subheader("Soil Information Used")

        soil_information = pd.DataFrame({
            "Soil Feature": [
                "Soil pH",
                "Organic Carbon",
                "Nitrogen",
                "Phosphorus",
                "Potassium",
                "Soil Moisture"
            ],
            "Value": [
                validated_inputs["soil_ph"],
                validated_inputs["organic_carbon"],
                validated_inputs["total_nitrogen"],
                validated_inputs["phosphorus"],
                validated_inputs["potassium"],
                validated_inputs["soil_moisture"]
            ],
            "Unit": [
                "pH value",
                "g/kg",
                "%",
                "ppm",
                "meq%",
                "%"
            ]
        })

        st.dataframe(
            soil_information,
            use_container_width=True,
            hide_index=True
        )

        st.divider()


        # Displaying the recommended crop alternatives
        st.subheader("Crop Recommendation")

        if not better_alternatives.empty:
            best_alternative = better_alternatives.iloc[0]

            recommended_crop = best_alternative["Crop Name"]
            recommended_yield = best_alternative[
                "Predicted Yield (kg)"
            ]
            expected_increase = best_alternative[
                "Expected Increase (kg)"
            ]
            expected_percentage = best_alternative[
                "Expected Increase (%)"
            ]

            recommendation_column_one, recommendation_column_two = (
                st.columns(2)
            )

            recommendation_column_one.metric(
                label="🌱 Recommended Crop",
                value=recommended_crop.title()
            )

            recommendation_column_two.metric(
                label="🌾 Predicted Yield",
                value=f"{recommended_yield:,.2f} kg",
                delta=(
                    f"+{expected_increase:,.2f} kg "
                    f"({expected_percentage:,.1f}%)"
                )
            )

            st.markdown(
            f"""
            <div class="recommendation-card">
            Under the same field size and soil conditions,
            <strong>{recommended_crop.title()}</strong>
            is predicted to produce
            <strong>{recommended_yield:,.2f} kg</strong>.

            This is approximately
            <strong>{expected_increase:,.2f} kg</strong>
            more than
            {validated_inputs["crop_name"].title()}.
            </div>
            """,
                unsafe_allow_html=True
            )

            st.subheader("Top 3 Higher-Yielding Alternatives")

            top_three = better_alternatives.head(3).copy()

            top_three = top_three[[
                "Crop Name",
                "Predicted Yield (kg)",
                "Expected Increase (kg)",
                "Expected Increase (%)"
            ]]

            top_three["Crop Name"] = (
                top_three["Crop Name"].str.title()
            )

            top_three["Predicted Yield (kg)"] = (
                top_three["Predicted Yield (kg)"].round(2)
            )

            top_three["Expected Increase (kg)"] = (
                top_three["Expected Increase (kg)"].round(2)
            )

            top_three["Expected Increase (%)"] = (
                top_three["Expected Increase (%)"].round(1)
            )

            st.dataframe(
                top_three,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info(
                f'{validated_inputs["crop_name"].title()} already '
                "has the highest predicted yield under the provided "
                "field and soil conditions."
            )


        # Displaying the prediction limitation
        st.markdown(
            """
            <div class="warning-card">
                <strong>Important:</strong>
                This recommendation is based on predicted crop yield.
                It does not consider market prices, production costs,
                weather changes, pests, diseases, or farmer preferences.
            </div>
            """,
            unsafe_allow_html=True
        )

    except ValueError as error:
        st.error(str(error))

    except FileNotFoundError as error:
        st.error("A required model file could not be found.")

        with st.expander("View technical details"):
            st.code(str(error))

    except Exception as error:
        st.error(
            "An unexpected error occurred while generating "
            "the prediction."
        )

        with st.expander("View technical details"):
            st.code(str(error))


