
import streamlit as st
import pandas as pd
import pickle


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 25px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
    border: 1px solid rgba(128, 128, 128, 0.25);
}

.result-price {
    font-size: 42px;
    font-weight: 700;
}

.info-box {
    padding: 15px;
    border-radius: 10px;
    border: 1px solid rgba(128, 128, 128, 0.25);
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    with open("california_housing_rf.pkl", "rb") as file:
        model = pickle.load(file)

    return model


try:

    model = load_model()

except Exception as e:

    st.error("❌ Unable to load the trained model.")

    st.write(
        "Please make sure california_housing_rf.pkl "
        "is present in the GitHub repository."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🏠 California House Price Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Estimate the median house value using a tuned Random Forest model'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏠 House Details")

st.sidebar.write(
    "Enter the information about the housing area below."
)

st.sidebar.info(
    "💡 Tip: The more accurate your information is, "
    "the more meaningful the prediction will be."
)


# ============================================================
# LOCATION
# ============================================================

st.header("📍 Location")

st.caption(
    "Enter the geographic coordinates of the housing area."
)

col1, col2 = st.columns(2)

with col1:

    longitude = st.number_input(
        "Longitude",
        min_value=-125.0,
        max_value=-114.0,
        value=-119.0,
        step=0.01,
        format="%.2f",
        help="California longitude usually falls between -125 and -114."
    )

with col2:

    latitude = st.number_input(
        "Latitude",
        min_value=32.0,
        max_value=42.0,
        value=35.0,
        step=0.01,
        format="%.2f",
        help="California latitude usually falls between 32 and 42."
    )


# ============================================================
# HOUSING INFORMATION
# ============================================================

st.header("🏡 Housing Information")

st.caption(
    "Provide details about the houses and households in the area."
)

col1, col2, col3 = st.columns(3)

with col1:

    housing_median_age = st.number_input(
        "Median House Age (years)",
        min_value=1,
        max_value=100,
        value=30,
        step=1,
        format="%d",
        help=(
            "Median age of houses in the area. "
            "This is not the age of one individual house."
        )
    )

with col2:

    total_rooms = st.number_input(
        "Total Rooms",
        min_value=1,
        max_value=50000,
        value=2000,
        step=1,
        format="%d",
        help="Approximate total number of rooms in the housing area."
    )

with col3:

    total_bedrooms = st.number_input(
        "Total Bedrooms",
        min_value=1,
        max_value=10000,
        value=400,
        step=1,
        format="%d",
        help="Approximate total number of bedrooms in the housing area."
    )


# ============================================================
# POPULATION INFORMATION
# ============================================================

st.header("👨‍👩‍👧‍👦 Community Information")

col1, col2, col3 = st.columns(3)

with col1:

    population = st.number_input(
        "Population",
        min_value=1,
        max_value=50000,
        value=1000,
        step=1,
        format="%d",
        help="Approximate number of people living in the area."
    )

with col2:

    households = st.number_input(
        "Number of Households",
        min_value=1,
        max_value=10000,
        value=400,
        step=1,
        format="%d",
        help="Approximate number of households in the area."
    )

with col3:

    median_income_dollars = st.number_input(
        "Median Household Income ($)",
        min_value=0,
        max_value=200000,
        value=40000,
        step=1000,
        format="%d",
        help=(
            "Median annual household income. "
            "Enter the actual dollar amount."
        )
    )


# ============================================================
# OCEAN PROXIMITY
# ============================================================

st.header("🌊 Location Type")

ocean_options = {
    "<1H OCEAN": "Less than 1 hour from the ocean",
    "INLAND": "Inland area",
    "ISLAND": "Island",
    "NEAR BAY": "Near a bay",
    "NEAR OCEAN": "Near the ocean"
}

ocean_proximity = st.selectbox(
    "How is the area located relative to the ocean?",
    options=list(ocean_options.keys()),
    format_func=lambda x: ocean_options[x],
    help="Choose the option that best describes the location."
)


# ============================================================
# CONVERT INCOME TO DATASET FORMAT
# ============================================================

# California Housing dataset stores median income
# in units of $10,000.

median_income = median_income_dollars / 10000


# ============================================================
# VALIDATION
# ============================================================

validation_error = False

if total_bedrooms > total_rooms:

    st.warning(
        "⚠️ Total bedrooms cannot be greater than total rooms. "
        "Please check your values."
    )

    validation_error = True


if households > population:

    st.warning(
        "⚠️ Number of households cannot be greater than population. "
        "Please check your values."
    )

    validation_error = True


# ============================================================
# CREATE MODEL INPUT
# ============================================================

input_data = pd.DataFrame({

    "longitude": [longitude],

    "latitude": [latitude],

    "housing_median_age": [housing_median_age],

    "total_rooms": [total_rooms],

    "total_bedrooms": [total_bedrooms],

    "population": [population],

    "households": [households],

    "median_income": [median_income],

    "<1H OCEAN": [
        1 if ocean_proximity == "<1H OCEAN" else 0
    ],

    "INLAND": [
        1 if ocean_proximity == "INLAND" else 0
    ],

    "ISLAND": [
        1 if ocean_proximity == "ISLAND" else 0
    ],

    "NEAR BAY": [
        1 if ocean_proximity == "NEAR BAY" else 0
    ],

    "NEAR OCEAN": [
        1 if ocean_proximity == "NEAR OCEAN" else 0
    ]

})


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict House Value",
    use_container_width=True,
    type="primary"
)


if predict_button:

    if validation_error:

        st.error(
            "❌ Please correct the highlighted input values "
            "before making a prediction."
        )

    else:

        try:

            # Model prediction is in units of $100,000
            prediction = model.predict(input_data)[0]
            st.write("### 🔎 Debug Input")
            st.dataframe(input_data)

            # Convert to actual dollars
            estimated_price = prediction * 100000


            # ====================================================
            # RESULT
            # ====================================================

            st.success("✅ Prediction completed successfully!")

            st.markdown(
                '<div class="result-box">'
                '<h2>🏠 Estimated House Value</h2>'
                f'<div class="result-price">${estimated_price:,.0f}</div>'
                '<p>Estimated median house value</p>'
                '</div>',
                unsafe_allow_html=True
            )


            # ====================================================
            # PREDICTION DETAILS
            # ====================================================

            st.subheader("📊 Prediction Details")

            metric1, metric2, metric3 = st.columns(3)

            with metric1:

                st.metric(
                    "Estimated Value",
                    f"${estimated_price:,.0f}"
                )

            with metric2:

                st.metric(
                    "Model",
                    "Random Forest"
                )

            with metric3:

                st.metric(
                    "R² Score",
                    "0.818"
                )


            # ====================================================
            # INPUT SUMMARY
            # ====================================================

            st.subheader("📋 Your Input")

            summary = pd.DataFrame({

                "Input": [
                    "Longitude",
                    "Latitude",
                    "Median House Age",
                    "Total Rooms",
                    "Total Bedrooms",
                    "Population",
                    "Households",
                    "Median Household Income",
                    "Location Type"
                ],

                "Value": [
                    f"{longitude:.2f}",
                    f"{latitude:.2f}",
                    f"{housing_median_age} years",
                    f"{total_rooms:,}",
                    f"{total_bedrooms:,}",
                    f"{population:,}",
                    f"{households:,}",
                    f"${median_income_dollars:,}",
                    ocean_options[ocean_proximity]
                ]

            })

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )


        except Exception as e:

            st.error(
                "❌ Prediction failed. Please check the model "
                "and input features."
            )

            st.exception(e)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.header("🤖 About This Model")

st.write(
    """
    This application uses a tuned Random Forest Regressor trained
    on the California Housing dataset.

    Several regression algorithms were compared before selecting
    Random Forest as the best-performing model. Hyperparameters
    were optimized using GridSearchCV.
    """
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("📈 Model Performance")

metric1, metric2, metric3 = st.columns(3)

with metric1:

    st.metric(
        "R² Score",
        "0.818"
    )

with metric2:

    st.metric(
        "MAE",
        "0.273"
    )

with metric3:

    st.metric(
        "RMSE",
        "0.423"
    )


st.caption(
    "Higher R² is better. Lower MAE and RMSE are better."
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader("🔍 What Influences the Prediction?")

feature_importance = pd.DataFrame({

    "Feature": [
        "Median Income",
        "Inland",
        "Longitude",
        "Latitude",
        "Housing Median Age",
        "Population",
        "Total Rooms",
        "Total Bedrooms",
        "Households",
        "Near Ocean",
        "<1H Ocean",
        "Near Bay",
        "Island"
    ],

    "Importance": [
        0.499238,
        0.144997,
        0.105563,
        0.101109,
        0.050860,
        0.030244,
        0.021021,
        0.020841,
        0.015844,
        0.006470,
        0.003149,
        0.000621,
        0.000043
    ]

})

st.bar_chart(
    feature_importance.set_index("Feature")
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "California Housing Price Prediction • "
    "Tuned Random Forest • Streamlit"
)

