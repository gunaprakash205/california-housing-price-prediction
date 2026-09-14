import streamlit as st
import pandas as pd
import pickle


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
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
    margin-bottom: 30px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
}

.metric-card {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
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


model = load_model()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🏠 California House Price Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict California housing prices using a tuned Random Forest model'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🏠 House Information")

st.sidebar.write(
    "Enter the property details below to generate a prediction."
)


# ============================================================
# INPUT FEATURES
# ============================================================

col1, col2 = st.columns(2)


# -----------------------------
# Column 1
# -----------------------------

with col1:

    st.subheader("📍 Location")

    longitude = st.number_input(
        "Longitude",
        min_value=-125.0,
        max_value=-114.0,
        value=-119.0,
        step=0.01
    )

    latitude = st.number_input(
        "Latitude",
        min_value=32.0,
        max_value=42.0,
        value=35.0,
        step=0.01
    )

    housing_median_age = st.number_input(
        "Housing Median Age",
        min_value=1.0,
        max_value=100.0,
        value=30.0,
        step=1.0
    )

    median_income = st.number_input(
        "Median Income",
        min_value=0.0,
        max_value=20.0,
        value=4.0,
        step=0.1
    )


# -----------------------------
# Column 2
# -----------------------------

with col2:

    st.subheader("🏡 Property Information")

    total_rooms = st.number_input(
        "Total Rooms",
        min_value=1.0,
        max_value=50000.0,
        value=2000.0,
        step=100.0
    )

    total_bedrooms = st.number_input(
        "Total Bedrooms",
        min_value=1.0,
        max_value=10000.0,
        value=400.0,
        step=10.0
    )

    population = st.number_input(
        "Population",
        min_value=1.0,
        max_value=50000.0,
        value=1000.0,
        step=50.0
    )

    households = st.number_input(
        "Households",
        min_value=1.0,
        max_value=10000.0,
        value=400.0,
        step=10.0
    )


# ============================================================
# OCEAN PROXIMITY
# ============================================================

st.subheader("🌊 Ocean Proximity")

ocean_proximity = st.selectbox(
    "Select location type",
    [
        "<1H OCEAN",
        "INLAND",
        "ISLAND",
        "NEAR BAY",
        "NEAR OCEAN"
    ]
)


# ============================================================
# CREATE INPUT DATAFRAME
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
# MAKE PREDICTION
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict House Value",
    use_container_width=True
)


if predict_button:

    prediction = model.predict(input_data)[0]


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    st.success("Prediction completed successfully!")

    st.markdown(
        '<div class="result-box">'
        '<h2>🏠 Estimated House Value</h2>'
        f'<h1>${prediction:.2f}</h1>'
        '<p>Predicted median house value</p>'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # ADDITIONAL INFORMATION
    # --------------------------------------------------------

    st.subheader("📊 Prediction Details")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "Predicted Value",
            f"${prediction:.2f}"
        )

    with metric2:

        st.metric(
            "Model",
            "Random Forest"
        )

    with metric3:

        st.metric(
            "Test R²",
            "0.818"
        )


    # --------------------------------------------------------
    # INPUT SUMMARY
    # --------------------------------------------------------

    st.subheader("📋 Input Summary")

    display_data = pd.DataFrame({
        "Feature": [
            "Longitude",
            "Latitude",
            "Housing Median Age",
            "Total Rooms",
            "Total Bedrooms",
            "Population",
            "Households",
            "Median Income",
            "Ocean Proximity"
        ],

        "Value": [
            longitude,
            latitude,
            housing_median_age,
            total_rooms,
            total_bedrooms,
            population,
            households,
            median_income,
            ocean_proximity
        ]
    })

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ABOUT MODEL
# ============================================================

st.divider()

st.subheader("🤖 About the Model")

st.write(
    """
    This application uses a tuned Random Forest Regressor trained
    on the California Housing dataset.

    The model was selected after comparing multiple regression
    algorithms and performing hyperparameter tuning using
    GridSearchCV.
    """
)

info1, info2, info3 = st.columns(3)

with info1:

    st.markdown("### 🏆 R²")
    st.write("0.818")

with info2:

    st.markdown("### 📉 MAE")
    st.write("0.273")

with info3:

    st.markdown("### 📊 RMSE")
    st.write("0.423")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader("🔍 Important Features")

feature_importance = pd.DataFrame({

    "Feature": [
        "median_income",
        "INLAND",
        "longitude",
        "latitude",
        "housing_median_age",
        "population",
        "total_rooms",
        "total_bedrooms",
        "households",
        "NEAR OCEAN",
        "<1H OCEAN",
        "NEAR BAY",
        "ISLAND"
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
