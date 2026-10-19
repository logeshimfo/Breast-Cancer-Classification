import streamlit as st
import pandas as pd
import joblib


# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("breast_cancer_model.pkl")


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🩺",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------
st.title("🩺 Breast Cancer Classification")

st.write(
    "Enter the tumor measurements below to predict whether "
    "the tumor is malignant or benign."
)

st.warning(
    "This application is for educational purposes only and "
    "is not a medical diagnosis tool."
)


# -----------------------------
# Input fields
# -----------------------------

st.subheader("Tumor Measurements")

col1, col2, col3 = st.columns(3)

with col1:

    mean_radius = st.number_input(
        "Mean Radius",
        min_value=0.0,
        value=14.0
    )

    mean_texture = st.number_input(
        "Mean Texture",
        min_value=0.0,
        value=19.0
    )

    mean_perimeter = st.number_input(
        "Mean Perimeter",
        min_value=0.0,
        value=90.0
    )

    mean_area = st.number_input(
        "Mean Area",
        min_value=0.0,
        value=600.0
    )

    mean_smoothness = st.number_input(
        "Mean Smoothness",
        min_value=0.0,
        value=0.1
    )

    mean_compactness = st.number_input(
        "Mean Compactness",
        min_value=0.0,
        value=0.1
    )

    mean_concavity = st.number_input(
        "Mean Concavity",
        min_value=0.0,
        value=0.1
    )

    mean_concave_points = st.number_input(
        "Mean Concave Points",
        min_value=0.0,
        value=0.05
    )

    mean_symmetry = st.number_input(
        "Mean Symmetry",
        min_value=0.0,
        value=0.18
    )

    mean_fractal_dimension = st.number_input(
        "Mean Fractal Dimension",
        min_value=0.0,
        value=0.06
    )


with col2:

    radius_error = st.number_input(
        "Radius Error",
        min_value=0.0,
        value=0.4
    )

    texture_error = st.number_input(
        "Texture Error",
        min_value=0.0,
        value=1.2
    )

    perimeter_error = st.number_input(
        "Perimeter Error",
        min_value=0.0,
        value=2.8
    )

    area_error = st.number_input(
        "Area Error",
        min_value=0.0,
        value=40.0
    )

    smoothness_error = st.number_input(
        "Smoothness Error",
        min_value=0.0,
        value=0.007
    )

    compactness_error = st.number_input(
        "Compactness Error",
        min_value=0.0,
        value=0.02
    )

    concavity_error = st.number_input(
        "Concavity Error",
        min_value=0.0,
        value=0.02
    )

    concave_points_error = st.number_input(
        "Concave Points Error",
        min_value=0.0,
        value=0.01
    )

    symmetry_error = st.number_input(
        "Symmetry Error",
        min_value=0.0,
        value=0.02
    )

    fractal_dimension_error = st.number_input(
        "Fractal Dimension Error",
        min_value=0.0,
        value=0.003
    )


with col3:

    worst_radius = st.number_input(
        "Worst Radius",
        min_value=0.0,
        value=16.0
    )

    worst_texture = st.number_input(
        "Worst Texture",
        min_value=0.0,
        value=25.0
    )

    worst_perimeter = st.number_input(
        "Worst Perimeter",
        min_value=0.0,
        value=105.0
    )

    worst_area = st.number_input(
        "Worst Area",
        min_value=0.0,
        value=800.0
    )

    worst_smoothness = st.number_input(
        "Worst Smoothness",
        min_value=0.0,
        value=0.13
    )

    worst_compactness = st.number_input(
        "Worst Compactness",
        min_value=0.0,
        value=0.25
    )

    worst_concavity = st.number_input(
        "Worst Concavity",
        min_value=0.0,
        value=0.3
    )

    worst_concave_points = st.number_input(
        "Worst Concave Points",
        min_value=0.0,
        value=0.1
    )

    worst_symmetry = st.number_input(
        "Worst Symmetry",
        min_value=0.0,
        value=0.29
    )

    worst_fractal_dimension = st.number_input(
        "Worst Fractal Dimension",
        min_value=0.0,
        value=0.08
    )


# -----------------------------
# Create input DataFrame
# -----------------------------

input_data = pd.DataFrame({
    "mean radius": [mean_radius],
    "mean texture": [mean_texture],
    "mean perimeter": [mean_perimeter],
    "mean area": [mean_area],
    "mean smoothness": [mean_smoothness],
    "mean compactness": [mean_compactness],
    "mean concavity": [mean_concavity],
    "mean concave points": [mean_concave_points],
    "mean symmetry": [mean_symmetry],
    "mean fractal dimension": [mean_fractal_dimension],

    "radius error": [radius_error],
    "texture error": [texture_error],
    "perimeter error": [perimeter_error],
    "area error": [area_error],
    "smoothness error": [smoothness_error],
    "compactness error": [compactness_error],
    "concavity error": [concavity_error],
    "concave points error": [concave_points_error],
    "symmetry error": [symmetry_error],
    "fractal dimension error": [fractal_dimension_error],

    "worst radius": [worst_radius],
    "worst texture": [worst_texture],
    "worst perimeter": [worst_perimeter],
    "worst area": [worst_area],
    "worst smoothness": [worst_smoothness],
    "worst compactness": [worst_compactness],
    "worst concavity": [worst_concavity],
    "worst concave points": [worst_concave_points],
    "worst symmetry": [worst_symmetry],
    "worst fractal dimension": [worst_fractal_dimension]
})


# -----------------------------
# Prediction
# -----------------------------

st.divider()

if st.button("🔍 Predict", use_container_width=True):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    # IMPORTANT:
    # Change these labels if your target encoding is different.
    if prediction == 0:
        result = "Malignant"
        probability_value = probability[0]
    else:
        result = "Benign"
        probability_value = probability[1]

    if result == "Malignant":
        st.error(f"Prediction: {result}")
    else:
        st.success(f"Prediction: {result}")

    st.write(
        f"Model probability for predicted class: "
        f"{probability_value:.2%}"
    )