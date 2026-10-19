import joblib
import numpy as np


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

MODEL_PATH = "models/breast_cancer_model.pkl"

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_breast_cancer(features):
    """
    Predict whether a breast tumor is malignant or benign.

    Parameters
    ----------
    features : list
        List of 30 breast cancer features.

    Returns
    -------
    prediction : int
        0 = malignant
        1 = benign

    probability : float
        Probability of the predicted class.
    """

    # Convert input to NumPy array
    features = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)[0]

    # Get probabilities
    probabilities = model.predict_proba(features)[0]

    probability = probabilities[prediction]

    return prediction, probability


# --------------------------------------------------
# Test prediction
# --------------------------------------------------

if __name__ == "__main__":

    # Example input using the first test sample
    # This is only for testing the prediction function.

    example_features = [
        17.99,
        10.38,
        122.8,
        1001.0,
        0.1184,
        0.2776,
        0.3001,
        0.1471,
        0.2419,
        0.07871,
        1.095,
        0.9053,
        8.589,
        153.4,
        0.006399,
        0.04904,
        0.05373,
        0.01587,
        0.03003,
        0.006193,
        25.38,
        17.33,
        184.6,
        2019.0,
        0.1622,
        0.6656,
        0.7119,
        0.2654,
        0.4601,
        0.1189
    ]

    prediction, probability = predict_breast_cancer(
        example_features
    )

    print("Prediction:", prediction)
    print("Probability:", probability)