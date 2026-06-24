import joblib
import pandas as pd
import numpy as np

# Load model files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")


def predict_mental_health(input_data):
    """
    input_data = pandas DataFrame with 51 features
    """

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)[0]

    return prediction


def get_recommendation(prediction):

    if str(prediction) == "1":

        return """
⚠️ Depression Risk Detected

Recommendations:
• Maintain proper sleep schedule
• Exercise regularly
• Reduce academic stress
• Talk with friends and family
• Consult a counselor if needed
"""

    else:

        return """
✅ No Depression Risk Detected

Recommendations:
• Continue healthy habits
• Maintain study-life balance
• Stay physically active
• Stay socially connected
"""