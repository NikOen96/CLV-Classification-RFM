import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model
# rf_model = joblib.load('rf_model.joblib')

try:
    model = joblib.load('rf_model.joblib')
except Exception as e:
    print("Load error:", e)


# Streamlit App
st.title("Random Forest Classifier - Prediction App")

# Example feature inputs (adjust these according to your model's features)
st.sidebar.header("Input Features")

def user_input_features():
    feature_1 = st.sidebar.number_input("Feature 1", min_value=0.0, value=1.0)
    feature_2 = st.sidebar.number_input("Feature 2", min_value=0.0, value=2.0)
    feature_3 = st.sidebar.number_input("Feature 3", min_value=0.0, value=3.0)
    feature_4 = st.sidebar.number_input("Feature 4", min_value=0.0, value=4.0)
    data = {
        'feature_1': feature_1,
        'feature_2': feature_2,
        'feature_3': feature_3,
        'feature_4': feature_4
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Prediction
if st.button('Predict'):
    prediction = rf_model.predict(input_df)
    prediction_proba = rf_model.predict_proba(input_df)
    
    st.write(f"Predicted Class: {prediction[0]}")
    st.write("Prediction Probabilities:")
    st.write(prediction_proba)

