import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Load the saved model
# rf_model = joblib.load('rf_model.joblib')

try:
    rf_model = joblib.load('rf_model.joblib')
except Exception as e:
    print("Load error:", e)


# Streamlit App
st.title("Random Forest Classifier - Prediction App")

# Example feature inputs (adjust these according to your model's features)
st.sidebar.header("Input Features")


def user_input_features():
    feature_1 = st.sidebar.number_input("Wine Sales ($)", min_value=0.0, value=11.0)
    feature_2 = st.sidebar.number_input("Meat Sales ($)", min_value=0.0, value=6.0)
    feature_3 = st.sidebar.number_input("Premium Product Sales ($)", min_value=0.0, value=6.0)
    feature_4 = st.sidebar.number_input("Sweet Sales ($)", min_value=0.0, value=1.0)
    feature_5 = st.sidebar.number_input("Fish Sales ($)", min_value=0.0, value=2.0)
    data = {
        'wines_sales': feature_1,
        'meatproducts_sales': feature_2,
        'goldprods_sales': feature_3,
        'sweetproducts_sales': feature_4,
        'fishproducts_sales': feature_5
    }
    return pd.DataFrame([data])

input_df = user_input_features()

class_labels = {
    0 : 'Bottom Tier',
    1 : 'Middle Tier',
    2 : 'Upper Tier'
}

# Prediction
if st.button('Predict'):
    prediction = rf_model.predict(input_df)
    prediction_proba = rf_model.predict_proba(input_df)
    predicted_label = class_labels.get(prediction[0])
    
    st.write(f"Predicted Class: {predicted_label}")
    st.write("Prediction Probabilities:")
    # st.write(prediction_proba)

    # Extract probabilities correctly
    probs = prediction_proba[0]
    prob_0 = round(probs[0] * 100, 2)
    prob_1 = round(probs[1] * 100, 2)
    prob_2 = round(probs[2] * 100, 2)
    
    # Plotly Bar Chart for Probabilities
    fig = go.Figure(data=[
        go.Bar(
            x=[class_labels[0], class_labels[1], class_labels[2]],
            y=[prob_0, prob_1, prob_2],
            text=[f"{prob_0}%", f"{prob_1}%", f"{prob_2}%"],
            textposition='auto',
            marker=dict(color=['green', 'red', 'blue'])
        )
    ])
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Customer Tier",
        yaxis_title="Probability (%)",
        template="plotly_white",
        yaxis=dict(range=[0, 100])  # Optional: force y-axis to max 100%
    )
    st.plotly_chart(fig)

