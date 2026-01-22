import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder

# ------------------------------
# Load saved model and scaler
# ------------------------------
model = joblib.load("heart_model1.pkl")
scaler = joblib.load("scaler1.pkl")

st.set_page_config(page_title="❤️ Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.write("This app predicts the likelihood of **heart disease** using machine learning models.")

# ------------------------------
# User Input Form
# ------------------------------
st.header("Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 45)
    sex = st.selectbox("Sex", ("Male", "Female"))
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 200, 120)
    chol = st.number_input("Cholesterol (chol)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", (0, 1))

with col2:
    restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate (thalach)", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina (exang)", (0, 1))
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of ST (slope)", [0, 1, 2])
    ca = st.selectbox("Major Vessels Colored by Fluoroscopy (ca)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# ------------------------------
# Prediction
# ------------------------------
if st.button("🔍 Predict"):
    # Convert inputs to dataframe
    input_data = pd.DataFrame([[
        age, 1 if sex=="Male" else 0, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]], 
    columns=['age','sex','cp','trestbps','chol','fbs','restecg',
             'thalach','exang','oldpeak','slope','ca','thal'])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    # Display results
    if prediction == 1:
        st.error(f"🚨 High Risk of Heart Disease! (Probability: {prob*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Heart Disease (Probability: {prob*100:.2f}%)")

st.write("---")
st.caption("Built with Streamlit • Machine Learning Models: Logistic Regression, Random Forest, XGBoost")
