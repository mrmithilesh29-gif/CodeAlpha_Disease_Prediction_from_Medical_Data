import streamlit as st
import pandas as pd
import joblib

# Load trained model (replace with your saved model path)
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")  # if you used scaling

st.title("🩺 Diabetes Disease Detection")
st.write("Enter patient details to predict diabetes risk.")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, value=30)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
smoking_history = st.selectbox("Smoking History", ["never", "former", "current", "No Info", "ever"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
HbA1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50, max_value=400, value=120)
cardio_risk = st.selectbox("Cardio Risk (hypertension+heart_disease)", [0, 1, 2])
gender = st.selectbox("Gender", ["Female", "Male", "Other"])

# One-hot encoding for gender
gender_Female = 1 if gender == "Female" else 0
gender_Male = 1 if gender == "Male" else 0
gender_Other = 1 if gender == "Other" else 0

# Encode smoking history (simple mapping)
smoking_map = {"never": 0, "former": 1, "current": 2, "No Info": 3, "ever": 4}
smoking_encoded = smoking_map[smoking_history]

# Create dataframe for prediction
input_data = pd.DataFrame({
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'smoking_history': [smoking_encoded],
    'bmi': [bmi],
    'HbA1c_level': [HbA1c_level],
    'blood_glucose_level': [blood_glucose_level],
    'cardio_risk': [cardio_risk],
    'gender_Female': [gender_Female],
    'gender_Male': [gender_Male],
    'gender_Other': [gender_Other]
})

# Scale numeric features if scaler was used
try:
    input_scaled = scaler.transform(input_data)
except:
    input_scaled = input_data

# Prediction
if st.button("Predict Diabetes"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Patient is predicted to have Diabetes (Risk Probability: {probability:.2f})")
    else:
        st.success(f"✅ Patient is predicted to NOT have Diabetes (Risk Probability: {probability:.2f})")
