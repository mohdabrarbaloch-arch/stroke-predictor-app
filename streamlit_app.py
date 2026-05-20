import streamlit as st
import joblib
import pandas as pd

model = joblib.load("stroke_prediction_pipeline.pkl")

st.set_page_config(page_title="Stroke Prediction", layout="centered")

st.title("🧠 Stroke Prediction App")
st.write("Fill patient details to predict stroke risk")

st.sidebar.header("Patient Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=50)
hypertension = st.sidebar.selectbox("Hypertension", [0, 1])
heart_disease = st.sidebar.selectbox("Heart Disease", [0, 1])
ever_married = st.sidebar.selectbox("Ever Married", ["Yes", "No"])
work_type = st.sidebar.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
residence_type = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.sidebar.number_input("Average Glucose Level", value=100.0)
bmi = st.sidebar.number_input("BMI", value=25.0)
smoking_status = st.sidebar.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

if st.button("Predict Stroke Risk", use_container_width=True, type="primary"):
    input_df = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }])

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Stroke")
    else:
        st.success("✅ Low Risk of Stroke")
