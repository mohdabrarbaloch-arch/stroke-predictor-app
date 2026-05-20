import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Stroke Prediction", layout="centered")

@st.cache_resource
def train_model():
    data = pd.read_csv("healthcare-dataset-stroke-data.csv")
    data = data.dropna(subset=["bmi"])
    
    cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    num_cols = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
    
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first"), cat_cols),
    ])
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)),
    ])
    
    X = data[num_cols + cat_cols]
    y = data["stroke"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline.fit(X_train, y_train)
    r2 = pipeline.score(X_test, y_test)
    return pipeline, round(r2, 4)

st.title("🧠 Stroke Prediction App")
st.write("Fill patient details to predict stroke risk")

with st.spinner("Loading model..."):
    model, accuracy = train_model()

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

st.sidebar.divider()
st.sidebar.success(f"Model Accuracy: {accuracy:.4f}")

if st.button("Predict Stroke Risk", use_container_width=True, type="primary"):
    input_df = pd.DataFrame([{
        "gender": gender, "age": age, "hypertension": hypertension,
        "heart_disease": heart_disease, "ever_married": ever_married,
        "work_type": work_type, "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level, "bmi": bmi,
        "smoking_status": smoking_status
    }])
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    if prediction == 1:
        st.error(f"⚠️ High Risk of Stroke ({proba*100:.1f}%)")
    else:
        st.success(f"✅ Low Risk of Stroke ({proba*100:.1f}%)")
