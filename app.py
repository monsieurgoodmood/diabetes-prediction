import streamlit as st
import xgboost as xgb
import pandas as pd
from util import handle_outliers
import os 
import pickle
# Set page configuration (must be the first Streamlit command)
st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="🩺", layout="wide")

# Load pre-trained model from local file
@st.cache_resource
def load_model():
    model_path = "diabetes_model.pkl"
    data_path = "TAIPEI_diabetes.csv"

    if not os.path.exists(model_path):
        if not os.path.exists(data_path):
            raise FileNotFoundError("Data file not found. Please ensure 'TAIPEI_diabetes.csv' exists in the working directory.")

        # Load and preprocess the data
        data = pd.read_csv(data_path)
        if "PatientID" in data.columns:
            data.drop("PatientID", axis=1, inplace=True)
        data = handle_outliers(data)  # Ensure this function is defined and works correctly

        # Split data into features and target
        X = data.drop(["Diabetic"], axis=1)
        y = data["Diabetic"]

        # Train the model
        model = xgb.XGBClassifier()
        model.fit(X, y)

        # Save the trained model to a file
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        return model
    else:
        # Load the model from the file
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()


st.markdown("""
<style>
    .main-title {
        font-size: 36px;
        font-weight: bold;
        color: #4CAF50;
    }
    .sub-title {
        font-size: 18px;
        color: #555;
    }
    .result-success {
        font-size: 20px;
        color: #4CAF50;
        font-weight: bold;
    }
    .result-error {
        font-size: 20px;
        color: #FF5722;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Welcome to the Diabetes Risk Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">This application predicts the probability of a patient having diabetes based on medical parameters.</div>', unsafe_allow_html=True)

# Input form
with st.form('patient_form'):
    st.header('📝 Patient Information')
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        pregnancies = st.number_input('🤰 Pregnancies', 0, 20, 1)
        plasma_glucose = st.number_input('🩸 Plasma Glucose (mg/dL)', 50, 300, 100)
        diastolic_bp = st.number_input('💓 Diastolic Blood Pressure (mm Hg)', 30, 120, 70)
        triceps_thickness = st.number_input('💪 Triceps Skin Fold Thickness (mm)', 0, 50, 20)
    
    with col2:
        serum_insulin = st.number_input('🧪 Serum Insulin (μU/mL)', 0, 300, 80)
        bmi = st.number_input('⚖️ BMI', 10.0, 50.0, 25.0, step=0.1)
        diabetes_pedigree = st.number_input('📊 Diabetes Pedigree Function', 0.0, 2.5, 0.5, step=0.01)
        age = st.number_input('⏳ Age', 20, 100, 30)
    
    submitted = st.form_submit_button('🔍 Predict Risk')

# Prediction when form is submitted
if submitted:
    # Create input dataframe
    input_data = pd.DataFrame([[pregnancies, plasma_glucose, diastolic_bp, triceps_thickness,
                               serum_insulin, bmi, diabetes_pedigree, age]],
                             columns=['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure',
                                      'TricepsThickness', 'SerumInsulin', 'BMI', 
                                      'DiabetesPedigree', 'Age'])
    input_data = handle_outliers(input_data)  # Ensure this function is defined and works correctly
    # Make prediction
    prediction = model.predict_proba(input_data)[0][1]
    risk_percentage = round(prediction * 100, 2)
    
    # Display results
    st.subheader('📊 Prediction Results')
    
    if risk_percentage < 50:
        st.markdown(f'<div class="result-success">✅ The patient do not have diabetes with a probability of {risk_percentage}%.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-error">⚠️ The patient have diabetes with a probability of {risk_percentage}%.</div>', unsafe_allow_html=True)
    
    # Visual progress bar
    st.progress(int(risk_percentage))
    
    st.markdown("""
    **Note:** This prediction is based on an XGBoost model. Always consult a healthcare professional for medical diagnosis.
    """)

# Sidebar with additional info
st.sidebar.header('ℹ️ About')
st.sidebar.info("""
This application uses an XGBoost model to predict diabetes risk based on patient data.
""")
st.sidebar.markdown("""
Developed by [DSTI Students](https://github.com/MahamadouKEITA/DSTI-Project-Diabeties-Prediction-In-Women).
""")