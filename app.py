import streamlit as st
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('diabetes_model.pkl', 'rb'))

# App title
st.title("🧠 AI-Powered Diabetes Risk Predictor")

st.write("Enter your health information to check if you're at risk of diabetes.")

# Input fields
pregnancies = st.number_input('Pregnancies', min_value=0)
glucose = st.number_input('Glucose Level')
bp = st.number_input('Blood Pressure')
skin = st.number_input('Skin Thickness')
insulin = st.number_input('Insulin Level')
bmi = st.number_input('BMI')
dpf = st.number_input('Diabetes Pedigree Function')
age = st.number_input('Age')

# When user clicks the button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)
    result = "🟥 Positive (High Risk)" if prediction[0] == 1 else "🟩 Negative (Low Risk)"
    st.success(f"Prediction Result: {result}")
