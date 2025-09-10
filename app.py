import streamlit as st
import pandas as pd
import joblib
import logging

# Set up logging for monitoring
logging.basicConfig(filename='predictions.log', level=logging.INFO)

# Load model and encoders
model = joblib.load('titanic_model.pkl')
le_sex = joblib.load('le_sex.pkl')
le_embarked = joblib.load('le_embarked.pkl')

# App UI
st.title("Titanic Survival Prediction")
st.write("Input passenger details to predict survival.")

# Input fields
pclass = st.selectbox("Passenger Class", options=[1, 2, 3], format_func=lambda x: f"Class {x}")
sex = st.selectbox("Sex", options=["male", "female"])
age = st.slider("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare Paid (£)", min_value=0.0, max_value=500.0, value=50.0, step=0.1)
embarked = st.selectbox("Port of Embarkation", options=["C", "Q", "S"], format_func=lambda x: {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}[x])

# Predict button
if st.button("Predict"):
    # Prepare input
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    })

    # Preprocess
    input_data['Sex'] = le_sex.transform(input_data['Sex'])
    input_data['Embarked'] = le_embarked.transform(input_data['Embarked'])

    # Predict
    prediction = model.predict(input_data)[0]
    result = "Survived" if prediction == 1 else "Did Not Survive"

    # Display and log
    st.success(f"Prediction: {result}")
    logging.info(f"Input: {input_data.to_dict()}, Prediction: {result}")