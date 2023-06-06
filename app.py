# import libraries
import streamlit as st
import pandas as pd
import joblib
from scipy.sparse import hstack

# --------- COLLECT JOBLIB FILES --------- 

# Load preprocessing and model files
preprocessing_steps = joblib.load("C:\Users\aswinram\Aswin's Data Science Portfolio\Credit Card Approval Prediction\models\preprocessing_steps.joblib")

model = joblib.load("C:/Users/aswinram/Aswin's Data Science Portfolio/Credit Card Approval Prediction/models/cca_model.joblib")



# --------- DEFINE PREDICTION FUNCTION --------- 

# Define the predict function
def predict_card_approval(CODE_GENDER, FLAG_EMAIL, FLAG_MOBIL, FLAG_OWN_CAR, FLAG_OWN_REALTY, 
        FLAG_PHONE, FLAG_WORK_PHONE, NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, 
        NAME_INCOME_TYPE, DAYS_BIRTH, DAYS_EMPLOYED):
    input_data = pd.DataFrame({
        "CODE_GENDER": [CODE_GENDER],
        "FLAG_EMAIL": [FLAG_EMAIL],
        "FLAG_MOBIL": [FLAG_MOBIL],
        "FLAG_OWN_CAR": [FLAG_OWN_CAR],
        "FLAG_OWN_REALTY": [FLAG_OWN_REALTY],
        "FLAG_PHONE" : [FLAG_PHONE],
        "FLAG_WORK_PHONE" : [FLAG_WORK_PHONE],
        "NAME_EDUCATION_TYPE" : [NAME_EDUCATION_TYPE],
        "NAME_FAMILY_STATUS" : [NAME_FAMILY_STATUS],
        "NAME_HOUSING_TYPE" : [NAME_HOUSING_TYPE],
        "NAME_INCOME_TYPE" : [NAME_INCOME_TYPE],
        "DAYS_BIRTH" : [DAYS_BIRTH],
        "DAYS_EMPLOYED" : [DAYS_EMPLOYED]
    })
    
    # Apply preprocessing steps to input data
    b_encoder = preprocessing_steps['binary_encoder']
    b_encoder_cols = preprocessing_steps['b_encoder_cols']
    encoder = preprocessing_steps['one_hot_encoder']
    encoder_cols = preprocessing_steps['encoder_cols']
    scaler = preprocessing_steps['scaler']
    numeric_features = preprocessing_steps['numeric_features']
    
    
    # Apply binary encoding
    input_data_b_encoded = b_encoder.transform(input_data[b_encoder_cols])
    
    # Apply one-hot encoding
    input_data_o_encoded = encoder.transform(input_data[encoder_cols])
    
    # Combine the encoded DataFrames
    input_data_encoded = hstack([input_data_b_encoded, input_data_o_encoded])
    
    # Scale the numerical columns
    input_data_scaled = scaler.transform(input_data[numeric_features])
    
    # Update the scaled dataframe by concatenating with encoded dfs
    input_data_scaled = hstack([input_data_encoded , input_data_scaled])
    

    # Make prediction using model
    prediction = model.predict(input_data_scaled)
    
    if prediction[0] == 0:
        return "This application will be denied."
    else:
        return "This application will be approved."

    
    
    
# --------- DESIGN GUI ---------     
    
# Set title
st.title("Credit Card Approval Predictor 💳")

    
# Define the input fields

CODE_GENDER = st.selectbox('Gender', ['M', 'F'], format_func=lambda x: 'Male' if x=='M' else 'Female')

FLAG_EMAIL = st.selectbox('Has Email?', ['0', '1'], format_func=lambda x: 'Yes' if x=='1' else 'No')

FLAG_MOBIL = st.selectbox('Has Mobile?', ['0', '1'], format_func=lambda x: 'Yes' if x=='1' else 'No')

FLAG_OWN_CAR = st.selectbox('Owns a Car?', ['Y', 'N'], format_func=lambda x: 'Yes' if x=='Y' else 'No')

FLAG_OWN_REALTY = st.selectbox('Owns Real Estate?', ['Y', 'N'], format_func=lambda x: 'Yes' if x=='Y' else 'No')

FLAG_PHONE = st.selectbox('Has Phone?', ['0', '1'], format_func=lambda x: 'Yes' if x=='1' else 'No')

FLAG_WORK_PHONE = st.selectbox('Has Work Phone?', ['0', '1'], format_func=lambda x: 'Yes' if x=='1' else 'No')

NAME_EDUCATION_TYPE = st.selectbox('Education Level', ['Higher education', 'Secondary / secondary special', 'Incomplete higher', 'Lower secondary', 'Academic degree'])

NAME_FAMILY_STATUS = st.selectbox('Family Status', ['Civil marriage', 'Married', 'Single / not married', 'Separated', 'Widow'])

NAME_HOUSING_TYPE = st.selectbox('Housing Type', ['Rented apartment', 'House / apartment', 'Municipal apartment', 'With parents', 'Co-op apartment', 'Office apartment'])

NAME_INCOME_TYPE = st.selectbox('Income Type', ['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Student'])

DAYS_BIRTH = st.slider("Age", min_value=18, max_value=100, step=1)

DAYS_EMPLOYED = st.text_input("Indicate Days Employed in (-ve) OR Days Unemployed in (+ve)", value=0)




# Convert the user-friendly inputs to the values that the model uses
DAYS_BIRTH = -(DAYS_BIRTH * 365)



# Define the predict button
if st.button("Predict"):
    prediction_result = predict_card_approval(CODE_GENDER, FLAG_EMAIL, FLAG_MOBIL, FLAG_OWN_CAR, FLAG_OWN_REALTY, 
        FLAG_PHONE, FLAG_WORK_PHONE, NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, 
        NAME_INCOME_TYPE, DAYS_BIRTH, DAYS_EMPLOYED)
    st.write(prediction_result)


