import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the models
lr = pickle.load(open('lr1_model_23_apr.pkl', 'rb'))
dt = pickle.load(open('dt1_model_23_apr.pkl', 'rb'))
rf = pickle.load(open('rf1_model_23_apr.pkl', 'rb'))

# Load the dataframe (without the target variable 'Price')
df = pickle.load(open('df_23_apr.pkl', 'rb')).drop(columns=['Price'])

# Streamlit app
st.title('Laptop Price Prediction')
st.header('Fill the laptop details to predict the price')

# Create input fields for user
Company = st.selectbox('Company', df['Company'].unique())
TypeName = st.selectbox('TypeName', df['TypeName'].unique())
Ram = st.selectbox('Ram', df['Ram'].unique())
Weight = st.number_input('Weight (Between 0.5 - 4.3)')
Touchscreen = st.selectbox('Touchscreen', df['Touchscreen'].unique())

# When the button is clicked, make predictions
if st.button('Predict Price'):
    # Process the input and make predictions
    input_data = pd.DataFrame([[Company, TypeName, Ram, Weight, Touchscreen]], columns=['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen'])
    
    # Ensure the input_data has the same feature names and order as the training data
    input_data_encoded = pd.get_dummies(input_data, columns=['Company', 'TypeName', 'Touchscreen'])
    
    # Align the input_data_encoded columns with the training data columns
    input_data_encoded = input_data_encoded.reindex(columns=df.columns, fill_value=0)
    
    # Make predictions using the models
    try:
        lr_prediction = lr.predict(input_data_encoded)[0]
        dt_prediction = dt.predict(input_data_encoded)[0]
        rf_prediction = rf.predict(input_data_encoded)[0]
        
        st.subheader(f'Linear Regression Prediction: ${lr_prediction:.2f}')
        st.subheader(f'Decision Tree Prediction: ${dt_prediction:.2f}')
        st.subheader(f'Random Forest Prediction: ${rf_prediction:.2f}')
    except ValueError as e:
        st.error(f"Error in prediction: {e}")
