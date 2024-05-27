import streamlit as st
import numpy as np
import pandas as pd
import pickle


lr=pickle.load(open('lr1_model_23_apr.pkl','rb'))
dt=pickle.load(open('dt1_model_23_apr.pkl','rb'))
rf=pickle.load(open('rf1_model_23_apr.pkl','rb'))

df=pickle.load(open('df_23_apr.pkl','rb'))

st.title('Laptop Price Prediction')
st.header('Fill the laptop details to predict the price')

Company=st.selectbox('Company',df['Company'].unique())
TypeName=st.selectbox('TypeName',df['TypeName'].unique())
Ram=st.selectbox('Ram',df['Ram'].unique())
Weight=st.number_input('Weight(Between 0.5 - 4.3)')
Touchscreen=st.selectbox('Touchscreen',df['Touchscreen'].unique())



# When the button is clicked, make predictions
if st.button('Predict Price'):
    # Process the input and make predictions
    input_data = pd.DataFrame([[Company, TypeName, Ram, Weight, Touchscreen]], columns=['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen'])
    
    # Make predictions using the models
    lr_prediction = lr.predict(input_data)[0]
    dt_prediction = dt.predict(input_data)[0]
    rf_prediction = rf.predict(input_data)[0]
    
    st.subheader(f'Linear Regression Prediction: ${lr_prediction:.2f}')
    st.subheader(f'Decision Tree Prediction: ${dt_prediction:.2f}')
    st.subheader(f'Random Forest Prediction: ${rf_prediction:.2f}')
