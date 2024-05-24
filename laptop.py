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


