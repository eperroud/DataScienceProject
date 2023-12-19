from joblib import load
import streamlit as st
from data_preprocess import Preprocess
import pandas as pd

# Now you can use MyClass in app.py
logistic_model = load('/Users/elenaperroud/Desktop/DataScienceProject/streamlit/LogRegFull.joblib')

user_input = st.text_input("Enter your sentence here:")
if type(user_input) != str:
    user_input = st.text_input("Please enter a string sentence")

data_input = Preprocess(pd.DataFrame(user_input, columns = ['sentence']))
data_input.data_preprocess

# Button to make predictions
if st.button("Predict"):
    pred_features = ['token_count_no_stop', 'cognate_count', 'nb_nouns', 'nb_adj']
    X_cam = pd.DataFrame(data_input['cam_pooled_embedding'].tolist())
    X = pd.concat([data_input[pred_features], X_cam], axis=1)
    X.columns = X.columns.astype(str)
    prediction = logistic_model.predict(X)
    st.write(f"Predicted Level: {prediction[0]}")

