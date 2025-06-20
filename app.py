#importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans
import streamlit as st

#load the model
kmeans = joblib.load("model.pkl")
df = pd.read_csv("Mall_Customers1.csv")
x=df[["Annual Income (k$)","Spending Score (1-100)"]]
x_array = x.values

st.set_page_config(page_title="Customer Cluster Prediction",layout="centered")
st.title("customer cluster prediction")
st.write("enter the customer spending score and annual income to predict the cluster")

#inputs
income = st.number_input("anunnal income of the customer",min_value=0,max_value=400,value=50)
spending_score = st.slider("spending score between 1-100",1,100,20)

#predict the cluster 
if st.button("predict cluster"):
    input_data = np.array([[income,spending_score]])
    cluster = kmeans.predict(input_data)[0]
    st.success(f"Predicted cluster is{cluster}")