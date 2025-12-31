import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load("gb_model.pkl")

selected_features = [
    'trip_distance_haversine',
    'hour',
    'passenger_count',
    'pickup_day',
    'VendorID_1','VendorID_2',
    'RatecodeID_1','RatecodeID_2','RatecodeID_3','RatecodeID_4','RatecodeID_5','RatecodeID_6','RatecodeID_99',
    'payment_type_1','payment_type_2','payment_type_3','payment_type_4'
]

st.title("Taxi Fare Prediction  (Gradient Boosting)")

trip_distance = st.slider("Trip Distance (haversine)", min_value=0.1, max_value=7.0, step=0.1)
hour = st.slider("Hour of Day", min_value=0, max_value=23, step=1)
passenger_count = st.slider("Passenger Count", min_value=1, max_value=6, step=1)
pickup_day = st.slider("Pickup Day (0=Monday ... 6=Sunday)", min_value=0, max_value=6, step=1)

vendor_id = st.selectbox("Vendor ID", options=[1,2])
ratecode_id = st.selectbox("Ratecode ID", options=[1,2,3,4,5,6,99])
payment_type = st.selectbox("Payment Type", options=[1,2,3,4])

user_data = pd.DataFrame({
    'trip_distance_haversine': [trip_distance],   
    'hour':[hour],
    'passenger_count':[passenger_count],
    'pickup_day':[pickup_day],
    'VendorID':[vendor_id],
    'RatecodeID':[ratecode_id],
    'payment_type':[payment_type]
})

vendor_dummies = pd.get_dummies(user_data['VendorID'], prefix='VendorID')
rate_dummies = pd.get_dummies(user_data['RatecodeID'], prefix='RatecodeID')
payment_dummies = pd.get_dummies(user_data['payment_type'], prefix='payment_type')

user_data = pd.concat([
    user_data[['trip_distance_haversine','hour','passenger_count','pickup_day']],
    vendor_dummies, rate_dummies, payment_dummies
], axis=1)

for col in selected_features:
    if col not in user_data.columns:
        user_data[col] = 0
user_data = user_data[selected_features]

prediction = model.predict(user_data)
st.success(f"Predicted Fare: {prediction[0]:.2f}")

