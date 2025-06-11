import streamlit as st
import pickle
import numpy as np

# Load Model & Encoders
with open('car_price_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('brand_encoder.pkl', 'rb') as f:
    le_brand = pickle.load(f)
with open('fuel_encoder.pkl', 'rb') as f:
    le_fuel = pickle.load(f)

# UI Title
st.title("🚗 Car Price Prediction App")

# Input Fields
brand_input = st.selectbox("Select Car Brand", le_brand.classes_)
fuel_input = st.selectbox("Select Fuel Type", le_fuel.classes_)
car_age = st.slider("Car Age (Years)", 0, 30, 5)
km_driven = st.number_input("KM Driven", min_value=0)

# Predict Button
if st.button("Predict Price"):
    brand_encoded = le_brand.transform([brand_input])[0]
    fuel_encoded = le_fuel.transform([fuel_input])[0]
    
    input_data = np.array([[brand_encoded, car_age, km_driven, fuel_encoded]])
    prediction = model.predict(input_data)[0]
    
    st.success(f"Estimated Selling Price: ₹{int(prediction):,}")
