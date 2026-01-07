import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# ---------------- CACHING ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

@st.cache_resource
def load_artifacts():
    with open("label_encoder_gender.pkl", "rb") as f:
        le_gender = pickle.load(f)
    with open("onehot_encoder_geo.pkl", "rb") as f:
        ohe_geo = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return le_gender, ohe_geo, scaler

model = load_model()
label_encoder_gender, onehot_encoder_geo, scaler = load_artifacts()

# ---------------- UI ----------------
st.title("Customer Churn Prediction")

geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# ---------------- PREDICTION ----------------
if st.button("Predict Churn"):
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [label_encoder_gender.transform([gender])[0]],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    })

    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
    )

    final_input = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)
    final_input_scaled = scaler.transform(final_input)

    prediction = model.predict(final_input_scaled)
    prediction_proba = prediction[0][0]

    st.subheader(f"Churn Probability: {prediction_proba:.2f}")

    if prediction_proba > 0.5:
        st.error("⚠️ The customer is likely to churn.")
    else:
        st.success("✅ The customer is not likely to churn.")
