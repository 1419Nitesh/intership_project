import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessor
model = joblib.load("bank_subscription_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.title("🏦 Bank Subscription Prediction App")

st.write("Enter customer details to predict subscription.")

# Input fields
customer_age = st.number_input("Customer Age", 18, 100, 30)
job_type = st.selectbox("Job Type", 
    ['admin.', 'blue-collar', 'technician', 'services', 'management'])

marital_status = st.selectbox("Marital Status", 
    ['married', 'single', 'divorced'])

education_level = st.selectbox("Education Level", 
    ['high school', 'university degree', 'professional course'])

call_duration_sec = st.number_input("Call Duration (seconds)", 0, 5000, 100)
campaign_contacts = st.number_input("Number of Contacts During Campaign", 1, 50, 1)
euribor_3m_rate = st.number_input("Euribor 3 Month Rate", 0.0, 10.0, 4.0)
consumer_confidence_index = st.number_input("Consumer Confidence Index", -100.0, 100.0, -30.0)

# Prediction button
if st.button("Predict"):

    # Create empty dataframe with correct columns
    input_data = pd.DataFrame(columns=preprocessor.feature_names_in_)

    # Add one row
    input_data.loc[0] = None

    # Assign values properly
    input_data['customer_age'] = int(customer_age)
    input_data['job_type'] = str(job_type)
    input_data['marital_status'] = str(marital_status)
    input_data['education_level'] = str(education_level)
    input_data['call_duration_sec'] = float(call_duration_sec)
    input_data['campaign_contacts'] = int(campaign_contacts)
    input_data['euribor_3m_rate'] = float(euribor_3m_rate)
    input_data['consumer_confidence_index'] = float(consumer_confidence_index)

    # Fill remaining columns with safe defaults
    for col in input_data.columns:
        if input_data[col].isnull().all():
            if col in ['credit_default','has_housing_loan','has_personal_loan']:
                input_data[col] = 'no'
            elif col in ['contact_type']:
                input_data[col] = 'cellular'
            elif col in ['last_contact_month']:
                input_data[col] = 'may'
            elif col in ['last_contact_day']:
                input_data[col] = 'mon'
            elif col in ['previous_campaign_outcome']:
                input_data[col] = 'nonexistent'
            else:
                input_data[col] = 0

    processed_data = preprocessor.transform(input_data)

    prediction = model.predict(processed_data)
    probability = model.predict_proba(processed_data)[0][1]

    if prediction[0] == 1:
        st.success(f"✅ Customer likely to Subscribe (Probability: {probability:.2f})")
    else:
        st.error(f"❌ Customer Not likely to Subscribe (Probability: {probability:.2f})")