import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and encoders
model = joblib.load('xgboost_modelNew1.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Define categorical and numeric columns
categorical_cols = ['bloodGroup', 'occupation', 'convTime', 'timesDonated', 'reaction', 'encouragement', 'convLocality']
numeric_cols = [ 'daysSinceLastDonation', 'prefferedFreq', 'rateService', 'yob']

st.title("🩸 Smart Donor Score Predictor")

# User input form
with st.form("donor_form"):
    bloodGroup = st.selectbox("Blood Group", ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'])
    occupation = st.selectbox('Select Occupation', ['Trader', 'Volunteer', 'Doctor', 'Student', 'Nurse'])
    yob = st.number_input('Enter Year of Birth', min_value=1960, max_value=2015, value=2000)
    timesDonated =  st.selectbox('Times Donated', ['1 to 3 times', '4 to 9 times', 'over 10 times'])
    daysSinceLastDonation = st.number_input('Days Since Last Donation', min_value=15, max_value=1050, value=160)
    convTime = st.selectbox("Convenient Time to Donate", ['morning', 'evening', 'anytime'])
    prefferedFreq = st.number_input('Preferred Frequency (Yearly)', min_value=1, max_value=24, value=4)
    rateService = st.slider('Rate Service (1-10)', 1, 10, 5)
    reaction = st.selectbox('Any Reaction Last Time?', ['YES', 'NO'])
    encouragement = st.selectbox("Encouragement Source", ['self', 'friends', 'family', 'events'])
    convLocality = st.selectbox('Convenient Location', ['anywhere', 'mvita', 'kisauni', 'nyali', 'changamwe', 'likoni', 'jomvu', 'mombasa'])

    submit = st.form_submit_button("Predict Donor Score")

#if submit:
    # Compute age
   #age = 2025 - yob

    # Create input DataFrame
    input_data = pd.DataFrame({
        'bloodGroup': [bloodGroup],
        'occupation': [occupation],
        'yob': [yob],
        'timesDonated': [timesDonated],
        'daysSinceLastDonation': [daysSinceLastDonation],
        'convTime': [convTime],
        'prefferedFreq': [prefferedFreq],
        'rateService': [rateService],
        'reaction': [reaction],
        'encouragement': [encouragement],
        'convLocality': [convLocality]
       # 'age': [age]
    })

    # Encode categorical values using saved encoders
    for col in categorical_cols:
        le = label_encoders[col]
        try:
            input_data[col] = le.transform(input_data[col])
        except ValueError:
            st.error(f"❌ Unknown value '{input_data[col][0]}' for column '{col}'. Try a different input.")
            st.stop()

    # Scale numeric values
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    # Predict
    prediction = model.predict(input_data)[0] / 10
    st.success(f"✅ Predicted Donor SCORE: **{prediction:.4f}**")
