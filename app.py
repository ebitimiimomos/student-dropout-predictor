import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Student Dropout Risk Predictor")

st.title("Student Dropout Risk Predictor")
st.markdown("Enter a student's profile to predict their risk of withdrawing from their course.")
st.markdown("---")

@st.cache_resource
def load_model():
    with open('dropout_model_light.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["M", "F"])
    age_band = st.selectbox("Age band", ["0-35", "35-55", "55<="])
    disability = st.selectbox("Disability", ["N", "Y"])
    highest_education = st.selectbox("Highest education", [
        "No Formal quals", "Lower Than A Level", "A Level or Equivalent",
        "HE Qualification", "Post Graduate Qualification"])

with col2:
    region = st.selectbox("Region", [
        "London Region", "South East Region", "North Western Region",
        "East Anglian Region", "West Midlands Region", "South Region",
        "North Region", "East Midlands Region", "South West Region",
        "Yorkshire Region", "Wales", "Scotland", "Ireland"])
    imd_band = st.selectbox("Deprivation band (IMD)", [
        "0-10%", "10-20", "20-30%", "30-40%", "40-50%",
        "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"])
    studied_credits = st.slider("Credits studied", 30, 600, 60, step=30)
    num_prev_attempts = st.slider("Previous attempts", 0, 6, 0)

st.markdown("---")

if st.button("Predict dropout risk", use_container_width=True):
    le = LabelEncoder()

    def encode(val, options):
        le.fit(options)
        return le.transform([val])[0]

    features = np.array([[
        encode(gender, ["F", "M"]),
        encode(region, ["East Anglian Region", "East Midlands Region", "Ireland",
                        "London Region", "North Region", "North Western Region",
                        "Scotland", "South East Region", "South Region",
                        "South West Region", "Wales", "West Midlands Region", "Yorkshire Region"]),
        encode(highest_education, ["A Level or Equivalent", "HE Qualification",
                                   "Lower Than A Level", "No Formal quals",
                                   "Post Graduate Qualification"]),
        encode(imd_band, ["0-10%", "10-20", "20-30%", "30-40%", "40-50%",
                          "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]),
        encode(age_band, ["0-35", "35-55", "55<="]),
        num_prev_attempts,
        studied_credits,
        encode(disability, ["N", "Y"])
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    risk_percent = round(probability * 100, 1)

    if risk_percent >= 50:
        st.error(f"⚠️ High dropout risk: {risk_percent}%")
        st.markdown("This student profile shows a high likelihood of withdrawal. Early intervention is recommended.")
    elif risk_percent >= 30:
        st.warning(f"🔶 Moderate dropout risk: {risk_percent}%")
        st.markdown("This student may benefit from additional support and check-ins.")
    else:
        st.success(f"✅ Low dropout risk: {risk_percent}%")
        st.markdown("This student profile shows a low likelihood of withdrawal.")

st.markdown("---")
st.caption("Built by Ebitimi Imomotebegha | MSc Data Science & AI | Based on the Open University Learning Analytics Dataset (OULAD)")
