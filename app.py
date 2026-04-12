import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Student Dropout Risk Predictor",
    layout="wide"
)

st.markdown("""
    <style>
        body { background-color: #ffffff; }
        .main { background-color: #ffffff; }
        
        .hero {
            background-color: #0a0a0a;
            padding: 3rem 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
        }
        .hero h1 {
            color: #ffffff;
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .hero p {
            color: #aaaaaa;
            font-size: 1rem;
            margin: 0;
        }
        .hero span {
            color: #4FC3F7;
        }

        .stat-box {
            background-color: #f5f5f5;
            border-left: 4px solid #0a0a0a;
            padding: 1rem 1.2rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .stat-box h2 {
            font-size: 1.8rem;
            font-weight: 700;
            color: #0a0a0a;
            margin: 0;
        }
        .stat-box p {
            font-size: 0.85rem;
            color: #666;
            margin: 0;
        }

        .section-label {
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #999;
            margin-bottom: 1rem;
            margin-top: 1.5rem;
        }

        .result-high {
            background-color: #fff0f0;
            border-left: 5px solid #e53935;
            padding: 1.5rem;
            border-radius: 8px;
        }
        .result-mid {
            background-color: #fff8e1;
            border-left: 5px solid #f9a825;
            padding: 1.5rem;
            border-radius: 8px;
        }
        .result-low {
            background-color: #f0fff4;
            border-left: 5px solid #2e7d32;
            padding: 1.5rem;
            border-radius: 8px;
        }
        .result-high h2, .result-mid h2, .result-low h2 {
            font-size: 2rem;
            font-weight: 700;
            margin: 0 0 0.3rem 0;
        }
        .result-high h2 { color: #e53935; }
        .result-mid h2 { color: #f9a825; }
        .result-low h2 { color: #2e7d32; }
        .result-high p, .result-mid p, .result-low p {
            color: #444;
            margin: 0;
            font-size: 0.95rem;
        }

        .divider {
            border: none;
            border-top: 1px solid #eeeeee;
            margin: 2rem 0;
        }

        .footer-note {
            font-size: 0.8rem;
            color: #aaa;
            text-align: center;
            margin-top: 3rem;
        }
        .footer-note a {
            color: #4FC3F7;
            text-decoration: none;
        }

        div[data-testid="stSelectbox"] label,
        div[data-testid="stSlider"] label {
            font-weight: 600;
            color: #0a0a0a;
            font-size: 0.9rem;
        }

        div[data-testid="stButton"] button {
            background-color: #0a0a0a;
            color: white;
            font-weight: 600;
            padding: 0.6rem 2rem;
            border-radius: 8px;
            border: none;
            width: 100%;
            font-size: 1rem;
        }
        div[data-testid="stButton"] button:hover {
            background-color: #333;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🎓 Student Dropout Risk Predictor</h1>
    <p>Built by <span>Ebitimi Imomotebegha</span> · MSc Data Science & AI · 
    Using the Open University Learning Analytics Dataset (32,593 students)</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="stat-box">
        <h2>31.2%</h2>
        <p>Overall withdrawal rate in dataset</p>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="stat-box">
        <h2>70.1%</h2>
        <p>Model accuracy on test data</p>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="stat-box">
        <h2>8</h2>
        <p>Features used to predict risk</p>
    </div>""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<p class="section-label">Enter student profile</p>', unsafe_allow_html=True)

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

st.markdown('<hr class="divider">', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open('dropout_model_light.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

if st.button("Predict dropout risk"):
    le = LabelEncoder()

    def encode(val, options):
        le.fit(options)
        return le.transform([val])[0]

    features = np.array([[
        encode(gender, ["F", "M"]),
        encode(region, ["East Anglian Region", "East Midlands Region", "Ireland",
                        "London Region", "North Region", "North Western Region",
                        "Scotland", "South East Region", "South Region",
                        "South West Region", "Wales", "West Midlands Region",
                        "Yorkshire Region"]),
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

    probability = model.predict_proba(features)[0][1]
    risk_percent = round(probability * 100, 1)

    st.markdown('<p class="section-label">Result</p>', unsafe_allow_html=True)

    if risk_percent >= 50:
        st.markdown(f"""
        <div class="result-high">
            <h2>⚠ {risk_percent}% dropout risk</h2>
            <p>High risk profile. This student is likely to withdraw without early intervention and targeted support.</p>
        </div>""", unsafe_allow_html=True)
    elif risk_percent >= 30:
        st.markdown(f"""
        <div class="result-mid">
            <h2>◆ {risk_percent}% dropout risk</h2>
            <p>Moderate risk profile. Additional check-ins and support could make a meaningful difference.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-low">
            <h2>✓ {risk_percent}% dropout risk</h2>
            <p>Low risk profile. This student is likely to complete their course successfully.</p>
        </div>""", unsafe_allow_html=True)

st.markdown("""
<p class="footer-note">
    Built by <a href="https://www.linkedin.com/in/ebitimi-imomotebegha-5a06b019a/" target="_blank">Ebitimi Imomotebegha</a> · 
    <a href="https://github.com/ebitimiimomos/student-dropout-predictor" target="_blank">GitHub</a> · 
    MSc Data Science & AI, University of Liverpool
</p>
""", unsafe_allow_html=True)
