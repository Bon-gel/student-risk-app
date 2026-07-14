import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Page Title and Styling
st.set_page_config(page_title="Student Academic Risk Monitor", layout="centered")
st.title("🎓 Student Academic Risk Monitor")
st.write("Enter a student's behavioral attributes to assess their failure risk.")

# 2. Global Artifact Loading (Runs once when app boots)
@st.cache_resource
def load_pipeline():
    model = joblib.load('student_risk_classifier.pkl')
    scaler = joblib.load('student_data_scaler.pkl')
    return model, scaler

try:
    model, scaler = load_pipeline()
except Exception as e:
    st.error(f"Error loading model files: {e}. Make sure the .pkl files are in the same directory.")

# 3. Build the Form Interface
# NOTE: Replace the input fields below with the exact features your X DataFrame uses.
with st.form("student_data_form"):
    st.subheader("Behavioral & Academic Profile")
    
    # Example input structures matching common dataset features:
    absences = st.slider("Total School Absences", min_value=0, max_value=93, value=0)
    failures = st.selectbox("Number of Past Class Failures", options=[0, 1, 2, 3, 4], index=0)
    studytime = st.slider("Weekly Study Time (Hours category: 1=low, 4=high)", min_value=1, max_value=4, value=2)
    goout = st.slider("Frequency of Going Out with Friends (1=low, 5=high)", min_value=1, max_value=5, value=3)
    
    # Submit button
    submitted = st.form_submit_button("Run Risk Assessment")

# 4. Processing & Prediction Logic
if submitted:
    # Build a dictionary containing the raw features. 
    # CRITICAL: The dictionary keys MUST match your training DataFrame columns exactly!
    input_data = {
        'absences': absences,
        'failures': failures,
        'studytime': studytime,
        'goout': goout
        # Add the rest of your features here in the exact order/names they were trained on
    }
    
    # Convert input into a single-row DataFrame
    raw_df = pd.DataFrame([input_data])
    
    # Apply standard preprocessing metrics
    scaled_features = scaler.transform(raw_df)
    
    # Extract failure probability (Class 0)
    fail_probability = model.predict_proba(scaled_features)[:, 0][0]
    
    # Display optimized results using your 25% safety threshold
    st.markdown("---")
    st.subheader("AI System Assessment Result:")
    
    if fail_probability >= 0.25:
        st.error(f"⚠️ **ALERT: Preemptive Intervention Suggested** (Failure Risk: {fail_probability*100:.1f}%)")
        st.write("This student crosses the 25.0% threshold boundary and exhibits behavioral metrics correlated with high academic vulnerability.")
    else:
        st.success(f"✅ **Safe: On Track** (Failure Risk: {fail_probability*100:.1f}%)")
        st.write("This student is operating comfortably within safe performance parameters.")