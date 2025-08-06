import streamlit as st
import pickle
import pandas as pd

# Display header image
st.image('images/heart_image.jpg', width=700)

# Load model
with open('models/catboost_model.txt', 'rb') as model_file:
    model = pickle.load(model_file)

# pipeline = joblib.load('./pipeline_heart.pkl')

st.sidebar.header("Enter the details for prediction:🔍")

# Input fields
age = st.sidebar.number_input("Enter your age", min_value=0, max_value=100, value=0)
sex = st.sidebar.number_input("Enter your sex", min_value=0, max_value=1, value=0)
cp = st.sidebar.number_input("Enter your cp", min_value=0, max_value=3, value=0)
trestbps = st.sidebar.number_input("Enter your trestbps", min_value=0, max_value=300, value=0)
chol = st.sidebar.number_input("Enter your chol", min_value=0, max_value=500, value=0)
# fbs = st.sidebar.number_input("Enter your fbs", min_value=0, max_value=1, value=0)
restecg = st.sidebar.number_input("Enter your restecg", min_value=0, max_value=2, value=0)
thalach = st.sidebar.number_input("Enter your thalach", min_value=0, max_value=200, value=0)
exang = st.sidebar.number_input("Enter your exang", min_value=0, max_value=1, value=0)
oldpeak = st.sidebar.number_input("Enter your oldpeak", min_value=0.0, max_value=5.0, value=0.0)
slope = st.sidebar.number_input("Enter your slope", min_value=0, max_value=2, value=0)
ca = st.sidebar.number_input("Enter your ca", min_value=0, max_value=4, value=0)
thal = st.sidebar.number_input("Enter your thal", min_value=0, max_value=3, value=0)

# Create DataFrame for model input
user_input = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    # 'fbs' : [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

# Predict
# prob = pipeline.predict_proba(user_input)[0][1]
prob = model.predict_proba(user_input)[0][1]
st.sidebar.markdown(f"🩺 **Probability of Heart Disease: `{prob:.2f}`**")

# Prediction button
if st.button("Predict"):
    # Main prediction result
    if prob >= 0.5:
        st.error("🚨 The patient has a high probability of having heart disease.")
    else:
        st.success("✅ The patient has a low probability of having heart disease.")

    st.markdown("### 🧠 Doctor AI Analysis")

    # Start analysis
    analysis = []

    if chol > 240:
        analysis.append("• **High cholesterol** detected. Risk of arterial plaque buildup.")
    else:
        analysis.append("• Cholesterol is within **normal limits**.")

    if trestbps > 140:
        analysis.append("• **Elevated blood pressure.** Consider monitoring or treatment.")
    else:
        analysis.append("• Blood pressure is in a **healthy range**.")

    if exang == 1:
        analysis.append("• **Exercise-induced angina present** — potential heart issue.")
    else:
        analysis.append("• No angina during exercise — **good sign**.")

    if thalach < 100:
        analysis.append("• **Low max heart rate** — may reflect low exercise capacity.")
    elif thalach > 140:
        analysis.append("• **Excellent heart rate response.**")
    else:
        analysis.append("• **Moderate heart rate** — consider further testing.")

    if oldpeak > 2:
        analysis.append("• **High ST depression (oldpeak > 2)** — sign of possible ischemia.")
    else:
        analysis.append("• ST depression is within **safe range**.")

    if slope == 0:
        analysis.append("• **Flat slope in ECG** — may indicate abnormal heart function.")

    if thal == 1:
        analysis.append("• **Fixed defect in Thallium scan** — risk of past heart damage.")

    # Check for high-risk combination
    high_risk_combo = (oldpeak > 2 and thal == 1) or (slope == 0 and thal == 1)
    if high_risk_combo:
        analysis.append("⚠️ **High-risk combination detected. Recommend cardiologist consultation.**")

    # Final AI summary
    if prob > 0.85 or high_risk_combo:
        st.warning("🔴 **Doctor AI Summary:** High risk — recommend medical attention.")
    elif prob > 0.6:
        st.info("🟠 **Doctor AI Summary:** Moderate to high risk — schedule medical checkup.")
    else:
        st.success("🟢 **Doctor AI Summary:** Low risk — keep a healthy lifestyle.")

    # Show analysis
    for line in analysis:
        st.markdown(line)


