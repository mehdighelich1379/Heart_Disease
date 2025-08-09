import streamlit as st
import joblib
import pandas as pd
import random

# ---------- Page config ----------
st.set_page_config(page_title="AI Heart Disease Prediction", page_icon="🫀", layout="wide")

# ---------- Styles ----------
st.markdown("""
<style>
.block-container{padding-top:0.6rem;}
[data-testid="stHeader"]{background: rgba(0,0,0,0);}
.hero{
  background:#f3f4f6;
  padding: 12px 16px;
  border-radius:12px;
  border:1px solid #e5e7eb;
  text-align:center;
  margin-bottom: 12px;
}
.hero h1{margin:0; font-size:1.6rem;}
.hero p{margin:.25rem 0 0 0; color:#4b5563;}
.figure-panel{
  background:#f8fafc;
  padding:10px;
  border-radius:12px;
  border:1px solid #e5e7eb;
  margin-bottom:12px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Language selector ----------
lang = st.sidebar.selectbox("🌐 Select Language / انتخاب زبان", ["فارسی", "English"])
def L(fa, en): return fa if lang == "فارسی" else en

if lang == "فارسی":
    st.markdown("<div dir='rtl'>", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown(f"""
<div class="hero">
  <h1>🫀 {L("پیش‌بینی بیماری قلبی با هوش مصنوعی", "AI Heart Disease Prediction")}</h1>
  <p>{L("این اپ با استفاده از یادگیری ماشین، احتمال ابتلا به بیماری قلبی را پیش‌بینی می‌کند.",
         "This app predicts the probability of heart disease using machine learning.")}</p>
</div>
""", unsafe_allow_html=True)

# ---------- Image panel ----------
st.markdown('<div class="figure-panel">', unsafe_allow_html=True)
st.image('images/heart_image.jpg', use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Load model ----------
model = joblib.load("heart_pipeline_model.pkl")

# ---------- Init defaults ----------
if "inputs" not in st.session_state:
    st.session_state.inputs = {
        'age': 47, 'sex': 0, 'cp': 1, 'trestbps': 122, 'chol': 224,
        'fbs': 0, 'restecg': 1, 'thalach': 168, 'exang': 0,
        'oldpeak': 0.6, 'slope': 2, 'ca': 0, 'thal': 2
    }

# ---------- Random data ----------
if st.sidebar.button(L("🔄 تولید داده تصادفی", "🔄 Generate Random Data")):
    st.session_state.inputs = {
        'age': random.randint(29, 77),
        'sex': random.randint(0, 1),
        'cp': random.randint(0, 3),
        'trestbps': random.randint(90, 180),
        'chol': random.randint(150, 350),
        'fbs': random.randint(0, 1),
        'restecg': random.randint(0, 2),
        'thalach': random.randint(80, 200),
        'exang': random.randint(0, 1),
        'oldpeak': round(random.uniform(0.0, 5.0), 1),
        'slope': random.randint(0, 2),
        'ca': random.randint(0, 4),
        'thal': random.randint(0, 3)
    }

# ---------- Sidebar inputs ----------
st.sidebar.header(L("🔍 جزئیات بیمار را وارد کنید", "Enter Patient Details 🔍"))
for key, label_fa, label_en, min_val, max_val, step in [
    ('age', "سن", "Age", 0, 100, 1),
    ('sex', "جنسیت (0=زن, 1=مرد)", "Sex (0=female, 1=male)", 0, 1, 1),
    ('cp', "نوع درد قفسه سینه (0-3)", "Chest Pain Type (0-3)", 0, 3, 1),
    ('trestbps', "فشار خون استراحت", "Resting BP", 0, 300, 1),
    ('chol', "کلسترول", "Cholesterol", 0, 500, 1),
    ('fbs', "قند خون ناشتا > 120 (0/1)", "FBS > 120 (0/1)", 0, 1, 1),
    ('restecg', "ECG در استراحت (0-2)", "Resting ECG (0-2)", 0, 2, 1),
    ('thalach', "حداکثر ضربان قلب", "Max Heart Rate (thalach)", 0, 200, 1),
    ('exang', "آنژین حین ورزش (0/1)", "Exercise-induced Angina (0/1)", 0, 1, 1),
    ('oldpeak', "افت ST (oldpeak)", "ST Depression (oldpeak)", 0.0, 6.0, 0.1),
    ('slope', "شیب ST (0-2)", "ST Slope (0-2)", 0, 2, 1),
    ('ca', "تعداد رگ‌های مسدود (0-4)", "Number of Major Vessels (0-4)", 0, 4, 1),
    ('thal', "Thal (0-3)", "Thal (0-3)", 0, 3, 1)
]:
    st.session_state.inputs[key] = st.sidebar.number_input(
        L(label_fa, label_en), min_val, max_val,
        value=st.session_state.inputs[key], step=step
    )

# ---------- Predict ----------
user_df = pd.DataFrame([st.session_state.inputs])
prob = 1 - model.predict_proba(user_df)[0][1]

# شرایط ترکیبی
oldpeak = st.session_state.inputs['oldpeak']
thal = st.session_state.inputs['thal']
slope = st.session_state.inputs['slope']
high_risk_combo = (oldpeak > 2 and thal == 1) or (slope == 0 and thal == 1)

# سطح ریسک هماهنگ
if prob > 0.85 or high_risk_combo:
    risk_level = "high"
elif prob > 0.6:
    risk_level = "medium"
else:
    risk_level = "low"

st.sidebar.markdown(L(
    f"🩺 **احتمال ابتلا به بیماری قلبی: `{prob:.2f}`**",
    f"🩺 **Heart Disease Probability: `{prob:.2f}`**"
))

if st.button(L("پیش‌بینی", "Predict")):
    # پیام اصلی
    if risk_level == "high":
        st.error(L("🚨 احتمال بالای بیماری قلبی.", "🚨 High probability of heart disease."))
    elif risk_level == "medium":
        st.warning(L("⚠️ احتمال متوسط بیماری قلبی.", "⚠️ Moderate probability of heart disease."))
    else:
        st.success(L("✅ احتمال پایین بیماری قلبی.", "✅ Low probability of heart disease."))

    # تحلیل دکتر هوشمند
    st.markdown(L("### 🧠 تحلیل دکتر هوشمند", "### 🧠 Doctor AI Analysis"))
    analysis = []

    if st.session_state.inputs['chol'] > 240:
        analysis.append(L("• **کلسترول بالا**؛ خطر رسوب پلاک در عروق.", "• **High cholesterol**; risk of arterial plaque."))
    else:
        analysis.append(L("• کلسترول در **محدوده طبیعی**.", "• Cholesterol within **normal limits**."))

    if st.session_state.inputs['trestbps'] > 140:
        analysis.append(L("• **فشار خون بالا**؛ نیاز به پایش/درمان.", "• **Elevated blood pressure**; consider monitoring/treatment."))
    else:
        analysis.append(L("• فشار خون در **محدوده سالم**.", "• Blood pressure in a **healthy range**."))

    if st.session_state.inputs['exang'] == 1:
        analysis.append(L("• **آنژین ناشی از ورزش**؛ احتمال مشکل قلبی.", "• **Exercise-induced angina**; potential heart issue."))
    else:
        analysis.append(L("• بدون آنژین ورزشی — **نشانه خوب**.", "• No exercise-induced angina — **good sign**."))

    if st.session_state.inputs['thalach'] < 100:
        analysis.append(L("• **حداکثر ضربان قلب پایین**؛ احتمال ظرفیت ورزشی کم.", "• **Low max heart rate**; possibly low exercise capacity."))
    elif st.session_state.inputs['thalach'] > 140:
        analysis.append(L("• **پاسخ قلبی بسیار خوب**.", "• **Excellent heart rate response.**"))
    else:
        analysis.append(L("• **حداکثر ضربان قلب متوسط**؛ بررسی تکمیلی توصیه می‌شود.", "• **Moderate heart rate** — consider further testing."))

    if oldpeak > 2:
        analysis.append(L("• **افت ST بالا (oldpeak>2)**؛ احتمال ایسکمی.", "• **High ST depression (oldpeak>2)**; possible ischemia."))
    else:
        analysis.append(L("• افت ST در **بازه امن**.", "• ST depression within **safe range**."))

    if slope == 0:
        analysis.append(L("• **شیب ST تخت** در ECG؛ می‌تواند غیرطبیعی باشد.", "• **Flat ST slope** on ECG; can be abnormal."))

    if thal == 1:
        analysis.append(L("• **نقص ثابت در Thallium**؛ احتمال آسیب قبلی قلب.", "• **Fixed defect on Thallium scan**; risk of prior heart damage."))

    # جمع‌بندی هماهنگ
    if risk_level == "high":
        st.warning(L("🔴 **جمع‌بندی:** ریسک بالا — مراجعه به متخصص قلب.", "🔴 **Summary:** High risk — cardiologist consultation recommended."))
    elif risk_level == "medium":
        st.info(L("🟠 **جمع‌بندی:** ریسک متوسط — معاینه توصیه می‌شود.", "🟠 **Summary:** Moderate risk — checkup advised."))
    else:
        st.success(L("🟢 **جمع‌بندی:** ریسک پایین — سبک زندگی سالم را ادامه دهید.", "🟢 **Summary:** Low risk — keep a healthy lifestyle."))

    for line in analysis:
        st.markdown(line)

if lang == "فارسی":
    st.markdown("</div>", unsafe_allow_html=True)
