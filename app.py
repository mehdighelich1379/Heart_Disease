import streamlit as st
import pickle
import joblib
import pandas as pd

# ---------- Page config ----------
st.set_page_config(page_title="AI Heart Disease Prediction", page_icon="🫀", layout="wide")

# ---------- Styles (gray header bar) ----------
st.markdown("""
<style>
.block-container{padding-top:0.6rem;}
[data-testid="stHeader"]{background: rgba(0,0,0,0);}
.hero{
  background:#f3f4f6; /* light gray */
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

# ---------- Language selector (in sidebar, before inputs) ----------
lang = st.sidebar.selectbox("🌐 Select Language / انتخاب زبان", ["فارسی", "English"])

def L(fa, en):
    return fa if lang == "فارسی" else en

# Optional RTL tweak for Persian body (keeps layout tidy)
if lang == "فارسی":
    st.markdown("<div dir='rtl'>", unsafe_allow_html=True)

# ---------- Header (gray background, language-aware) ----------
st.markdown(f"""
<div class="hero">
  <h1>🫀 {L("پیش‌بینی بیماری قلبی با هوش مصنوعی", "AI Heart Disease Prediction")}</h1>
  <p>{L("این اپ با استفاده از یادگیری ماشین، احتمال ابتلا به بیماری قلبی را پیش‌بینی می‌کند.",
         "This app predicts the probability of heart disease using machine learning.")}</p>
</div>
""", unsafe_allow_html=True)

# ---------- (Optional) Top image with soft background ----------
st.markdown('<div class="figure-panel">', unsafe_allow_html=True)
st.image('images/heart_image.jpg', use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Load model ----------
model = joblib.load("heart_pipeline_model.pkl")


# ---------- Sidebar inputs ----------
st.sidebar.header(L("🔍 جزئیات بیمار را وارد کنید", "Enter Patient Details 🔍"))

age      = st.sidebar.number_input(L("سن", "Age"), 0, 100, 0)
sex      = st.sidebar.number_input(L("جنسیت (0=زن, 1=مرد)", "Sex (0=female, 1=male)"), 0, 1, 0)
cp       = st.sidebar.number_input(L("نوع درد قفسه سینه (0-3)", "Chest Pain Type (0-3)"), 0, 3, 0)
trestbps = st.sidebar.number_input(L("فشار خون استراحت", "Resting BP"), 0, 300, 0)
chol     = st.sidebar.number_input(L("کلسترول", "Cholesterol"), 0, 500, 0)
fbs      = st.sidebar.number_input(L("قند خون ناشتا > 120 (0/1)", "FBS > 120 (0/1)"), 0, 1, 0)
restecg  = st.sidebar.number_input(L("ECG در استراحت (0-2)", "Resting ECG (0-2)"), 0, 2, 0)
thalach  = st.sidebar.number_input(L("حداکثر ضربان قلب", "Max Heart Rate (thalach)"), 0, 200, 0)
exang    = st.sidebar.number_input(L("آنژین حین ورزش (0/1)", "Exercise-induced Angina (0/1)"), 0, 1, 0)
oldpeak  = st.sidebar.number_input(L("افت ST (oldpeak)", "ST Depression (oldpeak)"), 0.0, 6.0, 0.0, step=0.1)
slope    = st.sidebar.number_input(L("شیب ST (0-2)", "ST Slope (0-2)"), 0, 2, 0)
ca       = st.sidebar.number_input(L("تعداد رگ‌های مسدود (0-4)", "Number of Major Vessels (0-4)"), 0, 4, 0)
thal     = st.sidebar.number_input(L("Thal (0-3)", "Thal (0-3)"), 0, 3, 0)

# ---------- Pack input for model ----------
user_df = pd.DataFrame([{
    'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
    'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
    'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
}])

# ---------- Predict probability ----------
prob = 1 - model.predict_proba(user_df)[0][1]
st.sidebar.markdown(L(
    f"🩺 **احتمال ابتلا به بیماری قلبی: `{prob:.2f}`**",
    f"🩺 **Heart Disease Probability: `{prob:.2f}`**"
))

# ---------- Predict button + Doctor AI (detailed) ----------
if st.button(L("پیش‌بینی", "Predict")):
    # Main result banner
    if prob >= 0.5:
        st.error(L("🚨 احتمال بالای بیماری قلبی وجود دارد.", "🚨 High probability of heart disease."))
    else:
        st.success(L("✅ احتمال پایین بیماری قلبی.", "✅ Low probability of heart disease."))

    st.markdown(L("### 🧠 تحلیل دکتر هوشمند", "### 🧠 Doctor AI Analysis"))

    analysis = []

    # Cholesterol
    if chol > 240:
        analysis.append(L("• **کلسترول بالا**؛ خطر رسوب پلاک در عروق.", "• **High cholesterol**; risk of arterial plaque."))
    else:
        analysis.append(L("• کلسترول در **محدوده طبیعی**.", "• Cholesterol within **normal limits**."))

    # Blood Pressure
    if trestbps > 140:
        analysis.append(L("• **فشار خون بالا**؛ نیاز به پایش/درمان.", "• **Elevated blood pressure**; consider monitoring/treatment."))
    else:
        analysis.append(L("• فشار خون در **محدوده سالم**.", "• Blood pressure in a **healthy range**."))

    # Exercise-induced angina
    if exang == 1:
        analysis.append(L("• **آنژین ناشی از ورزش**؛ احتمال مشکل قلبی.", "• **Exercise-induced angina**; potential heart issue."))
    else:
        analysis.append(L("• بدون آنژین ورزشی — **نشانه خوب**.", "• No exercise-induced angina — **good sign**."))

    # Max heart rate
    if thalach < 100:
        analysis.append(L("• **حداکثر ضربان قلب پایین**؛ احتمال ظرفیت ورزشی کم.", "• **Low max heart rate**; possibly low exercise capacity."))
    elif thalach > 140:
        analysis.append(L("• **پاسخ قلبی بسیار خوب**.", "• **Excellent heart rate response.**"))
    else:
        analysis.append(L("• **حداکثر ضربان قلب متوسط**؛ بررسی تکمیلی توصیه می‌شود.", "• **Moderate heart rate** — consider further testing."))

    # ST depression
    if oldpeak > 2:
        analysis.append(L("• **افت ST بالا (oldpeak>2)**؛ احتمال ایسکمی.", "• **High ST depression (oldpeak>2)**; possible ischemia."))
    else:
        analysis.append(L("• افت ST در **بازه امن**.", "• ST depression within **safe range**."))

    # Slope
    if slope == 0:
        analysis.append(L("• **شیب ST تخت** در ECG؛ می‌تواند غیرطبیعی باشد.", "• **Flat ST slope** on ECG; can be abnormal."))

    # Thal
    if thal == 1:
        analysis.append(L("• **نقص ثابت در Thallium**؛ احتمال آسیب قبلی قلب.", "• **Fixed defect on Thallium scan**; risk of prior heart damage."))

    # High-risk combinations
    high_risk_combo = (oldpeak > 2 and thal == 1) or (slope == 0 and thal == 1)

    # Summary
    if prob > 0.85 or high_risk_combo:
        st.warning(L("🔴 **جمع‌بندی:** ریسک بالا — مراجعه به متخصص قلب.", "🔴 **Summary:** High risk — cardiologist consultation recommended."))
    elif prob > 0.6:
        st.info(L("🟠 **جمع‌بندی:** ریسک متوسط تا بالا — معاینه را زمان‌بندی کنید.", "🟠 **Summary:** Moderate–high risk — schedule a checkup."))
    else:
        st.success(L("🟢 **جمع‌بندی:** ریسک پایین — سبک زندگی سالم را ادامه دهید.", "🟢 **Summary:** Low risk — keep a healthy lifestyle."))

    # Render bullet lines
    for line in analysis:
        st.markdown(line)

# Close RTL wrapper for Persian
if lang == "فارسی":
    st.markdown("</div>", unsafe_allow_html=True)
