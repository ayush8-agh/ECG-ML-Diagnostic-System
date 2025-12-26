import streamlit as st
import numpy as np
import joblib

def normalize_label(label):
    if label in ["Sinus Rhythm", "Normal ECG", "Sinus Rhythm / Normal ECG"]:
        return "Normal Sinus Rhythm"
    return label
def patient_explanation(label):
    explanations = {
        "Normal Sinus Rhythm":
            "Your heart rhythm is normal. Electrical signals are working properly and no significant abnormalities were detected.",

        "Sinus Tachycardia":
            "Your heart is beating faster than normal. This can happen due to stress, fever, exercise, or other conditions.",

        "Sinus Bradycardia":
            "Your heart is beating slower than normal. This can be normal in athletes, but should be reviewed if symptoms are present.",

        "Prolonged QT Interval":
            "The electrical recovery of your heart is slower than normal. This may increase the risk of abnormal heart rhythms.",

        "Wide QRS Complex":
            "The electrical signal in your heart is taking longer to travel through the ventricles. Further evaluation is recommended."
    }

    return explanations.get(
        label,
        "This ECG pattern should be reviewed by a healthcare professional."
    )


def risk_level(label):
    if label == "Normal Sinus Rhythm":
        return "Low", "#198754"      # Green
    if label in ["Sinus Tachycardia", "Sinus Bradycardia"]:
        return "Medium", "#fd7e14"   # Orange
    return "High", "#dc3545"         # Red


def safe_label(label):
    """
    Prevent NaN or invalid labels from being shown to user
    """
    if label is None:
        return "Sinus Rhythm / Normal ECG"
    if str(label).lower() in ["nan", "none", ""]:
        return "Sinus Rhythm / Normal ECG"
    return label


# -------------------------------------------------
# Page configuration
def clinical_override(hr, qtc, qrs):
    """
    Clinical rule-based ECG interpretation
    """
    if hr >= 100:
        return "Sinus Tachycardia", 95.0
    if hr <= 50:
        return "Sinus Bradycardia", 95.0
    if qtc >= 480:
        return "Prolonged QT Interval", 95.0
    if qrs >= 150:
        return "Wide QRS Complex", 95.0
    return None, None





# -------------------------------------------------
st.set_page_config(
    page_title="ECG ML Diagnostic System",
    page_icon="❤️",
    layout="wide"
)

# -------------------------------------------------
# Session state initialization
# -------------------------------------------------
if "predicted" not in st.session_state:
    st.session_state.predicted = False

# -------------------------------------------------
# Load model & encoder
# -------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

model, label_encoder = load_artifacts()

# -------------------------------------------------
# Title
# -------------------------------------------------
st.markdown(
    """
    <h1 style="text-align:center; color:#d6336c;">❤️ ECG Machine Learning Diagnostic System</h1>
    <p style="text-align:center; font-size:18px;">
        AI-assisted ECG classification using clinical parameters
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.header("ℹ️ About")
st.sidebar.markdown(
    """
    **Model:** Random Forest Classifier  
    **Data:** Real ECG clinical parameters  
    **Purpose:** Academic & research use  
    """
)

st.sidebar.warning(
    "⚠️ This tool does NOT replace a cardiologist.\n"
    "Always consult a medical professional."
)

# -------------------------------------------------
# Input Section
# -------------------------------------------------
st.subheader("🧾 Enter ECG Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    Age = st.number_input("Age (years)", 1, 120, 45)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    HR = st.number_input("Heart Rate (bpm)", 20, 250, 70)

with col2:
    P_ms = st.number_input("P duration (ms)", 0, 200, 90)
    PR_ms = st.number_input("PR interval (ms)", 0, 300, 160)
    QRS_ms = st.number_input("QRS duration (ms)", 0, 300, 100)

with col3:
    QT_ms = st.number_input("QT (ms)", 200, 600, 400)
    QTc_ms = st.number_input("QTc (ms)", 200, 600, 430)
    P_axis = st.number_input("P axis (°)", -180, 180, 60)

QRS_axis = st.number_input("QRS axis (°)", -180, 180, 50)
T_axis = st.number_input("T axis (°)", -180, 180, 70)
RV5 = st.number_input("RV5 (mV)", 0.0, 5.0, 1.0)
SV1 = st.number_input("SV1 (mV)", 0.0, 5.0, 1.0)

# -------------------------------------------------
# Buttons
# -------------------------------------------------
st.markdown("---")
col_btn1, col_btn2 = st.columns([3, 1])

with col_btn1:
    predict_clicked = st.button("🔍 Predict ECG Condition", use_container_width=True)

with col_btn2:
    reset_clicked = st.button("🔄 Reset", use_container_width=True)

# -------------------------------------------------
# Reset logic
# -------------------------------------------------
if reset_clicked:
    st.session_state.predicted = False

# -------------------------------------------------
# Prediction logic (ONLY on button click)
# -------------------------------------------------
if predict_clicked:
    st.session_state.predicted = True

    gender_encoded = 1 if Gender == "Male" else 0

    input_data = np.array([[
        Age,
        gender_encoded,
        HR,
        P_ms,
        PR_ms,
        QRS_ms,
        QT_ms,
        QTc_ms,
        P_axis,
        QRS_axis,
        T_axis,
        RV5,
        SV1
    ]])

    # ---- ML prediction ----
    ml_pred = model.predict(input_data)
    ml_proba = model.predict_proba(input_data)


    raw_label = label_encoder.inverse_transform(ml_pred)[0]
    ml_label = safe_label(raw_label)


    ml_confidence = float(np.max(ml_proba) * 100)

    # ---- Clinical override ----
    override_label, override_conf = clinical_override(
        HR, QTc_ms, QRS_ms
    )

    if override_label:
        st.session_state.final_label = override_label
        st.session_state.final_confidence = override_conf
        st.session_state.source = "Clinical Rule"
    else:
        st.session_state.final_label = ml_label
        st.session_state.final_confidence = ml_confidence
        st.session_state.source = "Machine Learning"

if st.session_state.predicted:

    final_label = normalize_label(st.session_state.final_label)
    explanation = patient_explanation(final_label)
    risk, risk_color = risk_level(final_label)

    # --- Card background ---
    st.markdown(
        f"""
        <div style="
            background-color:#f1f8ff;
            padding:24px;
            border-radius:14px;
            border-left:8px solid {risk_color};
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        ">
        """,
        unsafe_allow_html=True
    )

    # --- Clean, user-friendly content ---
    st.markdown(f"### 🩺 {final_label}")
    st.write(explanation)

    st.markdown(f"**Confidence:** {st.session_state.final_confidence:.1f}%")

    st.markdown(
        f"**Risk Level:** "
        f"<span style='color:{risk_color}; font-weight:bold;'>{risk}</span>",
        unsafe_allow_html=True
    )

    st.caption(f"Decision Source: {st.session_state.source}")

    # --- Close card ---
    st.markdown("</div>", unsafe_allow_html=True)
