import streamlit as st
import pickle
import json
import numpy as np
import os

st.set_page_config(page_title="Credit Card Compliance Checker", page_icon="💳", layout="centered")

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    with open(os.path.join(BASE_DIR, 'model', 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'model', 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    return model, metadata

model, metadata = load_model()

# UI
st.title("💳 Credit Card Compliance Checker")
st.markdown("Enter credit card product parameters to check RBI regulatory compliance using a trained ML model.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    interest_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=100.0, value=24.0, step=0.1, help="Compliant: 36% or below")
    annual_fee = st.number_input("Annual Fee (Rs)", min_value=0.0, max_value=50000.0, value=3000.0, step=100.0, help="Compliant: Rs 5,000 or below")
    min_payment = st.number_input("Minimum Payment (%)", min_value=1.0, max_value=100.0, value=5.0, step=0.5, help="Compliant: 5% or above")

with col2:
    late_payment_fee = st.number_input("Late Payment Fee (Rs)", min_value=0.0, max_value=10000.0, value=750.0, step=50.0, help="Compliant: Rs 1,000 or below")
    billing_cycle = st.number_input("Billing Cycle (days)", min_value=1, max_value=60, value=30, step=1, help="Compliant: 25 to 31 days")
    disclosure = st.selectbox("Disclosure Provided", ["Yes", "No"], help="Must be Yes for compliance")

st.divider()

if st.button("Run Compliance Check", type="primary", use_container_width=True):

    disc_val = 1 if disclosure == "Yes" else 0
    X = np.array([[interest_rate, late_payment_fee, annual_fee, billing_cycle, min_payment, disc_val]])

    prediction_idx = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = round(float(max(probabilities)) * 100, 1)
    label = metadata['classes'][prediction_idx]
    is_compliant = prediction_idx == 1

    # Result
    st.subheader("Prediction Result")
    col1, col2, col3 = st.columns(3)

    with col1:
        if is_compliant:
            st.success(f"✅ {label}")
        else:
            st.error(f"❌ {label}")

    with col2:
        st.metric("Confidence", f"{confidence}%")

    with col3:
        p_compliant = round(float(probabilities[1]) * 100, 1)
        p_noncompliant = round(float(probabilities[0]) * 100, 1)
        st.metric("P(Compliant)", f"{p_compliant}%")

    st.divider()

    # Rule violations
    violations = []
    if interest_rate > 36:
        violations.append(("Interest Rate", f"{interest_rate}%", "36% or below", "high" if interest_rate > 45 else "medium"))
    if late_payment_fee > 1000:
        violations.append(("Late Payment Fee", f"Rs {late_payment_fee:,.0f}", "Rs 1,000 or below", "high" if late_payment_fee > 1500 else "medium"))
    if annual_fee > 5000:
        violations.append(("Annual Fee", f"Rs {annual_fee:,.0f}", "Rs 5,000 or below", "medium"))
    if billing_cycle < 25 or billing_cycle > 31:
        violations.append(("Billing Cycle", f"{billing_cycle} days", "25 to 31 days", "medium"))
    if min_payment < 5:
        violations.append(("Minimum Payment", f"{min_payment}%", "5% or above", "medium"))
    if disc_val == 0:
        violations.append(("Disclosure", "Not Provided", "Must be Provided", "high"))

    st.subheader(f"Rule-Level Analysis — {len(violations)} violation{'s' if len(violations) != 1 else ''}")

    if len(violations) == 0:
        st.success("✅ All 6 compliance rules are satisfied")
    else:
        for field, actual, expected, severity in violations:
            icon = "🔴" if severity == "high" else "🟡"
            st.warning(f"{icon} **{field}** — Actual: `{actual}` · Expected: `{expected}` · Severity: `{severity}`")

    st.divider()

    # Feature importance
    st.subheader("Feature Importance — why this decision was made")
    fi = metadata['feature_importances']
    labels = {'interest_rate': 'Interest Rate', 'late_payment_fee': 'Late Payment Fee', 'annual_fee': 'Annual Fee', 'billing_cycle': 'Billing Cycle', 'min_payment': 'Minimum Payment', 'disclosure': 'Disclosure'}
    sorted_fi = sorted(fi.items(), key=lambda x: -x[1])
    for feat, imp in sorted_fi:
        st.progress(float(imp), text=f"{labels.get(feat, feat)}: {round(imp*100, 1)}%")

    st.caption(f"Model test accuracy: {round(metadata['accuracy']*100, 1)}%")
