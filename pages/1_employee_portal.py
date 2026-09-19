import streamlit as st
import pandas as pd
import time

st.set_page_config(page_title="Employee Portal", layout="centered")
st.title("🏛️ Secure Employee Portal")
st.markdown("### Ubuntu Behavioral Integrity Index (UBII)")
st.divider()

if 'start_time' not in st.session_state:
    st.session_state['start_time'] = time.time()

with st.form("ubii_form"):
    st.markdown("#### Part A: Likert Statements")
    
    q1 = st.slider("1. My duty to protect customer data outweighs my duty to help a relative in a financial dispute.", 1, 5, 3) #[cite: 1]
    q25 = st.slider("25. I would immediately report a client who offers me a cash gift to speed up a transaction.", 1, 5, 3) #[cite: 1]
    q49 = st.slider("49. I feel personally responsible for ensuring my branch maintains a flawless audit record.", 1, 5, 3) #[cite: 1]
    q73 = st.slider("73. I am comfortable questioning a branch manager if I believe their instruction violates banking laws.", 1, 5, 3) #[cite: 1]
    q97 = st.slider("97. I hold myself to strict ethical standards, regardless of how the bank treats me.", 1, 5, 3) #[cite: 1]
    
    st.divider()
    st.markdown("#### Part B: Situational Judgment Test (SJT)")
    
    q115 = st.radio(
        "115. Scenario: You find a loophole in the banking software that allows you to waive your own monthly account fees without triggering an alert.", #[cite: 1]
        options=[
            "A) Use the loophole; the bank charges too much in fees anyway.", #[cite: 1]
            "B) Use the loophole only for yourself, not for clients.", #[cite: 1]
            "C) Do not use the loophole, and report the software flaw to the IT department.", #[cite: 1]
            "D) Sell the knowledge of the loophole to other employees." #[cite: 1]
        ], index=None
    )

    submitted = st.form_submit_button("Submit Assessment", type="primary")

if submitted:
    if q115 is None:
        st.error("Please complete the SJT scenario.")
    else:
        # Biometric Keystroke Simulation
        time_taken = time.time() - st.session_state['start_time']
        simulated_keystroke_ms = min(400, max(80, (time_taken * 5) + 50)) 
        
        # 5-Dimension Base Scoring
        raw_likert = (q1 + q25 + q49 + q73 + q97) 
        base_score = (raw_likert / 25) * 100
        
        if "C)" in q115:
            base_score = min(100, base_score + 10)
        else:
            base_score = max(0, base_score - 20)
            
        new_emp_id = f"EMP_LIVE_{int(time.time())}"[-8:]
        
        new_data = pd.DataFrame([{
            "Employee_ID": new_emp_id, "Role": "Live Entry", 
            "CAT_After_Hours_Logins": 0, "RTGS_Zipit_Anomalies": 0, 
            "Keystroke_Variance_ms": simulated_keystroke_ms,
            "UBII_Org_Loyalty": base_score, "UBII_Corruption_Res": base_score, 
            "UBII_Communal_Acc": base_score, "UBII_Ethical_Courage": base_score, 
            "UBII_Moral_Disengage": (100 - base_score),
            "HR_Stressor_Flag": "None", "Actual_Threat": 0,
            "Behavioral_Score": base_score, "Technical_Score": 0.0, 
            "Integrated_Rule_Score": (base_score * 0.6)
        }])
        
        # Live ML Prediction
        features = st.session_state['ml_features']
        rf = st.session_state['rf_model']
        ml_pred = rf.predict_proba(new_data[features])[:, 1] * 100
        new_data['ML_Predictive_Risk'] = ml_pred
        new_data['Risk_Category'] = "MEDIUM" if ml_pred > 30 else "LOW"

        st.session_state['fbih_db'] = pd.concat([st.session_state['fbih_db'], new_data], ignore_index=True)
        
        st.success("✅ **Assessment Securely Submitted.**")
        st.session_state['start_time'] = time.time()
