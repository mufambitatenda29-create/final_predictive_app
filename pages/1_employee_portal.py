import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(page_title="UBII Assessment Portal", layout="wide")
st.title("🏛️ Enterprise Employee Evaluation Portal")
st.markdown("### Ubuntu Behavioral Integrity Index (UBII) - 120 Item Full Assessment")
st.divider()

if 'start_time' not in st.session_state:
    st.session_state['start_time'] = time.time()

# --- 120-Item Dynamic Rendering Logic ---
dimensions = [
    "Dimension 1: Organizational Loyalty vs. Communal Pressure",
    "Dimension 2: Corruption Resistance",
    "Dimension 3: Communal Accountability",
    "Dimension 4: Ethical Courage",
    "Dimension 5: Moral Disengagement"
]

responses = {}

with st.form("full_ubii_form"):
    st.info("Responses are encrypted and securely transmitted to the Risk & Compliance Hub.")
    
    # Generate the 120-item structure dynamically based on the UBII document
    for dim_idx, dimension in enumerate(dimensions):
        st.subheader(dimension)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Part A: Likert Statements (1-18)**")
            for i in range(1, 19):
                q_num = (dim_idx * 24) + i
                responses[f"q_{q_num}"] = st.slider(f"Q{q_num}: [Statement Placeholder - Refer to UBII pool]", 1, 5, 3, key=f"q_{q_num}")
                
        with col2:
            st.markdown("**Part B: Situational Judgment Tests (19-24)**")
            for i in range(19, 25):
                q_num = (dim_idx * 24) + i
                responses[f"q_{q_num}"] = st.radio(
                    f"Q{q_num}: [SJT Scenario Placeholder]",
                    options=["Option A (Compliance)", "Option B (Passive)", "Option C (Violation)", "Option D (Reporting)"],
                    index=None, key=f"q_{q_num}"
                )
        st.divider()

    submitted = st.form_submit_button("Submit 120-Item Assessment & Sync Telemetry", type="primary")

if submitted:
    # Validation: Check if any SJTs were left blank (which return None)
    if any(val is None for key, val in responses.items() if "SJT" in key or int(key.split('_')[1]) % 24 > 18):
        st.warning("⚠️ Please complete all Situational Judgment Tests before submitting.")
    else:
        with st.spinner("Processing 120 psychometric vectors and synchronizing with Sage X3 ERP..."):
            time.sleep(2.5) # Simulate API latency
            
            # Simulated Biometrics
            time_taken = time.time() - st.session_state['start_time']
            keystroke_variance = min(400, max(80, (time_taken * 1.5) + 50))
            
            # --- API SIMULATION: Fetching Live Technical Data ---
            # Instead of zeros, we simulate querying the bank's servers for this employee's recent activity
            live_cat_logins = int(np.random.choice([0, 1, 2, 5], p=[0.6, 0.2, 0.15, 0.05]))
            live_rtgs_anomalies = int(np.random.choice([0, 1, 3], p=[0.7, 0.2, 0.1]))
            
            # Simulate HRMS Data Fetch (Looking for Black Tax/Garnishment flags)
            live_hr_flag = np.random.choice(["None", "High Black Tax Pressure", "Hyperinflation Wage Garnishment"], p=[0.8, 0.15, 0.05])
            
            # Simulated 120-item psychometric scoring mapping
            base_integrity = np.random.uniform(40, 95) # Placeholder for the 120-item math compilation
            
            new_emp_id = f"EMP_LIVE_{int(time.time())}"[-8:]
            
            # Construct the complete data package
            new_data = pd.DataFrame([{
                "Employee_ID": new_emp_id, "Role": "Active User", 
                "CAT_After_Hours_Logins": live_cat_logins, 
                "RTGS_Zipit_Anomalies": live_rtgs_anomalies, 
                "Keystroke_Variance_ms": keystroke_variance,
                "UBII_Org_Loyalty": base_integrity, "UBII_Corruption_Res": base_integrity, 
                "UBII_Communal_Acc": base_integrity, "UBII_Ethical_Courage": base_integrity, 
                "UBII_Moral_Disengage": (100 - base_integrity),
                "HR_Stressor_Flag": live_hr_flag, "Actual_Threat": 0,
                "Behavioral_Score": base_integrity, "Technical_Score": 0.0, 
                "Integrated_Rule_Score": (base_score * 0.6) if 'base_score' in locals() else (base_integrity * 0.6)
            }])
            
            # Live Machine Learning Injection
            if 'rf_model' in st.session_state and 'ml_features' in st.session_state:
                features = st.session_state['ml_features']
                rf = st.session_state['rf_model']
                
                # Rule-based tech score calc to feed the ML
                t_score = 0
                if live_cat_logins > 2: t_score += min((live_cat_logins * 5), 35)
                if live_rtgs_anomalies > 0: t_score += min((live_rtgs_anomalies * 15), 50)
                if keystroke_variance > 250: t_score += 15
                new_data['Technical_Score'] = t_score
                new_data['Integrated_Rule_Score'] = (base_integrity * 0.6) + (t_score * 0.4)
                
                ml_pred = rf.predict_proba(new_data[features])[:, 1] * 100
                new_data['ML_Predictive_Risk'] = ml_pred
                new_data['Risk_Category'] = "HIGH" if ml_pred > 50 else "MEDIUM" if ml_pred > 30 else "LOW"

            st.session_state['fbih_db'] = pd.concat([st.session_state['fbih_db'], new_data], ignore_index=True)
            
            st.success("✅ **Assessment Securely Logged & ERP Data Synchronized.**")
            st.info(f"System Note: Technical Telemetry (Logins: {live_cat_logins}, Anomalies: {live_rtgs_anomalies}) successfully retrieved via simulated API and attached to {new_emp_id}.")
            st.session_state['start_time'] = time.time()
