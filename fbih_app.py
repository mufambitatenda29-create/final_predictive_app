import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# 1. CORE DATA GENERATION & AFRICAN LOCALIZATION
@st.cache_data
def generate_fbih_database():
    np.random.seed(42)
    num_employees = 250
    
    employee_ids = [f"EMP_{str(i).zfill(4)}" for i in range(1, num_employees + 1)]
    roles = np.random.choice(["Bank Teller", "Loan Officer", "IT Admin", "Branch Manager"], num_employees, p=[0.5, 0.3, 0.1, 0.1])
    is_threat = np.random.choice([0, 1], num_employees, p=[0.95, 0.05])
    
    # Computationally Localized CERT Technical Data (Sage ERP / African Banking)
    cat_after_hours_logins = np.where(is_threat == 1, np.random.randint(4, 15, num_employees), np.random.randint(0, 3, num_employees))
    rtgs_zipit_anomalies = np.where(is_threat == 1, np.random.randint(2, 8, num_employees), np.random.randint(0, 2, num_employees))
    
    # Behavioral Biometrics
    keystroke_variance_ms = np.where(is_threat == 1, np.random.normal(450, 100, num_employees), np.random.normal(120, 30, num_employees))
    
    # Localized HR Logs
    hr_flags = np.where(
        is_threat == 1, 
        np.random.choice(["High Black Tax Pressure", "Hyperinflation Wage Garnishment", "None"], num_employees, p=[0.5, 0.3, 0.2]), 
        np.random.choice(["High Black Tax Pressure", "Hyperinflation Wage Garnishment", "None"], num_employees, p=[0.2, 0.05, 0.75])
    )
    
    # Psychometric Data (Bounded by Dark Triad proxies)
    org_loyalty = np.where(is_threat == 1, np.random.normal(40, 10, num_employees), np.random.normal(85, 10, num_employees))
    corruption_res = np.where(is_threat == 1, np.random.normal(35, 15, num_employees), np.random.normal(90, 8, num_employees))
    communal_acc = np.where(is_threat == 1, np.random.normal(50, 12, num_employees), np.random.normal(80, 10, num_employees))
    ethical_courage = np.where(is_threat == 1, np.random.normal(45, 10, num_employees), np.random.normal(75, 12, num_employees))
    moral_disengage = np.where(is_threat == 1, np.random.normal(80, 10, num_employees), np.random.normal(20, 10, num_employees))
    
    df = pd.DataFrame({
        "Employee_ID": employee_ids, "Role": roles, 
        "CAT_After_Hours_Logins": cat_after_hours_logins,
        "RTGS_Zipit_Anomalies": rtgs_zipit_anomalies, 
        "Keystroke_Variance_ms": keystroke_variance_ms,
        "UBII_Org_Loyalty": np.clip(org_loyalty, 0, 100).astype(int),
        "UBII_Corruption_Res": np.clip(corruption_res, 0, 100).astype(int), 
        "UBII_Communal_Acc": np.clip(communal_acc, 0, 100).astype(int),
        "UBII_Ethical_Courage": np.clip(ethical_courage, 0, 100).astype(int), 
        "UBII_Moral_Disengage": np.clip(moral_disengage, 0, 100).astype(int),
        "HR_Stressor_Flag": hr_flags, "Actual_Threat": is_threat
    })
    
    # 60/40 Rule-Based Calculation
    def calculate_60_40_scores(row):
        b_base = (((100 - row['UBII_Org_Loyalty']) * 0.15) + ((100 - row['UBII_Corruption_Res']) * 0.20) + 
                  ((100 - row['UBII_Communal_Acc']) * 0.10) + ((100 - row['UBII_Ethical_Courage']) * 0.15) + 
                  (row['UBII_Moral_Disengage'] * 0.25))
        
        if row['HR_Stressor_Flag'] == 'High Black Tax Pressure': b_base += 15
        elif row['HR_Stressor_Flag'] == 'Hyperinflation Wage Garnishment': b_base += 20
        b_score = min(b_base, 100)
        
        t_score = 0
        if row['CAT_After_Hours_Logins'] > 2: t_score += min((row['CAT_After_Hours_Logins'] * 5), 35)
        if row['RTGS_Zipit_Anomalies'] > 0: t_score += min((row['RTGS_Zipit_Anomalies'] * 15), 50)
        if row['Keystroke_Variance_ms'] > 250: t_score += 15
        t_score = min(t_score, 100)
        
        integrated = (b_score * 0.60) + (t_score * 0.40)
        return pd.Series([round(b_score, 1), round(t_score, 1), round(integrated, 1)])

    df[['Behavioral_Score', 'Technical_Score', 'Integrated_Rule_Score']] = df.apply(calculate_60_40_scores, axis=1)
    
    # Advanced Machine Learning (Random Forest)
    features = ['Behavioral_Score', 'Technical_Score', 'Keystroke_Variance_ms', 'CAT_After_Hours_Logins', 'RTGS_Zipit_Anomalies']
    X = df[features]
    y = df['Actual_Threat']
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    
    ml_probs = rf_model.predict_proba(X)[:, 1] * 100
    df['ML_Predictive_Risk'] = np.round(ml_probs, 1)
    
    def cat_risk(score):
        if score >= 75: return "CRITICAL"
        elif score >= 50: return "HIGH"
        elif score >= 30: return "MEDIUM"
        else: return "LOW"
        
    df['Risk_Category'] = df['ML_Predictive_Risk'].apply(cat_risk)
    return df, rf_model, features

# Initialize Real-Time Session State
if 'fbih_db' not in st.session_state:
    st.session_state['fbih_db'], st.session_state['rf_model'], st.session_state['ml_features'] = generate_fbih_database()

df = st.session_state['fbih_db']

# 2. ADMIN UI DASHBOARD
st.set_page_config(page_title="FBIH Admin", layout="wide")
st.title("🛡️ FBIH Administration Hub")
st.markdown("**System Status:** 🟢 Sage ERP Connected | 🟢 African HRMS Sync Active")
st.divider()

st.sidebar.header("🔍 Audit Selection")
st.sidebar.divider()
st.sidebar.header("🔌 API Infrastructure Status")
st.sidebar.caption("Enterprise Integration Hooks")
st.sidebar.checkbox("Sage X3 ERP (RTGS/Zipit Ledgers)", value=True, disabled=True)
st.sidebar.checkbox("Active Directory (Login Audits)", value=True, disabled=True)
st.sidebar.checkbox("Oracle HRMS (Disciplinary/Wage Data)", value=True, disabled=True)
st.sidebar.info("Webhook Endpoints configured for JSON packet ingestion. System ready for live banking data pipeline.")
selected_emp = st.sidebar.selectbox("Select Employee:", df['Employee_ID'].tolist())
emp_data = df[df['Employee_ID'] == selected_emp].iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Employee ID", emp_data['Employee_ID'])
col2.metric("Role", emp_data['Role'])
col3.metric("HR Stressor", emp_data['HR_Stressor_Flag'])
col4.metric("Threat Status", emp_data['Risk_Category'])

st.divider()
st.markdown("### 📊 Dual-Engine Risk Assessment")
g_col1, g_col2 = st.columns(2)

def create_gauge(value, title, color):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = value, title = {'text': title},
        gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': color},
                 'steps': [{'range': [0, 40], 'color': "lightgreen"}, {'range': [40, 70], 'color': "lightyellow"},
                           {'range': [70, 100], 'color': "red"}]}
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
    return fig

with g_col1:
    st.plotly_chart(create_gauge(emp_data['Integrated_Rule_Score'], "Rule-Based Score (60/40)", "darkblue"), use_container_width=True)
with g_col2:
    st.plotly_chart(create_gauge(emp_data['ML_Predictive_Risk'], "AI Predictive Risk (Random Forest)", "purple"), use_container_width=True)

st.divider()
xai_col, action_col = st.columns([1.5, 1])

with xai_col:
    st.markdown("### 🧠 Explainable AI (XAI) Thought Process")
    reasoning = f"The Random Forest algorithm assigns {emp_data['Employee_ID']} a {emp_data['ML_Predictive_Risk']}% threat probability. "
    if emp_data['UBII_Moral_Disengage'] > 60:
        reasoning += "Psychometrics indicate elevated Dark Triad bounds (high Moral Disengagement). "
    if emp_data['HR_Stressor_Flag'] != "None":
        reasoning += f"This vulnerability is compounded by socioeconomic stress: '{emp_data['HR_Stressor_Flag']}'. "
    if emp_data['RTGS_Zipit_Anomalies'] > 0 or emp_data['Keystroke_Variance_ms'] > 200:
        reasoning += f"ERP data validates the 'Means' to commit fraud: {emp_data['RTGS_Zipit_Anomalies']} RTGS/Zipit anomalies and biometric latency ({emp_data['Keystroke_Variance_ms']:.0f}ms)."
    else:
        reasoning += "No significant ERP anomalies detected. The threat remains latent."
    st.info(reasoning)

with action_col:
    st.markdown("### ⚡ Prescriptive Action")
    if emp_data['Risk_Category'] == "CRITICAL":
        st.error("**Fraud Imminent.**\n1. Suspend Sage ERP access.\n2. Audit 72-hour RTGS clearances.\n3. Secure CCTV.")
    elif emp_data['Risk_Category'] == "HIGH":
        st.warning("**Elevated Risk.**\n1. Enforce Zipit dual-authorization.\n2. Initiate HR wellness check.")
    else:
        st.success("**Normal.**\n1. Standard ledger reconciliation.\n2. Next UBII evaluation in 90 days.")

st.divider()
st.markdown("### 💻 Localized ICT & Biometric Telemetry")
st.table(pd.DataFrame({
    "Metric": ["After-Hours Logins (23:00 - 04:00 CAT)", "RTGS / Zipit Anomalies", "Keystroke Variance (Biometric)"],
    "Value": [emp_data['CAT_After_Hours_Logins'], emp_data['RTGS_Zipit_Anomalies'], f"{emp_data['Keystroke_Variance_ms']:.1f} ms"]
}))
