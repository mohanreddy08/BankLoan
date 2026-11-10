import os
import json
import pandas as pd
import streamlit as st
from predict import predict_loan, COLUMNS

# Configure Streamlit page
st.set_page_config(
    page_title="Bank Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    .approved-badge {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: bold;
        display: inline-block;
    }
    .rejected-badge {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def load_results():
    if os.path.exists("results.json"):
        with open("results.json", "r") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_tests():
    if os.path.exists("tests_results.json"):
        with open("tests_results.json", "r") as f:
            return json.load(f)
    return {}

# Sidebar Navigation
st.sidebar.title("🏦 Bank Loan System")
page = st.sidebar.radio(
    "Navigation",
    ["Loan Approval Predictor", "Batch Evaluation", "Model Performance & Charts", "System Architecture & Docs"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Core Model:** Random Forest Classifier (200 trees)\n\n"
    "**Key Metrics:** Accuracy ~87.5% | F1-Score 0.90 | ROC-AUC 0.93\n\n"
    "**Currency:** All amounts in INR Lakhs (₹100,000s)"
)

# ----------------------------------------------------
# PAGE 1: LOAN APPROVAL PREDICTOR
# ----------------------------------------------------
if page == "Loan Approval Predictor":
    st.markdown('<div class="main-header">🏦 Loan Eligibility & Approval Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluate applicant creditworthiness and instant approval probability using our trained ML model.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📋 Applicant Information")
        with st.form("loan_form"):
            income = st.number_input(
                "Annual Applicant Income (in ₹ Lakhs)",
                min_value=0.5,
                max_value=150.0,
                value=8.5,
                step=0.5,
                help="Total gross annual income in Lakhs (e.g., 8.5 means ₹8,50,000/year)"
            )
            
            cibil = st.slider(
                "CIBIL Credit Score (300 - 900)",
                min_value=300,
                max_value=900,
                value=720,
                step=5,
                help="Credit Bureau score. <650: Risky | 650-749: Fair | 750+: Prime"
            )
            
            # CIBIL indicator tag
            if cibil >= 750:
                cibil_badge = "🟢 Excellent (Low Default Risk)"
            elif cibil >= 680:
                cibil_badge = "🟡 Good / Acceptable"
            elif cibil >= 600:
                cibil_badge = "🟠 Fair / Borderline"
            else:
                cibil_badge = "🔴 Poor / High Risk"
            st.caption(f"Score Rating: **{cibil_badge}**")
            
            loan_amount = st.number_input(
                "Requested Loan Amount (in ₹ Lakhs)",
                min_value=0.5,
                max_value=250.0,
                value=25.0,
                step=1.0,
                help="Desired total loan principal in Lakhs"
            )
            
            employment = st.selectbox(
                "Employment Status",
                ["Salaried", "Self-Employed", "Unemployed"],
                index=0
            )
            
            submit_btn = st.form_submit_button("🔍 Predict Loan Decision", use_container_width=True)
    
    with col2:
        st.subheader("📊 Decision & Risk Analysis")
        
        lti_ratio = round(loan_amount / income, 2)
        
        if submit_btn:
            try:
                status, prob = predict_loan(income, cibil, loan_amount, employment)
                
                # Result Card
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if status == "Approved":
                    st.markdown(f'<div class="approved-badge">✅ {status.upper()}</div>', unsafe_allow_html=True)
                    st.success(f"Loan is approved with **{prob*100:.1f}% confidence**.")
                else:
                    st.markdown(f'<div class="rejected-badge">❌ {status.upper()}</div>', unsafe_allow_html=True)
                    st.error(f"Loan is rejected with only **{prob*100:.1f}% approval probability**.")
                
                st.write("")
                st.write("**Approval Probability Gauge:**")
                st.progress(prob)
                st.caption(f"Calculated Probability: `{prob:.4f}` | Approval Threshold: `0.5000`")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Financial Health Breakdown
                st.markdown("### 🔍 Risk Factors Breakdown")
                
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Loan-to-Income", f"{lti_ratio}x", delta="-High Risk" if lti_ratio > 4.5 else "Healthy")
                with m2:
                    st.metric("Credit Score", f"{cibil}", delta=f"{cibil - 680:+d} vs Median")
                with m3:
                    st.metric("Employment", employment)
                
                # Actionable Recommendations
                st.markdown("#### 💡 Underwriting Insights:")
                insights = []
                if cibil < 680:
                    insights.append("⚠️ **Low CIBIL Score**: Credit score is below the institutional benchmark (680). Improving credit history will greatly enhance approval odds.")
                if lti_ratio > 4.0:
                    insights.append(f"⚠️ **High Loan-to-Income Ratio ({lti_ratio}x)**: The requested loan is more than 4x annual income. A higher down-payment or lower principal request is recommended.")
                if employment == "Unemployed":
                    insights.append("⚠️ **Employment Risk**: Unemployed applicants exhibit significantly lower repayment certainty unless backed by a salaried co-applicant or collateral.")
                if not insights and status == "Approved":
                    insights.append("✨ **Strong Profile**: Stable income-to-loan multiplier, excellent credit score, and low statistical risk of default.")
                
                for ins in insights:
                    st.markdown(f"- {ins}")
                    
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.info("Fill out applicant details on the left and click **Predict Loan Decision** to generate an automated assessment.")
            
            st.markdown("#### Example Benchmark Profiles:")
            demo_data = pd.DataFrame([
                {"Income (₹L)": 12.0, "CIBIL": 780, "Loan (₹L)": 20.0, "Employment": "Salaried", "Expected": "Approved (97%)"},
                {"Income (₹L)": 8.0, "CIBIL": 650, "Loan (₹L)": 25.0, "Employment": "Salaried", "Expected": "Rejected (26%)"},
                {"Income (₹L)": 4.0, "CIBIL": 540, "Loan (₹L)": 30.0, "Employment": "Self-Employed", "Expected": "Rejected (5%)"},
                {"Income (₹L)": 15.0, "CIBIL": 820, "Loan (₹L)": 40.0, "Employment": "Self-Employed", "Expected": "Approved (96%)"},
            ])
            st.table(demo_data)

# ----------------------------------------------------
# PAGE 2: BATCH EVALUATION
# ----------------------------------------------------
elif page == "Batch Evaluation":
    st.markdown('<div class="main-header">📁 Batch Loan Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a CSV containing multiple applicant profiles for bulk prediction and automated loan underwriting.</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    use_sample = st.checkbox("Or use demo dataset (first 20 records from data/loan_data.csv)", value=False)
    
    df_batch = None
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
    elif use_sample:
        if os.path.exists("data/loan_data.csv"):
            df_batch = pd.read_csv("data/loan_data.csv").dropna().head(20)
        else:
            st.warning("loan_data.csv not found.")
            
    if df_batch is not None:
        st.write(f"**Loaded {len(df_batch)} records:**")
        st.dataframe(df_batch.head(), use_container_width=True)
        
        required_cols = {"Applicant_Income", "CIBIL_Score", "Loan_Amount", "Employment_Status"}
        if not required_cols.issubset(set(df_batch.columns)):
            st.error(f"CSV missing one or more required columns: {required_cols - set(df_batch.columns)}")
        else:
            if st.button("🚀 Run Batch Prediction", type="primary"):
                results = []
                for _, row in df_batch.iterrows():
                    try:
                        status, prob = predict_loan(
                            row["Applicant_Income"],
                            row["CIBIL_Score"],
                            row["Loan_Amount"],
                            row["Employment_Status"]
                        )
                        results.append({
                            "Applicant_Income": row["Applicant_Income"],
                            "CIBIL_Score": int(row["CIBIL_Score"]),
                            "Loan_Amount": row["Loan_Amount"],
                            "Employment_Status": row["Employment_Status"],
                            "Prediction": status,
                            "Approval_Probability": f"{prob*100:.1f}%",
                            "Raw_Score": prob
                        })
                    except Exception as e:
                        results.append({
                            "Applicant_Income": row.get("Applicant_Income"),
                            "CIBIL_Score": row.get("CIBIL_Score"),
                            "Loan_Amount": row.get("Loan_Amount"),
                            "Employment_Status": row.get("Employment_Status"),
                            "Prediction": "Error",
                            "Approval_Probability": str(e),
                            "Raw_Score": 0.0
                        })
                        
                res_df = pd.DataFrame(results)
                
                app_count = sum(res_df["Prediction"] == "Approved")
                rej_count = sum(res_df["Prediction"] == "Rejected")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Evaluated", len(res_df))
                c2.metric("Approved", app_count, f"{app_count/len(res_df)*100:.1f}%")
                c3.metric("Rejected", rej_count, f"{rej_count/len(res_df)*100:.1f}%")
                
                st.subheader("Batch Results Table")
                st.dataframe(res_df.drop(columns=["Raw_Score"]), use_container_width=True)
                
                # Download CSV
                csv_data = res_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv_data,
                    file_name="loan_predictions.csv",
                    mime="text/csv"
                )

# ----------------------------------------------------
# PAGE 3: MODEL PERFORMANCE & CHARTS
# ----------------------------------------------------
elif page == "Model Performance & Charts":
    st.markdown('<div class="main-header">📈 Model Performance & Visualizations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Review cross-validation metrics, confusion matrices, ROC curves, and feature importance.</div>', unsafe_allow_html=True)
    
    results = load_results()
    if results and "results" in results:
        res_data = results["results"]
        metrics_summary = []
        for model_name, m in res_data.items():
            metrics_summary.append({
                "Model": model_name,
                "Accuracy": f"{m.get('Accuracy', 0):.3f}",
                "Precision": f"{m.get('Precision', 0):.3f}",
                "Recall": f"{m.get('Recall', 0):.3f}",
                "F1 Score": f"{m.get('F1', 0):.3f}",
                "ROC-AUC": f"{m.get('AUC', 0):.3f}" if m.get('AUC') is not None else "N/A",
                "5-Fold CV Mean": f"{m.get('CV_mean', 0):.3f}"
            })
        st.subheader("🏆 Model Comparison Metrics (Test Set)")
        st.table(pd.DataFrame(metrics_summary))
    
    st.subheader("📊 Visualizations Gallery")
    chart_tabs = st.tabs([
        "Model Comparison", "Confusion Matrix", "ROC Curves", 
        "Feature Importance", "EDA Analysis", "Correlation Matrix", "Decision Tree"
    ])
    
    with chart_tabs[0]:
        if os.path.exists("fig/compare.png"):
            st.image("fig/compare.png", caption="Model comparison across 5 classification metrics", use_container_width=True)
    with chart_tabs[1]:
        if os.path.exists("fig/confusion.png"):
            st.image("fig/confusion.png", caption="Confusion Matrix on held-out test set", use_container_width=True)
    with chart_tabs[2]:
        if os.path.exists("fig/roc.png"):
            st.image("fig/roc.png", caption="Receiver Operating Characteristic (ROC) curves", use_container_width=True)
    with chart_tabs[3]:
        if os.path.exists("fig/importance.png"):
            st.image("fig/importance.png", caption="Feature importance breakdown across Random Forest & Decision Tree", use_container_width=True)
    with chart_tabs[4]:
        if os.path.exists("fig/eda.png"):
            st.image("fig/eda.png", caption="Exploratory Data Analysis: Status, CIBIL, and Income Distributions", use_container_width=True)
    with chart_tabs[5]:
        if os.path.exists("fig/corr.png"):
            st.image("fig/corr.png", caption="Correlation heatmap of engineered features", use_container_width=True)
    with chart_tabs[6]:
        if os.path.exists("fig/tree.png"):
            st.image("fig/tree.png", caption="Top levels of the Decision Tree model", use_container_width=True)

# ----------------------------------------------------
# PAGE 4: SYSTEM ARCHITECTURE & DOCS
# ----------------------------------------------------
elif page == "System Architecture & Docs":
    st.markdown('<div class="main-header">🏛️ System Architecture & Automated Verification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Data flow diagrams, architectural blueprints, and unit test results.</div>', unsafe_allow_html=True)
    
    st.subheader("✅ Automated Test Suite Verification (13 Test Cases)")
    tests = load_tests()
    if tests:
        test_rows = []
        for test_id, info in tests.items():
            test_rows.append({
                "Test ID": test_id,
                "Status": "PASSED ✅" if info.get("pass") else "FAILED ❌",
                "Description / Result": info.get("actual")
            })
        st.dataframe(pd.DataFrame(test_rows), use_container_width=True)
    else:
        st.info("Run `python tests.py` to generate `tests_results.json`.")
    
    st.subheader("📐 Architecture & Flow Diagrams")
    doc_tabs = st.tabs(["System Architecture", "Application Flowchart", "DFD Level 0", "DFD Level 1"])
    
    with doc_tabs[0]:
        if os.path.exists("fig/arch.png"):
            st.image("fig/arch.png", caption="High-level System Architecture", use_container_width=True)
    with doc_tabs[1]:
        if os.path.exists("fig/flowchart.png"):
            st.image("fig/flowchart.png", caption="Loan Underwriting Flowchart", use_container_width=True)
    with doc_tabs[2]:
        if os.path.exists("fig/dfd0.png"):
            st.image("fig/dfd0.png", caption="Data Flow Diagram (Level 0 - Context)", use_container_width=True)
    with doc_tabs[3]:
        if os.path.exists("fig/dfd1.png"):
            st.image("fig/dfd1.png", caption="Data Flow Diagram (Level 1 - Detailed)", use_container_width=True)
