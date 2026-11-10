# 🏦 Bank Loan Eligibility & Approval Predictor

A complete machine learning solution for evaluating bank loan applications, assessing borrower creditworthiness, and predicting loan approval probabilities.

---

## 📌 Project Overview

This application predicts whether a loan applicant will be **Approved** or **Rejected** based on four primary underwriting attributes:
1. **Applicant Income** (in ₹ Lakhs per annum)
2. **CIBIL Credit Score** (300 to 900)
3. **Requested Loan Amount** (in ₹ Lakhs)
4. **Employment Status** (`Salaried`, `Self-Employed`, `Unemployed`)

### 🧠 Machine Learning Model
- **Algorithm**: `RandomForestClassifier` (200 decision trees, max depth 8)
- **Key Engineered Features**: `Loan_to_Income` ratio, One-hot encoded employment status
- **Model Accuracy**: **~87.5%**
- **F1 Score**: **0.900**
- **ROC-AUC**: **0.930**
- **5-Fold Cross-Validation Accuracy**: **89.6% (±1.6%)**

---

## 📂 Project Structure

```
D:\BankLoan\
│
├── app.py                  # 🌟 Streamlit Web Application (Interactive UI)
├── api.py                  # 🚀 FastAPI REST backend service (/predict, /health, /docs)
├── predict.py              # Core inference logic and CLI predictor
├── train.py                # Model training, cross-validation & chart generation
├── generate_data.py        # Synthetic loan dataset generator (seed 42)
├── tests.py                # Automated test suite (13 test cases)
│
├── data/
│   └── loan_data.csv       # Training & evaluation dataset (1,005 records)
│
├── model/
│   └── loan_model.pkl      # Serialized trained model bundle (Random Forest + feature columns)
│
├── fig/                    # High-resolution architectural & analytical visualizations
│   ├── arch.png            # High-level System Architecture diagram
│   ├── flowchart.png       # End-to-end Loan Decision Flowchart
│   ├── dfd0.png            # Data Flow Diagram (Level 0 - Context)
│   ├── dfd1.png            # Data Flow Diagram (Level 1 - Detailed)
│   ├── eda.png             # Exploratory Data Analysis distributions
│   ├── corr.png            # Feature correlation matrix
│   ├── compare.png         # Model performance comparisons
│   ├── confusion.png       # Test set confusion matrices
│   ├── roc.png             # Receiver Operating Characteristic curves
│   ├── importance.png      # Feature importance rankings
│   └── tree.png            # Sample Decision Tree visualization
│
├── results.json            # Model evaluation metrics & cross-validation scores
├── tests_results.json      # Verification outputs from running tests.py
├── requirements.txt        # Python dependency manifest
├── run_app.bat             # Windows 1-click launcher for Streamlit Web App
├── run_api.bat             # Windows 1-click launcher for FastAPI REST server
├── run_tests.bat           # Windows 1-click launcher for automated tests
└── README.md               # Project documentation
```

---

## ⚙️ Prerequisites & Setup

Ensure you have **Python 3.10+** installed.

### 1. Install Dependencies
Open a command prompt or PowerShell in `D:\BankLoan` and run:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Application

You have multiple convenient ways to run the project:

### Option 1: Interactive Web Application (Streamlit) — **Recommended**
Run the Streamlit application in your terminal:
```bash
streamlit run app.py
```
*(Or double-click **`run_app.bat`** on Windows)*

Once started, open your web browser at:
👉 **`http://localhost:8501`**

#### Web App Features:
- **Loan Approval Predictor**: Interactive sliders and inputs with instant approval badges, approval confidence gauge, and risk factor breakdown.
- **Batch Evaluation**: Upload a CSV of applicant records or use the demo dataset to predict hundreds of applications at once and download results as CSV.
- **Model Performance & Charts**: Browse all training charts (Confusion Matrix, ROC Curve, Feature Importance, Model Comparisons).
- **Architecture & Verification**: View system flowcharts, DFDs, and all 13 passing test results.

---

### Option 2: REST API (FastAPI)
Run the REST API server:
```bash
python api.py
```
*(Or double-click **`run_api.bat`**)*

The API server will run at:
👉 **`http://127.0.0.1:8000`**

Interactive Swagger API Documentation:
👉 **`http://127.0.0.1:8000/docs`**

#### Example API Request (cURL):
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "income": 12.0,
       "cibil": 780,
       "loan_amount": 20.0,
       "employment": "Salaried"
     }'
```

#### Example API Response:
```json
{
  "status": "Approved",
  "approval_probability": 0.9707,
  "loan_to_income_ratio": 1.67,
  "risk_assessment": "Low to Moderate Risk"
}
```

---

### Option 3: Command Line Interface (CLI)
You can directly run predictions from python in your terminal:
```bash
python predict.py
```
Or import in your own Python script:
```python
from predict import predict_loan

status, prob = predict_loan(
    income=12.0,        # ₹12 Lakhs/year
    cibil=780,          # 780 CIBIL score
    loan_amount=20.0,   # ₹20 Lakhs loan
    employment="Salaried"
)
print(f"Status: {status}, Confidence: {prob:.2%}")
# Output: Status: Approved, Confidence: 97.07%
```

---

## 🧪 Testing & Model Retraining

### Running Automated Tests
Run the 13 automated test cases covering data integrity, model behavior, monotonicity, and inference latency:
```bash
python tests.py
```
*(Or double-click **`run_tests.bat`**)*

### Retraining the Model
To retrain the models and re-generate all evaluation plots in `fig/`:
```bash
python train.py
```

### Regenerating the Synthetic Dataset
```bash
python generate_data.py
```
