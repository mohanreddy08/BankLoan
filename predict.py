import os
import pickle
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "loan_model.pkl")

with open(MODEL_PATH, "rb") as f:
    _bundle = pickle.load(f)

MODEL, COLUMNS = _bundle["model"], _bundle["columns"]
EMP_COL = {
    "Salaried": "Emp_Salaried",
    "Self-Employed": "Emp_Self_Employed",
    "Unemployed": "Emp_Unemployed",
}

def predict_loan(income, cibil, loan_amount, employment):
    """Return ('Approved'/'Rejected', approval probability) for one applicant.
    
    Parameters:
        income (float): Applicant income in INR Lakhs (> 0)
        cibil (int/float): CIBIL credit score (300 to 900)
        loan_amount (float): Desired loan amount in INR Lakhs (> 0)
        employment (str): 'Salaried', 'Self-Employed', or 'Unemployed'
        
    Returns:
        tuple: (status: str, probability: float)
    """
    if income <= 0 or loan_amount <= 0:
        raise ValueError("Income and loan amount must be positive (INR Lakhs).")
    if not 300 <= cibil <= 900:
        raise ValueError("CIBIL score must be between 300 and 900.")
    if employment not in EMP_COL:
        raise ValueError("Employment must be Salaried, Self-Employed or Unemployed.")
    
    row = {c: 0 for c in COLUMNS}
    row.update({
        "Applicant_Income": float(income),
        "CIBIL_Score": float(cibil),
        "Loan_Amount": float(loan_amount),
        "Loan_to_Income": round(float(loan_amount) / float(income), 4),
        EMP_COL[employment]: 1,
    })
    
    df_row = pd.DataFrame([row])[COLUMNS]
    prob = round(float(MODEL.predict_proba(df_row)[0, 1]), 6)
    status = "Approved" if prob >= 0.5 else "Rejected"
    return status, prob

if __name__ == "__main__":
    print("=" * 55)
    print("       BANK LOAN APPROVAL PREDICTOR (CLI DEMO)     ")
    print("=" * 55)
    test_cases = [
        (12.0, 780, 20.0, "Salaried"),
        (4.0, 540, 30.0, "Self-Employed"),
        (2.5, 710, 15.0, "Unemployed"),
        (8.0, 650, 25.0, "Salaried"),
        (15.0, 820, 40.0, "Self-Employed")
    ]
    for inc, cib, l_amt, emp in test_cases:
        status, prob = predict_loan(inc, cib, l_amt, emp)
        print(f"Income: {inc:5.1f}L | CIBIL: {cib:3d} | Loan: {l_amt:5.1f}L | Emp: {emp:13s} -> {status:8s} (Prob: {prob:.2%})")
    print("=" * 55)
