"""FastAPI Backend REST service for Bank Loan Approval Prediction.

Provides endpoints to evaluate loan applications and check model status.
"""
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from predict import predict_loan

app = FastAPI(
    title="Bank Loan Prediction API",
    description="REST API for evaluating loan approval probabilities using Random Forest model.",
    version="1.0.0"
)

class LoanRequest(BaseModel):
    income: float = Field(..., gt=0, description="Applicant Annual Income in INR Lakhs (e.g. 12.0)")
    cibil: int = Field(..., ge=300, le=900, description="CIBIL Credit Score (300 to 900)")
    loan_amount: float = Field(..., gt=0, description="Desired Loan Amount in INR Lakhs (e.g. 25.0)")
    employment: Literal["Salaried", "Self-Employed", "Unemployed"] = Field(
        ..., description="Employment category"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "income": 12.0,
                "cibil": 780,
                "loan_amount": 20.0,
                "employment": "Salaried"
            }
        }

class LoanResponse(BaseModel):
    status: str
    approval_probability: float
    loan_to_income_ratio: float
    risk_assessment: str

@app.get("/")
def root():
    return {
        "message": "Bank Loan Prediction API is online.",
        "documentation": "/docs",
        "endpoints": {
            "POST /predict": "Predict loan approval for an applicant",
            "GET /health": "Health and model readiness check"
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": "RandomForestClassifier", "features": 7}

@app.post("/predict", response_model=LoanResponse)
def evaluate_loan(req: LoanRequest):
    try:
        status, prob = predict_loan(req.income, req.cibil, req.loan_amount, req.employment)
        lti = round(req.loan_amount / req.income, 2)
        
        if status == "Approved":
            risk = "Low to Moderate Risk"
        else:
            risk = "High Risk / Poor Credit or High Debt Ratio"
            
        return LoanResponse(
            status=status,
            approval_probability=round(prob, 4),
            loan_to_income_ratio=lti,
            risk_assessment=risk
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
