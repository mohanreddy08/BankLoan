"""Generate the synthetic loan dataset used in this report (seed 42)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1000

emp = rng.choice(["Salaried", "Self-Employed", "Unemployed"], size=N, p=[0.58, 0.30, 0.12])
base_income = rng.lognormal(mean=np.log(6.0), sigma=0.45, size=N)      # INR Lakhs per year
income = np.where(emp == "Unemployed", base_income * 0.35, base_income)
income = np.clip(income, 0.8, 60).round(2)
cibil = np.clip(rng.normal(700, 85, size=N), 300, 900).round().astype(int)
loan = np.clip(income * rng.uniform(0.6, 6.0, size=N) + rng.normal(0, 1, size=N), 1, 150).round(2)

z = (0.030 * (cibil - 680)
     + 1.1 * (np.log(income) - np.log(loan / 3.0))
     + np.select([emp == "Salaried", emp == "Self-Employed"], [0.6, 0.0], -1.8)
     + rng.normal(0, 0.9, size=N))
status = np.where(z > 0, "Approved", "Rejected")

df = pd.DataFrame({"Applicant_Income": income, "CIBIL_Score": cibil, "Loan_Amount": loan,
                   "Employment_Status": emp, "Loan_Status": status})

# deliberately inject a few data-quality problems so the cleaning stage is exercised
miss_idx = rng.choice(N, size=15, replace=False)
df.loc[miss_idx[:9], "Applicant_Income"] = np.nan
df.loc[miss_idx[9:], "CIBIL_Score"] = np.nan
dups = df.sample(5, random_state=42)
df = pd.concat([df, dups], ignore_index=True)
df.to_csv("data/loan_data.csv", index=False)
print(df.shape, df["Loan_Status"].value_counts().to_dict())
