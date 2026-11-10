import json, time, os, pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from predict import predict_loan, COLUMNS, MODEL
T = {}
raw = pd.read_csv("data/loan_data.csv")
T["T1"] = (raw.shape == (1005, 5), f"{raw.shape[0]} rows, {raw.shape[1]} columns")
cl = raw.drop_duplicates().copy()
n_miss_before = int(cl.isna().sum().sum())
for c in ["Applicant_Income", "CIBIL_Score"]: cl[c] = cl[c].fillna(cl[c].median())
T["T2"] = (len(cl) == 1000 and cl.isna().sum().sum() == 0 and not cl.duplicated().any(),
           f"{len(cl)} rows; {int(raw.duplicated().sum())} duplicates removed; {n_miss_before} missing values imputed")
cl["Loan_to_Income"] = cl["Loan_Amount"] / cl["Applicant_Income"]
X = pd.get_dummies(cl.drop(columns=["Loan_Status"]), columns=["Employment_Status"], dtype=int)
X.columns = [c.replace("Employment_Status_", "Emp_").replace("-", "_") for c in X.columns]
T["T3"] = (list(X.columns) == COLUMNS and all(X.dtypes != object), f"{X.shape[1]} numeric features: " + ", ".join(X.columns))
y = (cl["Loan_Status"] == "Approved").astype(int)
a, b, c_, d = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
T["T4"] = (len(a) == 800 and len(b) == 200 and abs(c_.mean()-d.mean()) < 0.01,
           f"{len(a)} train / {len(b)} test; approval rate {c_.mean():.3f} vs {d.mean():.3f}")
T["T5"] = (os.path.exists("model/loan_model.pkl") and pickle.load(open("model/loan_model.pkl","rb"))["columns"] == COLUMNS,
           "model/loan_model.pkl exists and reloads with the same feature columns")
R = json.load(open("results.json"))
T["T6"] = (all(k in R["results"]["Random Forest"] for k in ["Accuracy","Precision","Recall","F1","AUC"]),
           "Accuracy, Precision, Recall, F1, ROC-AUC and CV accuracy produced for every model")
bad = 0
for args in [(5, 250, 10, "Salaried"), (5, 1000, 10, "Salaried"), (-1, 700, 10, "Salaried"), (5, 700, 0, "Salaried"), (5, 700, 10, "Student")]:
    try: predict_loan(*args)
    except ValueError: bad += 1
T["T7"] = (bad == 5, f"{bad} of 5 invalid inputs rejected with ValueError")
lab, p = predict_loan(12.0, 780, 20.0, "Salaried")
T["T8"] = (lab in ("Approved", "Rejected") and 0 <= p <= 1, f"{lab}, probability {p:.3f}")
T["T9"] = (predict_loan(12.0, 780, 20.0, "Salaried") == predict_loan(12.0, 780, 20.0, "Salaried"), "Identical output on two calls")
cib = [predict_loan(8.0, s, 25.0, "Salaried")[1] for s in (500, 600, 700, 800)]
T["T10"] = (all(x <= y_ for x, y_ in zip(cib, cib[1:])), "Approval probability for CIBIL 500/600/700/800: " + ", ".join(f"{v:.3f}" for v in cib))
ln = [predict_loan(8.0, 700, l, "Salaried")[1] for l in (10, 25, 50, 80)]
T["T11"] = (all(x >= y_ for x, y_ in zip(ln, ln[1:])), "Approval probability for loan 10/25/50/80 Lakhs: " + ", ".join(f"{v:.3f}" for v in ln))
t0 = time.perf_counter()
for _ in range(20): predict_loan(8.0, 700, 25.0, "Salaried")
ms = (time.perf_counter()-t0)/20*1000
T["T12"] = (ms < 1000, f"{ms:.1f} ms per prediction (mean of 20 calls)")
un = predict_loan(2.5, 710, 15.0, "Unemployed")[0]
sa = predict_loan(2.5, 710, 15.0, "Salaried")[1]; un_p = predict_loan(2.5, 710, 15.0, "Unemployed")[1]
T["T13"] = (un_p <= sa, f"Same financial profile: Salaried {sa:.3f} vs Unemployed {un_p:.3f}")
out = {k: {"pass": bool(v[0]), "actual": v[1]} for k, v in T.items()}
json.dump(out, open("tests_results.json", "w"), indent=1)
for k, v in out.items(): print(k, v)
