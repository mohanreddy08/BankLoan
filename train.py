import json, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, classification_report)

SEED = 42
R = {}
df = pd.read_csv("data/loan_data.csv")
R["raw_shape"] = list(df.shape)
R["missing"] = df.isna().sum().to_dict()
R["dups"] = int(df.duplicated().sum())
R["class_raw"] = df["Loan_Status"].value_counts().to_dict()
R["describe"] = df.describe().round(2).to_dict()
R["emp_counts"] = df["Employment_Status"].value_counts().to_dict()

# ---- cleaning
df = df.drop_duplicates()
for c in ["Applicant_Income", "CIBIL_Score"]:
    df[c] = df[c].fillna(df[c].median())
R["clean_shape"] = list(df.shape)
R["class_clean"] = df["Loan_Status"].value_counts().to_dict()

# ---- EDA figures
sns.set_theme(style="whitegrid", font="DejaVu Serif")
PAL={"Approved":"#2E75B6","Rejected":"#C0504D"}
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
sns.countplot(data=df, x="Loan_Status", hue="Loan_Status", palette=PAL, ax=ax[0], legend=False, order=["Approved","Rejected"])
ax[0].set_title("Loan status count"); ax[0].set_xlabel("")
sns.histplot(data=df, x="CIBIL_Score", hue="Loan_Status", bins=30, kde=True, palette=PAL, ax=ax[1])
ax[1].set_title("CIBIL score by loan status")
sns.boxplot(data=df, x="Employment_Status", y="Loan_Status".replace("Loan_Status","Applicant_Income"), hue="Loan_Status", palette=PAL, ax=ax[2])
ax[2].set_title("Income by employment and status"); ax[2].set_ylabel("Income (INR Lakhs)"); ax[2].set_xlabel("")
ax[2].tick_params(axis="x", labelsize=8)
plt.tight_layout(); plt.savefig("fig/eda.png", dpi=200); plt.close()

# ---- feature engineering
df["Loan_to_Income"] = (df["Loan_Amount"] / df["Applicant_Income"]).round(4)
y = (df["Loan_Status"] == "Approved").astype(int)
X = pd.get_dummies(df.drop(columns=["Loan_Status"]), columns=["Employment_Status"], dtype=int)
X.columns = [c.replace("Employment_Status_", "Emp_").replace("-", "_") for c in X.columns]
R["features"] = list(X.columns)

cm = X.copy(); cm["Approved"] = y
plt.figure(figsize=(7.2, 5.6))
sns.heatmap(cm.corr().round(2), annot=True, fmt=".2f", cmap="Blues", annot_kws={"size": 8}, cbar=True)
plt.title("Correlation Heat Map of Engineered Features"); plt.tight_layout()
plt.savefig("fig/corr.png", dpi=200); plt.close()
R["corr_target"] = cm.corr()["Approved"].round(2).to_dict()

# ---- split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
R["split"] = [len(Xtr), len(Xte)]
R["test_class"] = {"Approved": int(yte.sum()), "Rejected": int((1-yte).sum())}

models = {
    "Majority-class baseline": DummyClassifier(strategy="most_frequent"),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=SEED, n_jobs=-1),
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
res, probs, preds = {}, {}, {}
for n, m in models.items():
    m.fit(Xtr, ytr)
    p = m.predict(Xte); pr = m.predict_proba(Xte)[:, 1]
    cv = cross_val_score(m, Xtr, ytr, cv=skf, scoring="accuracy")
    res[n] = dict(Accuracy=accuracy_score(yte, p), Precision=precision_score(yte, p, zero_division=0),
                  Recall=recall_score(yte, p), F1=f1_score(yte, p), AUC=roc_auc_score(yte, pr),
                  CV_mean=cv.mean(), CV_std=cv.std(), Train_acc=accuracy_score(ytr, m.predict(Xtr)),
                  CM=confusion_matrix(yte, p).tolist())
    probs[n], preds[n] = pr, p
R["results"] = res
best = max([k for k in res if k != "Majority-class baseline"], key=lambda k: res[k]["F1"])
R["best"] = best

# ---- figures: confusion matrices
fig, ax = plt.subplots(1, 2, figsize=(9, 3.9))
for a, n in zip(ax, ["Decision Tree", "Random Forest"]):
    sns.heatmap(np.array(res[n]["CM"]), annot=True, fmt="d", cmap="Blues", cbar=False, ax=a,
                xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"], annot_kws={"size": 14})
    a.set_title(f"{n} (Acc = {res[n]['Accuracy']:.3f})"); a.set_xlabel("Predicted"); a.set_ylabel("Actual")
plt.tight_layout(); plt.savefig("fig/confusion.png", dpi=200); plt.close()

# ROC
plt.figure(figsize=(5.6, 4.6))
for n, c in [("Decision Tree", "#C0504D"), ("Random Forest", "#2E75B6")]:
    f, t, _ = roc_curve(yte, probs[n]); plt.plot(f, t, color=c, lw=2, label=f"{n} (AUC = {res[n]['AUC']:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves on the test set"); plt.legend(loc="lower right"); plt.tight_layout()
plt.savefig("fig/roc.png", dpi=200); plt.close()

# metric comparison
mets = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
x = np.arange(len(mets)); w = 0.27
plt.figure(figsize=(8, 4.2))
for i, (n, c) in enumerate([("Majority-class baseline", "#A6A6A6"), ("Decision Tree", "#C0504D"), ("Random Forest", "#2E75B6")]):
    v = [res[n][k] for k in mets]; b = plt.bar(x + (i-1)*w, v, w, label=n, color=c)
    for r_, vv in zip(b, v): plt.text(r_.get_x()+r_.get_width()/2, vv+0.01, f"{vv:.2f}", ha="center", fontsize=7)
plt.xticks(x, mets); plt.ylim(0, 1.12); plt.ylabel("Score"); plt.title("Model comparison on the held-out test set")
plt.legend(loc="lower right", fontsize=8); plt.tight_layout(); plt.savefig("fig/compare.png", dpi=200); plt.close()

# feature importance
fi = pd.DataFrame({"Decision Tree": models["Decision Tree"].feature_importances_,
                   "Random Forest": models["Random Forest"].feature_importances_}, index=X.columns)
R["importance"] = fi.round(3).to_dict()
fi.sort_values("Random Forest").plot(kind="barh", figsize=(8, 4.4), color=["#C0504D", "#2E75B6"])
plt.xlabel("Importance"); plt.title("Feature importance of the Decision Tree and Random Forest")
plt.tight_layout(); plt.savefig("fig/importance.png", dpi=200); plt.close()

# sample tree
plt.figure(figsize=(10, 4.8))
plot_tree(models["Decision Tree"], feature_names=list(X.columns), class_names=["Rejected", "Approved"],
          filled=True, rounded=True, max_depth=2, fontsize=10, impurity=False, proportion=False)
plt.title("Top two levels of the trained Decision Tree"); plt.tight_layout()
plt.savefig("fig/tree.png", dpi=200); plt.close()

# ---- save best model
final = models[best]
with open("model/loan_model.pkl", "wb") as f:
    pickle.dump({"model": final, "columns": list(X.columns)}, f)

# ---- prediction examples (reload from file)
bundle = pickle.load(open("model/loan_model.pkl", "rb"))
def predict(income, cibil, loan, emp):
    row = {c: 0 for c in bundle["columns"]}
    row.update({"Applicant_Income": income, "CIBIL_Score": cibil, "Loan_Amount": loan,
                "Loan_to_Income": round(loan/income, 4),
                {"Salaried":"Emp_Salaried","Self-Employed":"Emp_Self_Employed","Unemployed":"Emp_Unemployed"}[emp]: 1})
    d = pd.DataFrame([row])[bundle["columns"]]
    p = float(bundle["model"].predict_proba(d)[0, 1])
    return ("Approved" if p >= 0.5 else "Rejected"), p
cases = [(12.0, 780, 20.0, "Salaried"), (4.0, 540, 30.0, "Self-Employed"), (2.5, 710, 15.0, "Unemployed"),
         (8.0, 650, 25.0, "Salaried"), (15.0, 820, 40.0, "Self-Employed")]
R["examples"] = [dict(inp=c, out=predict(*c)) for c in cases]
R["predict_repeat"] = [predict(*cases[0]), predict(*cases[0])]
# monotonic sanity: raise CIBIL, keep rest fixed
R["mono_cibil"] = [(s, round(predict(8.0, s, 25.0, "Salaried")[1], 3)) for s in [500, 600, 700, 800]]
R["mono_loan"] = [(l, round(predict(8.0, 700, l, "Salaried")[1], 3)) for l in [10, 25, 50, 80]]
# held-out example
i = Xte.index[3]
R["heldout_example"] = dict(row=df.loc[i, ["Applicant_Income","CIBIL_Score","Loan_Amount","Employment_Status"]].to_dict(),
                            actual=df.loc[i,"Loan_Status"],
                            pred=predict(df.loc[i,"Applicant_Income"], df.loc[i,"CIBIL_Score"], df.loc[i,"Loan_Amount"], df.loc[i,"Employment_Status"])[0])
import sklearn, pandas, matplotlib
R["versions"] = dict(py="3.x", sklearn=sklearn.__version__, pandas=pandas.__version__, numpy=np.__version__)
json.dump(R, open("results.json", "w"), indent=1, default=str)
print(json.dumps({k: R[k] for k in ["clean_shape","split","best","class_clean"]}))
for n in res: print(n, {k: round(v,3) for k,v in res[n].items() if k!="CM"}, res[n]["CM"])
print(R["examples"]); print(R["mono_cibil"], R["mono_loan"]); print(R["heldout_example"])
