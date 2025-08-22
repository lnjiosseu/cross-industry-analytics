# scripts/churn_modeling.py
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ---- ABSOLUTE PATHS ----
ROOT = Path("/Users/caliboi/Desktop/Resumes/Github/Project 5")
DATA = ROOT / "industry_modeling_data.csv"
OUT  = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

print(f"[5] DATA: {DATA}")
print(f"[5] OUT : {OUT}")

# ---- Load ----
df = pd.read_csv(DATA)

# One-hot encode if column exists
if "industry" in df.columns:
    df = pd.get_dummies(df, columns=["industry"], drop_first=True)

# Features/target
X = df.drop(columns=[c for c in ["customer_id", "churn"] if c in df.columns])
y = df["churn"]

# ---- Train ----
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ---- Evaluate ----
y_pred = model.predict(X_test)
report_txt = classification_report(y_test, y_pred)
print(report_txt)
(OUT / "classification_report.txt").write_text(report_txt)
with open(OUT / "classification_report.json", "w") as f:
    json.dump(classification_report(y_test, y_pred, output_dict=True), f, indent=2)

cm = confusion_matrix(y_test, y_pred)
print("[5] Confusion matrix:\n", cm)

# ---- Feature Importance Plot ----
importances = pd.Series(model.feature_importances_, index=X.columns)

plt.figure(figsize=(10, 6))
plt.bar(importances.index, importances)   # vertical bars
plt.xticks(rotation=45, ha="right")              # rotate labels for readability
plt.ylabel("Importance")
plt.xlabel("Features")
plt.title("Feature Importance")
plt.tight_layout()

# Save plot
plt.savefig(OUT / "feature_importance.png", bbox_inches="tight")
plt.close()

print(f"[5] Wrote -> {OUT/'classification_report.txt'}")
print(f"[5] Wrote -> {OUT/'feature_importance.png'}")
