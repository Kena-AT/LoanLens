from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import pandas as pd
import os

# Load cleaned data
data = pd.read_csv("data/clean_credit_data.csv")
X = data.drop(columns=["SeriousDlqin2yrs"])
y = data["SeriousDlqin2yrs"]

# SMOTETomek balancing
smt = SMOTETomek(random_state=42)
X_balanced, y_balanced = smt.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42
)

# Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)

print("📊 Classification Report:\n")
print(report)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "notebooks/models/xgb_smote_model.pkl")

# Save classification report
os.makedirs("notebooks/outputs", exist_ok=True)
with open("notebooks/outputs/model_report.txt", "w") as f:
    f.write("XGBoost + SMOTETomek Classification Report\n")
    f.write("="*45 + "\n")
    f.write(report)
    f.write("\n✅ Model: notebooks/models/xgb_smote_model.pkl")