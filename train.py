import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import joblib

# Load dataset
df = pd.read_csv("data/german_credit_sample.csv")
target = "default"
X = df.drop(columns=[target])
y = df[target]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Scale numeric columns
num_cols = X_train.select_dtypes(include=['int','float']).columns
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])

# Train XGBoost model
model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:,1]
print(classification_report(y_test, pred))
print("ROC AUC:", roc_auc_score(y_test, proba))

# Save model + scaler
joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")
print("Saved model.joblib and scaler.joblib")

