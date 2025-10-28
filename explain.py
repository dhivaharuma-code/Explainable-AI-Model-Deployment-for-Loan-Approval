import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

# Load dataset again
df = pd.read_csv("data/german_credit_sample.csv")
X = df.drop(columns=["default"])

# Scale numeric columns
num_cols = X.select_dtypes(include=['int','float']).columns
X[num_cols] = scaler.transform(X[num_cols])

# Explain using SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Global feature importance
shap.summary_plot(shap_values, X, show=False)
plt.savefig("shap_summary.png")
print("Saved SHAP summary plot as shap_summary.png")

# Explain one sample loan
sample = X.iloc[[0]]
shap.initjs()
force_plot = shap.force_plot(explainer.expected_value, shap_values[0], sample)
shap.save_html("shap_force.html", force_plot)
print("Saved SHAP force plot for 1 prediction as shap_force.html")

