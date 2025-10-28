import pandas as pd
import numpy as np

np.random.seed(42)
N = 1000

df = pd.DataFrame({
    "age": np.random.randint(18, 75, size=N),
    "income": np.round(np.random.normal(40000, 15000, size=N)).astype(int),
    "loan_amount": np.round(np.random.normal(8000, 3000, size=N)).astype(int),
    "duration_months": np.random.randint(6, 60, size=N),
    "credit_history": np.random.choice([0,1,2], size=N),
    "purpose": np.random.choice([0,1,2,3], size=N),
})

# synthetic target: higher loan amount + low income → higher default chance
prob_default = (0.00002 * df.loan_amount) + (0.00001 * (50000 - df.income)) + (0.01 * (df.credit_history==2))
df['default'] = (np.random.rand(N) < prob_default).astype(int)

df.to_csv("data/german_credit_sample.csv", index=False)
print("Wrote data/german_credit_sample.csv (rows=%d)" % len(df))

