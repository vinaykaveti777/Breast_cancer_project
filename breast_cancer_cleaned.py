# breast_cancer_cleaned.py
import pandas as pd

df = pd.read_csv("breast_cancer_large.csv")

# Ensure target is integer (0 = malignant, 1 = benign)
df["target"] = df["target"].astype(int)

# Save the cleaned version
df.to_csv("breast_cancer_cleaned.csv", index=False)
print("✅ Cleaned dataset saved.")
