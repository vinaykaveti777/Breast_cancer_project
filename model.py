import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv("breast_cancer_cleaned.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")
print("✅ Model trained and saved.")

# Evaluate and save accuracy
accuracy = model.score(X_test, y_test)
print(f"✅ Model Accuracy: {accuracy:.2f}")

with open("model_accuracy.txt", "w") as f:
    f.write(str(round(accuracy * 100, 2)))
