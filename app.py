from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

# Feature names
features = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# Load accuracy
with open("model_accuracy.txt", "r") as f:
    accuracy = f.read()

@app.route('/')
def home():
    return render_template('index.html', accuracy=accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    values = [float(request.form[feature]) for feature in features]
    input_df = pd.DataFrame([values], columns=features)
    prediction = model.predict(input_df)[0]
    result = "Benign" if prediction == 1 else "Malignant"
    return render_template('index.html', result=result, accuracy=accuracy)

if __name__ == "__main__":
    app.run(debug=True)
