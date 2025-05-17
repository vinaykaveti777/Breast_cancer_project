# Breast Cancer Detection using Machine Learning

This project uses machine learning to detect the likelihood of breast cancer based on a dataset of medical attributes. It provides a web interface (using Flask) to input data and get predictions from a trained model.

## 🔍 Project Overview

- Uses a cleaned large dataset: `breast_cancer_large.csv`
- Trains a model using `scikit-learn`
- Web interface built with Flask
- Model serialized using `pickle` (`model.pkl`)

## 🚀 Features

- Breast cancer prediction from user input
- Cleaned and preprocessed dataset
- Trained machine learning model
- Flask web app with a user-friendly interface
- Model accuracy metrics

## 🧪 How to Run

1. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the app**
    ```bash
    python app.py
    ```

3. Open your browser and go to `http://127.0.0.1:5000`

## 📁 Project Structure

```
breast_cancer_project/
├── app.py
├── model.py
├── model.pkl
├── breast_cancer_large.csv
├── breast_cancer_cleaned.csv
├── breast_cancer_cleaned.py
├── model_accuracy.txt
├── requirements.txt
└── templates/
    └── index.html
```

## 📊 Model Performance

Check `model_accuracy.txt` for detailed performance metrics of the trained model.

## 📚 References

- Dataset: Custom dataset (`breast_cancer_large.csv`)
- Libraries: `scikit-learn`, `pandas`, `Flask`, `numpy`

---

> Project created for educational and academic purposes.
