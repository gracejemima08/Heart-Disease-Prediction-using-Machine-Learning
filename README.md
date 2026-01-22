# ❤️ Heart Disease Prediction using Machine Learning

This project predicts the likelihood of **heart disease** in a patient using **Machine Learning models** and provides an interactive **Streamlit web application** for real-time prediction.

The model is trained on heart disease clinical data and uses features such as age, cholesterol level, chest pain type, blood pressure, etc., to determine the risk of heart disease.

---

## 🚀 Live App Features

* User-friendly Streamlit web interface
* Real-time prediction
* Probability score of heart disease risk
* Uses trained ML models:

  * Logistic Regression
  * Random Forest
  * XGBoost
* Input scaling using StandardScaler

---

## 🧠 Machine Learning Workflow

1. Data Collection and Preprocessing
2. Feature Scaling using StandardScaler
3. Model Training using multiple algorithms
4. Model Selection based on accuracy
5. Model Saving using Joblib
6. Deployment using Streamlit

---

## 📊 Input Features Used

| Feature  | Description                 |
| -------- | --------------------------- |
| age      | Age of the patient          |
| sex      | Gender (Male/Female)        |
| cp       | Chest pain type             |
| trestbps | Resting blood pressure      |
| chol     | Cholesterol level           |
| fbs      | Fasting blood sugar         |
| restecg  | Resting ECG results         |
| thalach  | Maximum heart rate achieved |
| exang    | Exercise induced angina     |
| oldpeak  | ST depression               |
| slope    | Slope of ST segment         |
| ca       | Number of major vessels     |
| thal     | Thalassemia type            |

---

## 🛠️ Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Joblib

---

## 📁 Project Structure

```
Heart-Disease-Prediction/
│
├── app1.py                # Streamlit App
├── heart_model1.pkl      # Trained ML Model
├── scaler1.pkl           # Scaler
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run the App Locally

1. Install dependencies
```
pip install -r requirements.txt
```

2. Run the Streamlit app

```
streamlit run app1.py
```

3. Open browser at

```
http://localhost:8501
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn==1.5.1
xgboost
joblib
```

---

## 🎯 Project Objective

The goal of this project is to demonstrate how Machine Learning can assist in early detection of heart disease and how ML models can be deployed as an interactive web application.

---

## 👨‍💻 Author

**Grace**

