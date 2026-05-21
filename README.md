# 🩺 Early Diabetes Prediction System

A Machine Learning based web application that predicts the likelihood of diabetes using medical, demographic, and lifestyle data.

Developed as a Final Year Project at Kumaraguru College of Technology.

---

## 📘 Project Overview

Diabetes is a chronic health condition that often remains undiagnosed until severe complications arise.

This project aims to:

✔️ Detect diabetes risk at an early stage  
✔️ Provide real-time predictions  
✔️ Promote preventive healthcare awareness  

The system uses Machine Learning techniques to analyze user-provided health parameters and predicts diabetes risk instantly.

After evaluating multiple models, the **Random Forest Classifier** was selected due to its high accuracy and reliability.

---

## ❗ Problem Statement

Many individuals are unaware of their diabetic condition due to:

- Mild early symptoms  
- Lack of routine testing  
- Limited healthcare access  

This system provides:

✔️ Early risk prediction  
✔️ Quick and accessible screening  
✔️ Preventive health insights  

---

## 🎯 Objectives

- Perform data preprocessing  
- Conduct exploratory data analysis  
- Train ML models  
- Evaluate performance  
- Select best model  
- Integrate with web application  
- Deploy on cloud  
- Enable real-time prediction  

---

## 🏗 System Architecture

```
User Input
   ↓
Data Validation
   ↓
Preprocessing
   ↓
ML Model (Random Forest)
   ↓
Prediction
   ↓
Result Display
```

---

## 🚀 Key Features

- User Login & Signup  
- Real-time Diabetes Prediction  
- Gender-specific prediction forms  
- BMI Calculator  
- HbA1c estimation  
- Smart chatbot support  
- Cloud deployment  
- Responsive user interface  

---

## 🧰 Technology Stack

| Layer | Technology |
|------|------------|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python (Flask) |
| ML Library | Scikit-learn |
| Model | Random Forest |
| Deployment | Render |
| Model Storage | Joblib |

---

## 📊 Dataset Information

- Total Records: **5001**
- Features: **72**
- Target: Diabetes Status

### Feature Categories

- Age  
- Gender  
- BMI  
- Smoking History  
- Hypertension  
- Heart Disease  
- Blood Glucose Level  
- HbA1c Level  
- Geographic Data  

---

## 🔧 Data Preprocessing

- Missing value handling  
- Mean / Median imputation  
- Mode replacement  
- Categorical encoding  
- Train-Test Split  

---

## 📈 Exploratory Data Analysis

Analysis performed on:

- Class distribution  
- BMI trends  
- Feature correlation  
- Health indicators  

---

## 🤖 Machine Learning Models Used

- Logistic Regression  
- Basic Binary Classifiers  
- Random Forest  

---

## 🏆 Final Model

**Random Forest Classifier**

Reasons for selection:

- High accuracy  
- Handles complex datasets  
- Reduces overfitting  
- Provides feature importance  

Model stored using:

```
joblib
```

---

## 🌐 Web Application

Users input:

- Age  
- BMI  
- HbA1c  
- Blood Glucose  
- Smoking history  
- Medical conditions  

System predicts:

➡️ Diabetic / Non-Diabetic risk  

---

## ☁️ Deployment

Deployed on:

➡️ Render Cloud Platform

Steps followed:

1. Created requirements.txt  
2. Integrated ML model  
3. Configured Flask app  
4. Set environment variables  
5. Deployed online  

---

## 🧪 Testing

Performed:

- Functional Testing  
- Model Accuracy Testing  
- Integration Testing  
- Input Validation  
- Edge Case Testing  

---

## ✅ Advantages

- Quick prediction  
- User-friendly  
- Accessible online  
- Supports preventive care  

---

## ⚠️ Limitations

- Depends on dataset quality  
- Not a medical diagnosis replacement  
- Limited features  

---

## 🔮 Future Enhancements

- Patient history tracking  
- Advanced ML models  
- Visualization dashboards  
- Mobile application  
- Hospital integration  

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.x  
- Flask  
- Scikit-learn  

---

### Installation

Clone repository:

```bash
git clone <repo-link>
cd diabetes-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
python -m pip install -r requirements.txt
```

Run application:

```bash
python app.py
```

---

## 👩‍💻 Team Members

- Ragavan K (23BIT077)  
- Shakin Sakthi B O (23BIT100)  
- Pradakshina V (23BIT073)  
- Yokesh V (23BIT122)  
- Santhoshni S (23BIT097)  

Department of Information Technology  
Kumaraguru College of Technology  
Coimbatore  

---

## 📌 Conclusion

This project demonstrates how Machine Learning can be integrated with cloud-based web applications to support early diabetes detection and preventive healthcare.

---
