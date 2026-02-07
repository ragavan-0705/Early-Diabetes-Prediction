from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

# ==============================
# MODEL LOADING (SAFE METHOD)
# ==============================

# Get current directory of app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build full paths to model files
MODEL_PATH = os.path.join(BASE_DIR, "diabetes_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# Load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# ==============================
# ROUTES
# ==============================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # ------------------------------
    # INPUTS FROM FRONTEND
    # ------------------------------
    gender = data["gender"]
    age = float(data["age"])
    bmi = float(data["bmi"])
    glucose = float(data["glucose"])
    bp = float(data["bp"])
    pregnancies = float(data.get("pregnancies", 0))

    # ------------------------------
    # ENCODING
    # ------------------------------
    gender_val = 1 if gender == "male" else 0

    # ------------------------------
    # FEATURE ORDER (DO NOT CHANGE
    # unless ML engineer confirms)
    # ------------------------------
    features = np.array([[
        pregnancies,
        glucose,
        bp,
        bmi,
        age,
        gender_val
    ]])

    # ------------------------------
    # SCALING
    # ------------------------------
    features_scaled = scaler.transform(features)

    # ------------------------------
    # PREDICTION
    # ------------------------------
    prediction = model.predict(features_scaled)[0]

    result = "Diabetic" if prediction == 1 else "Not Diabetic"

    return jsonify({"result": result})


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    app.run(debug=True)
