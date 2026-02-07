from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import sqlite3
import numpy as np
import joblib
import os
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "supersecretkey"  # session security

# =====================
# LOAD MODEL
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "diabetes_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# =====================
# DATABASE SETUP
# =====================
def get_db():
    return sqlite3.connect("users.db")

def create_users_table():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

create_users_table()

# =====================
# AUTH ROUTES
# =====================

@app.route("/")
def root():
    return redirect("/login")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])

        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                        (name, email, password))
            conn.commit()
            conn.close()
            return redirect("/login")
        except:
            return "Email already exists"

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cur.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):
            session["user"] = user[1]
            return redirect("/home")
        else:
            return "Invalid credentials"

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# =====================
# PROTECTED HOME
# =====================
@app.route("/home")
def home():
    if "user" not in session:
        return redirect("/login")
    return render_template("index.html")

# =====================
# ML PREDICTION
# =====================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = np.array([[
        float(data.get("pregnancies", 0)),
        float(data["glucose"]),
        float(data["bp"]),
        float(data["bmi"]),
        float(data["age"]),
        1 if data["gender"] == "male" else 0
    ]])

    scaled = scaler.transform(features)
    result = model.predict(scaled)[0]

    return jsonify({"result": "Diabetic" if result == 1 else "Not Diabetic"})

# =====================
# MAIN
# =====================
if __name__ == "__main__":
    app.run(debug=True)
