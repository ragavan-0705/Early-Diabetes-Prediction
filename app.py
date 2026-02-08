from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import pandas as pd
import sqlite3
import numpy as np
import joblib
import os
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "supersecretkey"

# =====================
# LOAD MODEL
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "diabetes_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# Load feature columns from the CSV so we build a compatible input vector
CSV_PATH = os.path.join(BASE_DIR, "diabetes.csv")
try:
    _df_sample = pd.read_csv(CSV_PATH, nrows=5)
    FEATURE_COLS = [c for c in _df_sample.columns if c != "diabetes"]
except Exception:
    FEATURE_COLS = None

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
            cur.execute(
                "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                (name, email, password)
            )
            conn.commit()
            conn.close()
            flash("Account created successfully! Please login.", "success")
            return redirect("/login")
        except:
            flash("Email already exists!", "error")
            return redirect("/signup")

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
            session["email"] = user[2]
            return redirect("/home")
        else:
            flash("Invalid email or password", "error")
            return redirect("/login")

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
# PROFILE / ACCOUNT
# =====================
@app.route("/profile", methods=["GET", "POST"])
def profile():
    if "user" not in session:
        return redirect("/login")

    conn = get_db()
    cur = conn.cursor()

    if request.method == "POST":
        current = request.form.get("current_password")
        new = request.form.get("new_password")
        confirm = request.form.get("confirm_password")

        email = session.get("email")
        cur.execute("SELECT id, password FROM users WHERE email = ?", (email,))
        row = cur.fetchone()

        if not row or not check_password_hash(row[1], current):
            flash("Current password is incorrect", "error")
            conn.close()
            return redirect("/profile")

        if not new or new != confirm:
            flash("New passwords do not match", "error")
            conn.close()
            return redirect("/profile")

        hashed = generate_password_hash(new)
        cur.execute("UPDATE users SET password = ? WHERE id = ?", (hashed, row[0]))
        conn.commit()
        conn.close()

        flash("Password updated successfully", "success")
        return redirect("/profile")

    # GET
    name = session.get("user")
    email = session.get("email")
    conn.close()
    return render_template("profile.html", name=name, email=email)

# =====================
# ML PREDICTION
# =====================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}

    if FEATURE_COLS is None:
        return jsonify({"error": "Server missing feature metadata"}), 500

    # start with zeros/defaults for every column
    row = {c: 0 for c in FEATURE_COLS}

    # year
    row["year"] = int(data.get("year", 2019))

    # gender: dataset uses 1.0/0.0
    row["gender"] = 1.0 if data.get("gender") == "male" else 0.0

    # basic numeric fields
    if data.get("age") is not None:
        row["age"] = float(data.get("age"))
    if data.get("bmi") is not None:
        row["bmi"] = float(data.get("bmi"))
    if data.get("hbA1c_level") is not None:
        row["hbA1c_level"] = float(data.get("hbA1c_level"))
    if data.get("blood_glucose_level") is not None:
        row["blood_glucose_level"] = float(data.get("blood_glucose_level"))

    # health flags
    row["hypertension"] = 1 if data.get("hypertension") else 0
    row["heart_disease"] = 1 if data.get("heart_disease") else 0

    # race mapping (one-hot columns named like 'race:Asian')
    race = data.get("race")
    if race:
        race_col = f"race:{race}"
        if race_col in row:
            row[race_col] = 1

    # smoking history mapping
    smoke = data.get("smoking_history")
    if smoke:
        smoke_col = f"smoking_history_{smoke}"
        # note: one header contains a space 'smoking_history_not current'
        if smoke_col in row:
            row[smoke_col] = 1
        else:
            # try space variant
            alt = smoke_col.replace("_", " ", 1)
            if alt in row:
                row[alt] = 1

    # location mapping (columns: location_X)
    loc = data.get("location")
    if loc:
        loc_col = f"location_{loc}"
        if loc_col in row:
            row[loc_col] = 1

    # Build 2D features array in correct column order
    vals = [row[c] for c in FEATURE_COLS]
    features = np.array([vals], dtype=float)

    # scale and predict
    scaled = scaler.transform(features)
    result = model.predict(scaled)[0]

    return jsonify({"result": "Diabetic" if int(result) == 1 else "Not Diabetic"})

# =====================
# MAIN
# =====================
if __name__ == "__main__":
    app.run(debug=True)
