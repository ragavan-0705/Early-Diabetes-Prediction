from flask import Flask, render_template, request, redirect, session, jsonify, flash, send_file
import pandas as pd
import sqlite3
import numpy as np
import joblib
import os
import io
import datetime
import traceback
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage
)

from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

from reportlab.graphics.barcode.qr import QrCodeWidget
from reportlab.graphics.shapes import Drawing
from reportlab.graphics import renderPM

import uuid

from werkzeug.security import generate_password_hash, check_password_hash

# =========================
# FLASK APP
# =========================

app = Flask(__name__)
app.secret_key = "supersecretkey"

# =========================
# BASE DIRECTORY
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# LOAD MODEL & SCALER
# =========================

try:
    model_path = os.path.join(BASE_DIR, "diabetes_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print("✅ Model and scaler loaded successfully")

except Exception as e:
    print("❌ MODEL LOADING ERROR")
    print(str(e))
    traceback.print_exc()

    model = None
    scaler = None

# =========================
# LOAD FEATURE COLUMNS
# =========================

CSV_PATH = os.path.join(BASE_DIR, "diabetes_cleaned.csv")

try:
    df_sample = pd.read_csv(CSV_PATH, nrows=5)

    FEATURE_COLS = [c for c in df_sample.columns if c != "diabetes"]

    print("✅ Feature columns loaded")
    print("Total Features:", len(FEATURE_COLS))

except Exception as e:
    print("❌ CSV FEATURE ERROR")
    print(str(e))

    FEATURE_COLS = None

# =========================
# DATABASE
# =========================

DB_PATH = os.path.join(BASE_DIR, "users.db")


def get_db():
    return sqlite3.connect(DB_PATH)


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

# =========================
# ROOT
# =========================

@app.route("/")
def root():
    return redirect("/login")

# =========================
# SIGNUP
# =========================

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

            flash("Account created successfully!", "success")

            return redirect("/login")

        except Exception as e:

            print("SIGNUP ERROR:", str(e))

            flash("Email already exists!", "error")

            return redirect("/signup")

    return render_template("signup.html")

# =========================
# LOGIN
# =========================

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

# =========================
# LOGOUT
# =========================

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# =========================
# HOME
# =========================

@app.route("/home")
def home():

    if "user" not in session:
        return redirect("/login")

    return render_template("index.html")

# =========================
# PROFILE
# =========================

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

        cur.execute(
            "SELECT id, password FROM users WHERE email = ?",
            (email,)
        )

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

        cur.execute(
            "UPDATE users SET password = ? WHERE id = ?",
            (hashed, row[0])
        )

        conn.commit()
        conn.close()

        flash("Password updated successfully", "success")

        return redirect("/profile")

    name = session.get("user")
    email = session.get("email")

    conn.close()

    return render_template(
        "profile.html",
        name=name,
        email=email
    )

# =========================
# VERIFY PASSWORD
# =========================

@app.route("/verify-password", methods=["POST"])
def verify_password():

    if "user" not in session:
        return jsonify({
            "success": False,
            "message": "Not logged in"
        }), 401

    data = request.json or {}

    current_password = data.get("current_password", "")

    email = session.get("email")

    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        "SELECT password FROM users WHERE email = ?",
        (email,)
    )

    row = cur.fetchone()

    conn.close()

    if not row:

        return jsonify({
            "success": False,
            "message": "User not found"
        })

    if check_password_hash(row[0], current_password):

        return jsonify({
            "success": True,
            "message": "Password verified"
        })

    else:

        return jsonify({
            "success": False,
            "message": "Incorrect password"
        })

# =========================
# PREDICT
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        if model is None or scaler is None:

            return jsonify({
                "error": "Model not loaded"
            }), 500

        data = request.json or {}

        print("========== INCOMING DATA ==========")
        print(data)

        if FEATURE_COLS is None:

            return jsonify({
                "error": "Feature columns missing"
            }), 500

        # default values

        row = {c: 0 for c in FEATURE_COLS}

        # numeric fields

        row["year"] = int(data.get("year", 2019))

        row["gender"] = 1.0 if data.get("gender") == "male" else 0.0

        if data.get("age"):
            row["age"] = float(data.get("age"))

        if data.get("bmi"):
            row["bmi"] = float(data.get("bmi"))

        if data.get("hbA1c_level"):
            row["hbA1c_level"] = float(data.get("hbA1c_level"))

        if data.get("blood_glucose_level"):
            row["blood_glucose_level"] = float(
                data.get("blood_glucose_level")
            )

        row["hypertension"] = 1 if data.get("hypertension") else 0

        row["heart_disease"] = 1 if data.get("heart_disease") else 0

        # race

        race = data.get("race")

        if race:

            race_col = f"race:{race}"

            if race_col in row:
                row[race_col] = 1

        # smoking

        smoke = data.get("smoking_history")

        if smoke:

            smoke_col = f"smoking_history_{smoke}"

            if smoke_col in row:
                row[smoke_col] = 1

        # location

        loc = data.get("location")

        if loc:

            loc_col = f"location_{loc}"

            if loc_col in row:
                row[loc_col] = 1

        # create feature vector

        vals = [row[c] for c in FEATURE_COLS]

        print("Feature Count:", len(vals))

        if hasattr(scaler, "n_features_in_"):
            print("Scaler Expected:", scaler.n_features_in_)

        features = np.array([vals], dtype=float)

        print("Before Scaling")

        scaled = scaler.transform(features)

        print("After Scaling")

        result = model.predict(scaled)[0]

        print("Prediction:", result)

        probability = None

        try:

            probs = model.predict_proba(scaled)[0]

            if hasattr(model, 'classes_'):

                if 1 in model.classes_:

                    idx = list(model.classes_).index(1)

                    probability = float(probs[idx])

                else:

                    probability = float(probs.max())

            else:

                probability = float(probs.max())

        except Exception as prob_error:

            print("Probability Error:", str(prob_error))

        return jsonify({
            "result": "Diabetic" if int(result) == 1 else "Not Diabetic",
            "probability": probability
        })

    except Exception as e:

        print("========== PREDICT ERROR ==========")
        print(str(e))
        traceback.print_exc()
        print("===================================")

        return jsonify({
            "error": str(e)
        }), 500

# =========================
# REPORT
# =========================

@app.route("/report", methods=["POST"])
def report():

    try:

        payload = request.json or {}

        user = session.get('user', 'User')

        prediction = payload.get('prediction', 'Unknown')

        probability = payload.get('probability')

        pdf_buffer = io.BytesIO()

        doc = SimpleDocTemplate(pdf_buffer)

        styles = getSampleStyleSheet()

        story = []

        title = Paragraph(
            "Early Diabetes Prediction Report",
            styles['Title']
        )

        story.append(title)
        story.append(Spacer(1, 20))

        story.append(Paragraph(
            f"<b>User:</b> {user}",
            styles['BodyText']
        ))

        story.append(Paragraph(
            f"<b>Prediction:</b> {prediction}",
            styles['BodyText']
        ))

        if probability is not None:

            story.append(Paragraph(
                f"<b>Probability:</b> {round(float(probability)*100,2)}%",
                styles['BodyText']
            ))

        story.append(Spacer(1, 20))

        # QR

        report_id = str(uuid.uuid4())[:8]

        qr = QrCodeWidget(report_id)

        bounds = qr.getBounds()

        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]

        drawing = Drawing(60, 60)

        drawing.add(qr)

        png = renderPM.drawToString(drawing, fmt='PNG')

        qr_img = io.BytesIO(png)

        story.append(
            RLImage(qr_img, width=60, height=60)
        )

        story.append(Spacer(1, 20))

        story.append(Paragraph(
            "This report is AI generated and not a substitute for medical advice.",
            styles['Italic']
        ))

        doc.build(story)

        pdf_buffer.seek(0)

        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name="diabetes_report.pdf",
            mimetype="application/pdf"
        )

    except Exception as e:

        print("REPORT ERROR")
        print(str(e))

        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500

# =========================
# MAIN
# =========================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )