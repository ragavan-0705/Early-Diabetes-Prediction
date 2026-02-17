from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, send_file
import pandas as pd
import sqlite3
import numpy as np
import joblib
import os
import io
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
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

    # try to extract probability if available
    proba = None
    try:
        probs = model.predict_proba(scaled)[0]
        # probs ordering: [not_diabetic_prob, diabetic_prob] or vice-versa depending on model classes_
        if hasattr(model, 'classes_'):
            # find index for class 1
            if 1 in model.classes_:
                idx = list(model.classes_).index(1)
                proba = float(probs[idx])
            else:
                proba = float(probs.max())
        else:
            proba = float(probs.max())
    except Exception:
        proba = None

    return jsonify({
        "result": "Diabetic" if int(result) == 1 else "Not Diabetic",
        "probability": proba
    })


@app.route('/report', methods=['POST'])
def report():
    # Accept JSON describing the user's input and prediction
    payload = request.json or {}

    user = session.get('user', payload.get('name', 'User'))
    prediction = payload.get('prediction') or payload.get('result') or 'Unknown'
    probability = payload.get('probability')

    # build simple advice based on prediction
    if prediction == 'Diabetic':
        advice = "Your result suggests diabetes risk. See a healthcare professional for confirmation, start lifestyle changes (diet, exercise), and monitor blood sugar regularly."
    elif prediction == 'Not Diabetic':
        advice = "Your result suggests low immediate risk. Maintain healthy habits: balanced diet, regular exercise, routine checkups."
    else:
        advice = "No clear prediction available. Consider filling all fields and trying again."

    # Create small charts using matplotlib
    images = []
    try:
        # Chart 1: HbA1c and Blood Glucose if provided
        hb = None
        bg = None
        try:
            hb = float(payload.get('hbA1c_level')) if payload.get('hbA1c_level') else None
        except Exception:
            hb = None
        try:
            bg = float(payload.get('blood_glucose_level')) if payload.get('blood_glucose_level') else None
        except Exception:
            bg = None

        if hb is not None or bg is not None:
            fig, ax = plt.subplots(figsize=(6,3))
            labels = []
            vals = []
            colors = []
            if hb is not None:
                labels.append('HbA1c')
                vals.append(hb)
                colors.append('#ff7f0e')
            if bg is not None:
                labels.append('Blood Glucose')
                vals.append(bg)
                colors.append('#1f77b4')
            ax.bar(labels, vals, color=colors)
            ax.set_title('Measured Indicators')
            ax.axhline(5.7, color='grey', linestyle='--', linewidth=0.7)
            buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            images.append(buf)

        # Chart 2: simple pie for predicted probability
        fig2, ax2 = plt.subplots(figsize=(4,3))
        if probability is None:
            # fallback: show predicted class as full
            if prediction == 'Diabetic':
                vals = [1.0]
                labels = ['Diabetic']
                colors = ['#d62728']
            else:
                vals = [1.0]
                labels = ['Not Diabetic']
                colors = ['#2ca02c']
        else:
            p = float(probability)
            vals = [p, 1-p]
            labels = ['Diabetic', 'Not Diabetic']
            colors = ['#d62728', '#2ca02c']

        ax2.pie(vals, labels=labels, autopct=lambda pct: f"{pct:.0f}%", colors=colors, startangle=90)
        ax2.set_title('Model Confidence')
        buf2 = io.BytesIO()
        fig2.tight_layout()
        fig2.savefig(buf2, format='png')
        plt.close(fig2)
        buf2.seek(0)
        images.append(buf2)
    except Exception as e:
        # continue without images
        images = []

    # Build PDF in memory
    pdf_buf = io.BytesIO()
    c = pdf_canvas.Canvas(pdf_buf, pagesize=letter)
    width, height = letter

    c.setFont('Helvetica-Bold', 18)
    c.drawString(40, height - 60, 'Diabetes Risk Report')
    c.setFont('Helvetica', 12)
    c.drawString(40, height - 90, f'Name: {user}')
    c.drawString(40, height - 110, f'Date: {datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}')

    c.setFont('Helvetica-Bold', 14)
    c.drawString(40, height - 150, 'Prediction:')
    c.setFont('Helvetica', 14)
    c.drawString(140, height - 150, f'{prediction}')
    if probability is not None:
        c.setFont('Helvetica', 12)
        c.drawString(40, height - 170, f'Confidence: {probability*100:.1f}%')

    # Advice
    text = c.beginText(40, height - 200)
    text.setFont('Helvetica', 11)
    text.textLines(['Advice:', advice])
    c.drawText(text)

    # Insert images if any
    img_y = height - 360
    for img_buf in images:
        try:
            img = ImageReader(img_buf)
            c.drawImage(img, 40, img_y, width=500, preserveAspectRatio=True, mask='auto')
            img_y -= 200
        except Exception:
            pass

    c.showPage()
    c.save()
    pdf_buf.seek(0)

    filename = f"diabetes_report_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(pdf_buf, mimetype='application/pdf', as_attachment=True, download_name=filename)

# =====================
# MAIN
# =====================
if __name__ == "__main__":
    app.run(debug=True)
