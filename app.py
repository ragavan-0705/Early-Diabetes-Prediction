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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.graphics.barcode.qr import QrCodeWidget
from reportlab.graphics.shapes import Drawing
from reportlab.graphics import renderPM
import uuid
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
    try:
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
                plt_colors = []
                if hb is not None:
                    labels.append('HbA1c')
                    vals.append(hb)
                    plt_colors.append('#ff7f0e')
                if bg is not None:
                    labels.append('Blood Glucose')
                    vals.append(bg)
                    plt_colors.append('#1f77b4')
                ax.bar(labels, vals, color=plt_colors)
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
                    pie_colors = ['#d62728']
                else:
                    vals = [1.0]
                    labels = ['Not Diabetic']
                    pie_colors = ['#2ca02c']
            else:
                p = float(probability)
                vals = [p, 1-p]
                labels = ['Diabetic', 'Not Diabetic']
                pie_colors = ['#d62728', '#2ca02c']

            ax2.pie(vals, labels=labels, autopct=lambda pct: f"{pct:.0f}%", colors=pie_colors, startangle=90)
            ax2.set_title('Model Confidence')
            buf2 = io.BytesIO()
            fig2.tight_layout()
            fig2.savefig(buf2, format='png')
            plt.close(fig2)
            buf2.seek(0)
            images.append(buf2)
        except Exception:
            # continue without images
            images = []

        # Build a professional PDF using ReportLab platypus
        pdf_buf = io.BytesIO()

        styles = getSampleStyleSheet()
        normal = styles['Normal']
        h1 = ParagraphStyle('h1', parent=styles['Heading1'], alignment=1, fontSize=18, leading=22)
        h2 = ParagraphStyle('h2', parent=styles['Heading2'], fontSize=14, leading=18)
        small = ParagraphStyle('small', parent=styles['Normal'], fontSize=9, textColor=colors.grey)

        report_id = f"EDPR-{datetime.datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        model_name = type(model).__name__ if model is not None else 'Model'
        gen_time = datetime.datetime.utcnow().strftime('%d %b %Y %H:%M UTC')

        doc = SimpleDocTemplate(pdf_buf, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        story = []

        # Header
        story.append(Paragraph('Early Diabetes Risk Assessment Report', h1))
        story.append(Spacer(1, 6))
        meta = f"Generated on: {gen_time} | Report ID: {report_id} | Model: {model_name}"
        story.append(Paragraph(meta, small))
        story.append(Spacer(1, 12))

        # Patient summary
        story.append(Paragraph('Patient Summary', h2))
        patient_rows = []
        patient_rows.append(['Patient Name', user or '—'])
        if payload.get('age'):
            patient_rows.append(['Age', str(payload.get('age'))])
        if payload.get('gender'):
            patient_rows.append(['Gender', str(payload.get('gender'))])
        if session.get('email'):
            patient_rows.append(['Email', session.get('email')])

        t = Table(patient_rows, hAlign='LEFT', colWidths=[120, 360])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
            ('BOX', (0,0), (-1,-1), 0.5, colors.grey),
            ('INNERGRID', (0,0), (-1,-1), 0.25, colors.grey),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))

        # Input parameters table with normal ranges
        story.append(Paragraph('Input Parameters', h2))
        normal_ranges = {
            'blood_glucose_level': '70–140 mg/dL (fasting normal <100)',
            'hbA1c_level': '<5.7% (normal)',
            'bmi': '18.5–24.9',
            'age': '—',
            'hypertension': 'No / Yes',
            'heart_disease': 'No / Yes'
        }

        param_rows = [['Parameter', 'Value', 'Normal Range']]
        for key, label in [('blood_glucose_level','Glucose (mg/dL)'), ('hbA1c_level','HbA1c (%)'), ('bmi','BMI'), ('age','Age'), ('hypertension','Hypertension'), ('heart_disease','Heart Disease')]:
            val = payload.get(key)
            val_display = str(val) if val is not None and val != '' else '—'
            param_rows.append([label, val_display, normal_ranges.get(key, '—')])

        pt = Table(param_rows, colWidths=[160, 120, 200])
        pt.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f0f4ff')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#2438c7')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(pt)
        story.append(Spacer(1, 12))

        # Prediction block
        story.append(Paragraph('AI Prediction Result', h2))
        prob_text = f"Predicted Probability of Diabetes: {float(probability)*100:.1f}%" if probability is not None else 'Predicted Probability: N/A'
        # Determine risk level
        risk = 'UNKNOWN'
        risk_color = colors.grey
        try:
            p = float(probability) if probability is not None else None
            if p is None:
                risk = 'HIGH RISK' if prediction == 'Diabetic' else 'LOW RISK'
                risk_color = colors.red if prediction == 'Diabetic' else colors.green
            else:
                if p >= 0.7:
                    risk = 'HIGH RISK'
                    risk_color = colors.red
                elif p >= 0.4:
                    risk = 'MODERATE RISK'
                    risk_color = colors.orange
                else:
                    risk = 'LOW RISK'
                    risk_color = colors.green
        except Exception:
            pass

        pred_table = Table([[Paragraph(f'<b>{risk}</b>', ParagraphStyle('pred', fontSize=16, alignment=1, textColor=colors.white))]], colWidths=[440], rowHeights=[40])
        pred_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), risk_color),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(pred_table)
        story.append(Spacer(1, 6))
        story.append(Paragraph(prob_text, normal))
        story.append(Spacer(1, 12))

        # Interpretation & contributing factors
        story.append(Paragraph('Risk Interpretation', h2))
        factors = []
        try:
            if payload.get('blood_glucose_level') and float(payload.get('blood_glucose_level')) >= 126:
                factors.append('Elevated blood glucose')
            if payload.get('hbA1c_level') and float(payload.get('hbA1c_level')) >= 6.5:
                factors.append('High HbA1c')
            if payload.get('bmi') and float(payload.get('bmi')) >= 30:
                factors.append('High BMI (Obesity)')
            if payload.get('hypertension') in [1, '1', True, 'true', 'True']:
                factors.append('Hypertension')
            if payload.get('heart_disease') in [1, '1', True, 'true', 'True']:
                factors.append('History of heart disease')
        except Exception:
            pass

        interp_lines = ['Based on the provided parameters, the model indicates:']
        interp_lines += ['- ' + f for f in factors] if factors else ['- No major single factor detected from inputs.']
        interp_para = Paragraph('<br/>'.join(interp_lines), normal)
        story.append(interp_para)
        story.append(Spacer(1, 12))

        # Recommendations
        story.append(Paragraph('Personalized Recommendations', h2))
        recs = [
            'Maintain a balanced, low-sugar diet and reduce processed foods.',
            'Aim for at least 30 minutes of moderate exercise most days.',
            'Schedule HbA1c and fasting glucose tests as advised by a clinician.',
            'Consult a healthcare professional for personalized care.'
        ]
        # Add targeted recs
        if 'High BMI' in ' '.join(factors) or any('BMI' in f for f in factors):
            recs.insert(0, 'Weight management program: reduce caloric intake and increase activity.')

        rec_para = Paragraph('<br/>'.join(['• ' + r for r in recs]), normal)
        story.append(rec_para)
        story.append(Spacer(1, 12))

        # Charts
        if images:
            story.append(Paragraph('Visuals', h2))
            for img_buf in images:
                try:
                    img_buf.seek(0)
                    rl_img = RLImage(img_buf, width=5.5*inch, height=2.2*inch)
                    story.append(rl_img)
                    story.append(Spacer(1, 6))
                except Exception:
                    pass

        # QR Code and disclaimer
        story.append(Spacer(1, 10))
        qr = QrCodeWidget(report_id)
        b = qr.getBounds()
        w = b[2] - b[0]
        h = b[3] - b[1]
        d = Drawing(60, 60)
        d.add(qr)
        # render to PNG then include
        try:
            png = renderPM.drawToString(d, fmt='PNG')
            qr_img = io.BytesIO(png)
            story.append(RLImage(qr_img, width=60, height=60))
        except Exception:
            pass

        story.append(Spacer(1, 12))
        disclaimer = ('Disclaimer: This report is generated by an AI-based predictive system and is not a substitute for professional '
                      'medical diagnosis. Please consult a healthcare professional for confirmation and personalized advice.')
        story.append(Paragraph(disclaimer, ParagraphStyle('disc', fontSize=8, textColor=colors.grey)))

        # Build PDF
        doc.build(story)
        pdf_buf.seek(0)

        filename = f"EDPR_{report_id}.pdf"
        return send_file(pdf_buf, mimetype='application/pdf', as_attachment=True, download_name=filename)
    except Exception as e:
        app.logger.exception('Report generation failed')
        # Return JSON error so client can show details during debugging
        return jsonify({'error': 'Report generation failed', 'detail': str(e)}), 500

# =====================
# MAIN
# =====================
if __name__ == "__main__":
    app.run(debug=True)
