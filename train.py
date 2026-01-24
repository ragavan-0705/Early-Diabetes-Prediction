import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1️⃣ Load dataset
df = pd.read_csv("diabetes.csv")

# ✅ FIX 1: Handle missing values
df = df.fillna(df.median())

# ✅ FIX 2: Convert bool to int
df = df.replace({True: 1, False: 0})

print(df.info())

# 2️⃣ Split features & target
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# 3️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4️⃣ Scaling (for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
results["Logistic Regression"] = accuracy_score(y_test, lr.predict(X_test_scaled))

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
results["Decision Tree"] = accuracy_score(y_test, dt.predict(X_test))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
results["Random Forest"] = accuracy_score(y_test, rf.predict(X_test))

# 5️⃣ Print results
print("\nModel Accuracy Comparison:")
for model, acc in results.items():
    print(f"{model}: {acc:.4f}")

# 6️⃣ Save best model
os.makedirs("model", exist_ok=True)
joblib.dump(rf, "model/diabetes_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\n✅ Model and scaler saved successfully!")
