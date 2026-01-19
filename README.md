# Early-Diabetes-Prediction

CLEANED CODE :
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from google.colab import files

uploaded = files.upload()
df = pd.read_csv(list(uploaded.keys())[0])

print(df.info())
print(df.isnull().sum())

df.replace(['No Info', 'no info', 'Unknown', 'unknown', 'NA', ''], np.nan, inplace=True)

numeric_cols = ['age', 'bmi', 'hbA1c_level', 'blood_glucose_level']
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

categorical_cols = ['gender', 'location', 'smoking_history']
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

df.drop_duplicates(inplace=True)

df['age'] = df['age'].astype(int)
df['hypertension'] = df['hypertension'].astype(int)
df['heart_disease'] = df['heart_disease'].astype(int)
df['diabetes'] = df['diabetes'].astype(int)


df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})


df = pd.get_dummies(df, columns=['smoking_history'], drop_first=True)
df = pd.get_dummies(df, columns=['location'], drop_first=True)
X = df.drop('diabetes', axis=1)
y = df['diabetes']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


print("Final dataset shape:", df.shape)
print("Missing values after cleaning:\n", df.isnull().sum())
print("Scaled feature shape:", X_scaled.shape)


DOWNLOAD CODE:
from google.colab import files

# Save the cleaned DataFrame to a CSV file
df.to_csv('cleaned_diabetes_dataset.csv', index=False)

# Download the file
files.download('cleaned_diabetes_dataset.csv')

print("Cleaned dataset downloaded as 'cleaned_diabetes_dataset.csv'")