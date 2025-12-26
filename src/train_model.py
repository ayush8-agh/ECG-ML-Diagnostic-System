import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load dataset
# -----------------------------
DATA_PATH = "data/ecg_data.csv"
df = pd.read_csv(DATA_PATH)

print("📊 Original dataset shape:", df.shape)

# -----------------------------
# Clean Diagnosis
# -----------------------------
df["Diagnosis"] = df["Diagnosis"].astype(str).str.strip()

# Take ONLY the first diagnosis (most important)
df["Diagnosis"] = df["Diagnosis"].str.split("|").str[0].str.strip()

print("📌 Unique diagnoses:", df["Diagnosis"].nunique())
print(df["Diagnosis"].value_counts().head())

# -----------------------------
# Encode Gender
# -----------------------------
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

# Fill missing gender with most common
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])

# -----------------------------
# Handle missing numeric values
# -----------------------------
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# -----------------------------
# Encode target labels
# -----------------------------
le = LabelEncoder()
df["label"] = le.fit_transform(df["Diagnosis"])

# -----------------------------
# Prepare features and target
# -----------------------------
X = df.drop(["Diagnosis", "label"], axis=1)
y = df["label"]

print("📌 Samples:", X.shape[0], "Features:", X.shape[1])
print("📌 Classes:", le.classes_)

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Train model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=18,
    random_state=42,
    n_jobs=-1
)

print("🚀 Training model...")
model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_test)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📄 Classification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# Save artifacts
# -----------------------------
joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\n💾 Model saved as model.pkl")
print("💾 Label encoder saved as label_encoder.pkl")


