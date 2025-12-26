❤️ ECG Machine Learning Diagnostic System

An end-to-end ECG analysis and diagnostic support system that combines Machine Learning with clinical rule-based logic to provide meaningful, patient-friendly ECG interpretations through an interactive Streamlit dashboard.

📌 Project Overview

Electrocardiogram (ECG) interpretation is a critical task in healthcare. This project aims to assist ECG interpretation by:

Extracting ECG parameters from PDF reports

Training a machine learning model on clinical ECG data

Applying clinical rules for safety-critical conditions

Presenting results in a user-friendly dashboard with explanations and risk levels

The system is designed as a decision-support tool, not a replacement for medical professionals.

✨ Key Features

📄 Automatic PDF → CSV conversion

🧠 Machine Learning–based ECG classification

🩺 Clinical rule-based safety overrides

📊 Confidence score for predictions

🚦 Risk level indicator (Low / Medium / High)

👨‍⚕️ Patient-friendly explanations

🎨 Clean and readable Streamlit UI

🔍 Explainable decision source (ML or Clinical Rule)

🧪 ECG Parameters Used

The system uses the following ECG and patient parameters:

Age

Gender

Heart Rate (HR)

P duration (ms)

PR interval (ms)

QRS duration (ms)

QT interval (ms)

QTc interval (ms)

P axis (°)

QRS axis (°)

T axis (°)

RV5 (mV)

SV1 (mV)

🧠 System Architecture
ECG PDF Reports
        ↓
PDF → CSV Conversion
        ↓
Data Cleaning & Feature Engineering
        ↓
Machine Learning Model (Random Forest)
        ↓
Clinical Rule Validation
        ↓
Streamlit Dashboard (User Interface)

🛠️ Technologies Used

Python

Pandas & NumPy – data processing

Scikit-learn – machine learning

Streamlit – interactive dashboard

Joblib – model persistence

pdfplumber / pypdfium2 – PDF parsing

📦 Installation & Setup
🔹 Prerequisites

Python 3.9 or higher

Git (optional)

Windows / Linux / macOS

🔹 Step 1: Clone the Repository
git clone https://github.com/your-username/ECG-ML-Projec.git
cd ECG-ML-Projec

🔹 Step 2: Create a Virtual Environment

Windows (PowerShell):

python -m venv venv
.\venv\Scripts\Activate.ps1


Linux / macOS:

python3 -m venv venv
source venv/bin/activate

🔹 Step 3: Install Dependencies
pip install -r requirements.txt

🔹 Step 4: Convert ECG PDFs to CSV

Place ECG PDF files inside the data/ directory:

python src/pdf_to_csv.py


This generates ecg_data.csv.

🔹 Step 5: Train the Machine Learning Model
python src/train_model.py


This creates:

model.pkl

label_encoder.pkl

🔹 Step 6: Run the Application
streamlit run app.py


Open your browser at:

http://localhost:8501

📊 Output Interpretation
Example Output:

Diagnosis: Normal Sinus Rhythm

Explanation: Patient-friendly summary

Confidence: Model confidence percentage

Risk Level: Low / Medium / High

Decision Source: Machine Learning or Clinical Rule

🚦 Risk Classification Logic
Condition	Risk Level
Normal Sinus Rhythm	Low
Sinus Tachycardia / Bradycardia	Medium
Prolonged QT / Wide QRS	High
⚠️ Disclaimer

This project is intended for educational and research purposes only.
It is not a medical device and must not be used for real clinical diagnosis or treatment decisions.

Always consult a qualified healthcare professional.

🎓 Academic Value

This project demonstrates:

Real-world data preprocessing

Handling class imbalance in medical ML

Hybrid ML + rule-based decision systems

Explainable AI concepts

User-centered medical UI design

📌 Future Improvements

ECG signal waveform analysis

PDF auto-parameter extraction accuracy improvements

Feature importance visualization

Online deployment (Streamlit Cloud / Hugging Face)

Integration with wearable ECG devices

👤 Author

Ayush Trivedi
Mechanical Engineering Undergraduate
Interest Areas: AI, Robotics, Healthcare ML

⭐ Acknowledgements

Open-source Python community

Medical ECG interpretation references

Streamlit & Scikit-learn documentation