import pdfplumber
import pandas as pd
import re

PDF_PATH = "data/pdf24_merged (1).pdf"


OUTPUT_CSV = "data/ecg_data.csv"

records = []

def find(pattern, text):
    match = re.search(pattern, text)
    return match.group(1).strip() if match else None

with pdfplumber.open(PDF_PATH) as pdf:
    print(f"📄 Total pages found: {len(pdf.pages)}")

    for i, page in enumerate(pdf.pages, start=1):
        text = page.extract_text()
        if not text:
            continue

        record = {}

        # Demographics
        record["Age"] = find(r"(\d+)\s*Years", text)
        record["Gender"] = find(r"Years\s*(Male|Female)", text)

        # ECG Parameters
        record["HR"] = find(r"HR\s*:\s*(\d+)", text)
        record["P_ms"] = find(r"P\s*:\s*(\d+)\s*ms", text)
        record["PR_ms"] = find(r"PR\s*:\s*(\d+)\s*ms", text)
        record["QRS_ms"] = find(r"QRS\s*:\s*(\d+)\s*ms", text)

        qt_match = re.search(r"QT/QTc\s*:\s*(\d+)/(\d+)", text)
        record["QT_ms"] = qt_match.group(1) if qt_match else None
        record["QTc_ms"] = qt_match.group(2) if qt_match else None

        axis = re.search(r"P/QRS/T\s*:\s*([-\d]+)/([-\d]+)/([-\d]+)", text)
        record["P_axis"] = axis.group(1) if axis else None
        record["QRS_axis"] = axis.group(2) if axis else None
        record["T_axis"] = axis.group(3) if axis else None

        rv = re.search(r"RV5/SV1\s*:\s*([\d.]+)/([\d.]+)", text)
        record["RV5"] = rv.group(1) if rv else None
        record["SV1"] = rv.group(2) if rv else None

        # Diagnosis (MULTI-LINE SAFE)
        diagnosis_block = re.search(
            r"Diagnosis Information:\s*(.*?)\nReport Confirmed by:",
            text,
            re.DOTALL
        )

        if diagnosis_block:
            diagnosis = diagnosis_block.group(1)
            diagnosis = " | ".join(
                line.strip() for line in diagnosis.splitlines() if line.strip()
            )
            record["Diagnosis"] = diagnosis
        else:
            record["Diagnosis"] = None

        # Save only valid ECGs
        if record["HR"] is not None:
            records.append(record)

        if i % 50 == 0:
            print(f"✅ Processed {i} pages")

# Create DataFrame
df = pd.DataFrame(records)

# Save CSV
df.to_csv(OUTPUT_CSV, index=False)

print("\n DONE!")
print(f"📊 Total ECG records extracted: {len(df)}")
print(f" Saved to: {OUTPUT_CSV}")
print(df.head())
