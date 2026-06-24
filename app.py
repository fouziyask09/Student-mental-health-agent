import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Student Mental Health Agent")

st.title("🎓 Student Mental Health Assessment Agent")

# Inputs
age = st.number_input("Age", min_value=18, max_value=40, value=20)

marital_status = st.selectbox(
    "Marital Status",
    ["0", "1"]
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

course = st.selectbox(
    "Course",
    [
        "ala",
        "banking studies",
        "bcs",
        "benl",
        "biomedical science",
        "biotechnology",
        "bit",
        "business administration",
        "communication",
        "cts",
        "diploma nursing",
        "diploma tesl",
        "econs",
        "engin",
        "engine",
        "engineering",
        "enm",
        "fiqh",
        "fiqh fatwa",
        "human resources",
        "human sciences",
        "irkhs",
        "islamic education",
        "it",
        "kenms",
        "kirkhs",
        "koe",
        "kop",
        "law",
        "laws",
        "malcom",
        "marine science",
        "mathemathics",
        "mhsc",
        "nursing",
        "pendidikan islam",
        "psychology",
        "radiography",
        "taasl",
        "usuluddin"
    ]
)

year = st.selectbox(
    "Year",
    ["year 1", "year 2", "year 3", "year 4"]
)

cgpa = st.selectbox(
    "CGPA",
    [
        "0 - 1.99",
        "2.00 - 2.49",
        "2.50 - 2.99",
        "3.00 - 3.49",
        "3.50 - 4.00"
    ]
)

if st.button("Assess Mental Health"):

    # Create empty feature vector
    input_data = pd.DataFrame(
        np.zeros((1, len(feature_names))),
        columns=feature_names
    )

    # Numerical feature
    if "age" in input_data.columns:
        input_data.loc[0, "age"] = age

    # Gender
    if "gender_male" in input_data.columns:
        input_data.loc[0, "gender_male"] = 1 if gender == "Male" else 0

    # Marital Status
    if "marital_status_1" in input_data.columns:
        input_data.loc[0, "marital_status_1"] = 1 if marital_status == "1" else 0

    # Course
    course_col = f"course_{course}"
    if course_col in input_data.columns:
        input_data.loc[0, course_col] = 1

    # Year
    year_col = f"year_{year}"
    if year_col in input_data.columns:
        input_data.loc[0, year_col] = 1

    # CGPA
    cgpa_col = f"cgpa_{cgpa}"
    if cgpa_col in input_data.columns:
        input_data.loc[0, cgpa_col] = 1

    # Reorder columns exactly like training
    input_data = input_data[feature_names]

    # Scale
    scaled_data = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(scaled_data)[0]

    st.subheader("Prediction Result")

    if int(prediction) == 1:
        st.error("⚠️ Depression Risk Detected")

        st.subheader("Recommendations")
        st.write("""
- 😴 Maintain a proper sleep schedule
- 🏃 Exercise regularly
- 📚 Reduce academic stress
- 👨‍👩‍👧 Talk with friends and family
- 🧑‍⚕️ Consult a counselor if needed
""")

    else:
        st.success("✅ No Depression Risk Detected")

        st.subheader("Recommendations")
        st.write("""
- 😊 Continue healthy habits
- ⚖️ Maintain work-life balance
- 🏃 Stay physically active
- 🤝 Stay socially connected
""")

    # Save Record
    record = pd.DataFrame({
        "Age": [age],
        "Marital_Status": [marital_status],
        "Gender": [gender],
        "Course": [course],
        "Year": [year],
        "CGPA": [cgpa],
        "Prediction": [prediction]
    })

    try:
        old = pd.read_csv("student_records.csv")
        pd.concat([old, record], ignore_index=True).to_csv("student_records.csv", index=False)
    except FileNotFoundError:
        record.to_csv("student_records.csv", index=False)

    st.success("✅ Record Saved Successfully")