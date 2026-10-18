import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Student Performance Predictor", page_icon="📚", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "students.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def train_model(data):
    features = ["study_hours", "attendance", "previous_marks", "sleep_hours"]
    X = data[features]
    y = data["final_marks"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return model, mae, r2, y_test, predictions

st.title("📚 Student Performance Analysis & Prediction")
st.caption("A student project demonstrating data analysis and Linear Regression.")

data = load_data()
model, mae, r2, y_test, predictions = train_model(data)

overview, predictor, charts = st.tabs(["Dataset Overview", "Predict Marks", "Visual Analysis"])

with overview:
    c1, c2, c3 = st.columns(3)
    c1.metric("Students in sample dataset", len(data))
    c2.metric("Average final marks", f"{data['final_marks'].mean():.1f}")
    c3.metric("Average attendance", f"{data['attendance'].mean():.1f}%")
    st.subheader("Sample dataset")
    st.dataframe(data, use_container_width=True)
    st.subheader("Data quality checks")
    q1, q2 = st.columns(2)
    q1.metric("Missing cells", int(data.isna().sum().sum()))
    q2.metric("Duplicate rows", int(data.duplicated().sum()))

with predictor:
    st.write("Enter a student's information to estimate their final marks.")
    left, right = st.columns(2)
    with left:
        study_hours = st.slider("Daily study hours", 0.0, 12.0, 5.0, 0.5)
        attendance = st.slider("Attendance (%)", 0, 100, 85)
    with right:
        previous_marks = st.slider("Previous marks (out of 100)", 0, 100, 75)
        sleep_hours = st.slider("Sleep per day (hours)", 0.0, 12.0, 8.0, 0.5)
    if st.button("Predict final marks", type="primary"):
        new_student = pd.DataFrame([{
            "study_hours": study_hours,
            "attendance": attendance,
            "previous_marks": previous_marks,
            "sleep_hours": sleep_hours
        }])
        prediction = float(model.predict(new_student)[0])
        prediction = max(0.0, min(100.0, prediction))
        st.success(f"Estimated final marks: {prediction:.1f} / 100")
        st.info("This is an educational estimate from a tiny synthetic sample, not a reliable prediction of a real student's result.")

with charts:
    st.subheader("Study hours vs final marks")
    fig, ax = plt.subplots()
    ax.scatter(data["study_hours"], data["final_marks"])
    ax.set_xlabel("Daily study hours")
    ax.set_ylabel("Final marks")
    ax.set_title("Study Hours vs Final Marks")
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Average final marks by study-hours group")
    grouped = data.groupby("study_hours", as_index=False)["final_marks"].mean()
    fig2, ax2 = plt.subplots()
    ax2.bar(grouped["study_hours"].astype(str), grouped["final_marks"])
    ax2.set_xlabel("Daily study hours")
    ax2.set_ylabel("Average final marks")
    ax2.set_title("Average Marks by Study Hours")
    st.pyplot(fig2)
    plt.close(fig2)

st.divider()
st.subheader("Model evaluation (small sample)")
m1, m2 = st.columns(2)
m1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
m2.metric("R² score", f"{r2:.2f}")
st.caption("The dataset contains only 20 illustrative records. Evaluation scores can be unstable and must not be treated as evidence of real-world accuracy.")
