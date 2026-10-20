import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide"
)

DATA_PATH = Path(__file__).parent / "data" / "students.csv"

FEATURES = [
    "study_hours",
    "attendance",
    "previous_marks",
    "sleep_hours"
]
TARGET = "final_marks"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def prepare_data(data):
    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    clean = data[required].copy()

    for col in required:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean = clean.dropna()

    if len(clean) < 5:
        raise ValueError("At least 5 valid rows are required.")

    return clean


@st.cache_resource
def train_models(data):
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=4, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
    }

    trained = {}
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)

        # R² can be misleading for a very small test set.
        r2 = (
            r2_score(y_test, predictions)
            if len(y_test) >= 2 else float("nan")
        )

        trained[name] = model

        results.append({
            "Model": name,
            "MAE": round(mae, 2),
            "R² Score": round(r2, 3)
        })

    return trained, pd.DataFrame(results)


st.title("🎓 Student Performance Analysis & Prediction")
st.caption(
    "An educational machine-learning dashboard for exploring "
    "student data and estimating final marks."
)

st.info(
    "Demo project: the included dataset contains only 20 illustrative "
    "records. Predictions and evaluation scores are not reliable "
    "evidence of real-world student performance."
)

with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Choose a dataset for analysis.")

    uploaded_file = st.file_uploader(
        "Upload CSV dataset",
        type=["csv"]
    )

    model_choice = st.selectbox(
        "Prediction model",
        [
            "Linear Regression",
            "Decision Tree",
            "Random Forest"
        ]
    )

try:
    if uploaded_file is not None:
        raw_data = pd.read_csv(uploaded_file)
    else:
        raw_data = load_data()

    data = prepare_data(raw_data)

except Exception as error:
    st.error(f"Unable to load the dataset: {error}")
    st.stop()

try:
    models, results = train_models(data)
except Exception as error:
    st.error(f"Unable to train models: {error}")
    st.stop()

overview, predictor, analysis, comparison = st.tabs([
    "📊 Dataset Overview",
    "🔮 Predict Marks",
    "📈 Visual Analysis",
    "🤖 Model Comparison"
])

with overview:
    st.subheader("Dataset Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Valid Students", len(data))
    c2.metric("Average Final Marks", f"{data[TARGET].mean():.1f}")
    c3.metric("Average Attendance", f"{data['attendance'].mean():.1f}%")
    c4.metric("Missing Cells", int(raw_data.isna().sum().sum()))

    st.subheader("Student Dataset")
    st.dataframe(data, use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.dataframe(data.describe().round(2), use_container_width=True)

    st.subheader("Data Quality")
    q1, q2 = st.columns(2)
    q1.metric("Duplicate Rows", int(data.duplicated().sum()))
    q2.metric("Invalid Rows Removed", max(0, len(raw_data) - len(data)))

    csv_data = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Clean Dataset",
        data=csv_data,
        file_name="clean_student_dataset.csv",
        mime="text/csv"
    )

with predictor:
    st.subheader("Estimate Final Marks")

    left, right = st.columns(2)

    with left:
        study_hours = st.slider(
            "Daily study hours", 0.0, 12.0, 5.0, 0.5
        )
        attendance = st.slider(
            "Attendance (%)", 0, 100, 85
        )

    with right:
        previous_marks = st.slider(
            "Previous marks (out of 100)", 0, 100, 75
        )
        sleep_hours = st.slider(
            "Sleep per day (hours)", 0.0, 12.0, 8.0, 0.5
        )

    if st.button("🔮 Predict Final Marks", type="primary"):
        student = pd.DataFrame([{
            "study_hours": study_hours,
            "attendance": attendance,
            "previous_marks": previous_marks,
            "sleep_hours": sleep_hours
        }])

        selected_model = models[model_choice]
        prediction = float(selected_model.predict(student)[0])
        prediction = max(0.0, min(100.0, prediction))

        st.metric("Estimated Final Marks", f"{prediction:.1f} / 100")

        if prediction < 40:
            st.warning(
                "The estimate suggests the student may benefit from "
                "additional academic support. It is not a final judgment."
            )
        else:
            st.success(
                "Use this estimate as a starting point for discussion, "
                "not as a guarantee of actual results."
            )

        st.caption(
            "This estimate comes from a small illustrative dataset "
            "and should not be used to make decisions about real students."
        )

with analysis:
    st.subheader("Study Hours vs Final Marks")

    fig, ax = plt.subplots()
    ax.scatter(data["study_hours"], data["final_marks"])
    ax.set_xlabel("Daily Study Hours")
    ax.set_ylabel("Final Marks")
    ax.set_title("Study Hours vs Final Marks")
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Average Marks by Study Hours")

    grouped = data.groupby("study_hours")["final_marks"].mean()

    fig, ax = plt.subplots()
    grouped.plot(kind="bar", ax=ax)
    ax.set_xlabel("Daily Study Hours")
    ax.set_ylabel("Average Final Marks")
    ax.set_title("Average Marks by Study Hours")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Feature Relationships")

    corr = data[FEATURES + [TARGET]].corr()

    fig, ax = plt.subplots()
    image = ax.imshow(corr, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    ax.set_title("Correlation Matrix")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with comparison:
    st.subheader("Compare Machine-Learning Models")

    st.dataframe(results, use_container_width=True)

    st.caption(
        "Lower MAE generally indicates smaller prediction errors. "
        "R² can be unstable with very small test datasets. "
        "These scores are illustrative, not a reliable model ranking."
    )

    fig, ax = plt.subplots()
    ax.bar(results["Model"], results["MAE"])
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("Model Error Comparison")
    ax.tick_params(axis="x", rotation=15)
    st.pyplot(fig)
    plt.close(fig)

    report = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Model Comparison Report",
        data=report,
        file_name="model_comparison_report.csv",
        mime="text/csv"
    )

st.divider()
st.caption(
    "Student Performance AI | Educational BCA Data Science project"
)
