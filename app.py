import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide",
)

DATA_PATH = Path(__file__).parent / "data" / "students.csv"

FEATURES = [
    "study_hours",
    "attendance",
    "previous_marks",
    "sleep_hours",
]

TARGET = "final_marks"

REQUIRED_COLUMNS = FEATURES + [TARGET]


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

@st.cache_data
def load_default_data():
    return pd.read_csv(DATA_PATH)


def prepare_data(raw_data):
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in raw_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns missing: "
            + ", ".join(missing_columns)
            + ". Expected columns: "
            + ", ".join(REQUIRED_COLUMNS)
        )

    clean_data = raw_data[REQUIRED_COLUMNS].copy()

    for column in REQUIRED_COLUMNS:
        clean_data[column] = pd.to_numeric(
            clean_data[column],
            errors="coerce",
        )

    clean_data = clean_data.replace(
        [float("inf"), float("-inf")],
        float("nan"),
    )

    clean_data = clean_data.dropna()
    clean_data = clean_data.drop_duplicates().reset_index(drop=True)

    # Basic validation of feature ranges.
    valid_rows = (
        clean_data["study_hours"].between(0, 24)
        & clean_data["attendance"].between(0, 100)
        & clean_data["previous_marks"].between(0, 100)
        & clean_data["sleep_hours"].between(0, 24)
        & clean_data["final_marks"].between(0, 100)
    )

    clean_data = clean_data.loc[valid_rows].reset_index(drop=True)

    if len(clean_data) < 10:
        raise ValueError(
            "At least 10 valid records are recommended "
            "for this demo. Check your CSV data."
        )

    return clean_data


# --------------------------------------------------
# MACHINE LEARNING
# --------------------------------------------------

def train_models(data):
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model_definitions = {
        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            max_depth=4,
            min_samples_leaf=2,
            random_state=42,
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=5,
            min_samples_leaf=2,
            random_state=42,
        ),
    }

    trained_models = {}
    report = []

    for name, model in model_definitions.items():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)

        rmse = mean_squared_error(
            y_test,
            predictions,
        ) ** 0.5

        # R² needs at least two test observations.
        if len(y_test) >= 2:
            r2 = r2_score(y_test, predictions)
        else:
            r2 = float("nan")

        trained_models[name] = model

        report.append({
            "Model": name,
            "MAE": round(float(mae), 2),
            "RMSE": round(float(rmse), 2),
            "R² Score": round(float(r2), 3),
        })

    return (
        trained_models,
        pd.DataFrame(report),
        X_test,
        y_test,
    )


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🎓 Student Performance Analysis & Prediction")

st.markdown(
    """
    Explore educational data, visualize relationships between features,
    compare regression models, and generate experimental mark estimates.
    """
)

st.warning(
    "Educational demo only. Predictions are estimates, not guarantees. "
    "Do not use this application alone to make decisions about students."
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("⚙️ Dashboard Settings")

    uploaded_file = st.file_uploader(
        "Upload a CSV dataset",
        type=["csv"],
        help=(
            "The CSV must contain study_hours, attendance, "
            "previous_marks, sleep_hours and final_marks."
        ),
    )

    st.divider()

    st.caption("Required CSV columns")

    for column in REQUIRED_COLUMNS:
        st.code(column, language=None)


# --------------------------------------------------
# LOAD AND VALIDATE DATA
# --------------------------------------------------

try:
    if uploaded_file is not None:
        raw_data = pd.read_csv(uploaded_file)
    else:
        raw_data = load_default_data()

    data = prepare_data(raw_data)

except FileNotFoundError:
    st.error(
        "Dataset not found. Check that the file exists at "
        "`data/students.csv` in your GitHub repository."
    )
    st.stop()

except Exception as error:
    st.error(f"Unable to process the dataset: {error}")
    st.stop()


# --------------------------------------------------
# TRAIN MODELS
# --------------------------------------------------

try:
    # Include data content in the cache key so changed datasets
    # do not accidentally reuse an older training result.
    dataset_hash = hashlib.sha256(
        data.to_csv(index=False).encode("utf-8")
    ).hexdigest()

    @st.cache_resource
    def cached_train_models(dataset_csv, dataset_version):
        training_data = pd.read_csv(
            pd.io.common.StringIO(dataset_csv)
        )
        return train_models(training_data)

    (
        models,
        results,
        X_test,
        y_test,
    ) = cached_train_models(
        data.to_csv(index=False),
        dataset_hash,
    )

except Exception as error:
    st.error(f"Unable to train the models: {error}")
    st.stop()


# --------------------------------------------------
# TABS
# --------------------------------------------------

overview, predictor, analysis, comparison, quality = st.tabs([
    "📊 Dataset Overview",
    "🔮 Predict Marks",
    "📈 Visual Analysis",
    "🤖 Model Comparison",
    "🧹 Data Quality",
])


# --------------------------------------------------
# TAB 1: DATASET OVERVIEW
# --------------------------------------------------

with overview:
    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Valid Records",
        len(data),
    )

    col2.metric(
        "Average Final Marks",
        f"{data[TARGET].mean():.2f}",
    )

    col3.metric(
        "Average Attendance",
        f"{data['attendance'].mean():.1f}%",
    )

    col4.metric(
        "Average Study Hours",
        f"{data['study_hours'].mean():.1f}",
    )

    st.divider()

    st.subheader("Student Dataset")

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Descriptive Statistics")

    st.dataframe(
        data.describe().round(2),
        use_container_width=True,
    )

    st.download_button(
        "📥 Download Clean Dataset",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="clean_student_dataset.csv",
        mime="text/csv",
    )


# --------------------------------------------------
# TAB 2: PREDICTION
# --------------------------------------------------

with predictor:
    st.subheader("🔮 Estimate Final Marks")

    st.caption(
        "Adjust the input values and choose a model to calculate "
        "an experimental estimate."
    )

    left, right = st.columns(2)

    with left:
        study_hours = st.slider(
            "Daily Study Hours",
            min_value=0.0,
            max_value=12.0,
            value=5.0,
            step=0.5,
        )

        attendance = st.slider(
            "Attendance (%)",
            min_value=0,
            max_value=100,
            value=85,
        )

    with right:
        previous_marks = st.slider(
            "Previous Marks",
            min_value=0,
            max_value=100,
            value=75,
        )

        sleep_hours = st.slider(
            "Daily Sleep Hours",
            min_value=0.0,
            max_value=12.0,
            value=8.0,
            step=0.5,
        )

    selected_model_name = st.selectbox(
        "Choose Prediction Model",
        list(models.keys()),
    )

    if st.button(
        "🔮 Predict Final Marks",
        type="primary",
        use_container_width=True,
    ):
        student = pd.DataFrame([{
            "study_hours": study_hours,
            "attendance": attendance,
            "previous_marks": previous_marks,
            "sleep_hours": sleep_hours,
        }])

        selected_model = models[selected_model_name]

        prediction = float(
            selected_model.predict(student)[0]
        )

        prediction = max(0.0, min(100.0, prediction))

        st.metric(
            "Estimated Final Marks",
            f"{prediction:.1f} / 100",
        )

        st.caption(
            f"Model used: {selected_model_name}. "
            "The estimate may be inaccurate, especially when the "
            "training dataset is small or synthetic."
        )

        if prediction < 40:
            st.info(
                "Consider reviewing learning needs and offering "
                "appropriate academic support. This estimate alone "
                "does not determine a student's actual performance."
            )
        else:
            st.success(
                "Prediction generated. Compare it with real outcomes "
                "before drawing conclusions."
            )


# --------------------------------------------------
# TAB 3: VISUAL ANALYSIS
# --------------------------------------------------

with analysis:
    st.subheader("📈 Visual Analysis")

    chart1, chart2 = st.columns(2)

    with chart1:
        st.markdown("**Study Hours vs Final Marks**")

        fig, ax = plt.subplots()

        ax.scatter(
            data["study_hours"],
            data["final_marks"],
        )

        ax.set_xlabel("Study Hours per Day")
        ax.set_ylabel("Final Marks")
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

    with chart2:
        st.markdown("**Attendance vs Final Marks**")

        fig, ax = plt.subplots()

        ax.scatter(
            data["attendance"],
            data["final_marks"],
        )

        ax.set_xlabel("Attendance (%)")
        ax.set_ylabel("Final Marks")
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    chart3, chart4 = st.columns(2)

    with chart3:
        st.markdown("**Distribution of Final Marks**")

        fig, ax = plt.subplots()

        ax.hist(
            data["final_marks"],
            bins=10,
            edgecolor="black",
        )

        ax.set_xlabel("Final Marks")
        ax.set_ylabel("Number of Records")

        st.pyplot(fig)
        plt.close(fig)

    with chart4:
        st.markdown("**Feature Correlation Matrix**")

        correlation = data[REQUIRED_COLUMNS].corr()

        fig, ax = plt.subplots()

        image = ax.imshow(
            correlation,
            aspect="auto",
            vmin=-1,
            vmax=1,
        )

        ax.set_xticks(range(len(correlation.columns)))
        ax.set_xticklabels(
            correlation.columns,
            rotation=45,
            ha="right",
        )

        ax.set_yticks(range(len(correlation.columns)))
        ax.set_yticklabels(correlation.columns)

        fig.colorbar(image, ax=ax)
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    st.caption(
        "Correlation shows association, not proof that one variable "
        "causes another."
    )


# --------------------------------------------------
# TAB 4: MODEL COMPARISON
# --------------------------------------------------

with comparison:
    st.subheader("🤖 Model Performance Comparison")

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True,
    )

    best_index = results["MAE"].idxmin()
    best_model = results.loc[best_index, "Model"]
    best_mae = results.loc[best_index, "MAE"]

    st.metric(
        "Lowest Test MAE in This Split",
        f"{best_mae:.2f}",
        help=(
            "This is the lowest MAE among the tested models on one "
            "train/test split, not proof of general superiority."
        ),
    )

    st.caption(
        f"Model with the lowest MAE on this split: {best_model}. "
        "Results can change with more data or a different test split."
    )

    fig, ax = plt.subplots()

    ax.bar(
        results["Model"],
        results["MAE"],
    )

    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("Test MAE by Model")
    ax.tick_params(axis="x", rotation=15)

    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots()

    ax.bar(
        results["Model"],
        results["RMSE"],
    )

    ax.set_ylabel("Root Mean Squared Error")
    ax.set_title("Test RMSE by Model")
    ax.tick_params(axis="x", rotation=15)

    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "📥 Download Model Comparison Report",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="model_comparison_report.csv",
        mime="text/csv",
    )

    st.info(
        "MAE and RMSE measure prediction error; lower is generally "
        "better. R² measures fit relative to a baseline. Small "
        "datasets can make all three metrics unstable."
    )


# --------------------------------------------------
# TAB 5: DATA QUALITY
# --------------------------------------------------

with quality:
    st.subheader("🧹 Data Quality Report")

    missing_counts = raw_data[REQUIRED_COLUMNS].isna().sum()

    numeric_data = raw_data[REQUIRED_COLUMNS].copy()

    for column in REQUIRED_COLUMNS:
        numeric_data[column] = pd.to_numeric(
            numeric_data[column],
            errors="coerce",
        )

    invalid_numeric_counts = (
        numeric_data.isna().sum() - missing_counts
    ).clip(lower=0)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Original Rows",
        len(raw_data),
    )

    c2.metric(
        "Valid Rows Used",
        len(data),
    )

    c3.metric(
        "Rows Excluded",
        max(0, len(raw_data) - len(data)),
    )

    st.markdown("**Missing values by required column**")

    st.dataframe(
        missing_counts.rename("Missing Values").to_frame(),
        use_container_width=True,
    )

    st.markdown("**Non-numeric values by required column**")

    st.dataframe(
        invalid_numeric_counts.rename(
            "Non-numeric Values"
        ).to_frame(),
        use_container_width=True,
    )

    st.caption(
        "The model uses complete, valid records. Excluded rows are "
        "not automatically imputed."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Student Performance AI | BCA Data Science & AI Project | "
    "Educational demonstration"
)
