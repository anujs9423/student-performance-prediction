import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

st.set_page_config(page_title="Student Performance Analytics", page_icon="🎓", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "students.csv"
FEATURES = ["study_time_level", "absences", "period1_grade", "period2_grade"]
TARGET = "final_grade"
COLUMNS = FEATURES + [TARGET]

st.markdown("""
<style>
.block-container {padding-top: 1.8rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: #f4f7fb; padding: 16px; border-radius: 12px; border: 1px solid #e4eaf3;}
div[data-testid="stTabs"] button {font-weight: 600;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_default_data():
    return pd.read_csv(DATA_PATH)

def prepare_data(raw):
    missing = [c for c in COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    frame = raw[COLUMNS].copy()
    for c in COLUMNS:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    frame = frame.replace([float("inf"), float("-inf")], float("nan")).dropna()
    frame = frame.drop_duplicates().reset_index(drop=True)
    frame = frame[
        frame["study_time_level"].between(1, 4)
        & frame["absences"].ge(0)
        & frame["period1_grade"].between(0, 20)
        & frame["period2_grade"].between(0, 20)
        & frame["final_grade"].between(0, 20)
    ].reset_index(drop=True)
    if len(frame) < 20:
        raise ValueError("At least 20 valid rows are recommended.")
    return frame

def train_models(dataset_csv, version):
    from io import StringIO
    frame = pd.read_csv(StringIO(dataset_csv))
    X, y = frame[FEATURES], frame[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    definitions = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, min_samples_leaf=4, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=3, random_state=42),
    }
    models, rows = {}, []
    for name, model in definitions.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        models[name] = model
        rows.append({
            "Model": name,
            "MAE": round(mean_absolute_error(y_test, pred), 3),
            "RMSE": round(mean_squared_error(y_test, pred) ** 0.5, 3),
            "R²": round(r2_score(y_test, pred), 3),
        })
    return models, pd.DataFrame(rows), X_test, y_test

st.title("🎓 Student Performance Analytics")
st.caption("A data science dashboard for exploring school grades and evaluating regression models.")
st.warning("Dataset scope: UCI Student Performance data for Portuguese secondary-school students. This app uses the Mathematics subset (395 records). It is not college/BCA student data. Predictions are educational estimates, not decisions about real students.")

with st.sidebar:
    st.header("⚙️ Data & Model")
    uploaded = st.file_uploader("Upload a compatible CSV", type=["csv"])
    model_name = st.selectbox("Prediction model", ["Random Forest", "Linear Regression", "Decision Tree"])
    st.markdown("---")
    st.markdown("**Expected columns**")
    for col in COLUMNS:
        st.code(col)

try:
    raw = pd.read_csv(uploaded) if uploaded is not None else load_default_data()
    df = prepare_data(raw)
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.stop()

csv_text = df.to_csv(index=False)
version = hashlib.sha256(csv_text.encode()).hexdigest()
models, metrics, X_test, y_test = train_models(csv_text, version)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview", "🔎 Explore Data", "🔮 Predict Grade", "📈 Analytics", "🤖 Model Evaluation"
])

with tab1:
    st.subheader("At a glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Valid records", f"{len(df):,}")
    c2.metric("Average final grade", f"{df[TARGET].mean():.1f} / 20")
    c3.metric("Average period 2 grade", f"{df['period2_grade'].mean():.1f} / 20")
    c4.metric("Average absences", f"{df['absences'].mean():.1f}")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Final grade distribution")
        fig, ax = plt.subplots()
        ax.hist(df[TARGET], bins=11, edgecolor="black")
        ax.set_xlabel("Final grade (0–20)")
        ax.set_ylabel("Number of students")
        ax.grid(axis="y", alpha=.25)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("#### Average final grade by study-time level")
        grouped = df.groupby("study_time_level")[TARGET].mean()
        fig, ax = plt.subplots()
        ax.bar(grouped.index.astype(str), grouped.values)
        ax.set_xlabel("Study-time level (1–4)")
        ax.set_ylabel("Average final grade")
        ax.grid(axis="y", alpha=.25)
        st.pyplot(fig)
        plt.close(fig)
    st.download_button("⬇️ Download clean dataset", csv_text, "student_math_clean.csv", "text/csv")

with tab2:
    st.subheader("Explore records")
    f1, f2 = st.columns(2)
    with f1:
        min_grade = st.slider("Minimum final grade", 0, 20, 0)
    with f2:
        max_absences = st.slider("Maximum absences", 0, int(max(1, df.absences.max())), int(max(1, df.absences.max())))
    filtered = df[(df.final_grade >= min_grade) & (df.absences <= max_absences)]
    st.caption(f"Showing {len(filtered):,} of {len(df):,} records")
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.markdown("#### Descriptive statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)

with tab3:
    st.subheader("Estimate a final grade")
    st.caption("The study-time feature is an ordinal category, not a number of hours.")
    c1, c2 = st.columns(2)
    with c1:
        study = st.select_slider("Weekly study time level", options=[1, 2, 3, 4], value=2,
                                 format_func=lambda x: {1:"< 2 hours", 2:"2–5 hours", 3:"5–10 hours", 4:"> 10 hours"}[x])
        absences = st.number_input("Number of school absences", min_value=0, max_value=100, value=4)
    with c2:
        g1 = st.slider("Period 1 grade (0–20)", 0, 20, 10)
        g2 = st.slider("Period 2 grade (0–20)", 0, 20, 11)
    if st.button("Generate estimate", type="primary", use_container_width=True):
        sample = pd.DataFrame([{
            "study_time_level": study, "absences": absences,
            "period1_grade": g1, "period2_grade": g2
        }])
        estimate = float(models[model_name].predict(sample)[0])
        estimate = min(20, max(0, estimate))
        st.metric("Estimated final grade", f"{estimate:.1f} / 20")
        st.caption(f"Model: {model_name}. Estimates are uncertain and should not be used to label or make decisions about individual students.")

with tab4:
    st.subheader("Patterns in the dataset")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Period 2 vs final grade")
        fig, ax = plt.subplots()
        ax.scatter(df.period2_grade, df.final_grade, alpha=.7)
        ax.set_xlabel("Period 2 grade")
        ax.set_ylabel("Final grade")
        ax.grid(alpha=.25)
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        st.markdown("#### Absences vs final grade")
        fig, ax = plt.subplots()
        ax.scatter(df.absences, df.final_grade, alpha=.7)
        ax.set_xlabel("Absences")
        ax.set_ylabel("Final grade")
        ax.grid(alpha=.25)
        st.pyplot(fig)
        plt.close(fig)
    st.markdown("#### Feature correlation")
    corr = df[COLUMNS].corr()
    fig, ax = plt.subplots()
    image = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Correlation describes association, not causation.")

with tab5:
    st.subheader("Compare regression models")
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    best = metrics.loc[metrics.MAE.idxmin()]
    a, b = st.columns(2)
    a.metric("Lowest MAE on this test split", f"{best.MAE:.2f}", help="Lower is better; this single split does not prove general superiority.")
    b.metric("Best model on this split", best.Model)
    fig, ax = plt.subplots()
    ax.bar(metrics.Model, metrics.MAE)
    ax.set_ylabel("MAE (lower is better)")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=.25)
    st.pyplot(fig)
    plt.close(fig)
    selected = models[model_name]
    pred = selected.predict(X_test)
    fig, ax = plt.subplots()
    ax.scatter(y_test, pred, alpha=.75)
    ax.plot([0, 20], [0, 20], linestyle="--")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_xlabel("Actual final grade")
    ax.set_ylabel("Predicted final grade")
    ax.set_title(f"Actual vs predicted — {model_name}")
    ax.grid(alpha=.25)
    st.pyplot(fig)
    plt.close(fig)
    importance_model = models["Random Forest"]
    importance = pd.DataFrame({"Feature": FEATURES, "Importance": importance_model.feature_importances_}).sort_values("Importance", ascending=False)
    st.markdown("#### Random Forest feature importance")
    fig, ax = plt.subplots()
    ax.barh(importance.Feature[::-1], importance.Importance[::-1])
    ax.set_xlabel("Relative importance")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Feature importance reflects this model's use of features; it does not show causation.")
    st.download_button("⬇️ Download model report", metrics.to_csv(index=False), "model_comparison_report.csv", "text/csv")

st.divider()
st.caption("Built with Streamlit, Pandas, Matplotlib and scikit-learn | UCI Student Performance dataset")
