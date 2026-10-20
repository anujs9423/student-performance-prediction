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

st.set_page_config(page_title="Student Performance AI", page_icon="🎓", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "students.csv"
FEATURES = ["study_time_level", "absences", "period1_grade", "period2_grade"]
TARGET = "final_grade"
COLUMNS = FEATURES + [TARGET]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root {--blue:#3b82f6;--violet:#8b5cf6;--ink:#172554;--muted:#64748b;}
.stApp {
 background:radial-gradient(ellipse at 8% 0%,rgba(96,165,250,.19),transparent 34%),
 radial-gradient(ellipse at 95% 12%,rgba(167,139,250,.20),transparent 30%),
 linear-gradient(145deg,#f8fbff 0%,#f5f3ff 52%,#eff6ff 100%);
 color:var(--ink);font-family:'DM Sans',sans-serif;
}
.block-container{padding-top:1.2rem;padding-bottom:2.6rem;max-width:1480px}
[data-testid="stHeader"]{background:rgba(248,250,252,.72)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#eef5ff,#f5f3ff);border-right:1px solid rgba(148,163,184,.25)}
h1,h2,h3,h4{font-family:'Manrope',sans-serif!important;letter-spacing:-.035em}
h1{color:#172554!important;font-weight:800!important}
h2,h3,h4{color:#243b80!important;font-weight:750!important}
p,label,.stCaption{color:#52617a}
[data-testid="stMetric"]{background:linear-gradient(145deg,rgba(255,255,255,.96),rgba(239,246,255,.88));padding:19px 20px 17px;border:1px solid rgba(191,219,254,.85);border-radius:19px;box-shadow:0 12px 32px rgba(59,130,246,.075);transition:transform .22s ease,box-shadow .22s ease}
[data-testid="stMetric"]:hover{transform:translateY(-3px);box-shadow:0 16px 34px rgba(99,102,241,.14)}
[data-testid="stMetricLabel"]{color:#64748b!important;font-weight:700}
[data-testid="stMetricValue"]{color:#4f46e5!important;font-weight:800}
div[data-testid="stTabs"] button{font-weight:700;border-radius:12px 12px 0 0;padding:.72rem .85rem}
div[data-testid="stTabs"] button[aria-selected="true"]{color:#4f46e5!important;border-bottom:3px solid #8b5cf6!important;background:#ede9fe}
.stButton>button,.stDownloadButton>button{border:0;border-radius:12px;color:#fff;background:linear-gradient(105deg,#2563eb,#7c3aed);font-weight:750;box-shadow:0 8px 20px rgba(79,70,229,.2);transition:transform .2s ease,box-shadow .2s ease}
.stButton>button:hover,.stDownloadButton>button:hover{color:#fff;border:0;transform:translateY(-2px);box-shadow:0 12px 25px rgba(79,70,229,.28)}
[data-testid="stDataFrame"],[data-testid="stTable"]{border:1px solid #dbeafe;border-radius:14px;overflow:hidden}
[data-testid="stAlert"]{border-radius:14px}
hr{border-color:rgba(148,163,184,.28)}
.hero{position:relative;overflow:hidden;border-radius:28px;padding:34px;margin:6px 0 25px;color:white;background:linear-gradient(120deg,#1d4ed8,#4f46e5 48%,#7c3aed);box-shadow:0 22px 55px rgba(79,70,229,.24)}
.hero:before,.hero:after{content:"";position:absolute;border-radius:50%;pointer-events:none}
.hero:before{width:280px;height:280px;right:-65px;top:-145px;background:rgba(255,255,255,.13);animation:drift 8s ease-in-out infinite}
.hero:after{width:190px;height:190px;right:170px;bottom:-130px;background:rgba(196,181,253,.20);animation:drift 10s ease-in-out infinite reverse}
.hero-inner{position:relative;z-index:1;display:flex;gap:24px;align-items:center;justify-content:space-between;flex-wrap:wrap}
.hero-eyebrow{display:inline-block;padding:6px 11px;border:1px solid rgba(255,255,255,.3);background:rgba(255,255,255,.12);border-radius:999px;font-size:.76rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#e0e7ff}
.hero h1{font-size:clamp(2rem,4.3vw,3.35rem)!important;line-height:1.07;margin:.75rem 0 .8rem;color:#fff!important}
.hero p{color:#e0e7ff;font-size:1.02rem;line-height:1.65;max-width:720px;margin:0}
.hero-badge{width:112px;height:112px;display:flex;align-items:center;justify-content:center;border-radius:30px;font-size:3.2rem;background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.27);box-shadow:inset 0 1px 0 rgba(255,255,255,.2);animation:float 4.5s ease-in-out infinite}
.hero-chips{display:flex;gap:9px;flex-wrap:wrap;margin-top:20px}
.hero-chip{font-size:.78rem;font-weight:700;padding:8px 11px;border-radius:10px;background:rgba(255,255,255,.13);border:1px solid rgba(255,255,255,.18);color:white}
.section-kicker{font-size:.75rem;letter-spacing:.14em;text-transform:uppercase;color:#6366f1;font-weight:800;margin-bottom:5px}
.glass-card{border:1px solid rgba(191,219,254,.8);background:rgba(255,255,255,.76);border-radius:18px;padding:19px 20px;box-shadow:0 10px 30px rgba(37,99,235,.055)}
.glass-card h4{margin:.1rem 0 .55rem}
.glass-card p{margin:0;line-height:1.6}
.feature-icon{font-size:1.5rem;margin-bottom:8px}
.team-card{height:100%;border:1px solid rgba(196,181,253,.65);border-radius:18px;padding:20px;background:linear-gradient(145deg,rgba(255,255,255,.95),rgba(245,243,255,.86));box-shadow:0 10px 28px rgba(79,70,229,.07);transition:transform .22s ease,box-shadow .22s ease}
.team-card:hover{transform:translateY(-4px);box-shadow:0 16px 35px rgba(79,70,229,.13)}
.team-avatar{width:48px;height:48px;border-radius:15px;display:flex;align-items:center;justify-content:center;color:white;font-weight:800;font-size:1rem;background:linear-gradient(135deg,#2563eb,#8b5cf6);margin-bottom:14px;box-shadow:0 7px 16px rgba(99,102,241,.2)}
.team-name{font-family:'Manrope',sans-serif;font-weight:800;color:#1e2b58;font-size:1.02rem;margin-bottom:5px}
.team-role{color:#6366f1;font-weight:750;font-size:.79rem;line-height:1.45;margin-bottom:8px}
.team-desc{font-size:.83rem;color:#64748b;line-height:1.55}
.footer{text-align:center;padding:22px 8px 4px;color:#64748b;font-size:.82rem}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
@keyframes drift{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(-16px,15px) scale(1.05)}}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important}}
@media(max-width:768px){.block-container{padding-left:1rem;padding-right:1rem}.hero{padding:25px 21px;border-radius:22px}.hero-badge{width:76px;height:76px;border-radius:22px;font-size:2.3rem}.hero h1{font-size:2rem!important}}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor":"white","axes.facecolor":"#fbfdff","axes.edgecolor":"#cbd5e1",
    "axes.labelcolor":"#334155","xtick.color":"#475569","ytick.color":"#475569",
    "text.color":"#172554","axes.titleweight":"bold","axes.titlecolor":"#1e3a8a",
    "grid.color":"#dbeafe",
})

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

@st.cache_data(show_spinner="Training regression models…")
def train_models(dataset_csv, version):
    from io import StringIO
    frame = pd.read_csv(StringIO(dataset_csv))
    X, y = frame[FEATURES], frame[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
        rows.append({"Model":name,"MAE":round(mean_absolute_error(y_test,pred),3),
                     "RMSE":round(mean_squared_error(y_test,pred)**0.5,3),"R²":round(r2_score(y_test,pred),3)})
    return models, pd.DataFrame(rows), X_test, y_test

st.markdown("""
<div class="hero"><div class="hero-inner"><div style="flex:1;min-width:240px">
<span class="hero-eyebrow">Student Performance • Data Science Project</span>
<h1>Understand the data.<br>Unlock the potential.</h1>
<p>An interactive analytics workspace to explore student grade patterns, estimate outcomes with machine learning, and compare regression models — all in one place.</p>
<div class="hero-chips"><span class="hero-chip">✦ Machine Learning</span><span class="hero-chip">◈ Interactive Analytics</span><span class="hero-chip">⌁ Model Evaluation</span></div>
</div><div class="hero-badge">🎓</div></div></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🎓 Student Performance AI")
    st.caption("ANALYTICS WORKSPACE")
    st.markdown("---")
    uploaded = st.file_uploader("Upload a compatible CSV", type=["csv"])
    model_name = st.selectbox("Prediction model", ["Random Forest","Linear Regression","Decision Tree"])
    st.markdown("---")
    st.markdown("**Expected columns**")
    for col in COLUMNS:
        st.code(col, language=None)
    st.markdown("---")
    st.caption("Built for learning, analysis and responsible experimentation.")

st.warning("**Dataset scope:** UCI Student Performance data for Portuguese secondary-school students. This app uses the Mathematics subset. It is not college/BCA student data. Predictions are educational estimates, not decisions about real students.")

try:
    raw = pd.read_csv(uploaded) if uploaded is not None else load_default_data()
    df = prepare_data(raw)
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.stop()

csv_text = df.to_csv(index=False)
version = hashlib.sha256(csv_text.encode()).hexdigest()
models, metrics, X_test, y_test = train_models(csv_text, version)

feature_cols = st.columns(3)
for col, (icon, title, desc) in zip(feature_cols, [
    ("🧭","Explore","Filter records, inspect distributions and review descriptive statistics."),
    ("🔮","Predict","Estimate a final grade using a selected regression model."),
    ("📈","Evaluate","Compare MAE, RMSE and R² across three regression approaches."),
]):
    with col:
        st.markdown(f'<div class="glass-card"><div class="feature-icon">{icon}</div><h4>{title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Overview","🔎 Explore Data","🔮 Predict Grade","📈 Analytics","🤖 Model Evaluation","👥 Meet the Team"
])

with tab1:
    st.markdown('<div class="section-kicker">Dashboard / Overview</div>', unsafe_allow_html=True)
    st.subheader("Your dataset, at a glance")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Valid records",f"{len(df):,}")
    c2.metric("Average final grade",f"{df[TARGET].mean():.1f} / 20")
    c3.metric("Average period 2 grade",f"{df['period2_grade'].mean():.1f} / 20")
    c4.metric("Average absences",f"{df['absences'].mean():.1f}")
    left,right = st.columns(2)
    with left:
        st.markdown("#### Final grade distribution")
        fig,ax = plt.subplots(figsize=(7,4))
        ax.hist(df[TARGET],bins=11,color="#6366f1",edgecolor="white",linewidth=1.2)
        ax.set_xlabel("Final grade (0–20)"); ax.set_ylabel("Number of students"); ax.grid(axis="y",alpha=.25)
        st.pyplot(fig,use_container_width=True); plt.close(fig)
    with right:
        st.markdown("#### Average final grade by study-time level")
        grouped = df.groupby("study_time_level")[TARGET].mean()
        fig,ax = plt.subplots(figsize=(7,4))
        ax.bar(grouped.index.astype(str),grouped.values,color=["#2563eb","#4f46e5","#7c3aed","#a78bfa"][:len(grouped)])
        ax.set_xlabel("Study-time level (1–4)"); ax.set_ylabel("Average final grade"); ax.grid(axis="y",alpha=.25)
        st.pyplot(fig,use_container_width=True); plt.close(fig)
    st.download_button("⬇️ Download clean dataset",csv_text,"student_math_clean.csv","text/csv")

with tab2:
    st.markdown('<div class="section-kicker">Dataset / Explore</div>',unsafe_allow_html=True)
    st.subheader("Explore records")
    f1,f2 = st.columns(2)
    with f1: min_grade = st.slider("Minimum final grade",0,20,0)
    with f2: max_absences = st.slider("Maximum absences",0,int(max(1,df.absences.max())),int(max(1,df.absences.max())))
    filtered = df[(df.final_grade>=min_grade)&(df.absences<=max_absences)]
    st.caption(f"Showing {len(filtered):,} of {len(df):,} records")
    st.dataframe(filtered,use_container_width=True,hide_index=True)
    st.markdown("#### Descriptive statistics")
    st.dataframe(df.describe().round(2),use_container_width=True)

with tab3:
    st.markdown('<div class="section-kicker">Machine Learning / Prediction</div>',unsafe_allow_html=True)
    st.subheader("Estimate a final grade")
    st.caption("The study-time feature is an ordinal category, not a number of hours.")
    c1,c2 = st.columns(2)
    with c1:
        study = st.select_slider("Weekly study time level",options=[1,2,3,4],value=2,format_func=lambda x:{1:"< 2 hours",2:"2–5 hours",3:"5–10 hours",4:"> 10 hours"}[x])
        absences = st.number_input("Number of school absences",min_value=0,max_value=100,value=4)
    with c2:
        g1 = st.slider("Period 1 grade (0–20)",0,20,10)
        g2 = st.slider("Period 2 grade (0–20)",0,20,11)
    if st.button("✨ Generate estimate",type="primary",use_container_width=True):
        sample = pd.DataFrame([{"study_time_level":study,"absences":absences,"period1_grade":g1,"period2_grade":g2}])
        estimate = min(20,max(0,float(models[model_name].predict(sample)[0])))
        st.metric("Estimated final grade",f"{estimate:.1f} / 20")
        st.caption(f"Model: {model_name}. Estimates are uncertain and should not be used to label or make decisions about individual students.")

with tab4:
    st.markdown('<div class="section-kicker">Insights / Analytics</div>',unsafe_allow_html=True)
    st.subheader("Patterns in the dataset")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("#### Period 2 vs final grade")
        fig,ax = plt.subplots(figsize=(7,4))
        ax.scatter(df.period2_grade,df.final_grade,alpha=.78,color="#4f46e5",edgecolors="white",linewidths=.5)
        ax.set_xlabel("Period 2 grade"); ax.set_ylabel("Final grade"); ax.grid(alpha=.25)
        st.pyplot(fig,use_container_width=True); plt.close(fig)
    with c2:
        st.markdown("#### Absences vs final grade")
        fig,ax = plt.subplots(figsize=(7,4))
        ax.scatter(df.absences,df.final_grade,alpha=.78,color="#0ea5e9",edgecolors="white",linewidths=.5)
        ax.set_xlabel("Absences"); ax.set_ylabel("Final grade"); ax.grid(alpha=.25)
        st.pyplot(fig,use_container_width=True); plt.close(fig)
    st.markdown("#### Feature correlation")
    corr = df[COLUMNS].corr()
    fig,ax = plt.subplots(figsize=(8,5))
    image = ax.imshow(corr,vmin=-1,vmax=1,aspect="auto",cmap="Purples")
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns,rotation=35,ha="right")
    ax.set_yticks(range(len(corr.columns))); ax.set_yticklabels(corr.columns)
    fig.colorbar(image,ax=ax); fig.tight_layout()
    st.pyplot(fig,use_container_width=True); plt.close(fig)
    st.caption("Correlation describes association, not causation.")

with tab5:
    st.markdown('<div class="section-kicker">Machine Learning / Evaluation</div>',unsafe_allow_html=True)
    st.subheader("Compare regression models")
    st.dataframe(metrics,use_container_width=True,hide_index=True)
    best = metrics.loc[metrics.MAE.idxmin()]
    a,b = st.columns(2)
    a.metric("Lowest MAE on this test split",f"{best.MAE:.2f}",help="Lower is better; this single split does not prove general superiority.")
    b.metric("Best model on this split",best.Model)
    fig,ax = plt.subplots(figsize=(8,4))
    ax.bar(metrics.Model,metrics.MAE,color=["#2563eb","#6366f1","#7c3aed"])
    ax.set_ylabel("MAE (lower is better)"); ax.tick_params(axis="x",rotation=15); ax.grid(axis="y",alpha=.25)
    st.pyplot(fig,use_container_width=True); plt.close(fig)
    selected = models[model_name]; pred = selected.predict(X_test)
    fig,ax = plt.subplots(figsize=(7,5))
    ax.scatter(y_test,pred,alpha=.8,color="#4f46e5",edgecolors="white",linewidths=.5)
    ax.plot([0,20],[0,20],linestyle="--",color="#0f172a")
    ax.set_xlim(0,20); ax.set_ylim(0,20); ax.set_xlabel("Actual final grade"); ax.set_ylabel("Predicted final grade")
    ax.set_title(f"Actual vs predicted — {model_name}"); ax.grid(alpha=.25)
    st.pyplot(fig,use_container_width=True); plt.close(fig)
    importance = pd.DataFrame({"Feature":FEATURES,"Importance":models["Random Forest"].feature_importances_}).sort_values("Importance",ascending=False)
    st.markdown("#### Random Forest feature importance")
    fig,ax = plt.subplots(figsize=(8,4))
    ax.barh(importance.Feature[::-1],importance.Importance[::-1],color="#7c3aed")
    ax.set_xlabel("Relative importance"); st.pyplot(fig,use_container_width=True); plt.close(fig)
    st.caption("Feature importance reflects this model's use of features; it does not show causation.")
    st.download_button("⬇️ Download model report",metrics.to_csv(index=False),"model_comparison_report.csv","text/csv")

with tab6:
    st.markdown('<div class="section-kicker">The people behind the project</div>',unsafe_allow_html=True)
    st.title("Meet the Team")
    st.write("A collaborative student project focused on data exploration, machine learning and responsible interpretation.")
    team = [
        {"initials":"AS","name":"Anuj Shukla","role":"Project Lead & Lead Developer","desc":"Leads overall project development, dashboard implementation, machine-learning workflow and deployment."},
        {"initials":"JY","name":"Janhavi Yadav","role":"Research & Data Analytics Associate","desc":"Supports research, data interpretation and communicating analytical findings."},
        {"initials":"KU","name":"Kajal Upadhyay","role":"Research & Documentation Associate","desc":"Supports project research, written documentation and presentation of project details."},
        {"initials":"AS","name":"Aditi Singh","role":"Testing & Quality Support Associate","desc":"Supports review, testing and quality checks for a clear, reliable user experience."},
    ]
    team_cols = st.columns(4)
    for col,member in zip(team_cols,team):
        with col:
            st.markdown(f'<div class="team-card"><div class="team-avatar">{member["initials"]}</div><div class="team-name">{member["name"]}</div><div class="team-role">{member["role"]}</div><div class="team-desc">{member["desc"]}</div></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    st.info("Team roles are suggested project-role descriptions. Edit them if the actual contributions differ.")

st.markdown("""
<div class="footer"><div style="font-weight:800;color:#4338ca;font-size:.92rem">🎓 Student Performance AI</div>
Built with Streamlit · Pandas · Matplotlib · scikit-learn<br>
Data source: UCI Student Performance dataset · Educational use only</div>
""",unsafe_allow_html=True)
