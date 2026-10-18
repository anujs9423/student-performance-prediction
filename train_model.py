from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).parent
df = pd.read_csv(ROOT / "data" / "students.csv")

features = ["study_hours", "attendance", "previous_marks", "sleep_hours"]
X = df[features]
y = df["final_marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"Rows: {len(df)}")
print(f"Missing cells: {int(df.isna().sum().sum())}")
print(f"Duplicate rows: {int(df.duplicated().sum())}")
print(f"MAE: {mean_absolute_error(y_test, predictions):.2f}")
print(f"R2: {r2_score(y_test, predictions):.2f}")

example = pd.DataFrame([{
    "study_hours": 5,
    "attendance": 85,
    "previous_marks": 75,
    "sleep_hours": 8
}])
print(f"Example estimated marks: {model.predict(example)[0]:.1f}")
