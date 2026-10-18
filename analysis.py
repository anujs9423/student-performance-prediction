from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
df = pd.read_csv(ROOT / "data" / "students.csv")

print("First five rows:")
print(df.head())
print("\nDataset information:")
df.info()
print("\nStatistical summary:")
print(df.describe())
print("\nMissing values by column:")
print(df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nAverage final marks:", round(df["final_marks"].mean(), 2))
print("Highest final marks:", df["final_marks"].max())
print("Lowest final marks:", df["final_marks"].min())

plt.figure(figsize=(8, 5))
plt.scatter(df["study_hours"], df["final_marks"])
plt.title("Study Hours vs Final Marks")
plt.xlabel("Daily Study Hours")
plt.ylabel("Final Marks")
plt.grid(True)
plt.tight_layout()
plt.show()
