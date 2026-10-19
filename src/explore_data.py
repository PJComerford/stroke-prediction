
from pathlib import Path

import pandas as pd

# Locate the project root and raw dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "healthcare-dataset-stroke-data.csv"
)

# Load the original dataset
df = pd.read_csv(DATA_PATH)

print("\n--- DATASET OVERVIEW ---")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- COLUMN TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

print("\n--- STROKE DISTRIBUTION ---")
print(df["stroke"].value_counts())
print(
    f"Stroke prevalence: "
    f"{df['stroke'].mean() * 100:.2f}%"
)

print("\n--- NUMERICAL SUMMARY ---")
print(df.describe())

# Inspect categorical columns
print("\n--- CATEGORICAL VALUE COUNTS ---")

categorical_columns = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False))
