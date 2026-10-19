
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "healthcare-dataset-stroke-data.csv"
)

RANDOM_STATE = 28
TEST_SIZE = 0.15
DEV_SIZE = 0.15


# --------------------------------------------------
# 2. Load raw dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("--- ORIGINAL DATASET ---")
print(f"Number of rows: {len(df)}")
print(f"Number of stroke cases: {df['stroke'].sum()}")


# --------------------------------------------------
# 3. Remove gender = Other
# --------------------------------------------------

other_gender_count = (df["gender"] == "Other").sum()

df = df[df["gender"] != "Other"].copy()

print("\n--- DATA CLEANING ---")
print(f"Records removed (gender = Other): {other_gender_count}")
print(f"Rows remaining: {len(df)}")
print(f"Stroke cases remaining: {df['stroke'].sum()}")


# --------------------------------------------------
# 4. Separate features and target
# --------------------------------------------------

# Remove ID because it is an identifier, not a predictor.
# Keep stroke separately as the target variable.

X = df.drop(columns=["id", "stroke"])
y = df["stroke"]


# --------------------------------------------------
# 5. Split off the test set (15%)
# --------------------------------------------------

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)


# --------------------------------------------------
# 6. Split remaining data into train (70%) and
#    development (15%) overall
# --------------------------------------------------

dev_fraction_of_remaining = DEV_SIZE / (1 - TEST_SIZE)

X_train, X_dev, y_train, y_dev = train_test_split(
    X_temp,
    y_temp,
    test_size=dev_fraction_of_remaining,
    random_state=RANDOM_STATE,
    stratify=y_temp,
)


# --------------------------------------------------
# 7. Report split sizes and stroke prevalence
# --------------------------------------------------

splits = {
    "Train": (X_train, y_train),
    "Development": (X_dev, y_dev),
    "Test": (X_test, y_test),
}

print("\n--- DATA SPLITS ---")

for name, (X_split, y_split) in splits.items():
    print(
        f"{name}: {len(X_split)} rows | "
        f"{int(y_split.sum())} stroke cases | "
        f"{y_split.mean() * 100:.2f}% stroke prevalence"
    )


# --------------------------------------------------
# 8. Check missing BMI values before imputation
# --------------------------------------------------

print("\n--- MISSING BMI VALUES BEFORE IMPUTATION ---")

for name, (X_split, _) in splits.items():
    print(f"{name}: {X_split['bmi'].isna().sum()}")


# --------------------------------------------------
# 9. Validate split totals
# --------------------------------------------------

total_split_rows = (
    len(X_train)
    + len(X_dev)
    + len(X_test)
)

print("\n--- VALIDATION ---")
print(f"Rows after cleaning: {len(df)}")
print(f"Rows across splits: {total_split_rows}")
print(f"Split totals match: {total_split_rows == len(df)}")

print("\nRaw dataset has not been modified.")
print("No processed files have been written yet.")
