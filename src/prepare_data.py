
from pathlib import Path

import numpy as np
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


# Expected dataset structure
EXPECTED_COLUMNS = [
    "id",
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke",
]

NUMERIC_COLUMNS = [
    "id",
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
    "stroke",
]

CATEGORICAL_VALUES = {
    "gender": {"Male", "Female", "Other"},
    "ever_married": {"Yes", "No"},
    "work_type": {
        "Private",
        "Self-employed",
        "Govt_job",
        "children",
        "Never_worked",
    },
    "Residence_type": {"Urban", "Rural"},
    "smoking_status": {
        "formerly smoked",
        "never smoked",
        "smokes",
        "Unknown",
    },
}


# --------------------------------------------------
# 2. Load raw dataset
# --------------------------------------------------

print("--- LOADING DATASET ---")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Download the Kaggle dataset and place it in data/raw/."
    )

df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded successfully: {len(df)} rows")


# --------------------------------------------------
# 3. Validate dataset structure
# --------------------------------------------------

print("\n--- VALIDATING DATASET STRUCTURE ---")

missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
unexpected_columns = set(df.columns) - set(EXPECTED_COLUMNS)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

if unexpected_columns:
    raise ValueError(
        f"Unexpected columns found: {sorted(unexpected_columns)}"
    )

print("All expected columns are present.")
print(f"Number of columns: {len(df.columns)}")


# --------------------------------------------------
# 4. Validate data types
# --------------------------------------------------

print("\n--- VALIDATING DATA TYPES ---")

for column in NUMERIC_COLUMNS:
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(
            f"Column '{column}' must contain numeric values."
        )

print("All expected numerical columns have numeric data types.")


# --------------------------------------------------
# 5. Validate missing values
# --------------------------------------------------

print("\n--- VALIDATING MISSING VALUES ---")

missing_values = df.isna().sum()

print(missing_values)

# BMI is the only column where missing values are permitted.
unexpected_missing = {
    column: int(count)
    for column, count in missing_values.items()
    if count > 0 and column != "bmi"
}

if unexpected_missing:
    raise ValueError(
        f"Unexpected missing values found: {unexpected_missing}"
    )

print(
    f"BMI missing values: {missing_values['bmi']}"
)
print("Missing-value validation passed.")


# --------------------------------------------------
# 6. Validate duplicates
# --------------------------------------------------

print("\n--- VALIDATING DUPLICATES ---")

duplicate_rows = int(df.duplicated().sum())
duplicate_ids = int(df["id"].duplicated().sum())

print(f"Duplicate rows: {duplicate_rows}")
print(f"Duplicate IDs: {duplicate_ids}")

if duplicate_rows > 0:
    raise ValueError("Duplicate records found in the dataset.")

if duplicate_ids > 0:
    raise ValueError("Duplicate IDs found in the dataset.")

print("Duplicate validation passed.")


# --------------------------------------------------
# 7. Validate numerical ranges and binary values
# --------------------------------------------------

print("\n--- VALIDATING NUMERICAL VALUES ---")

for column in NUMERIC_COLUMNS:
    values = df[column].dropna().to_numpy()

    if not np.isfinite(values).all():
        raise ValueError(
            f"Non-finite values found in column '{column}'."
        )

# Age should be between 0 and 120.
if not df["age"].between(0, 120).all():
    raise ValueError("Age values outside the range 0-120 found.")

# Glucose must be positive.
if (df["avg_glucose_level"] <= 0).any():
    raise ValueError("Non-positive glucose values found.")

# BMI must be positive when recorded.
if (df["bmi"].dropna() <= 0).any():
    raise ValueError("Non-positive BMI values found.")

# These columns must contain only 0 or 1.
binary_columns = [
    "hypertension",
    "heart_disease",
    "stroke",
]

for column in binary_columns:
    if not set(df[column].unique()).issubset({0, 1}):
        raise ValueError(
            f"Invalid values found in binary column '{column}'."
        )

print("Numerical range and binary-value checks passed.")


# --------------------------------------------------
# 8. Validate categorical values
# --------------------------------------------------

print("\n--- VALIDATING CATEGORICAL VALUES ---")

for column, allowed_values in CATEGORICAL_VALUES.items():

    actual_values = set(df[column].dropna().unique())

    invalid_values = actual_values - allowed_values

    if invalid_values:
        raise ValueError(
            f"Unexpected values in '{column}': "
            f"{sorted(invalid_values)}"
        )

    print(f"{column}: valid values confirmed.")

print("Categorical validation passed.")


# --------------------------------------------------
# 9. Remove gender = Other
# --------------------------------------------------

print("\n--- DATA CLEANING ---")

other_gender_count = int(
    (df["gender"] == "Other").sum()
)

df = df[df["gender"] != "Other"].copy()

print(f"Records removed (gender = Other): {other_gender_count}")
print(f"Rows remaining: {len(df)}")
print(f"Stroke cases remaining: {int(df['stroke'].sum())}")


# --------------------------------------------------
# 10. Separate features and target
# --------------------------------------------------

# ID is an identifier, not a predictive feature.
# Stroke is kept separately as the target.

X = df.drop(columns=["id", "stroke"])
y = df["stroke"]


# --------------------------------------------------
# 11. Create stratified train/dev/test splits
# --------------------------------------------------

print("\n--- CREATING DATA SPLITS ---")

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

dev_fraction_of_remaining = DEV_SIZE / (1 - TEST_SIZE)

X_train, X_dev, y_train, y_dev = train_test_split(
    X_temp,
    y_temp,
    test_size=dev_fraction_of_remaining,
    random_state=RANDOM_STATE,
    stratify=y_temp,
)


# --------------------------------------------------
# 12. Validate the splits
# --------------------------------------------------

print("\n--- VALIDATING DATA SPLITS ---")

splits = {
    "Train": (X_train, y_train),
    "Development": (X_dev, y_dev),
    "Test": (X_test, y_test),
}

# Check that all rows are accounted for.
total_split_rows = sum(
    len(X_split)
    for X_split, _ in splits.values()
)

if total_split_rows != len(df):
    raise ValueError("Split row totals do not match cleaned data.")

# Check that no records appear in more than one split.
train_indices = set(X_train.index)
dev_indices = set(X_dev.index)
test_indices = set(X_test.index)

if train_indices & dev_indices:
    raise ValueError("Training and development sets overlap.")

if train_indices & test_indices:
    raise ValueError("Training and test sets overlap.")

if dev_indices & test_indices:
    raise ValueError("Development and test sets overlap.")

# Check that each split contains both target classes.
for name, (_, y_split) in splits.items():
    if set(y_split.unique()) != {0, 1}:
        raise ValueError(
            f"{name} split does not contain both target classes."
        )

print("Split totals match.")
print("No overlap between splits.")
print("Both target classes are present in every split.")


# --------------------------------------------------
# 13. Report split sizes and class balance
# --------------------------------------------------

print("\n--- FINAL DATA SPLITS ---")

for name, (X_split, y_split) in splits.items():

    print(
        f"{name}: {len(X_split)} rows | "
        f"{int(y_split.sum())} stroke cases | "
        f"{y_split.mean() * 100:.2f}% stroke prevalence"
    )


# --------------------------------------------------
# 14. Report missing BMI values
# --------------------------------------------------

print("\n--- MISSING BMI VALUES BEFORE IMPUTATION ---")

for name, (X_split, _) in splits.items():
    print(
        f"{name}: {int(X_split['bmi'].isna().sum())}"
    )
# --------------------------------------------------
# 15. Save processed datasets
# --------------------------------------------------

print("\n--- SAVING PROCESSED DATASETS ---")

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

for name, (X_split, y_split) in splits.items():

    # Combine features and target into one DataFrame.
    split_df = X_split.copy()
    split_df["stroke"] = y_split

    # Save the prepared split.
    output_file = PROCESSED_PATH / f"{name.lower()}.csv"
    split_df.to_csv(output_file, index=False)

    print(f"{name} saved: {output_file}")

print("All processed datasets saved successfully.")

# --------------------------------------------------
# 16. Completion summary
# --------------------------------------------------

print("\n--- PIPELINE PREPARATION COMPLETE ---")

print(f"Original rows: {len(pd.read_csv(DATA_PATH))}")
print(f"Rows after cleaning: {len(df)}")
print(f"Rows across splits: {total_split_rows}")
print(f"Random seed: {RANDOM_STATE}")

print("\nRaw dataset has not been modified.")
print("BMI imputation is handled later within model pipelines.")
print("No processed files have been written yet.")
