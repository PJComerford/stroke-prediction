
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
# 2. Load and clean data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

# Exclude the single record with gender = Other
other_count = (df["gender"] == "Other").sum()
df = df[df["gender"] != "Other"].copy()

print("--- DATA ---")
print(f"Original rows: {len(df) + other_count}")
print(f"Records excluded (gender = Other): {other_count}")
print(f"Rows used: {len(df)}")
print(f"Stroke cases: {df['stroke'].sum()}")


# --------------------------------------------------
# 3. Separate predictors and target
# --------------------------------------------------

# ID is an identifier and is excluded from the model.
X = df.drop(columns=["id", "stroke"])
y = df["stroke"]

categorical_features = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]

numerical_features = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
]


# --------------------------------------------------
# 4. Split into train, development and test sets
# --------------------------------------------------

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

dev_fraction = DEV_SIZE / (1 - TEST_SIZE)

X_train, X_dev, y_train, y_dev = train_test_split(
    X_temp,
    y_temp,
    test_size=dev_fraction,
    random_state=RANDOM_STATE,
    stratify=y_temp,
)

print("\n--- DATA SPLITS ---")
for name, X_part, y_part in [
    ("Train", X_train, y_train),
    ("Development", X_dev, y_dev),
    ("Test", X_test, y_test),
]:
    print(
        f"{name}: {len(X_part)} records, "
        f"{int(y_part.sum())} stroke cases, "
        f"{y_part.mean() * 100:.2f}% prevalence"
    )


# --------------------------------------------------
# 5. Preprocessing pipeline
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first",
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)


# --------------------------------------------------
# 6. Create and train baseline model
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

# Only the training set is used to fit preprocessing
# and the logistic regression model.
model.fit(X_train, y_train)

print("\n--- MODEL TRAINED ---")
print("Model: Logistic Regression")
print("BMI imputation: training-set median")
print("Categorical variables: one-hot encoded")


# --------------------------------------------------
# 7. Evaluate on development set
# --------------------------------------------------

# Use predicted probabilities for ranking metrics.
dev_probabilities = model.predict_proba(X_dev)[:, 1]

# Default classification threshold is 0.5.
dev_predictions = (dev_probabilities >= 0.5).astype(int)

print("\n--- DEVELOPMENT RESULTS ---")
print(
    "ROC-AUC:",
    round(roc_auc_score(y_dev, dev_probabilities), 4),
)
print(
    "PR-AUC (average precision):",
    round(average_precision_score(y_dev, dev_probabilities), 4),
)

print("\nClassification report:")
print(
    classification_report(
        y_dev,
        dev_predictions,
        zero_division=0,
    )
)

print("Confusion matrix:")
print(confusion_matrix(y_dev, dev_predictions))


# --------------------------------------------------
# 8. Save development results
# --------------------------------------------------

reports_dir = PROJECT_ROOT / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

dev_results = X_dev.copy()
dev_results["actual_stroke"] = y_dev
dev_results["predicted_probability"] = dev_probabilities
dev_results["predicted_stroke_at_0_5"] = dev_predictions

results_path = reports_dir / "development_predictions.csv"
dev_results.to_csv(results_path, index=False)

print(f"\nDevelopment predictions saved to: {results_path}")
print("\nThe test set has not been used for model evaluation.")
