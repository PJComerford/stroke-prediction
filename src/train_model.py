
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline


# ============================================================
# 1. PATHS AND DATA
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = PROCESSED_PATH / "train.csv"
DEV_PATH = PROCESSED_PATH / "development.csv"
TEST_PATH = PROCESSED_PATH / "test.csv"

REPORT_PATH = PROJECT_ROOT / "reports" / "development_predictions_smotenc.csv"

# Check that the processed datasets exist
for path in [TRAIN_PATH, DEV_PATH, TEST_PATH]:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. "
            "Run python src/prepare_data.py first."
        )

# Load the saved datasets
train_df = pd.read_csv(TRAIN_PATH)
dev_df = pd.read_csv(DEV_PATH)
test_df = pd.read_csv(TEST_PATH)

# Separate features and target
X_train = train_df.drop(columns=["stroke"])
y_train = train_df["stroke"].astype(int)

X_dev = dev_df.drop(columns=["stroke"])
y_dev = dev_df["stroke"].astype(int)

X_test = test_df.drop(columns=["stroke"])
y_test = test_df["stroke"].astype(int)

print("Processed datasets loaded successfully.")


# ============================================================
# 2. FEATURE TYPES
# ============================================================

categorical_features = [
    "gender",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]

numerical_features = [
    "age",
    "avg_glucose_level",
    "bmi",
]


# ============================================================
# 3. IMPUTATION
# ============================================================

# Fit imputers using training data only
numeric_imputer = SimpleImputer(strategy="median")
categorical_imputer = SimpleImputer(strategy="most_frequent")

X_train[numerical_features] = numeric_imputer.fit_transform(
    X_train[numerical_features]
)

X_dev[numerical_features] = numeric_imputer.transform(
    X_dev[numerical_features]
)

X_test[numerical_features] = numeric_imputer.transform(
    X_test[numerical_features]
)

X_train[categorical_features] = categorical_imputer.fit_transform(
    X_train[categorical_features].astype(str)
)

X_dev[categorical_features] = categorical_imputer.transform(
    X_dev[categorical_features].astype(str)
)

X_test[categorical_features] = categorical_imputer.transform(
    X_test[categorical_features].astype(str)
)


# ============================================================
# 4. SMOTENC RESAMPLING
# ============================================================

# Identify categorical feature positions
categorical_indices = [
    X_train.columns.get_loc(column)
    for column in categorical_features
]

smote = SMOTENC(
    categorical_features=categorical_indices,
    random_state=28,
)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train,
    y_train,
)


# ============================================================
# 5. PREPROCESSING AND MODEL
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features,
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first",
            ),
            categorical_features,
        ),
    ]
)

model = LogisticRegression(
    max_iter=2000,
    random_state=28,
)

pipeline = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)


# ============================================================
# 6. TRAIN MODEL
# ============================================================

pipeline.fit(
    X_train_resampled,
    y_train_resampled,
)


# ============================================================
# 7. DEVELOPMENT SET EVALUATION
# ============================================================

dev_probabilities = pipeline.predict_proba(X_dev)[:, 1]

roc_auc = roc_auc_score(y_dev, dev_probabilities)
average_precision = average_precision_score(
    y_dev,
    dev_probabilities,
)

print("\n--- DEVELOPMENT SET PERFORMANCE ---")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Average Precision: {average_precision:.4f}")


# Evaluate different classification thresholds
thresholds = [0.50, 0.30, 0.20, 0.10, 0.05]

print("\n--- THRESHOLD COMPARISON ---")

for threshold in thresholds:
    predictions = (
        dev_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_dev,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_dev,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_dev,
        predictions,
        zero_division=0,
    )

    print(f"\nThreshold: {threshold:.2f}")
    print(f"Predicted positive cases: {predictions.sum()}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")


# Detailed report at threshold 0.50
default_predictions = (
    dev_probabilities >= 0.50
).astype(int)

print("\n--- CLASSIFICATION REPORT (THRESHOLD 0.50) ---")
print(
    classification_report(
        y_dev,
        default_predictions,
        zero_division=0,
    )
)

print("\n--- CONFUSION MATRIX (THRESHOLD 0.50) ---")
print(confusion_matrix(y_dev, default_predictions))


# ============================================================
# 8. SAVE DEVELOPMENT PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "actual_stroke": y_dev.values,
    "predicted_probability": dev_probabilities,
    "predicted_stroke_0.50": default_predictions,
})

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

results.to_csv(REPORT_PATH, index=False)

print(f"\nDevelopment predictions saved to: {REPORT_PATH}")


# ============================================================
# 9. DATA AND RESAMPLING SUMMARY
# ============================================================

print("\n--- DATA SUMMARY ---")

total_records = len(train_df) + len(dev_df) + len(test_df)
total_strokes = int(y_train.sum() + y_dev.sum() + y_test.sum())

print(f"Total records used: {total_records}")
print(f"Total stroke cases: {total_strokes}")

print("\n--- DATA SPLITS ---")

for name, labels in [
    ("Train", y_train),
    ("Development", y_dev),
    ("Test", y_test),
]:
    print(
        f"{name}: {len(labels)} records | "
        f"{int(labels.sum())} stroke cases | "
        f"{labels.mean():.2%} prevalence"
    )

print("\n--- SMOTENC RESAMPLING ---")
print(f"Training records before SMOTENC: {len(y_train)}")
print(f"Training stroke cases before SMOTENC: {int(y_train.sum())}")
print(f"Training records after SMOTENC: {len(y_train_resampled)}")
print(
    "Training stroke cases after SMOTENC: "
    f"{int(y_train_resampled.sum())}"
)

print("\nTraining and development evaluation complete.")

