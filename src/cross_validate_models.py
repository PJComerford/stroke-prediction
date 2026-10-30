

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.over_sampling import SMOTENC


# --------------------------------------------------
# 1. LOAD SAVED TRAINING DATA
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "cross_validation_results.csv"
)

if not TRAIN_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {TRAIN_PATH}. "
        "Run python src/prepare_data.py first."
    )

train_df = pd.read_csv(TRAIN_PATH)

if "stroke" not in train_df.columns:
    raise ValueError("The training dataset must contain a 'stroke' column.")

X_train = train_df.drop(columns=["stroke"])
y_train = train_df["stroke"].astype(int)

print("\n--- TRAINING DATA ---")
print(f"Training records: {len(X_train)}")
print(f"Stroke cases: {int(y_train.sum())}")
print(f"Non-stroke cases: {int((y_train == 0).sum())}")
print(f"Stroke prevalence: {y_train.mean():.2%}")


# --------------------------------------------------
# 2. DEFINE FEATURE TYPES
# --------------------------------------------------

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

required_features = categorical_features + numerical_features

missing_features = [
    column
    for column in required_features
    if column not in X_train.columns
]

if missing_features:
    raise ValueError(
        f"Missing expected features: {missing_features}"
    )

thresholds = [0.50, 0.10, 0.05]


# --------------------------------------------------
# 3. PREPROCESSING HELPER
# --------------------------------------------------

def impute_features(X_fit, X_valid):
    """Fit imputers on a fold's training data only."""

    numeric_imputer = SimpleImputer(strategy="median")

    categorical_imputer = SimpleImputer(
        strategy="most_frequent"
    )

    numeric_fit = pd.DataFrame(
        numeric_imputer.fit_transform(
            X_fit[numerical_features]
        ),
        columns=numerical_features,
        index=X_fit.index,
    )

    numeric_valid = pd.DataFrame(
        numeric_imputer.transform(
            X_valid[numerical_features]
        ),
        columns=numerical_features,
        index=X_valid.index,
    )

    categorical_fit = pd.DataFrame(
        categorical_imputer.fit_transform(
            X_fit[categorical_features].astype(str)
        ),
        columns=categorical_features,
        index=X_fit.index,
    )

    categorical_valid = pd.DataFrame(
        categorical_imputer.transform(
            X_valid[categorical_features].astype(str)
        ),
        columns=categorical_features,
        index=X_valid.index,
    )

    # Treat categorical and binary features as categories.
    categorical_fit = categorical_fit.astype(str)
    categorical_valid = categorical_valid.astype(str)

    X_fit_imputed = pd.concat(
        [numeric_fit, categorical_fit],
        axis=1,
    )

    X_valid_imputed = pd.concat(
        [numeric_valid, categorical_valid],
        axis=1,
    )

    return X_fit_imputed, X_valid_imputed


def make_preprocessor():
    """Create a fresh encoder and scaler for each fold."""

    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )


# --------------------------------------------------
# 4. RUN STRATIFIED 5-FOLD CROSS-VALIDATION
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=28,
)

results = []

print("\n--- 5-FOLD CROSS-VALIDATION ---")
print(f"Training records: {len(X_train)}")
print(f"Training stroke cases: {int(y_train.sum())}")
print("Comparing baseline logistic regression vs SMOTENC")

for fold_number, (train_idx, valid_idx) in enumerate(
    cv.split(X_train, y_train),
    start=1,
):

    X_fold_train = X_train.iloc[train_idx].copy()
    X_fold_valid = X_train.iloc[valid_idx].copy()

    y_fold_train = y_train.iloc[train_idx]
    y_fold_valid = y_train.iloc[valid_idx]

    # Fit imputers using only this fold's training records.
    X_fit, X_valid = impute_features(
        X_fold_train,
        X_fold_valid,
    )

    # ----------------------------------------------
    # MODEL A: BASELINE LOGISTIC REGRESSION
    # ----------------------------------------------

    baseline_preprocessor = make_preprocessor()

    X_baseline_fit = baseline_preprocessor.fit_transform(
        X_fit
    )

    X_baseline_valid = baseline_preprocessor.transform(
        X_valid
    )

    baseline_model = LogisticRegression(
        max_iter=2000,
        random_state=28,
    )

    baseline_model.fit(
        X_baseline_fit,
        y_fold_train,
    )

    baseline_probabilities = baseline_model.predict_proba(
        X_baseline_valid
    )[:, 1]

    # ----------------------------------------------
    # MODEL B: SMOTENC + LOGISTIC REGRESSION
    # ----------------------------------------------

    smotenc = SMOTENC(
        categorical_features=categorical_features,
        random_state=28,
    )

    X_resampled, y_resampled = smotenc.fit_resample(
        X_fit,
        y_fold_train,
    )

    smotenc_preprocessor = make_preprocessor()

    X_smotenc_fit = smotenc_preprocessor.fit_transform(
        X_resampled
    )

    X_smotenc_valid = smotenc_preprocessor.transform(
        X_valid
    )

    smotenc_model = LogisticRegression(
        max_iter=2000,
        random_state=28,
    )

    smotenc_model.fit(
        X_smotenc_fit,
        y_resampled,
    )

    smotenc_probabilities = smotenc_model.predict_proba(
        X_smotenc_valid
    )[:, 1]

    # ----------------------------------------------
    # EVALUATE BOTH MODELS ON THE SAME VALIDATION FOLD
    # ----------------------------------------------

    for model_name, probabilities in [
        ("Baseline", baseline_probabilities),
        ("SMOTENC", smotenc_probabilities),
    ]:

        row = {
            "fold": fold_number,
            "model": model_name,
            "roc_auc": roc_auc_score(
                y_fold_valid,
                probabilities,
            ),
            "average_precision": average_precision_score(
                y_fold_valid,
                probabilities,
            ),
        }

        for threshold in thresholds:

            predictions = (
                probabilities >= threshold
            ).astype(int)

            row[f"precision_{threshold:.2f}"] = precision_score(
                y_fold_valid,
                predictions,
                zero_division=0,
            )

            row[f"recall_{threshold:.2f}"] = recall_score(
                y_fold_valid,
                predictions,
                zero_division=0,
            )

            row[f"f1_{threshold:.2f}"] = f1_score(
                y_fold_valid,
                predictions,
                zero_division=0,
            )

        results.append(row)

    print(
        f"Fold {fold_number}/5 complete | "
        f"Validation records: {len(valid_idx)} | "
        f"Stroke cases: {int(y_fold_valid.sum())}"
    )


# --------------------------------------------------
# 5. SUMMARISE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

metric_columns = [
    "roc_auc",
    "average_precision",
    "precision_0.50",
    "recall_0.50",
    "f1_0.50",
    "precision_0.10",
    "recall_0.10",
    "f1_0.10",
    "precision_0.05",
    "recall_0.05",
    "f1_0.05",
]

summary = results_df.groupby("model")[metric_columns].agg(
    ["mean", "std"]
)

print("\n--- CROSS-VALIDATION RESULTS ---")
print("Values are mean (standard deviation) across five folds.")
print(summary.round(3).to_string())


# --------------------------------------------------
# 6. SAVE RESULTS
# --------------------------------------------------

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results_df.to_csv(
    REPORT_PATH,
    index=False,
)

print(f"\nFold results saved to: {REPORT_PATH}")
print("Cross-validation used the saved training dataset only.")
print("Development and test datasets were not accessed.")
