
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
    / "model_comparison_results.csv"
)

if not TRAIN_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {TRAIN_PATH}. "
        "Run python src/prepare_data.py first."
    )

train_df = pd.read_csv(TRAIN_PATH)

if "stroke" not in train_df.columns:
    raise ValueError(
        "The training dataset must contain a 'stroke' column."
    )

X_train = train_df.drop(columns=["stroke"])
y_train = train_df["stroke"].astype(int)

print("\n--- TRAINING DATA ---")
print(f"Training records: {len(X_train)}")
print(f"Stroke cases: {int(y_train.sum())}")
print(f"Non-stroke cases: {int((y_train == 0).sum())}")
print(f"Stroke prevalence: {y_train.mean():.2%}")


# --------------------------------------------------
# 2. FEATURE DEFINITIONS
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
# 3. FOLD-SPECIFIC IMPUTATION
# --------------------------------------------------

def impute_fold(X_fit, X_valid):
    """Fit imputers on fold training data only."""

    numeric_imputer = SimpleImputer(
        strategy="median"
    )

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
    ).astype(str)

    categorical_valid = pd.DataFrame(
        categorical_imputer.transform(
            X_valid[categorical_features].astype(str)
        ),
        columns=categorical_features,
        index=X_valid.index,
    ).astype(str)

    fit = pd.concat(
        [numeric_fit, categorical_fit],
        axis=1,
    )

    valid = pd.concat(
        [numeric_valid, categorical_valid],
        axis=1,
    )

    return fit, valid


# --------------------------------------------------
# 4. PREPROCESSORS
# --------------------------------------------------

def make_standard_preprocessor():
    """For baseline and class-weighted models."""

    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first",
            ),
        ),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numerical_features),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ])


def make_smotenc_preprocessor():
    """Encode categories after SMOTENC.

    Numeric features are already scaled before resampling.
    """

    categorical_pipeline = Pipeline([
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first",
            ),
        ),
    ])

    return ColumnTransformer([
        ("numeric", "passthrough", numerical_features),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ])


# --------------------------------------------------
# 5. FIVE-FOLD STRATIFIED CROSS-VALIDATION
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=28,
)

results = []

print("\n--- 5-FOLD MODEL COMPARISON ---")
print("Models: Baseline, Class-weighted, SMOTENC")
print("Cross-validation uses saved training data only.")

for fold, (train_idx, valid_idx) in enumerate(
    cv.split(X_train, y_train),
    start=1,
):

    X_fold_train = X_train.iloc[train_idx].copy()
    X_fold_valid = X_train.iloc[valid_idx].copy()

    y_fold_train = y_train.iloc[train_idx]
    y_fold_valid = y_train.iloc[valid_idx]

    X_fit, X_valid = impute_fold(
        X_fold_train,
        X_fold_valid,
    )

    # ----------------------------------------------
    # A. BASELINE LOGISTIC REGRESSION
    # ----------------------------------------------

    baseline_preprocessor = make_standard_preprocessor()

    X_base_fit = baseline_preprocessor.fit_transform(X_fit)
    X_base_valid = baseline_preprocessor.transform(X_valid)

    baseline = LogisticRegression(
        max_iter=2000,
        random_state=28,
    )

    baseline.fit(X_base_fit, y_fold_train)

    baseline_probs = baseline.predict_proba(
        X_base_valid
    )[:, 1]

    # ----------------------------------------------
    # B. CLASS-WEIGHTED LOGISTIC REGRESSION
    # ----------------------------------------------

    weighted_preprocessor = make_standard_preprocessor()

    X_weighted_fit = weighted_preprocessor.fit_transform(X_fit)
    X_weighted_valid = weighted_preprocessor.transform(X_valid)

    weighted = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=28,
    )

    weighted.fit(X_weighted_fit, y_fold_train)

    weighted_probs = weighted.predict_proba(
        X_weighted_valid
    )[:, 1]

    # ----------------------------------------------
    # C. SMOTENC + LOGISTIC REGRESSION
    # ----------------------------------------------

    # Scale numeric features before SMOTENC so that
    # distance calculations are not dominated by units.

    numeric_scaler = StandardScaler()

    X_smotenc_fit = X_fit.copy()
    X_smotenc_valid = X_valid.copy()

    X_smotenc_fit[numerical_features] = (
        numeric_scaler.fit_transform(
            X_fit[numerical_features]
        )
    )

    X_smotenc_valid[numerical_features] = (
        numeric_scaler.transform(
            X_valid[numerical_features]
        )
    )

    smotenc = SMOTENC(
        categorical_features=categorical_features,
        random_state=28,
    )

    X_resampled, y_resampled = smotenc.fit_resample(
        X_smotenc_fit,
        y_fold_train,
    )

    smotenc_preprocessor = make_smotenc_preprocessor()

    X_smote_fit = smotenc_preprocessor.fit_transform(
        X_resampled
    )

    X_smote_valid = smotenc_preprocessor.transform(
        X_smotenc_valid
    )

    smotenc_model = LogisticRegression(
        max_iter=2000,
        random_state=28,
    )

    smotenc_model.fit(X_smote_fit, y_resampled)

    smotenc_probs = smotenc_model.predict_proba(
        X_smote_valid
    )[:, 1]

    # ----------------------------------------------
    # D. EVALUATE ALL MODELS ON THE SAME FOLD
    # ----------------------------------------------

    fold_models = [
        ("Baseline", baseline_probs),
        ("Class-weighted", weighted_probs),
        ("SMOTENC", smotenc_probs),
    ]

    for model_name, probabilities in fold_models:

        row = {
            "fold": fold,
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
        f"Fold {fold}/5 complete | "
        f"Validation records: {len(valid_idx)} | "
        f"Stroke cases: {int(y_fold_valid.sum())}"
    )


# --------------------------------------------------
# 6. SUMMARISE RESULTS
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
# 7. SAVE RESULTS
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
