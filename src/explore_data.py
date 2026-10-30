

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

FIGURES_PATH = PROJECT_ROOT / "reports" / "figures"
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


# --------------------------------------------------
# 2. Load the raw dataset
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}\n"
        "Download the dataset from Kaggle and place it in data/raw/."
    )

df = pd.read_csv(DATA_PATH)

print("\n--- DATASET OVERVIEW ---")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print(f"Column names: {df.columns.tolist()}")

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- COLUMN TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

print("\n--- STROKE DISTRIBUTION ---")
print(df["stroke"].value_counts(dropna=False))
print(f"Stroke prevalence: {df['stroke'].mean() * 100:.2f}%")

print("\n--- NUMERICAL SUMMARY ---")
print(df.describe())

categorical_columns = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]

print("\n--- CATEGORICAL VALUE COUNTS ---")

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False))


# --------------------------------------------------
# 3. Chart 1: Stroke class balance
# --------------------------------------------------

stroke_counts = (
    df["stroke"]
    .map({0: "No stroke", 1: "Stroke"})
    .fillna("Unknown")
    .value_counts()
    .reindex(["No stroke", "Stroke", "Unknown"], fill_value=0)
)

fig, ax = plt.subplots(figsize=(7, 5))

sns.barplot(
    x=stroke_counts.index,
    y=stroke_counts.values,
    ax=ax,
)

ax.set_title("Distribution of Stroke Outcomes")
ax.set_xlabel("Stroke outcome")
ax.set_ylabel("Number of records")

for i, count in enumerate(stroke_counts.values):
    ax.text(i, count, f"{count:,}", ha="center", va="bottom")

fig.tight_layout()
fig.savefig(FIGURES_PATH / "stroke_distribution.png", dpi=300)
plt.close(fig)


# --------------------------------------------------
# 4. Chart 2: Age distribution by stroke outcome
# --------------------------------------------------

age_df = df.copy()
age_df["stroke_label"] = age_df["stroke"].map(
    {0: "No stroke", 1: "Stroke"}
)

fig, ax = plt.subplots(figsize=(9, 5))

sns.histplot(
    data=age_df,
    x="age",
    hue="stroke_label",
    bins=25,
    element="step",
    stat="density",
    common_norm=False,
    ax=ax,
)

ax.set_title("Age Distribution by Stroke Outcome")
ax.set_xlabel("Age (years)")
ax.set_ylabel("Density")

fig.tight_layout()
fig.savefig(FIGURES_PATH / "age_by_stroke.png", dpi=300)
plt.close(fig)


# --------------------------------------------------
# 5. Chart 3: BMI distribution and missingness
# --------------------------------------------------

bmi_df = df.copy()
bmi_df["bmi_status"] = bmi_df["bmi"].isna().map(
    {False: "BMI recorded", True: "BMI missing"}
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(
    data=bmi_df,
    x="bmi",
    bins=30,
    ax=axes[0],
)

axes[0].set_title("BMI Distribution")
axes[0].set_xlabel("BMI")
axes[0].set_ylabel("Number of records")

bmi_counts = (
    bmi_df["bmi_status"]
    .value_counts()
    .reindex(["BMI recorded", "BMI missing"], fill_value=0)
)

sns.barplot(
    x=bmi_counts.index,
    y=bmi_counts.values,
    ax=axes[1],
)

axes[1].set_title("BMI Data Completeness")
axes[1].set_xlabel("")
axes[1].set_ylabel("Number of records")
axes[1].tick_params(axis="x", rotation=10)

for i, count in enumerate(bmi_counts.values):
    axes[1].text(i, count, f"{count:,}", ha="center", va="bottom")

fig.tight_layout()
fig.savefig(FIGURES_PATH / "bmi_distribution_missingness.png", dpi=300)
plt.close(fig)


# --------------------------------------------------
# 6. Chart 4: Glucose distribution by stroke outcome
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))

sns.histplot(
    data=age_df,
    x="avg_glucose_level",
    hue="stroke_label",
    bins=30,
    element="step",
    stat="density",
    common_norm=False,
    ax=ax,
)

ax.set_title("Average Glucose Distribution by Stroke Outcome")
ax.set_xlabel("Average glucose level")
ax.set_ylabel("Density")

fig.tight_layout()
fig.savefig(
    FIGURES_PATH / "glucose_by_stroke.png",
    dpi=300,
)
plt.close(fig)


# --------------------------------------------------
# 7. Completion summary
# --------------------------------------------------

print("\n--- EDA COMPLETE ---")
print(f"Figures saved to: {FIGURES_PATH}")
print("1. stroke_distribution.png")
print("2. age_by_stroke.png")
print("3. bmi_distribution_missingness.png")
print("4. glucose_by_stroke.png")
