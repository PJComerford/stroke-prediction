# stroke-prediction
1. Project Overview
   In this project I will aim to use Machine Learning to predict stroke occurence from health related and demographic data. The project uses a dataset found of kaggle. I will investigate the quality of the data, process it, and build the model using the dataset. The target variable is whether a stroke was recordedmor not.

2. Dataset
   The source dataset can be found at: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset?resource=download
   The dataset contains 5,110 records and 12 columns, including demographic characteristics, health indicators, lifestyle information and the binary stroke target. The dataset has some missing BMI values and a pretty big class imbalance as only 4.9% of the results had a stroke.

3. Repository Structure


The project is organised into the following directories and files:

```text
stroke-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── reports/
│   └── figures/
│
├── src/
│   ├── explore_data.py
│   ├── prepare_data.py
│   ├── train_model.py
│   ├── cross_validate_models.py
│   └── compare_class_weighting.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

data/raw/: Original dataset downloaded from Kaggle.

data/processed/: Generated training, development and test datasets.

reports/: Generated evaluation results and prediction outputs.

reports/figures/: Visualisations produced during exploratory data analysis.

src/explore_data.py: Performs exploratory data analysis and generates visualisations.

src/prepare_data.py: Validates the dataset, handles the excluded record and creates the stratified data splits.

src/train_model.py: Trains the SMOTENC logistic regression model and evaluates it on the development dataset.

src/cross_validate_models.py: Compares baseline logistic regression and SMOTENC using five-fold cross-validation.

src/compare_class_weighting.py: Compares baseline, class-weighted and SMOTENC approaches using five-fold cross-validation.

4. Environment Setup
   Python and a virtual environment are used to isolate project dependencies.

From the project root, create and activate a virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install the project dependencies:

python -m pip install -r requirements.txt

Ensure the downloaded dataset is in the expected location before running scripts that require it.

5. Running The Project:
   All commands should be executed from the project root with the virtual environment activated.


7. Data Handling and Preparation

The original dataset is validated before modelling.

Validation checks include:

- Expected columns and data types.
- Missing values.
- Duplicate records and identifiers.
- Numerical ranges and binary variables.
- Valid categorical values.

The single record with `gender = Other` is excluded from the modelling dataset. The original raw dataset is retained unchanged.

The remaining 5,109 records are divided using stratified sampling:

| Dataset | Records | Stroke cases | Percentage |
|---|---:|---:|---:|
| Training | 3,575 | 175 | 70% |
| Development | 767 | 37 | 15% |
| Test | 767 | 37 | 15% |
| **Total** | **5,109** | **249** | **100% |

A random seed of 28 is used to ensure reproducibility.

Missing BMI values are retained during data preparation and imputed using the median, fitted on the relevant training data. Categorical missing values are handled using the most frequent category.

The development and test datasets are kept separate from the cross-validation experiments.


7. Model Evaluation

The project investigates different approaches to handling the imbalance between stroke and non-stroke records.

Three approaches are compared:

1. **Baseline logistic regression:** Uses the original training class distribution.
2. **Class-weighted logistic regression:** Applies balanced class weights to give greater importance to the minority class.
3. **SMOTENC:** Generates synthetic minority-class records while accounting for categorical features.

Stratified five-fold cross-validation is performed using the training dataset.

The main evaluation metrics are:

- **ROC-AUC:** Measures the model's ability to distinguish between the two classes across classification thresholds.
- **Average Precision:** Summarises precision and recall performance, this is very useful in my case as the positive class is rare.
- **Precision:** Measures how many predicted stroke cases were actually stroke cases.
- **Recall:** Measures how many recorded stroke cases were identified.
- **F1-score:** Combines precision and recall into a single measure.

Different classification thresholds are also investigated to understand the trade-off between identifying stroke cases and generating false positives.

### Cross-validation results

The following table shows the mean results across five folds:

| Metric | Baseline | Class-weighted | SMOTENC |
|---|---:|---:|---:|
| ROC-AUC | 0.836 | 0.835 | 0.792 |
| Average Precision | 0.221 | 0.202 | 0.181 |
| Recall (0.50 threshold) | 0.011 | 0.794 | 0.600 |
| Recall (0.10 threshold) | 0.571 | 0.994 | 0.966 |
| Recall (0.05 threshold) | 0.777 | 0.994 | 0.989 |

The baseline model achieves the highest ROC-AUC and Average Precision. Class weighting substantially increases recall, although this comes at the cost of lower precision.

These results demonstrate the trade-offs involved in modelling an imbalanced classification dataset.

The test dataset has not been used for model selection or cross-validation, it has been reserved for the final evaluation.

8. Limitations and next steps
The datasaet has several limitations:
    - The positive stroke class is quite small
    - Missing BMI values must be imputed
    - Recorded associations do not necessarily mean relationships are causal.
Future work will include further model development, threshold evaluation and final evaluation on the test dataset.

9. Reproducibility
Fixed random seeds and saved data splits were used to support reproducibility

All Python dependencies are listed in requirements.txt

The raw dataset must be downloaded separately from Kaggle. Processed datasets, figures and evaluation outputs are generated by running the scripts in the order described above.
