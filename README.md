# stroke-prediction
1. Project Overview
   In this project I will aim to use Machine Learning to predict stroke occurence from health related and demographic data. The project uses a dataset found of kaggle. I will investigate the quality of the data, process it, and build the model using the dataset. The target variable is whether a stroke was recordedmor not.

2. Dataset
   The source dataset can be found at: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset?resource=download
   The dataset contains 5,110 records and 12 columns, including demographic characteristics, health indicators, lifestyle information and the binary stroke target. The dataset has some missing BMI values and a pretty big class imbalance as only 4.9% of the results had a stroke.

3. Repository Structure
   stroke-prediction/
├── data/
│   ├── raw/
│   └── processed/
├── reports/
├── src/
│   ├── explore_data.py
│   ├── prepare_data.py
│   ├── train_model.py
│   ├── cross_validate_models.py
│   └── compare_class_weighting.py
├── .gitignore
├── README.md
└── requirements.txt

data/raw/: original dataset downloaded from Kaggle.

data/processed/: location for processed data, where applicable.

src/explore_data.py: exploratory data analysis.

src/prepare_data.py: data preparation.

src/train_model.py: initial model training and development-set evaluation.

src/cross_validate_models.py: cross-validation experiments.

src/compare_class_weighting.py: comparison of baseline, class-weighted and SMOTENC approaches.

reports/: generated model evaluation results and prediction outputs.

4. Environment Setup
   Python and a virtual environment are used to isolate project dependencies.

From the project root, create and activate a virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install the project dependencies:

python -m pip install -r requirements.txt

Ensure the downloaded dataset is in the expected location before running scripts that require it.

5. Running The Project
   Run commands from the project root with the virtual environment activated.

Explore the dataset:

python src/explore_data.py

Prepare the data:

python src/prepare_data.py

Train the initial model:

python src/train_model.py

Run cross-validation:

python src/cross_validate_models.py

Compare class-weighting and resampling approaches:

python src/compare_class_weighting.py

The scripts generate analysis or evaluation outputs as implemented in the source code. Refer to the console output and the reports/ directory for the results.

6. Data Handling and Evaluation
   The project uses stratified training, development and test split. The number 28 was used as a random seed to ensure reproducibility. The current split is 70% training, 15% dev and 15% test. Model experimentation is done solely on training data. Stroke cases are rare so I am experimenting with SMOTENC which is a way of creating sythetic datapoints for datasets with class imbalances, we have a big class imbalance in this dataset due to the infrequency of stroke cases. I am using ROC-AUC for evaluation which is a way of evaluating binary classification models. 
