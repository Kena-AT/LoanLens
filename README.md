# LoanLens

A machine learning-based loan default prediction system with explainable AI (XAI) capabilities using SHAP.

## Overview

LoanLens is a comprehensive credit risk assessment tool that:
- Predicts loan default probability using XGBoost models
- Provides SHAP-based explainability for predictions
- Offers an interactive Streamlit web interface
- Supports multiple model comparison

## Features

- **Model Training**: XGBoost with SMOTETomek for class imbalance handling
- **Prediction Interface**: Upload CSV data and get default probabilities
- **Explainability**: SHAP summary plots and individual record analysis
- **Threshold Tuning**: Interactive threshold adjustment with confusion matrix
- **Model Comparison**: Compare multiple trained models simultaneously

## Project Structure

```
LoanLens/
├── app.py                    # Streamlit web application
├── config.yaml               # Configuration settings
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore               # Git ignore rules
├── data/                     # Data directory
│   ├── credit_data.csv      # Raw credit data
│   └── clean_credit_data.csv # Preprocessed data
├── notebooks/               # Jupyter notebooks
│   ├── 01_EDA.ipynb        # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb # Data preprocessing
│   ├── 03_Modeling.ipynb    # Model training
│   ├── 04_Explainability.ipynb # SHAP analysis
│   ├── 05_Report.ipynb      # Final report
│   └── models/              # Saved models
│       ├── xgb_smote_model.pkl
│       └── scaler.joblib
└── src/                     # Source code
    ├── __init__.py
    ├── config.py            # Configuration management
    ├── data_preprocessing.py # Data preprocessing utilities
    ├── train_model.py       # Model training script
    └── utils.py             # Utility functions
```

## Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd LoanLens
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Web Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Training a New Model

```bash
python src/train_model.py
```

## Data Features

The model uses the following features:

| Feature | Description |
|---------|-------------|
| RevolvingUtilizationOfUnsecuredLines | Credit card balance / credit limits |
| age | Age of borrower in years |
| NumberOfTime30-59DaysPastDueNotWorse | Times 30-59 days late (last 2 years) |
| DebtRatio | Monthly debt / monthly income |
| MonthlyIncome | Monthly income |
| NumberOfOpenCreditLinesAndLoans | Open loans and lines of credit |
| NumberOfTimes90DaysLate | Times 90+ days late |
| NumberRealEstateLoansOrLines | Real estate loans or lines |
| NumberOfTime60-89DaysPastDueNotWorse | Times 60-89 days late |
| NumberOfDependents | Number of dependents |

## Model Details

- **Algorithm**: XGBoost Classifier
- **Balancing**: SMOTETomek for handling class imbalance
- **Evaluation**: ROC AUC, Precision, Recall, F1-Score
- **Explainability**: SHAP (SHapley Additive exPlanations)

## Development

- `app.py`: Main Streamlit application
- `src/data_preprocessing.py`: Data pipeline
- `src/train_model.py`: Model training
- `src/utils.py`: Helper functions

## License

Academic project developed by Group 21.