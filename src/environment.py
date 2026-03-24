"""Environment-based configuration management."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# Application Settings
APP_NAME = os.getenv("APP_NAME", "LoanLens")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
APP_ENV = os.getenv("APP_ENV", "development")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Streamlit Settings
STREAMLIT_SERVER_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
STREAMLIT_SERVER_ADDRESS = os.getenv("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")

# Data Paths (can be overridden by environment)
DATA_RAW_PATH = os.getenv("DATA_RAW_PATH", "data/credit_data.csv")
DATA_CLEAN_PATH = os.getenv("DATA_CLEAN_PATH", "data/clean_credit_data.csv")

# Model Settings
MODEL_RANDOM_STATE = int(os.getenv("MODEL_RANDOM_STATE", "42"))
MODEL_TEST_SIZE = float(os.getenv("MODEL_TEST_SIZE", "0.2"))
MODEL_DEFAULT_THRESHOLD = float(os.getenv("MODEL_DEFAULT_THRESHOLD", "0.5"))

# Model Paths
MODELS_DIR = os.getenv("MODELS_DIR", "notebooks/models")
DEFAULT_MODEL_PATH = os.getenv("DEFAULT_MODEL_PATH", "notebooks/models/xgb_smote_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "notebooks/models/scaler.joblib")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/loanlens.log")

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "default-secret-key")

# Feature Configuration (remain static)
REQUIRED_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

DROP_COLUMNS = ["Unnamed: 0", "SeriousDlqin2yrs"]
TARGET_COLUMN = "SeriousDlqin2yrs"
