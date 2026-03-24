import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    CLEAN_DATA_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(path: str = None) -> pd.DataFrame:
    """Load data from CSV file."""
    if path is None:
        path = RAW_DATA_PATH

    if not Path(path).exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    logger.info(f"Loaded data from {path}, shape: {df.shape}")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    logger.info("Starting data preprocessing...")

    # Drop rows with missing values
    initial_rows = len(df)
    df = df.dropna()
    dropped = initial_rows - len(df)
    logger.info(f"Dropped {dropped} rows with missing values")

    # Additional preprocessing steps can be added here
    logger.info(f"Final data shape: {df.shape}")
    return df


def split_data(df: pd.DataFrame, target_col: str = None, test_size: float = None, random_state: int = None) -> tuple:
    """Split data into train and test sets."""
    if target_col is None:
        target_col = TARGET_COLUMN
    if test_size is None:
        test_size = TEST_SIZE
    if random_state is None:
        random_state = RANDOM_STATE

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    logger.info(f"Training set size: {X_train.shape}")
    logger.info(f"Test set size: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def save_clean_data(df: pd.DataFrame, path: str = None):
    """Save cleaned data to CSV."""
    if path is None:
        path = CLEAN_DATA_PATH

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved clean data to {path}")


if __name__ == "__main__":
    try:
        # Load raw data
        df = load_data()

        # Preprocess
        df_clean = preprocess_data(df)

        # Save clean data
        save_clean_data(df_clean)

        # Split data
        X_train, X_test, y_train, y_test = split_data(df_clean)

        logger.info("Data preprocessing completed successfully!")

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise
