import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

import logging

import joblib
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config import (
    CLEAN_DATA_PATH,
    DEFAULT_MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(path: str = None) -> pd.DataFrame:
    """Load cleaned data."""
    if path is None:
        path = CLEAN_DATA_PATH

    if not Path(path).exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    logger.info(f"Loaded data from {path}, shape: {df.shape}")
    return df


def balance_data(X: pd.DataFrame, y: pd.Series, random_state: int = None):
    """Apply SMOTETomek balancing."""
    if random_state is None:
        random_state = RANDOM_STATE

    logger.info("Applying SMOTETomek balancing...")
    smt = SMOTETomek(random_state=random_state)
    X_balanced, y_balanced = smt.fit_resample(X, y)

    logger.info(f"Original class distribution: {y.value_counts().to_dict()}")
    logger.info(f"Balanced class distribution: {pd.Series(y_balanced).value_counts().to_dict()}")

    return X_balanced, y_balanced


def train_model(X_train, y_train, random_state: int = None):
    """Train XGBoost model."""
    if random_state is None:
        random_state = RANDOM_STATE

    logger.info("Training XGBoost model...")
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state)
    model.fit(X_train, y_train)
    logger.info("Model training completed")

    return model


def evaluate_model(model, X_test, y_test) -> str:
    """Evaluate model and return classification report."""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    logger.info("Model evaluation completed")
    return report


def save_model(model, path: str = None):
    """Save trained model."""
    if path is None:
        path = DEFAULT_MODEL_PATH

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def save_report(report: str, path: str = None):
    """Save classification report."""
    if path is None:
        path = Path(MODELS_DIR) / ".." / "outputs" / "model_report.txt"

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("XGBoost + SMOTETomek Classification Report\n")
        f.write("=" * 45 + "\n")
        f.write(report)
        f.write("\nModel trained successfully")

    logger.info(f"Report saved to {path}")


from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler

# ... (inside main after splitting data)


def main():
    """Main training pipeline."""
    try:
        # Load data
        data = load_data()
        X = data.drop(columns=[TARGET_COLUMN])
        y = data[TARGET_COLUMN]

        # Split data FIRST (avoid leakage)
        X_train_raw, X_test, y_train_raw, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        logger.info(f"Training set (raw): {X_train_raw.shape}, Test set: {X_test.shape}")

        # Create pipeline: SMOTE -> Scaler -> XGBoost
        # Note: ImbPipeline only resamples during .fit()
        pipeline = ImbPipeline(
            [
                ("smote", SMOTETomek(random_state=RANDOM_STATE)),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    XGBClassifier(
                        use_label_encoder=False,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.1,
                    ),
                ),
            ]
        )

        logger.info("Fitting pipeline (SMOTE + Scaler + XGBoost)...")
        pipeline.fit(X_train_raw, y_train_raw)

        # Evaluate
        y_pred = pipeline.predict(X_test)
        report = classification_report(y_test, y_pred)
        logger.info(f"Classification Report:\n{report}")

        # Save outputs
        save_model(pipeline)
        save_report(report)

        logger.info("Training pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
