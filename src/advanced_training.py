"""Advanced model training with hyperparameter tuning and feature engineering."""

import sys
from pathlib import Path

src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

import json
import logging
from datetime import datetime
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
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


class FeatureEngineer:
    """Feature engineering for loan data."""

    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features."""
        df = df.copy()

        # Debt burden ratio
        df["DebtBurden"] = df["DebtRatio"] * df["MonthlyIncome"]

        # Late payment intensity
        df["TotalLatePayments"] = (
            df["NumberOfTime30-59DaysPastDueNotWorse"]
            + df["NumberOfTime60-89DaysPastDueNotWorse"]
            + df["NumberOfTimes90DaysLate"]
        )

        # Credit utilization per age
        df["CreditUtilPerAge"] = df["RevolvingUtilizationOfUnsecuredLines"] / (df["age"] + 1)

        # Income per dependent
        df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)

        # Real estate ratio
        df["RealEstateRatio"] = df["NumberRealEstateLoansOrLines"] / (df["NumberOfOpenCreditLinesAndLoans"] + 1)

        # Risk score (composite)
        df["RiskScore"] = (
            df["TotalLatePayments"] * 0.4
            + df["RevolvingUtilizationOfUnsecuredLines"] * 0.3
            + df["DebtRatio"] * 0.2
            + (df["NumberOfDependents"] / 5) * 0.1
        )

        logger.info(f"Created {len(df.columns) - 10} new features")
        return df


class HyperparameterTuner:
    """Hyperparameter tuning for XGBoost."""

    def __init__(self):
        self.param_distributions = {
            "classifier__n_estimators": [100, 200, 300, 500],
            "classifier__max_depth": [3, 4, 5, 6, 7, 8],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
            "classifier__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "classifier__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "classifier__min_child_weight": [1, 3, 5, 7],
            "classifier__gamma": [0, 0.1, 0.2, 0.3, 0.4],
            "classifier__reg_alpha": [0, 0.001, 0.01, 0.1],
            "classifier__reg_lambda": [1, 1.5, 2, 3],
            "classifier__scale_pos_weight": [1, 2, 3, 5, 10],
        }

        self.best_params = None
        self.best_score = None

    def tune(self, X_train, y_train, n_iter=50) -> Dict[str, Any]:
        """Perform randomized search for best hyperparameters."""
        logger.info(f"Starting hyperparameter tuning with {n_iter} iterations...")

        # Create pipeline with SMOTE and XGBoost
        pipeline = ImbPipeline(
            [
                ("smote", SMOTETomek(random_state=RANDOM_STATE)),
                ("scaler", StandardScaler()),
                (
                    "feature_selection",
                    SelectFromModel(XGBClassifier(n_estimators=100, random_state=RANDOM_STATE), threshold="median"),
                ),
                (
                    "classifier",
                    XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1),
                ),
            ]
        )

        # Randomized search
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions=self.param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1,
        )

        random_search.fit(X_train, y_train)

        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_

        logger.info(f"Best CV Score: {self.best_score:.4f}")
        logger.info(f"Best Parameters: {self.best_params}")

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "cv_results": pd.DataFrame(random_search.cv_results_),
        }


class AdvancedTrainer:
    """Advanced model training pipeline."""

    def __init__(self):
        self.model = None
        self.metrics = {}
        self.training_metadata = {}
        self.feature_engineer = FeatureEngineer()
        self.tuner = HyperparameterTuner()

    def load_data(self, path: str = None) -> pd.DataFrame:
        """Load and prepare data."""
        if path is None:
            path = CLEAN_DATA_PATH

        df = pd.read_csv(path)
        logger.info(f"Loaded data: {df.shape}")
        return df

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data with feature engineering."""
        # Feature engineering
        df = self.feature_engineer.create_features(df)

        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        logger.info(f"Features after engineering: {X.shape[1]}")
        return X, y

    def find_optimal_threshold(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Find optimal classification threshold using precision-recall curve."""
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

        # F1 score for each threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        f1_scores = np.nan_to_num(f1_scores)

        optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last point
        optimal_threshold = thresholds[optimal_idx]

        logger.info(f"Optimal threshold: {optimal_threshold:.3f} (F1: {f1_scores[optimal_idx]:.3f})")

        return optimal_threshold

    def train(self, tune_hyperparams: bool = True, n_iter: int = 50) -> Dict[str, Any]:
        """Full training pipeline."""
        start_time = datetime.now()

        # Load and prepare data
        df = self.load_data()
        X, y = self.prepare_data(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

        if tune_hyperparams:
            # Hyperparameter tuning
            tuning_results = self.tuner.tune(X_train, y_train, n_iter=n_iter)

            # Train final model with best params
            self.model = self._train_final_model(X_train, y_train, tuning_results["best_params"])
        else:
            # Train with default params
            self.model = self._train_default_model(X_train, y_train)

        # Evaluate
        self.metrics = self._evaluate_model(X_test, y_test)

        # Find optimal threshold
        y_proba = self.model.predict_proba(X_test)[:, 1]
        optimal_threshold = self.find_optimal_threshold(y_test, y_proba)
        self.metrics["optimal_threshold"] = optimal_threshold

        # Store metadata
        self.training_metadata = {
            "training_date": start_time.isoformat(),
            "duration_seconds": (datetime.now() - start_time).total_seconds(),
            "data_shape": X.shape,
            "features": list(X.columns),
            "tuned": tune_hyperparams,
            "n_samples": len(df),
        }

        if tune_hyperparams:
            self.training_metadata["best_params"] = self.tuner.best_params_
            self.training_metadata["best_cv_score"] = self.tuner.best_score

        return {"model": self.model, "metrics": self.metrics, "metadata": self.training_metadata}

    def _train_final_model(self, X_train, y_train, best_params: Dict) -> Any:
        """Train model with best hyperparameters."""
        logger.info("Training final model with best parameters...")

        # Extract classifier params
        classifier_params = {
            k.replace("classifier__", ""): v for k, v in best_params.items() if k.startswith("classifier__")
        }

        # Create final pipeline
        pipeline = ImbPipeline(
            [
                ("smote", SMOTETomek(random_state=RANDOM_STATE)),
                ("scaler", StandardScaler()),
                (
                    "feature_selection",
                    SelectFromModel(XGBClassifier(n_estimators=100, random_state=RANDOM_STATE), threshold="median"),
                ),
                (
                    "classifier",
                    XGBClassifier(
                        use_label_encoder=False,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        **classifier_params,
                    ),
                ),
            ]
        )

        pipeline.fit(X_train, y_train)

        logger.info("Final model training completed")
        return pipeline

    def _train_default_model(self, X_train, y_train) -> Any:
        """Train model with default parameters."""
        logger.info("Training model with default parameters...")

        # Apply SMOTE
        smote = SMOTETomek(random_state=RANDOM_STATE)
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_balanced)

        # Train model
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
        )

        model.fit(X_scaled, y_balanced)

        logger.info("Default model training completed")
        return model

    def _evaluate_model(self, X_test, y_test) -> Dict[str, Any]:
        """Evaluate model performance."""
        logger.info("Evaluating model...")

        y_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)

        roc_auc = roc_auc_score(y_test, y_proba)
        report = classification_report(y_test, y_pred, output_dict=True)

        metrics = {
            "roc_auc": roc_auc,
            "accuracy": report["accuracy"],
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1_score": report["1"]["f1-score"],
            "classification_report": report,
        }

        logger.info(f"ROC AUC: {roc_auc:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")

        return metrics

    def save_model(self, model_name: str = None, version: str = "1.0.0"):
        """Save model and metadata."""
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"xgb_optimized_{timestamp}"

        # Create versioned directory
        version_dir = Path(MODELS_DIR) / f"v{version}"
        version_dir.mkdir(parents=True, exist_ok=True)

        model_path = version_dir / f"{model_name}.pkl"
        joblib.dump(self.model, model_path)

        # Save metadata
        metadata = {
            "model_name": model_name,
            "version": version,
            "metrics": self.metrics,
            "training_metadata": self.training_metadata,
        }

        metadata_path = version_dir / f"{model_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {metadata_path}")

        return str(model_path)


def main():
    """Main training script."""
    trainer = AdvancedTrainer()

    # Train with hyperparameter tuning
    results = trainer.train(tune_hyperparams=True, n_iter=30)

    # Save model
    model_path = trainer.save_model(version="1.0.0")

    # Print summary
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 50)
    print(f"ROC AUC: {results['metrics']['roc_auc']:.4f}")
    print(f"F1 Score: {results['metrics']['f1_score']:.4f}")
    print(f"Optimal Threshold: {results['metrics']['optimal_threshold']:.3f}")
    print(f"Model saved: {model_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
