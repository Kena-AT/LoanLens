"""SHAP explainability module."""

import logging
from typing import Any, List, Union

import numpy as np
import pandas as pd
import shap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExplainabilityAnalyzer:
    """Handles SHAP explainability for model predictions."""

    def __init__(self, model: Any):
        """Initialize with a trained model.

        Args:
            model: Trained model (tree-based)
        """
        self.model = model
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        self._setup_explainer()

    def _setup_explainer(self):
        """Initialize SHAP TreeExplainer."""
        try:
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("SHAP explainer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            raise

    def compute_shap_values(self, df: pd.DataFrame):
        """Compute SHAP values for the dataset.

        Args:
            df: Input DataFrame
        """
        try:
            shap_values_raw = self.explainer.shap_values(df)
            expected_value = self.explainer.expected_value

            # Handle binary/multiclass models
            if isinstance(shap_values_raw, list):
                self.shap_values = shap_values_raw[1]  # Class 1 (default)
                self.expected_value = expected_value[1]
            else:
                self.shap_values = shap_values_raw
                self.expected_value = expected_value

            logger.info(f"SHAP values computed: shape {self.shap_values.shape}")
        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            raise

    def get_summary_plot(self, df: pd.DataFrame, **kwargs):
        """Get SHAP summary plot.

        Args:
            df: Input DataFrame
            **kwargs: Additional plot arguments

        Returns:
            Matplotlib figure
        """
        if self.shap_values is None:
            self.compute_shap_values(df)

        return shap.summary_plot(self.shap_values, df, show=False, **kwargs)

    def get_explanation(self, record_idx: int, df: pd.DataFrame) -> shap.Explanation:
        """Get SHAP explanation for a single record.

        Args:
            record_idx: Index of the record
            df: Full DataFrame

        Returns:
            SHAP Explanation object
        """
        if self.shap_values is None:
            self.compute_shap_values(df)

        record_values = self.shap_values[record_idx]
        record_data = df.iloc[record_idx].values

        explanation = shap.Explanation(
            values=np.array(record_values),
            base_values=np.array([self.expected_value]),
            data=np.array([record_data]),
            feature_names=df.columns.tolist(),
        )

        return explanation

    def get_bar_plot(self, explanation: shap.Explanation, **kwargs):
        """Get SHAP bar plot for an explanation.

        Args:
            explanation: SHAP Explanation object
            **kwargs: Additional plot arguments

        Returns:
            Matplotlib figure
        """
        return shap.plots.bar(explanation, **kwargs)

    def get_feature_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get feature importance based on mean absolute SHAP values.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with feature importance
        """
        if self.shap_values is None:
            self.compute_shap_values(df)

        importance = np.abs(self.shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({"feature": df.columns, "importance": importance}).sort_values(
            "importance", ascending=False
        )

        return feature_importance
