"""Data validation and schema module."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error."""

    pass


class DataType(Enum):
    """Supported data types."""

    NUMERIC = "numeric"
    INTEGER = "integer"
    FLOAT = "float"
    CATEGORICAL = "categorical"


@dataclass
class FeatureSchema:
    """Schema definition for a feature."""

    name: str
    dtype: DataType
    min_value: float = None
    max_value: float = None
    nullable: bool = False
    default_value: Any = None


class DataValidator:
    """Validates data against defined schemas."""

    def __init__(self, schemas: List[FeatureSchema]):
        self.schemas = {s.name: s for s in schemas}
        self.validation_errors = []

    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate entire DataFrame against schema.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (is_valid, error_messages)
        """
        self.validation_errors = []

        # Check required columns
        missing_cols = set(self.schemas.keys()) - set(df.columns)
        if missing_cols:
            self.validation_errors.append(f"Missing required columns: {missing_cols}")
            return False, self.validation_errors

        # Validate each column
        for col_name, schema in self.schemas.items():
            if col_name in df.columns:
                self._validate_column(df[col_name], schema)

        is_valid = len(self.validation_errors) == 0
        return is_valid, self.validation_errors

    def _validate_column(self, series: pd.Series, schema: FeatureSchema):
        """Validate a single column against its schema."""
        # Check nulls
        if not schema.nullable and series.isnull().any():
            null_count = series.isnull().sum()
            self.validation_errors.append(f"Column '{schema.name}' has {null_count} null values but is non-nullable")

        # Check data type
        if schema.dtype == DataType.NUMERIC:
            if not pd.api.types.is_numeric_dtype(series):
                self.validation_errors.append(f"Column '{schema.name}' should be numeric, got {series.dtype}")
        elif schema.dtype == DataType.INTEGER:
            if not pd.api.types.is_integer_dtype(series):
                self.validation_errors.append(f"Column '{schema.name}' should be integer, got {series.dtype}")

        # Check value ranges
        if schema.min_value is not None:
            if series.min() < schema.min_value:
                self.validation_errors.append(f"Column '{schema.name}' has values below minimum {schema.min_value}")

        if schema.max_value is not None:
            if series.max() > schema.max_value:
                self.validation_errors.append(f"Column '{schema.name}' has values above maximum {schema.max_value}")

    def get_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get validation summary statistics."""
        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.to_dict(),
            "numeric_summary": {},
        }

        for col in df.select_dtypes(include=[np.number]).columns:
            summary["numeric_summary"][col] = {
                "min": df[col].min(),
                "max": df[col].max(),
                "mean": df[col].mean(),
                "std": df[col].std(),
            }

        return summary


def create_default_schemas(required_features: List[str]) -> List[FeatureSchema]:
    """Create default feature schemas for loan data.

    Args:
        required_features: List of required feature names

    Returns:
        List of FeatureSchema objects
    """
    schemas = [
        FeatureSchema(name="RevolvingUtilizationOfUnsecuredLines", dtype=DataType.FLOAT, min_value=0.0, nullable=False),
        FeatureSchema(name="age", dtype=DataType.INTEGER, min_value=18, max_value=120, nullable=False),
        FeatureSchema(name="NumberOfTime30-59DaysPastDueNotWorse", dtype=DataType.INTEGER, min_value=0, nullable=False),
        FeatureSchema(name="DebtRatio", dtype=DataType.FLOAT, min_value=0.0, nullable=False),
        FeatureSchema(name="MonthlyIncome", dtype=DataType.FLOAT, min_value=0.0, nullable=True),
        FeatureSchema(name="NumberOfOpenCreditLinesAndLoans", dtype=DataType.INTEGER, min_value=0, nullable=False),
        FeatureSchema(name="NumberOfTimes90DaysLate", dtype=DataType.INTEGER, min_value=0, nullable=False),
        FeatureSchema(name="NumberRealEstateLoansOrLines", dtype=DataType.INTEGER, min_value=0, nullable=False),
        FeatureSchema(name="NumberOfTime60-89DaysPastDueNotWorse", dtype=DataType.INTEGER, min_value=0, nullable=False),
        FeatureSchema(name="NumberOfDependents", dtype=DataType.INTEGER, min_value=0, nullable=True),
    ]

    return schemas
