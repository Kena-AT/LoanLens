"""Unit tests for LoanLens modules."""
import sys
from pathlib import Path
import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.config import REQUIRED_FEATURES, DROP_COLUMNS
from src.data_handler import DataHandler
from src.models import ModelManager
from src.validation import DataValidator, create_default_schemas, FeatureSchema, DataType


class TestDataHandler(unittest.TestCase):
    """Test cases for DataHandler."""
    
    def setUp(self):
        self.handler = DataHandler(REQUIRED_FEATURES, DROP_COLUMNS)
        self.sample_data = pd.DataFrame({
            'RevolvingUtilizationOfUnsecuredLines': [0.5, 0.3],
            'age': [35, 45],
            'NumberOfTime30-59DaysPastDueNotWorse': [0, 1],
            'DebtRatio': [0.3, 0.5],
            'MonthlyIncome': [5000, 6000],
            'NumberOfOpenCreditLinesAndLoans': [5, 8],
            'NumberOfTimes90DaysLate': [0, 0],
            'NumberRealEstateLoansOrLines': [1, 2],
            'NumberOfTime60-89DaysPastDueNotWorse': [0, 0],
            'NumberOfDependents': [2, 3],
            'Unnamed: 0': [1, 2],
            'SeriousDlqin2yrs': [0, 1]
        })
    
    def test_validate_features_success(self):
        """Test feature validation with all required features."""
        is_valid, missing = self.handler.validate_features(self.sample_data)
        self.assertTrue(is_valid)
        self.assertEqual(len(missing), 0)
    
    def test_validate_features_missing(self):
        """Test feature validation with missing features."""
        df_missing = self.sample_data.drop(columns=['age'])
        is_valid, missing = self.handler.validate_features(df_missing)
        self.assertFalse(is_valid)
        self.assertIn('age', missing)
    
    def test_clean_data(self):
        """Test data cleaning removes specified columns."""
        cleaned = self.handler.clean_data(self.sample_data)
        self.assertNotIn('Unnamed: 0', cleaned.columns)
        self.assertNotIn('SeriousDlqin2yrs', cleaned.columns)
        self.assertIn('age', cleaned.columns)
    
    def test_get_feature_names(self):
        """Test getting feature names."""
        self.handler.df = self.sample_data
        names = self.handler.get_feature_names()
        self.assertEqual(len(names), len(self.sample_data.columns))


class TestModelManager(unittest.TestCase):
    """Test cases for ModelManager."""
    
    def setUp(self):
        self.manager = ModelManager()
        # Create a mock model
        self.mock_model = MagicMock()
        self.mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.6, 0.4]])
    
    def test_add_model(self):
        """Test adding a model."""
        self.manager.add_model(self.mock_model, "test_model")
        self.assertEqual(len(self.manager), 1)
        self.assertEqual(self.manager.model_names[0], "test_model")
    
    def test_predict(self):
        """Test prediction."""
        self.manager.add_model(self.mock_model, "test_model")
        df = pd.DataFrame({'feature': [1, 2]})
        result = self.manager.predict(df)
        self.assertEqual(len(result), 2)
        self.mock_model.predict_proba.assert_called_once()
    
    def test_predict_all(self):
        """Test prediction from all models."""
        self.manager.add_model(self.mock_model, "model1")
        self.manager.add_model(self.mock_model, "model2")
        df = pd.DataFrame({'feature': [1, 2]})
        results = self.manager.predict_all(df)
        self.assertEqual(len(results.columns), 2)
    
    def test_clear(self):
        """Test clearing models."""
        self.manager.add_model(self.mock_model, "test_model")
        self.manager.clear()
        self.assertEqual(len(self.manager), 0)


class TestDataValidator(unittest.TestCase):
    """Test cases for DataValidator."""
    
    def setUp(self):
        self.schemas = [
            FeatureSchema(name='age', dtype=DataType.INTEGER, min_value=18, max_value=120),
            FeatureSchema(name='income', dtype=DataType.FLOAT, min_value=0),
        ]
        self.validator = DataValidator(self.schemas)
    
    def test_validate_dataframe_success(self):
        """Test validation with valid data."""
        df = pd.DataFrame({
            'age': [25, 35, 45],
            'income': [5000.0, 6000.0, 7000.0]
        })
        is_valid, errors = self.validator.validate_dataframe(df)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_validate_dataframe_missing_columns(self):
        """Test validation with missing columns."""
        df = pd.DataFrame({'age': [25, 35]})
        is_valid, errors = self.validator.validate_dataframe(df)
        self.assertFalse(is_valid)
        self.assertTrue(any('income' in err for err in errors))
    
    def test_validate_dataframe_invalid_range(self):
        """Test validation with values out of range."""
        df = pd.DataFrame({
            'age': [15, 125],  # Out of range
            'income': [5000.0, 6000.0]
        })
        is_valid, errors = self.validator.validate_dataframe(df)
        self.assertFalse(is_valid)
        self.assertTrue(any('below minimum' in err or 'above maximum' in err for err in errors))


class TestCreateDefaultSchemas(unittest.TestCase):
    """Test cases for default schema creation."""
    
    def test_create_default_schemas(self):
        """Test creating default schemas for loan data."""
        schemas = create_default_schemas(REQUIRED_FEATURES)
        self.assertEqual(len(schemas), len(REQUIRED_FEATURES))
        
        # Check specific schema
        age_schema = next(s for s in schemas if s.name == 'age')
        self.assertEqual(age_schema.dtype, DataType.INTEGER)
        self.assertEqual(age_schema.min_value, 18)
        self.assertEqual(age_schema.max_value, 120)


if __name__ == '__main__':
    unittest.main()
