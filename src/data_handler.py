"""Data handling and validation module."""
import pandas as pd
import logging
from typing import List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataHandler:
    """Handles data loading, validation, and preprocessing."""
    
    def __init__(self, required_features: List[str], drop_columns: List[str]):
        self.required_features = required_features
        self.drop_columns = drop_columns
        self.df = None
    
    def load_csv(self, file) -> pd.DataFrame:
        """Load data from CSV file.
        
        Args:
            file: File object or path
            
        Returns:
            Loaded DataFrame
        """
        try:
            df = pd.read_csv(file)
            logger.info(f"Loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def validate_features(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate that required features are present.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (is_valid, missing_features)
        """
        missing = [f for f in self.required_features if f not in df.columns]
        is_valid = len(missing) == 0
        if not is_valid:
            logger.warning(f"Missing features: {missing}")
        return is_valid, missing
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data by dropping unnecessary columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        columns_to_drop = [col for col in self.drop_columns if col in df_clean.columns]
        if columns_to_drop:
            df_clean = df_clean.drop(columns=columns_to_drop, errors='ignore')
            logger.info(f"Dropped columns: {columns_to_drop}")
        return df_clean
    
    def prepare_data(self, file) -> Optional[pd.DataFrame]:
        """Complete data preparation pipeline.
        
        Args:
            file: Input CSV file
            
        Returns:
            Prepared DataFrame or None if validation fails
        """
        try:
            df = self.load_csv(file)
            is_valid, missing = self.validate_features(df)
            
            if not is_valid:
                logger.error(f"Missing required features: {missing}")
                return None
            
            df = self.clean_data(df)
            self.df = df
            return df
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            return None
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names from loaded data."""
        if self.df is None:
            return []
        return self.df.columns.tolist()
    
    def get_record_count(self) -> int:
        """Get number of records in loaded data."""
        if self.df is None:
            return 0
        return len(self.df)
