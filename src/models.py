"""Model management module for loading and prediction."""
import joblib
import pandas as pd
from typing import List, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model loading and predictions."""
    
    def __init__(self):
        self.models = []
        self.model_names = []
    
    def load_model(self, model_file) -> Tuple[Any, str]:
        """Load a model from file.
        
        Args:
            model_file: File object or path to model
            
        Returns:
            Tuple of (model, model_name)
        """
        try:
            model = joblib.load(model_file)
            name = getattr(model_file, 'name', str(model_file))
            logger.info(f"Successfully loaded model: {name}")
            return model, name
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def add_model(self, model, name: str):
        """Add a model to the manager."""
        self.models.append(model)
        self.model_names.append(name)
    
    def predict(self, df: pd.DataFrame, model_idx: int = 0) -> pd.Series:
        """Get predictions from a specific model.
        
        Args:
            df: Input DataFrame
            model_idx: Index of model to use
            
        Returns:
            Series of probabilities
        """
        if not self.models:
            raise ValueError("No models loaded")
        
        model = self.models[model_idx]
        probabilities = model.predict_proba(df)[:, 1]
        return pd.Series(probabilities, name=f"Default_Prob_{self.model_names[model_idx]}")
    
    def predict_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get predictions from all loaded models.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with predictions from all models
        """
        results = pd.DataFrame()
        for i, model in enumerate(self.models):
            proba = model.predict_proba(df)[:, 1]
            results[self.model_names[i]] = proba
        return results
    
    def get_model(self, idx: int):
        """Get a specific model by index."""
        return self.models[idx]
    
    def get_model_name(self, idx: int) -> str:
        """Get model name by index."""
        return self.model_names[idx]
    
    def clear(self):
        """Clear all loaded models."""
        self.models = []
        self.model_names = []
    
    def __len__(self) -> int:
        return len(self.models)
