"""Data drift detection module."""
import sys
from pathlib import Path

src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import json

from config import REQUIRED_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Data drift detection report."""
    feature_name: str
    drift_detected: bool
    method: str
    statistic: float
    p_value: float
    threshold: float
    drift_score: float
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float


class DataDriftDetector:
    """Detects data drift in model features."""
    
    def __init__(self, reference_data: pd.DataFrame = None, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.reference_data = reference_data
        self.reference_statistics = {}
        
        if reference_data is not None:
            self._compute_reference_statistics()
    
    def _compute_reference_statistics(self):
        """Compute and store reference data statistics."""
        if self.reference_data is None:
            return
        
        for col in self.reference_data.columns:
            if pd.api.types.is_numeric_dtype(self.reference_data[col]):
                self.reference_statistics[col] = {
                    'mean': self.reference_data[col].mean(),
                    'std': self.reference_data[col].std(),
                    'min': self.reference_data[col].min(),
                    'max': self.reference_data[col].max(),
                    'q25': self.reference_data[col].quantile(0.25),
                    'q75': self.reference_data[col].quantile(0.75),
                    'median': self.reference_data[col].median()
                }
        
        logger.info(f"Computed reference statistics for {len(self.reference_statistics)} features")
    
    def kolmogorov_smirnov_test(self, reference: pd.Series, current: pd.Series) -> DriftReport:
        """Perform KS test for drift detection."""
        statistic, p_value = stats.ks_2samp(reference.dropna(), current.dropna())
        
        return DriftReport(
            feature_name=reference.name,
            drift_detected=p_value < self.significance_level,
            method='kolmogorov_smirnov',
            statistic=statistic,
            p_value=p_value,
            threshold=self.significance_level,
            drift_score=1 - p_value,
            reference_mean=reference.mean(),
            current_mean=current.mean(),
            reference_std=reference.std(),
            current_std=current.std()
        )
    
    def population_stability_index(
        self, 
        reference: pd.Series, 
        current: pd.Series, 
        bins: int = 10
    ) -> DriftReport:
        """Calculate Population Stability Index (PSI)."""
        # Create bins based on reference data
        min_val = reference.min()
        max_val = reference.max()
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        # Calculate distributions
        ref_hist, _ = np.histogram(reference, bins=bin_edges)
        curr_hist, _ = np.histogram(current, bins=bin_edges)
        
        # Convert to percentages
        ref_pct = ref_hist / len(reference)
        curr_pct = curr_hist / len(current)
        
        # Calculate PSI
        psi = 0
        for i in range(bins):
            if ref_pct[i] > 0:
                psi += (curr_pct[i] - ref_pct[i]) * np.log(curr_pct[i] / ref_pct[i])
        
        # PSI thresholds: <0.1 stable, 0.1-0.2 moderate, >0.2 significant drift
        drift_detected = psi > 0.2
        
        return DriftReport(
            feature_name=reference.name,
            drift_detected=drift_detected,
            method='population_stability_index',
            statistic=psi,
            p_value=None,
            threshold=0.2,
            drift_score=psi,
            reference_mean=reference.mean(),
            current_mean=current.mean(),
            reference_std=reference.std(),
            current_std=current.std()
        )
    
    def wasserstein_distance(self, reference: pd.Series, current: pd.Series) -> DriftReport:
        """Calculate Wasserstein distance for drift detection."""
        from scipy.stats import wasserstein_distance as wd
        
        distance = wd(reference.dropna(), current.dropna())
        
        # Normalize by reference standard deviation
        normalized_distance = distance / (reference.std() + 1e-10)
        
        # Threshold based on normalized distance
        drift_detected = normalized_distance > 0.1
        
        return DriftReport(
            feature_name=reference.name,
            drift_detected=drift_detected,
            method='wasserstein_distance',
            statistic=distance,
            p_value=None,
            threshold=0.1,
            drift_score=normalized_distance,
            reference_mean=reference.mean(),
            current_mean=current.mean(),
            reference_std=reference.std(),
            current_std=current.std()
        )
    
    def detect_drift(
        self, 
        current_data: pd.DataFrame, 
        method: str = 'ks_test'
    ) -> Dict[str, DriftReport]:
        """Detect drift in all features."""
        if self.reference_data is None:
            raise ValueError("Reference data not set")
        
        results = {}
        
        for col in self.reference_data.columns:
            if col not in current_data.columns:
                continue
            
            if not pd.api.types.is_numeric_dtype(self.reference_data[col]):
                continue
            
            reference = self.reference_data[col]
            current = current_data[col]
            
            if method == 'ks_test':
                results[col] = self.kolmogorov_smirnov_test(reference, current)
            elif method == 'psi':
                results[col] = self.population_stability_index(reference, current)
            elif method == 'wasserstein':
                results[col] = self.wasserstein_distance(reference, current)
            else:
                raise ValueError(f"Unknown method: {method}")
        
        return results
    
    def get_drift_summary(self, drift_results: Dict[str, DriftReport]) -> Dict[str, Any]:
        """Get summary of drift detection results."""
        total_features = len(drift_results)
        drifted_features = [name for name, report in drift_results.items() if report.drift_detected]
        
        # Calculate overall drift score
        if drift_results:
            avg_drift_score = np.mean([r.drift_score for r in drift_results.values()])
        else:
            avg_drift_score = 0
        
        return {
            'total_features': total_features,
            'drifted_features_count': len(drifted_features),
            'drifted_features': drifted_features,
            'drift_percentage': (len(drifted_features) / total_features * 100) if total_features > 0 else 0,
            'average_drift_score': avg_drift_score,
            'drift_detected': len(drifted_features) > 0
        }
    
    def save_reference(self, path: str):
        """Save reference data and statistics."""
        data = {
            'statistics': self.reference_statistics,
            'significance_level': self.significance_level,
            'created_at': datetime.now().isoformat()
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved reference statistics to {path}")
    
    def load_reference(self, path: str):
        """Load reference statistics from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.reference_statistics = data['statistics']
        self.significance_level = data.get('significance_level', 0.05)
        
        logger.info(f"Loaded reference statistics from {path}")


class PredictionDriftMonitor:
    """Monitor prediction distribution drift."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.predictions = []
        self.probabilities = []
        self.timestamps = []
    
    def add_prediction(self, prediction: int, probability: float):
        """Add a new prediction to the monitor."""
        self.predictions.append(prediction)
        self.probabilities.append(probability)
        self.timestamps.append(datetime.now())
        
        # Keep only recent predictions
        if len(self.predictions) > self.window_size:
            self.predictions = self.predictions[-self.window_size:]
            self.probabilities = self.probabilities[-self.window_size:]
            self.timestamps = self.timestamps[-self.window_size:]
    
    def get_distribution_stats(self) -> Dict[str, Any]:
        """Get statistics about prediction distribution."""
        if not self.predictions:
            return {}
        
        pred_series = pd.Series(self.predictions)
        prob_series = pd.Series(self.probabilities)
        
        return {
            'total_predictions': len(self.predictions),
            'positive_rate': pred_series.mean(),
            'avg_probability': prob_series.mean(),
            'std_probability': prob_series.std(),
            'min_probability': prob_series.min(),
            'max_probability': prob_series.max(),
            'prediction_entropy': self._calculate_entropy(pred_series)
        }
    
    def _calculate_entropy(self, series: pd.Series) -> float:
        """Calculate entropy of prediction distribution."""
        value_counts = series.value_counts(normalize=True)
        entropy = -sum(p * np.log2(p) for p in value_counts if p > 0)
        return entropy
    
    def check_drift(self, reference_stats: Dict[str, Any], threshold: float = 0.1) -> bool:
        """Check if prediction distribution has drifted."""
        current_stats = self.get_distribution_stats()
        
        if not current_stats or not reference_stats:
            return False
        
        # Check if positive rate has changed significantly
        if 'positive_rate' in reference_stats and 'positive_rate' in current_stats:
            ref_rate = reference_stats['positive_rate']
            curr_rate = current_stats['positive_rate']
            
            if abs(curr_rate - ref_rate) > threshold:
                return True
        
        # Check if average probability has changed significantly
        if 'avg_probability' in reference_stats and 'avg_probability' in current_stats:
            ref_prob = reference_stats['avg_probability']
            curr_prob = current_stats['avg_probability']
            
            if abs(curr_prob - ref_prob) > threshold:
                return True
        
        return False


class DriftDetectionPipeline:
    """Complete drift detection pipeline."""
    
    def __init__(self, reference_data: pd.DataFrame = None, storage_path: str = None):
        if storage_path is None:
            storage_path = str(Path(__file__).parent.parent / "data" / "drift_detection")
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.feature_detector = DataDriftDetector(reference_data)
        self.prediction_monitor = PredictionDriftMonitor()
        self.drift_history = []
    
    def run_detection(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Run complete drift detection."""
        logger.info("Running drift detection...")
        
        # Feature drift detection
        feature_drift = self.feature_detector.detect_drift(current_data, method='ks_test')
        feature_summary = self.feature_detector.get_drift_summary(feature_drift)
        
        # Prediction drift
        pred_stats = self.prediction_monitor.get_distribution_stats()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'feature_drift': {
                'summary': feature_summary,
                'details': {name: {
                    'drift_detected': r.drift_detected,
                    'method': r.method,
                    'statistic': r.statistic,
                    'p_value': r.p_value,
                    'drift_score': r.drift_score
                } for name, r in feature_drift.items()}
            },
            'prediction_drift': pred_stats,
            'overall_drift': feature_summary['drift_detected']
        }
        
        self.drift_history.append(report)
        
        # Save report
        report_file = self.storage_path / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Drift detection completed. Drift detected: {report['overall_drift']}")
        
        return report
    
    def should_retrain(self, threshold_drifted_features: int = 3) -> bool:
        """Determine if model should be retrained based on drift."""
        if not self.drift_history:
            return False
        
        latest_report = self.drift_history[-1]
        drifted_count = latest_report['feature_drift']['summary']['drifted_features_count']
        
        return drifted_count >= threshold_drifted_features
