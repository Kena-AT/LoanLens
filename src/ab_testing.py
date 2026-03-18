"""A/B testing framework for model comparison."""
import sys
from pathlib import Path

src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

import json
import random
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import numpy as np
from scipy import stats

from model_registry import ModelRegistry, get_registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AssignmentStrategy(Enum):
    """A/B test assignment strategies."""
    RANDOM = "random"
    HASH = "hash_based"
    USER_ID = "user_id_based"


@dataclass
class ABTestConfig:
    """A/B test configuration."""
    test_id: str
    name: str
    model_a_name: str
    model_a_version: str
    model_b_name: str
    model_b_version: str
    traffic_split: float = 0.5  # Percentage to model B
    assignment_strategy: AssignmentStrategy = AssignmentStrategy.RANDOM
    start_date: str = None
    end_date: Optional[str] = None
    status: str = "draft"  # draft, running, paused, completed
    metrics: List[str] = None
    min_sample_size: int = 1000


@dataclass
class PredictionRecord:
    """Record of a prediction in A/B test."""
    test_id: str
    variant: str  # 'A' or 'B'
    model_name: str
    model_version: str
    input_features: Dict[str, Any]
    prediction: int
    probability: float
    timestamp: str
    user_id: Optional[str] = None
    actual_label: Optional[int] = None


class ABTestingFramework:
    """A/B testing framework for model comparison."""
    
    def __init__(self, storage_path: str = None):
        if storage_path is None:
            storage_path = str(Path(__file__).parent.parent / "data" / "ab_tests")
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.configs_file = self.storage_path / "configs.json"
        self.results_file = self.storage_path / "results.json"
        
        self.configs = self._load_configs()
        self.results = self._load_results()
        
        self.registry = get_registry()
    
    def _load_configs(self) -> Dict:
        """Load test configurations."""
        if self.configs_file.exists():
            with open(self.configs_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_results(self) -> Dict:
        """Load test results."""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_configs(self):
        """Save test configurations."""
        with open(self.configs_file, 'w') as f:
            json.dump(self.configs, f, indent=2, default=str)
    
    def _save_results(self):
        """Save test results."""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def create_test(self, config: ABTestConfig) -> str:
        """Create a new A/B test."""
        test_id = config.test_id
        
        # Validate models exist
        if not self.registry.get_metadata(config.model_a_name, config.model_a_version):
            raise ValueError(f"Model A not found: {config.model_a_name} v{config.model_a_version}")
        
        if not self.registry.get_metadata(config.model_b_name, config.model_b_version):
            raise ValueError(f"Model B not found: {config.model_b_name} v{config.model_b_version}")
        
        if config.start_date is None:
            config.start_date = datetime.now().isoformat()
        
        if config.metrics is None:
            config.metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        self.configs[test_id] = asdict(config)
        self.results[test_id] = {'A': [], 'B': []}
        
        self._save_configs()
        self._save_results()
        
        logger.info(f"Created A/B test: {test_id}")
        return test_id
    
    def assign_variant(self, test_id: str, user_id: str = None, features: Dict = None) -> str:
        """Assign a user/request to variant A or B."""
        if test_id not in self.configs:
            raise ValueError(f"Test {test_id} not found")
        
        config = ABTestConfig(**self.configs[test_id])
        
        if config.status != "running":
            raise ValueError(f"Test {test_id} is not running")
        
        strategy = config.assignment_strategy
        split = config.traffic_split
        
        if strategy == AssignmentStrategy.RANDOM:
            return 'B' if random.random() < split else 'A'
        
        elif strategy == AssignmentStrategy.HASH:
            # Hash-based assignment for consistent user experience
            if features is None:
                features = {}
            
            hash_input = json.dumps(features, sort_keys=True)
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            return 'B' if (hash_value % 100) < (split * 100) else 'A'
        
        elif strategy == AssignmentStrategy.USER_ID:
            if user_id is None:
                raise ValueError("User ID required for USER_ID strategy")
            
            hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            return 'B' if (hash_value % 100) < (split * 100) else 'A'
        
        return 'A'
    
    def get_model_for_variant(self, test_id: str, variant: str) -> Tuple[Any, str, str]:
        """Get the model for a specific variant."""
        config = ABTestConfig(**self.configs[test_id])
        
        if variant == 'A':
            model = self.registry.get_model(config.model_a_name, config.model_a_version)
            return model, config.model_a_name, config.model_a_version
        else:
            model = self.registry.get_model(config.model_b_name, config.model_b_version)
            return model, config.model_b_name, config.model_b_version
    
    def record_prediction(
        self,
        test_id: str,
        variant: str,
        input_features: Dict,
        prediction: int,
        probability: float,
        user_id: str = None,
        actual_label: int = None
    ):
        """Record a prediction for analysis."""
        config = ABTestConfig(**self.configs[test_id])
        model_name = config.model_a_name if variant == 'A' else config.model_b_name
        model_version = config.model_a_version if variant == 'A' else config.model_b_version
        
        record = PredictionRecord(
            test_id=test_id,
            variant=variant,
            model_name=model_name,
            model_version=model_version,
            input_features=input_features,
            prediction=prediction,
            probability=probability,
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            actual_label=actual_label
        )
        
        self.results[test_id][variant].append(asdict(record))
        self._save_results()
    
    def start_test(self, test_id: str):
        """Start an A/B test."""
        if test_id not in self.configs:
            raise ValueError(f"Test {test_id} not found")
        
        self.configs[test_id]['status'] = 'running'
        self.configs[test_id]['start_date'] = datetime.now().isoformat()
        self._save_configs()
        
        logger.info(f"Started A/B test: {test_id}")
    
    def pause_test(self, test_id: str):
        """Pause an A/B test."""
        if test_id not in self.configs:
            raise ValueError(f"Test {test_id} not found")
        
        self.configs[test_id]['status'] = 'paused'
        self._save_configs()
        
        logger.info(f"Paused A/B test: {test_id}")
    
    def stop_test(self, test_id: str):
        """Stop/complet an A/B test."""
        if test_id not in self.configs:
            raise ValueError(f"Test {test_id} not found")
        
        self.configs[test_id]['status'] = 'completed'
        self.configs[test_id]['end_date'] = datetime.now().isoformat()
        self._save_configs()
        
        logger.info(f"Completed A/B test: {test_id}")
    
    def get_test_status(self, test_id: str) -> Dict:
        """Get current test status and statistics."""
        if test_id not in self.configs:
            raise ValueError(f"Test {test_id} not found")
        
        config = ABTestConfig(**self.configs[test_id])
        results = self.results[test_id]
        
        stats = {
            'config': asdict(config),
            'statistics': {
                'variant_A': len(results['A']),
                'variant_B': len(results['B']),
                'total': len(results['A']) + len(results['B'])
            }
        }
        
        # Calculate metrics if we have actual labels
        for variant in ['A', 'B']:
            variant_results = results[variant]
            labeled_results = [r for r in variant_results if r.get('actual_label') is not None]
            
            if labeled_results:
                predictions = [r['prediction'] for r in labeled_results]
                actuals = [r['actual_label'] for r in labeled_results]
                
                # Calculate accuracy
                correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
                accuracy = correct / len(labeled_results)
                
                stats['statistics'][f'{variant}_accuracy'] = accuracy
                stats['statistics'][f'{variant}_labeled_samples'] = len(labeled_results)
        
        return stats
    
    def analyze_results(self, test_id: str, metric: str = 'accuracy') -> Dict:
        """Statistical analysis of A/B test results."""
        if test_id not in self.results:
            raise ValueError(f"Test {test_id} not found")
        
        results = self.results[test_id]
        
        # Extract metric values
        a_values = []
        b_values = []
        
        for record in results['A']:
            if metric == 'prediction' and 'prediction' in record:
                a_values.append(record['prediction'])
            elif metric == 'probability' and 'probability' in record:
                a_values.append(record['probability'])
        
        for record in results['B']:
            if metric == 'prediction' and 'prediction' in record:
                b_values.append(record['prediction'])
            elif metric == 'probability' and 'probability' in record:
                b_values.append(record['probability'])
        
        if not a_values or not b_values:
            return {'error': 'Insufficient data for analysis'}
        
        # Statistical tests
        t_stat, p_value = stats.ttest_ind(a_values, b_values)
        
        # Effect size (Cohen's d)
        mean_a, mean_b = np.mean(a_values), np.mean(b_values)
        std_a, std_b = np.std(a_values), np.std(b_values)
        pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
        cohens_d = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0
        
        return {
            'metric': metric,
            'variant_A': {
                'n': len(a_values),
                'mean': mean_a,
                'std': std_a
            },
            'variant_B': {
                'n': len(b_values),
                'mean': mean_b,
                'std': std_b
            },
            'difference': mean_b - mean_a,
            'percent_change': ((mean_b - mean_a) / mean_a * 100) if mean_a != 0 else 0,
            'statistical_test': {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            },
            'effect_size': {
                'cohens_d': cohens_d,
                'interpretation': (
                    'small' if abs(cohens_d) < 0.5 else
                    'medium' if abs(cohens_d) < 0.8 else
                    'large'
                )
            }
        }
    
    def get_active_tests(self) -> List[str]:
        """Get list of currently running tests."""
        return [
            test_id for test_id, config in self.configs.items()
            if config.get('status') == 'running'
        ]
    
    def recommend_winner(self, test_id: str) -> Optional[str]:
        """Recommend winning variant based on analysis."""
        try:
            analysis = self.analyze_results(test_id)
            
            if 'error' in analysis:
                return None
            
            # Check if statistically significant
            if not analysis['statistical_test']['significant']:
                return None
            
            # Recommend based on metric improvement
            if analysis['difference'] > 0:
                return 'B'  # Model B is better
            else:
                return 'A'  # Model A is better
            
        except Exception as e:
            logger.error(f"Error recommending winner: {e}")
            return None
