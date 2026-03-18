"""Automated retraining pipeline."""
import sys
from pathlib import Path

src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import schedule
import time
import threading
from dataclasses import dataclass

from advanced_training import AdvancedTrainer
from drift_detection import DriftDetectionPipeline
from model_registry import get_registry
from config import CLEAN_DATA_PATH, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrainingConfig:
    """Configuration for automated retraining."""
    trigger_on_drift: bool = True
    drift_threshold: int = 3  # Number of drifted features
    trigger_on_schedule: bool = False
    schedule_interval: str = 'weekly'  # daily, weekly, monthly
    min_new_samples: int = 1000
    performance_threshold: float = 0.05  # Retrain if performance drops by 5%
    max_models_to_keep: int = 5
    enable_ab_testing: bool = True


class RetrainingPipeline:
    """Automated model retraining pipeline."""
    
    def __init__(self, config: RetrainingConfig = None):
        if config is None:
            config = RetrainingConfig()
        
        self.config = config
        self.trainer = AdvancedTrainer()
        self.drift_detector = None
        self.registry = get_registry()
        
        self.retraining_history = []
        self.is_running = False
        self.scheduler_thread = None
    
    def initialize_drift_detector(self, reference_data: pd.DataFrame):
        """Initialize drift detector with reference data."""
        self.drift_detector = DriftDetectionPipeline(reference_data)
        logger.info("Initialized drift detector")
    
    def check_retraining_needed(self, current_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Check if retraining is needed based on various triggers."""
        triggers = []
        
        # Check data drift
        if self.config.trigger_on_drift and self.drift_detector:
            if current_data is not None:
                drift_report = self.drift_detector.run_detection(current_data)
                
                if drift_report['overall_drift']:
                    drifted_count = drift_report['feature_drift']['summary']['drifted_features_count']
                    if drifted_count >= self.config.drift_threshold:
                        triggers.append({
                            'type': 'data_drift',
                            'severity': 'high' if drifted_count > self.config.drift_threshold * 2 else 'medium',
                            'details': f'{drifted_count} features showing drift'
                        })
        
        # Check model performance (if we have recent metrics)
        if self.retraining_history:
            last_model = self.retraining_history[-1]
            if 'metrics' in last_model:
                last_roc_auc = last_model['metrics'].get('roc_auc', 0)
                # In real scenario, we'd compare with current performance on validation set
        
        # Check if enough new data
        if current_data is not None:
            if len(current_data) >= self.config.min_new_samples:
                triggers.append({
                    'type': 'new_data',
                    'severity': 'low',
                    'details': f'{len(current_data)} new samples available'
                })
        
        return {
            'retraining_needed': len(triggers) > 0,
            'triggers': triggers,
            'timestamp': datetime.now().isoformat()
        }
    
    def run_retraining(self, data_path: str = None) -> Dict[str, Any]:
        """Execute the full retraining pipeline."""
        logger.info("Starting automated retraining...")
        start_time = datetime.now()
        
        try:
            # Run training with hyperparameter tuning
            results = self.trainer.train(tune_hyperparams=True, n_iter=30)
            
            # Get previous model for comparison
            previous_model = None
            if self.retraining_history:
                previous_model = self.retraining_history[-1]
            
            # Compare with previous model
            comparison = self._compare_models(previous_model, results)
            
            # Save new model
            version = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = self.trainer.save_model(version=version)
            
            # Register in model registry
            parent_version = previous_model.get('version') if previous_model else None
            
            self.registry.register_model(
                model_path=model_path,
                name="loan_default_predictor",
                version=version,
                metrics=results['metrics'],
                parameters=self.trainer.training_metadata.get('best_params', {}),
                features=self.trainer.training_metadata.get('features', []),
                description=f"Auto-retrained model {version}",
                tags=['automated', 'retrained'],
                parent_version=parent_version
            )
            
            # Record retraining event
            retraining_record = {
                'version': version,
                'timestamp': start_time.isoformat(),
                'duration_seconds': (datetime.now() - start_time).total_seconds(),
                'metrics': results['metrics'],
                'metadata': results['metadata'],
                'comparison': comparison,
                'model_path': model_path
            }
            
            self.retraining_history.append(retraining_record)
            
            # Cleanup old models
            self._cleanup_old_models()
            
            # Start A/B test if enabled
            if self.config.enable_ab_testing and previous_model:
                self._setup_ab_test(previous_model['version'], version)
            
            logger.info(f"Retraining completed successfully. New version: {version}")
            
            return {
                'success': True,
                'version': version,
                'metrics': results['metrics'],
                'comparison': comparison
            }
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _compare_models(self, previous_model: Dict, new_results: Dict) -> Dict:
        """Compare new model with previous version."""
        if previous_model is None:
            return {'is_first_model': True}
        
        prev_metrics = previous_model.get('metrics', {})
        new_metrics = new_results['metrics']
        
        comparison = {}
        
        for metric in ['roc_auc', 'f1_score', 'precision', 'recall']:
            prev_val = prev_metrics.get(metric, 0)
            new_val = new_metrics.get(metric, 0)
            
            diff = new_val - prev_val
            pct_change = (diff / prev_val * 100) if prev_val != 0 else 0
            
            comparison[metric] = {
                'previous': prev_val,
                'new': new_val,
                'difference': diff,
                'percent_change': pct_change,
                'improved': diff > 0
            }
        
        # Overall assessment
        improved_metrics = sum(1 for m in comparison.values() if isinstance(m, dict) and m.get('improved', False))
        total_metrics = len([m for m in comparison.values() if isinstance(m, dict)])
        
        comparison['overall'] = {
            'improved_metrics': improved_metrics,
            'total_metrics': total_metrics,
            'is_better': improved_metrics > total_metrics / 2
        }
        
        return comparison
    
    def _cleanup_old_models(self):
        """Remove old model versions to save space."""
        if len(self.retraining_history) <= self.config.max_models_to_keep:
            return
        
        # Keep only recent models
        models_to_remove = self.retraining_history[:-self.config.max_models_to_keep]
        
        for model_record in models_to_remove:
            model_path = model_record.get('model_path')
            if model_path and Path(model_path).exists():
                try:
                    Path(model_path).unlink()
                    logger.info(f"Cleaned up old model: {model_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove old model {model_path}: {e}")
    
    def _setup_ab_test(self, version_a: str, version_b: str):
        """Set up A/B test between old and new model."""
        from ab_testing import ABTestingFramework, ABTestConfig, AssignmentStrategy
        
        ab_framework = ABTestingFramework()
        
        test_config = ABTestConfig(
            test_id=f"retrain_{version_a}_vs_{version_b}",
            name="Automated Retraining A/B Test",
            model_a_name="loan_default_predictor",
            model_a_version=version_a,
            model_b_name="loan_default_predictor",
            model_b_version=version_b,
            traffic_split=0.2,  # 20% to new model initially
            assignment_strategy=AssignmentStrategy.RANDOM,
            min_sample_size=500
        )
        
        ab_framework.create_test(test_config)
        ab_framework.start_test(test_config.test_id)
        
        logger.info(f"Started A/B test: {test_config.test_id}")
    
    def start_scheduler(self):
        """Start the automated retraining scheduler."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        
        if self.config.trigger_on_schedule:
            if self.config.schedule_interval == 'daily':
                schedule.every().day.at("02:00").do(self._scheduled_check)
            elif self.config.schedule_interval == 'weekly':
                schedule.every().week.at("02:00").do(self._scheduled_check)
            elif self.config.schedule_interval == 'monthly':
                schedule.every(30).days.at("02:00").do(self._scheduled_check)
        
        def run_scheduler():
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Started automated retraining scheduler")
    
    def stop_scheduler(self):
        """Stop the scheduler."""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Stopped automated retraining scheduler")
    
    def _scheduled_check(self):
        """Scheduled check for retraining needs."""
        logger.info("Running scheduled retraining check...")
        
        # Load current data
        try:
            current_data = pd.read_csv(CLEAN_DATA_PATH)
            
            check_result = self.check_retraining_needed(current_data)
            
            if check_result['retraining_needed']:
                logger.info("Retraining triggers detected, starting retraining...")
                result = self.run_retraining()
                
                if result['success']:
                    logger.info(f"Scheduled retraining completed: {result['version']}")
                else:
                    logger.error(f"Scheduled retraining failed: {result.get('error')}")
            else:
                logger.info("No retraining needed at this time")
                
        except Exception as e:
            logger.error(f"Error in scheduled check: {e}")
    
    def get_retraining_history(self) -> List[Dict]:
        """Get history of all retraining events."""
        return self.retraining_history
    
    def get_latest_model_info(self) -> Optional[Dict]:
        """Get information about the latest model."""
        if not self.retraining_history:
            return None
        
        return self.retraining_history[-1]


def main():
    """Run the automated retraining pipeline once."""
    config = RetrainingConfig(
        trigger_on_drift=True,
        drift_threshold=3,
        enable_ab_testing=True
    )
    
    pipeline = RetrainingPipeline(config)
    
    # Initialize with reference data
    reference_data = pd.read_csv(CLEAN_DATA_PATH)
    pipeline.initialize_drift_detector(reference_data)
    
    # Check if retraining is needed
    check_result = pipeline.check_retraining_needed(reference_data)
    
    print(f"Retraining needed: {check_result['retraining_needed']}")
    print(f"Triggers: {check_result['triggers']}")
    
    if check_result['retraining_needed']:
        result = pipeline.run_retraining()
        print(f"Retraining result: {result}")


if __name__ == '__main__':
    main()
