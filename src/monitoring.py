"""Monitoring and health check module."""

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthCheck:
    """System health monitoring."""

    def __init__(self):
        self.start_time = time.time()
        self.checks = {}

    def get_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime": self._get_uptime(),
            "system": self._get_system_metrics(),
            "checks": self._run_health_checks(),
        }

    def _get_uptime(self) -> str:
        """Get application uptime."""
        uptime_seconds = int(time.time() - self.start_time)
        hours = uptime_seconds // 3600
        minutes = (uptime_seconds % 3600) // 60
        seconds = uptime_seconds % 60
        return f"{hours}h {minutes}m {seconds}s"

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system resource metrics."""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage("/").percent,
                "cpu_count": psutil.cpu_count(),
            }
        except Exception as e:
            logger.warning(f"Could not get system metrics: {e}")
            return {}

    def _run_health_checks(self) -> Dict[str, Any]:
        """Run individual health checks."""
        checks = {}

        # Check model files
        checks["models"] = self._check_model_files()

        # Check data files
        checks["data"] = self._check_data_files()

        # Check disk space
        checks["disk_space"] = self._check_disk_space()

        return checks

    def _check_model_files(self) -> Dict[str, Any]:
        """Check if model files exist."""
        from src.config import DEFAULT_MODEL_PATH

        model_exists = os.path.exists(DEFAULT_MODEL_PATH)
        return {
            "status": "healthy" if model_exists else "unhealthy",
            "model_path": DEFAULT_MODEL_PATH,
            "exists": model_exists,
        }

    def _check_data_files(self) -> Dict[str, Any]:
        """Check if data files exist."""
        from src.config import CLEAN_DATA_PATH, RAW_DATA_PATH

        raw_exists = os.path.exists(RAW_DATA_PATH)
        clean_exists = os.path.exists(CLEAN_DATA_PATH)

        return {
            "status": "healthy" if (raw_exists or clean_exists) else "unhealthy",
            "raw_data_exists": raw_exists,
            "clean_data_exists": clean_exists,
        }

    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        try:
            disk = psutil.disk_usage("/")
            free_percent = (disk.free / disk.total) * 100

            status = "healthy"
            if free_percent < 10:
                status = "critical"
            elif free_percent < 20:
                status = "warning"

            return {
                "status": status,
                "free_gb": disk.free / (1024**3),
                "total_gb": disk.total / (1024**3),
                "free_percent": free_percent,
            }
        except Exception as e:
            return {"status": "unknown", "error": str(e)}


class ModelMonitor:
    """Monitor model performance and predictions."""

    def __init__(self):
        self.prediction_count = 0
        self.error_count = 0
        self.latency_sum = 0
        self.prediction_history = []

    def log_prediction(self, latency_ms: float, error: bool = False):
        """Log a prediction event."""
        self.prediction_count += 1
        self.latency_sum += latency_ms

        if error:
            self.error_count += 1

        self.prediction_history.append(
            {"timestamp": datetime.now().isoformat(), "latency_ms": latency_ms, "error": error}
        )

        # Keep only last 1000 predictions
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]

    def get_metrics(self) -> Dict[str, Any]:
        """Get monitoring metrics."""
        avg_latency = (self.latency_sum / self.prediction_count) if self.prediction_count > 0 else 0
        error_rate = (self.error_count / self.prediction_count) if self.prediction_count > 0 else 0

        return {
            "prediction_count": self.prediction_count,
            "error_count": self.error_count,
            "error_rate": error_rate,
            "average_latency_ms": avg_latency,
            "recent_predictions": len(self.prediction_history),
        }


# Global instances
health_check = HealthCheck()
model_monitor = ModelMonitor()
