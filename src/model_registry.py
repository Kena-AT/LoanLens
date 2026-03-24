"""Model versioning and registry system."""

import sys
from pathlib import Path

src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

import hashlib
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib

from config import MODELS_DIR, RANDOM_STATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Model metadata for registry."""

    name: str
    version: str
    description: str
    author: str
    created_at: str
    metrics: Dict[str, Any]
    parameters: Dict[str, Any]
    features: List[str]
    hash: str
    path: str
    tags: List[str]
    status: str  # 'staging', 'production', 'archived'
    parent_version: Optional[str] = None


class ModelRegistry:
    """Model versioning and registry management."""

    def __init__(self, registry_path: str = None):
        if registry_path is None:
            registry_path = str(Path(MODELS_DIR) / "registry")

        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.registry_path / "registry.json"
        self.models_dir = self.registry_path / "models"
        self.models_dir.mkdir(exist_ok=True)

        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """Load registry from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {"models": {}, "versions": []}

    def _save_registry(self):
        """Save registry to file."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.registry, f, indent=2, default=str)

    def _compute_hash(self, model_path: str) -> str:
        """Compute model file hash."""
        with open(model_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]

    def register_model(
        self,
        model_path: str,
        name: str,
        version: str,
        metrics: Dict[str, Any],
        parameters: Dict[str, Any],
        features: List[str],
        description: str = "",
        author: str = "system",
        tags: List[str] = None,
        parent_version: str = None,
    ) -> ModelMetadata:
        """Register a new model version."""
        if tags is None:
            tags = []

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Compute hash
        model_hash = self._compute_hash(str(model_path))

        # Create version directory
        version_dir = self.models_dir / name / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Copy model file
        dest_path = version_dir / "model.pkl"
        shutil.copy2(model_path, dest_path)

        # Create metadata
        metadata = ModelMetadata(
            name=name,
            version=version,
            description=description,
            author=author,
            created_at=datetime.now().isoformat(),
            metrics=metrics,
            parameters=parameters,
            features=features,
            hash=model_hash,
            path=str(dest_path),
            tags=tags,
            status="staging",
            parent_version=parent_version,
        )

        # Update registry
        if name not in self.registry["models"]:
            self.registry["models"][name] = {}

        self.registry["models"][name][version] = asdict(metadata)

        if version not in self.registry["versions"]:
            self.registry["versions"].append(version)

        self._save_registry()

        logger.info(f"Registered model {name} version {version}")
        return metadata

    def get_model(self, name: str, version: str) -> Optional[Any]:
        """Load a registered model."""
        if name not in self.registry["models"]:
            return None

        if version not in self.registry["models"][name]:
            return None

        model_path = self.registry["models"][name][version]["path"]
        return joblib.load(model_path)

    def get_metadata(self, name: str, version: str) -> Optional[ModelMetadata]:
        """Get model metadata."""
        if name not in self.registry["models"]:
            return None

        if version not in self.registry["models"][name]:
            return None

        data = self.registry["models"][name][version]
        return ModelMetadata(**data)

    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self.registry["models"].keys())

    def list_versions(self, name: str) -> List[str]:
        """List all versions of a model."""
        if name not in self.registry["models"]:
            return []
        return list(self.registry["models"][name].keys())

    def promote_to_production(self, name: str, version: str):
        """Promote a model version to production."""
        if name not in self.registry["models"]:
            raise ValueError(f"Model {name} not found")

        if version not in self.registry["models"][name]:
            raise ValueError(f"Version {version} not found for model {name}")

        # Demote current production version
        for ver, data in self.registry["models"][name].items():
            if data["status"] == "production":
                data["status"] = "archived"
                logger.info(f"Archived model {name} version {ver}")

        # Promote new version
        self.registry["models"][name][version]["status"] = "production"
        self._save_registry()

        logger.info(f"Promoted model {name} version {version} to production")

    def compare_versions(self, name: str, version1: str, version2: str) -> Dict:
        """Compare two model versions."""
        meta1 = self.get_metadata(name, version1)
        meta2 = self.get_metadata(name, version2)

        if not meta1 or not meta2:
            raise ValueError("One or both versions not found")

        comparison = {"version1": version1, "version2": version2, "metrics_comparison": {}, "parameter_changes": {}}

        # Compare metrics
        for metric in set(meta1.metrics.keys()) | set(meta2.metrics.keys()):
            val1 = meta1.metrics.get(metric)
            val2 = meta2.metrics.get(metric)

            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                diff = val2 - val1
                pct_change = (diff / val1 * 100) if val1 != 0 else 0
                comparison["metrics_comparison"][metric] = {
                    "v1": val1,
                    "v2": val2,
                    "difference": diff,
                    "percent_change": pct_change,
                }

        # Compare parameters
        for param in set(meta1.parameters.keys()) | set(meta2.parameters.keys()):
            val1 = meta1.parameters.get(param)
            val2 = meta2.parameters.get(param)

            if val1 != val2:
                comparison["parameter_changes"][param] = {"from": val1, "to": val2}

        return comparison

    def get_production_model(self, name: str) -> Optional[ModelMetadata]:
        """Get the current production version of a model."""
        if name not in self.registry["models"]:
            return None

        for version, data in self.registry["models"][name].items():
            if data["status"] == "production":
                return ModelMetadata(**data)

        return None

    def get_model_lineage(self, name: str, version: str) -> List[ModelMetadata]:
        """Get the lineage of a model version."""
        lineage = []
        current = self.get_metadata(name, version)

        while current:
            lineage.append(current)
            if current.parent_version:
                current = self.get_metadata(name, current.parent_version)
            else:
                break

        return lineage[::-1]  # Return in chronological order

    def search_models(self, tags: List[str] = None, status: str = None) -> List[ModelMetadata]:
        """Search models by tags or status."""
        results = []

        for name, versions in self.registry["models"].items():
            for version, data in versions.items():
                # Filter by tags
                if tags:
                    if not any(tag in data.get("tags", []) for tag in tags):
                        continue

                # Filter by status
                if status and data.get("status") != status:
                    continue

                results.append(ModelMetadata(**data))

        return results


# Global registry instance
_registry = None


def get_registry() -> ModelRegistry:
    """Get or create global registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
