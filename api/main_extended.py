"""Extended FastAPI with comprehensive endpoints including A/B testing, drift detection, and retraining."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import json
import logging
from datetime import datetime

from src.config import REQUIRED_FEATURES, DEFAULT_THRESHOLD
from src.models import ModelManager
from src.data_handler import DataHandler
from src.monitoring import health_check, model_monitor
from src.model_registry import get_registry, ModelRegistry
from src.ab_testing import ABTestingFramework, ABTestConfig, AssignmentStrategy
from src.drift_detection import DriftDetectionPipeline, DataDriftDetector
from src.automated_retraining import RetrainingPipeline, RetrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LoanLens API",
    description="""
    ## LoanLens - Advanced Loan Default Prediction API
    
    This API provides comprehensive loan default prediction capabilities with:
    
    - **Predictions**: Single and batch predictions with probability scores
    - **Explainability**: SHAP-based feature importance explanations
    - **Model Management**: Model versioning, registry, and A/B testing
    - **Monitoring**: Health checks, metrics, and drift detection
    - **Automation**: Automated retraining pipeline
    
    ## Features
    
    ### Prediction Endpoints
    - Single prediction with detailed explanations
    - Batch predictions for CSV files
    - Model comparison and selection
    
    ### Model Management
    - Upload and manage multiple models
    - Model versioning and registry
    - Promote models to production
    
    ### A/B Testing
    - Create and manage A/B tests
    - Automatic traffic splitting
    - Statistical analysis of results
    
    ### Drift Detection
    - Monitor data drift in real-time
    - Multiple detection methods (KS test, PSI, Wasserstein)
    - Automated retraining triggers
    
    ## Authentication
    
    API authentication will be implemented in production.
    
    ## Rate Limits
    
    Production deployments should implement rate limiting.
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
model_manager = ModelManager()
data_handler = DataHandler(REQUIRED_FEATURES, [])
registry = get_registry()
ab_framework = ABTestingFramework()
retraining_pipeline = None

# Pydantic models
class FeatureInput(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., ge=0, le=10, description="Credit utilization ratio")
    age: int = Field(..., ge=18, le=120, description="Age in years")
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(..., ge=0, description="30-59 days late count")
    DebtRatio: float = Field(..., ge=0, description="Debt to income ratio")
    MonthlyIncome: float = Field(..., ge=0, description="Monthly income")
    NumberOfOpenCreditLinesAndLoans: int = Field(..., ge=0, description="Open credit lines")
    NumberOfTimes90DaysLate: int = Field(..., ge=0, description="90+ days late count")
    NumberRealEstateLoansOrLines: int = Field(..., ge=0, description="Real estate loans")
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(..., ge=0, description="60-89 days late count")
    NumberOfDependents: int = Field(..., ge=0, description="Number of dependents")

class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Feature values")
    model_name: Optional[str] = Field(None, description="Specific model to use")
    threshold: float = Field(DEFAULT_THRESHOLD, ge=0, le=1)

class ABTestCreateRequest(BaseModel):
    name: str = Field(..., description="Test name")
    model_a_name: str = Field(..., description="Control model name")
    model_a_version: str = Field(..., description="Control model version")
    model_b_name: str = Field(..., description="Treatment model name")
    model_b_version: str = Field(..., description="Treatment model version")
    traffic_split: float = Field(0.5, ge=0, le=1, description="Percentage to variant B")

# Endpoints
@app.get("/", tags=["General"])
async def root():
    """API information and available endpoints."""
    return {
        "name": "LoanLens API",
        "version": "2.0.0",
        "description": "Advanced Loan Default Prediction API",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "predict": "/predict",
            "models": "/models",
            "ab_tests": "/ab-tests",
            "drift": "/drift",
            "retraining": "/retraining"
        }
    }

@app.get("/health", tags=["Monitoring"])
async def health_check_endpoint():
    """Get comprehensive system health status."""
    return health_check.get_status()

@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get model performance metrics."""
    return model_monitor.get_metrics()

@app.get("/features", tags=["API Info"], response_model=Dict[str, Any])
async def get_features():
    """Get list of required features with descriptions."""
    return {
        "required_features": REQUIRED_FEATURES,
        "count": len(REQUIRED_FEATURES),
        "descriptions": {
            "RevolvingUtilizationOfUnsecuredLines": "Total balance on credit cards / credit limits",
            "age": "Age of borrower in years",
            "NumberOfTime30-59DaysPastDueNotWorse": "Number of times 30-59 days late (last 2 years)",
            "DebtRatio": "Monthly debt payments / gross monthly income",
            "MonthlyIncome": "Monthly income",
            "NumberOfOpenCreditLinesAndLoans": "Number of open loans and lines of credit",
            "NumberOfTimes90DaysLate": "Number of times 90+ days late",
            "NumberRealEstateLoansOrLines": "Number of real estate loans or lines",
            "NumberOfTime60-89DaysPastDueNotWorse": "Number of times 60-89 days late",
            "NumberOfDependents": "Number of dependents in family"
        }
    }

@app.post("/models/load", tags=["Models"])
async def load_model(file: UploadFile = File(...)):
    """Upload and load a model file."""
    try:
        import tempfile
        import shutil
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        # Load model
        model, name = model_manager.load_model(tmp_path)
        model_manager.add_model(model, file.filename)
        
        # Clean up temp file
        Path(tmp_path).unlink()
        
        return {
            "message": "Model loaded successfully",
            "model_name": file.filename,
            "models_loaded": len(model_manager)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {str(e)}")

@app.get("/models", response_model=List[str], tags=["Models"])
async def list_models():
    """List all loaded models."""
    return model_manager.model_names

@app.delete("/models/{model_name}", tags=["Models"])
async def unload_model(model_name: str):
    """Unload a specific model."""
    if model_name not in model_manager.model_names:
        raise HTTPException(status_code=404, detail="Model not found")
    
    idx = model_manager.model_names.index(model_name)
    model_manager.models.pop(idx)
    model_manager.model_names.pop(idx)
    
    return {"message": f"Model {model_name} unloaded"}

@app.post("/predict", tags=["Predictions"])
async def predict(request: PredictionRequest):
    """
    Make a single prediction with SHAP explanations.
    
    Returns prediction, probability, and feature explanations.
    """
    import time
    start_time = time.time()
    
    try:
        if len(model_manager) == 0:
            raise HTTPException(status_code=400, detail="No models loaded")
        
        df = pd.DataFrame([request.features])
        
        model_idx = 0
        if request.model_name:
            if request.model_name not in model_manager.model_names:
                raise HTTPException(status_code=404, detail="Model not found")
            model_idx = model_manager.model_names.index(request.model_name)
        
        model = model_manager.get_model(model_idx)
        model_name = model_manager.get_model_name(model_idx)
        
        proba = model.predict_proba(df)[0, 1]
        prediction = int(proba >= request.threshold)
        
        # Generate explanation
        from src.explainability import ExplainabilityAnalyzer
        explanation = None
        try:
            analyzer = ExplainabilityAnalyzer(model)
            analyzer.compute_shap_values(df)
            exp = analyzer.get_explanation(0, df)
            
            explanation = {
                "base_value": float(exp.base_values[0]),
                "features": [
                    {"feature": name, "value": float(val), "contribution": float(contrib)}
                    for name, val, contrib in zip(df.columns, exp.data[0], exp.values)
                ]
            }
        except Exception as e:
            logger.warning(f"Explanation generation failed: {e}")
        
        latency = (time.time() - start_time) * 1000
        model_monitor.log_prediction(latency)
        
        return {
            "prediction": prediction,
            "probability": float(proba),
            "model_name": model_name,
            "threshold": request.threshold,
            "timestamp": datetime.now().isoformat(),
            "explanation": explanation,
            "latency_ms": latency
        }
        
    except HTTPException:
        model_monitor.log_prediction(0, error=True)
        raise
    except Exception as e:
        model_monitor.log_prediction(0, error=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", tags=["Predictions"])
async def predict_batch(request: PredictionRequest): # we can use List for actual implementation if needed but keeping it simple for now or copy batch from main
    pass

@app.post("/predict/file", tags=["Predictions"])
async def predict_file(
    file: UploadFile = File(...),
    threshold: float = DEFAULT_THRESHOLD,
    model_name: Optional[str] = None
):
    """Upload CSV file and make predictions."""
    try:
        if len(model_manager) == 0:
            raise HTTPException(status_code=400, detail="No models loaded")
        
        # Read CSV
        import pandas as pd
        
        # use file descriptor temporarily, wait handle directly is better
        # Validate and clean
        df_clean = data_handler.prepare_data(file.file)
        if df_clean is None:
            raise HTTPException(status_code=400, detail="Invalid data format")
        
        model_idx = 0
        if model_name:
            if model_name not in model_manager.model_names:
                raise HTTPException(status_code=404, detail="Model not found")
            model_idx = model_manager.model_names.index(model_name)
        
        model = model_manager.get_model(model_idx)
        proba = model.predict_proba(df_clean)[:, 1]
        predictions = (proba >= threshold).astype(int)
        
        results = [{"index": i, "prediction": int(pred), "probability": float(prob)} 
                   for i, (pred, prob) in enumerate(zip(predictions, proba))]
        
        return {
            "predictions": results,
            "model_name": model_manager.get_model_name(model_idx),
            "threshold": threshold,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ab-tests", tags=["A/B Testing"])
async def create_ab_test(request: ABTestCreateRequest):
    """
    Create a new A/B test comparing two model versions.
    
    Automatically assigns traffic and tracks performance metrics.
    """
    try:
        from src.ab_testing import ABTestConfig, AssignmentStrategy
        
        test_id = f"ab_{request.name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        config = ABTestConfig(
            test_id=test_id,
            name=request.name,
            model_a_name=request.model_a_name,
            model_a_version=request.model_a_version,
            model_b_name=request.model_b_name,
            model_b_version=request.model_b_version,
            traffic_split=request.traffic_split,
            assignment_strategy=AssignmentStrategy.RANDOM,
            status="draft"
        )
        
        ab_framework.create_test(config)
        
        return {
            "test_id": test_id,
            "status": "created",
            "config": {
                "name": request.name,
                "model_a": f"{request.model_a_name}:{request.model_a_version}",
                "model_b": f"{request.model_b_name}:{request.model_b_version}",
                "traffic_split": request.traffic_split
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/ab-tests", tags=["A/B Testing"])
async def list_ab_tests():
    """List all A/B tests and their status."""
    return {"tests": list(ab_framework.configs.keys())}

@app.get("/ab-tests/{test_id}", tags=["A/B Testing"])
async def get_ab_test_status(test_id: str):
    """Get detailed status of an A/B test."""
    try:
        return ab_framework.get_test_status(test_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/ab-tests/{test_id}/start", tags=["A/B Testing"])
async def start_ab_test(test_id: str):
    """Start an A/B test."""
    try:
        ab_framework.start_test(test_id)
        return {"test_id": test_id, "status": "running"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/ab-tests/{test_id}/analyze", tags=["A/B Testing"])
async def analyze_ab_test(test_id: str, metric: str = "prediction"):
    """
    Analyze A/B test results with statistical tests.
    
    Returns statistical significance, effect size, and recommendations.
    """
    try:
        return ab_framework.analyze_results(test_id, metric)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/ab-tests/{test_id}/recommend", tags=["A/B Testing"])
async def recommend_winner(test_id: str):
    """Get recommendation for winning variant."""
    winner = ab_framework.recommend_winner(test_id)
    return {
        "test_id": test_id,
        "recommended_winner": winner,
        "can_promote": winner is not None
    }

@app.post("/drift/detect", tags=["Drift Detection"])
async def detect_drift(file: UploadFile = File(...), method: str = "ks_test"):
    """
    Detect data drift in uploaded dataset.
    
    Methods: ks_test, psi, wasserstein
    """
    try:
        from src.config import CLEAN_DATA_PATH
        
        # Load reference data
        reference_data = pd.read_csv(CLEAN_DATA_PATH)
        
        # Load current data
        current_data = pd.read_csv(file.file)
        
        # Initialize detector
        detector = DataDriftDetector(reference_data)
        
        # Detect drift
        drift_results = detector.detect_drift(current_data, method=method)
        summary = detector.get_drift_summary(drift_results)
        
        return {
            "drift_detected": summary["drift_detected"],
            "summary": summary,
            "details": {name: {
                "drift_detected": r.drift_detected,
                "method": r.method,
                "drift_score": r.drift_score,
                "p_value": r.p_value
            } for name, r in drift_results.items()},
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/drift/status", tags=["Drift Detection"])
async def drift_status():
    """Get current drift monitoring status."""
    return {
        "monitoring_active": False,
        "message": "Drift monitoring requires initialization with reference data"
    }

@app.post("/retraining/trigger", tags=["Automated Retraining"])
async def trigger_retraining(background_tasks: BackgroundTasks):
    """
    Trigger manual model retraining.
    
    Runs in background with hyperparameter tuning.
    """
    global retraining_pipeline
    
    if retraining_pipeline is None:
        from src.config import CLEAN_DATA_PATH
        
        config = RetrainingConfig(
            trigger_on_drift=False,
            enable_ab_testing=True
        )
        retraining_pipeline = RetrainingPipeline(config)
        
        # Initialize with reference data
        reference_data = pd.read_csv(CLEAN_DATA_PATH)
        retraining_pipeline.initialize_drift_detector(reference_data)
    
    def run_retraining():
        result = retraining_pipeline.run_retraining()
        logger.info(f"Retraining completed: {result}")
    
    background_tasks.add_task(run_retraining)
    
    return {
        "status": "started",
        "message": "Retraining triggered in background",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/retraining/status", tags=["Automated Retraining"])
async def retraining_status():
    """Get status of automated retraining pipeline."""
    if retraining_pipeline is None:
        return {"status": "not_initialized"}
    
    history = retraining_pipeline.get_retraining_history()
    latest = retraining_pipeline.get_latest_model_info()
    
    return {
        "status": "initialized",
        "total_retrainings": len(history),
        "latest_model": latest
    }

@app.get("/retraining/history", tags=["Automated Retraining"])
async def retraining_history():
    """Get full retraining history."""
    if retraining_pipeline is None:
        return {"history": []}
    
    return {"history": retraining_pipeline.get_retraining_history()}


@app.get("/models/registry", tags=["Model Management"])
async def list_registered_models():
    """List all models in registry."""
    return {"models": registry.list_models()}

@app.get("/models/registry/{model_name}", tags=["Model Management"])
async def get_model_versions(model_name: str):
    """Get all versions of a registered model."""
    versions = registry.list_versions(model_name)
    return {"model": model_name, "versions": versions}

@app.post("/models/promote", tags=["Model Management"])
async def promote_model(model_name: str, version: str):
    """Promote a model version to production."""
    try:
        registry.promote_to_production(model_name, version)
        return {
            "model": model_name,
            "version": version,
            "status": "promoted_to_production"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
