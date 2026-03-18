"""FastAPI backend for LoanLens with REST endpoints."""
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import json
import logging
from datetime import datetime
import uvicorn

from src.config import REQUIRED_FEATURES, DEFAULT_THRESHOLD
from src.data_handler import DataHandler
from src.models import ModelManager
from src.explainability import ExplainabilityAnalyzer
from src.monitoring import health_check, model_monitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LoanLens API",
    description="REST API for loan default prediction with explainable AI",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model manager
model_manager = ModelManager()
data_handler = DataHandler(REQUIRED_FEATURES, [])


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Feature values for prediction")
    model_name: Optional[str] = Field(None, description="Specific model to use")
    threshold: float = Field(DEFAULT_THRESHOLD, ge=0.0, le=1.0, description="Classification threshold")


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_name: str
    threshold: float
    timestamp: str
    explanation: Optional[Dict[str, Any]] = None


class BatchPredictionRequest(BaseModel):
    data: List[Dict[str, float]]
    model_name: Optional[str] = None
    threshold: float = DEFAULT_THRESHOLD


class BatchPredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]
    model_name: str
    count: int
    timestamp: str


class ModelInfo(BaseModel):
    name: str
    version: str
    metrics: Dict[str, Any]
    loaded_at: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime: str
    system: Dict[str, Any]
    checks: Dict[str, Any]


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load default model on startup."""
    try:
        from src.config import DEFAULT_MODEL_PATH
        if Path(DEFAULT_MODEL_PATH).exists():
            model = joblib.load(DEFAULT_MODEL_PATH)
            model_manager.add_model(model, "default")
            logger.info("Default model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load default model: {e}")


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health():
    """Get system health status."""
    return health_check.get_status()


# Model management endpoints
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


# Prediction endpoints
@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest):
    """Make a single prediction."""
    import time
    
    start_time = time.time()
    
    try:
        if len(model_manager) == 0:
            raise HTTPException(status_code=400, detail="No models loaded")
        
        # Prepare data
        df = pd.DataFrame([request.features])
        
        # Validate features
        is_valid, missing = data_handler.validate_features(df)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
        
        # Select model
        model_idx = 0
        if request.model_name:
            if request.model_name not in model_manager.model_names:
                raise HTTPException(status_code=404, detail="Model not found")
            model_idx = model_manager.model_names.index(request.model_name)
        
        # Make prediction
        model = model_manager.get_model(model_idx)
        model_name = model_manager.get_model_name(model_idx)
        
        proba = model.predict_proba(df)[0, 1]
        prediction = int(proba >= request.threshold)
        
        # Get SHAP explanation
        explanation = None
        try:
            analyzer = ExplainabilityAnalyzer(model)
            analyzer.compute_shap_values(df)
            exp = analyzer.get_explanation(0, df)
            
            explanation = {
                "base_value": float(exp.base_values[0]),
                "features": [
                    {
                        "feature": name,
                        "value": float(val),
                        "contribution": float(contrib)
                    }
                    for name, val, contrib in zip(
                        df.columns,
                        exp.data[0],
                        exp.values
                    )
                ]
            }
        except Exception as e:
            logger.warning(f"Could not generate explanation: {e}")
        
        # Log metrics
        latency = (time.time() - start_time) * 1000
        model_monitor.log_prediction(latency)
        
        return PredictionResponse(
            prediction=prediction,
            probability=float(proba),
            model_name=model_name,
            threshold=request.threshold,
            timestamp=datetime.now().isoformat(),
            explanation=explanation
        )
        
    except HTTPException:
        model_monitor.log_prediction(0, error=True)
        raise
    except Exception as e:
        model_monitor.log_prediction(0, error=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions."""
    try:
        if len(model_manager) == 0:
            raise HTTPException(status_code=400, detail="No models loaded")
        
        # Prepare data
        df = pd.DataFrame(request.data)
        
        # Validate features
        is_valid, missing = data_handler.validate_features(df)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
        
        # Select model
        model_idx = 0
        if request.model_name:
            if request.model_name not in model_manager.model_names:
                raise HTTPException(status_code=404, detail="Model not found")
            model_idx = model_manager.model_names.index(request.model_name)
        
        # Make predictions
        model = model_manager.get_model(model_idx)
        model_name = model_manager.get_model_name(model_idx)
        
        proba = model.predict_proba(df)[:, 1]
        predictions = (proba >= request.threshold).astype(int).tolist()
        
        return BatchPredictionResponse(
            predictions=predictions,
            probabilities=proba.tolist(),
            model_name=model_name,
            count=len(predictions),
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        df = pd.read_csv(file.file)
        
        # Validate and clean
        df_clean = data_handler.prepare_data(file.file)
        if df_clean is None:
            raise HTTPException(status_code=400, detail="Invalid data format")
        
        # Select model
        model_idx = 0
        if model_name:
            if model_name not in model_manager.model_names:
                raise HTTPException(status_code=404, detail="Model not found")
            model_idx = model_manager.model_names.index(model_name)
        
        # Predict
        model = model_manager.get_model(model_idx)
        proba = model.predict_proba(df_clean)[:, 1]
        predictions = (proba >= threshold).astype(int)
        
        # Prepare response
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, proba)):
            results.append({
                "index": i,
                "prediction": int(pred),
                "probability": float(prob)
            })
        
        return {
            "predictions": results,
            "model_name": model_manager.get_model_name(model_idx),
            "threshold": threshold,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# SHAP explainability endpoint
@app.post("/explain", tags=["Explainability"])
async def explain(request: PredictionRequest):
    """Get SHAP explanation for a prediction."""
    try:
        if len(model_manager) == 0:
            raise HTTPException(status_code=400, detail="No models loaded")
        
        # Prepare data
        df = pd.DataFrame([request.features])
        
        # Select model
        model_idx = 0
        if request.model_name:
            if request.model_name not in model_manager.model_names:
                raise HTTPException(status_code=404, detail="Model not found")
            model_idx = model_manager.model_names.index(request.model_name)
        
        model = model_manager.get_model(model_idx)
        
        # Generate explanation
        analyzer = ExplainabilityAnalyzer(model)
        analyzer.compute_shap_values(df)
        
        # Feature importance
        importance = analyzer.get_feature_importance(df)
        
        return {
            "model_name": model_manager.get_model_name(model_idx),
            "feature_importance": importance.to_dict('records'),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Monitoring endpoints
@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get model performance metrics."""
    return model_monitor.get_metrics()


@app.get("/features", tags=["API Info"])
async def get_features():
    """Get list of required features."""
    return {
        "required_features": REQUIRED_FEATURES,
        "count": len(REQUIRED_FEATURES)
    }


# Root endpoint
@app.get("/")
async def root():
    """API root with basic info."""
    return {
        "name": "LoanLens API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "models_loaded": len(model_manager)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
