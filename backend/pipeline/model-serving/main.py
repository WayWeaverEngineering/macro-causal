#!/usr/bin/env python3
"""
Model Serving Application
Serves Hybrid Causal Inference ML models via FastAPI endpoints
"""

import os
import json
import logging
import joblib
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

import boto3
import structlog
import numpy as np
import pandas as pd
import torch
import sklearn
import econml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Initialize FastAPI app
app = FastAPI(
    title="Model Serving API",
    description="ML Model Inference Service",
    version="1.0.0"
)

# AWS clients
dynamodb = boto3.client('dynamodb')
s3 = boto3.client('s3')

# Environment variables
GOLD_BUCKET = os.getenv('GOLD_BUCKET')
ARTIFACTS_BUCKET = os.getenv('ARTIFACTS_BUCKET')
MODEL_REGISTRY_TABLE = os.getenv('MODEL_REGISTRY_TABLE')
EXECUTION_MODE = os.getenv('EXECUTION_MODE', 'standalone')
PIPELINE_EXECUTION_ID = os.getenv('PIPELINE_EXECUTION_ID', 'unknown')
EXECUTION_START_TIME = os.getenv('EXECUTION_START_TIME', 'unknown')

# Global model cache
model_cache: Dict[str, Any] = {}

# Model types supported
SUPPORTED_MODEL_TYPES = ['hybrid_causal_model']

def _scan_all(**kwargs):
    """Helper function to scan DynamoDB with pagination"""
    items, resp = [], dynamodb.scan(**kwargs)
    items.extend(resp.get("Items", []))
    while "LastEvaluatedKey" in resp:
        resp = dynamodb.scan(ExclusiveStartKey=resp["LastEvaluatedKey"], **kwargs)
        items.extend(resp.get("Items", []))
    return items

def validate_environment() -> bool:
    """Validate that all required environment variables and S3 paths are accessible"""
    try:
        # Check required environment variables
        required_vars = ['GOLD_BUCKET', 'ARTIFACTS_BUCKET', 'MODEL_REGISTRY_TABLE']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error("Missing required environment variables", missing_vars=missing_vars)
            return False
        
        # Validate S3 bucket access
        for bucket_name in [GOLD_BUCKET, ARTIFACTS_BUCKET]:
            try:
                s3.head_bucket(Bucket=bucket_name)
                logger.info("Successfully accessed bucket", bucket=bucket_name)
            except Exception as e:
                logger.error("Failed to access bucket", bucket=bucket_name, error=str(e))
                return False
        
        # Validate DynamoDB table access
        try:
            dynamodb.describe_table(TableName=MODEL_REGISTRY_TABLE)
            logger.info("Successfully accessed DynamoDB table", table=MODEL_REGISTRY_TABLE)
        except Exception as e:
            logger.error("Failed to access DynamoDB table", table=MODEL_REGISTRY_TABLE, error=str(e))
            return False
        
        return True
        
    except Exception as e:
        logger.error("Error validating environment", error=str(e))
        return False

def ensure_model_folders() -> None:
    """Ensure required model folders exist in S3"""
    try:
        # Create models folder in artifacts bucket
        models_folder = f"models/{datetime.now().strftime('%Y%m%d')}/"
        try:
            s3.put_object(
                Bucket=ARTIFACTS_BUCKET,
                Key=f"{models_folder}.keep",
                Body=b'',
                ContentType='text/plain'
            )
            logger.info("Ensured models folder exists", folder=models_folder)
        except Exception as e:
            logger.warning("Could not create models folder", folder=models_folder, error=str(e))
        
        # Create inference results folder in gold bucket
        inference_folder = f"inference_results/{datetime.now().strftime('%Y%m%d')}/"
        try:
            s3.put_object(
                Bucket=GOLD_BUCKET,
                Key=f"{inference_folder}.keep",
                Body=b'',
                ContentType='text/plain'
            )
            logger.info("Ensured inference results folder exists", folder=inference_folder)
        except Exception as e:
            logger.warning("Could not create inference results folder", folder=inference_folder, error=str(e))
            
    except Exception as e:
        logger.warning("Could not ensure model folders", error=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize the model serving application"""
    logger.info("Starting Model Serving Application", 
                execution_mode=EXECUTION_MODE,
                pipeline_execution_id=PIPELINE_EXECUTION_ID)
    
    if EXECUTION_MODE == 'step-functions':
        logger.info("Running in Step Functions execution mode",
                   execution_start_time=EXECUTION_START_TIME)
    elif EXECUTION_MODE == 'standalone':
        logger.info("Running as persistent model serving service")
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed. Exiting.")
        exit(1)

    # Ensure model folders exist
    ensure_model_folders()

    # Initialize models from registry
    await initialize_models()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "execution_mode": EXECUTION_MODE,
        "pipeline_execution_id": PIPELINE_EXECUTION_ID
    }

@app.get("/models")
async def list_models():
    """List available models"""
    try:
        response_items = _scan_all(
            TableName=MODEL_REGISTRY_TABLE,
            ProjectionExpression="model_id, model_name, model_version, status, created_at"
        )
        
        models = []
        for item in response_items:
            models.append({
                "model_id": item.get('model_id', {}).get('S'),
                "model_name": item.get('model_name', {}).get('S'),
                "model_version": item.get('model_version', {}).get('S'),
                "status": item.get('status', {}).get('S'),
                "created_at": item.get('created_at', {}).get('S')
            })
        
        return {"models": models, "count": len(models)}
    
    except Exception as e:
        logger.error("Failed to list models", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve models")

@app.get("/models/{model_id}/info")
async def get_model_info(model_id: str):
    """Get detailed information about a specific model"""
    try:
        # Check if model is loaded
        if model_id not in model_cache:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found or not loaded")
        
        model_info = model_cache[model_id]
        
        # Return model information
        return {
            "model_id": model_id,
            "type": model_info.get('type'),
            "feature_columns": model_info.get('feature_columns', []),
            "n_features": len(model_info.get('feature_columns', [])),
            "loaded_at": model_info.get('loaded_at'),
            "training_results": model_info.get('training_results', {}),
            "versions": model_info.get('versions', {}),
            "status": model_info.get('status')
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get model info", 
                    model_id=model_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve model info")

@app.post("/reload/{model_id}")
async def reload_model(model_id: str, request: Request):
    """Reload a model from S3 (useful during development)"""
    try:
        data = await request.json()
        model_path = data.get('model_path')
        
        if not model_path:
            raise HTTPException(status_code=400, detail="model_path is required")
        
        # Load the model
        await load_model(model_id, model_path)
        
        logger.info("Model reloaded successfully", model_id=model_id, model_path=model_path)
        return {
            "status": "reloaded",
            "model_id": model_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to reload model", 
                    model_id=model_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail="Failed to reload model")

@app.post("/predict/{model_id}")
async def predict(model_id: str, request: Request):
    """Make prediction using specified model"""
    try:
        # Get request data
        data = await request.json()
        
        # Check if model is available
        if model_id not in model_cache:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found or not loaded")
        
        # Get model info
        model_info = model_cache[model_id]
        
        # Make prediction (placeholder - implement actual inference logic)
        prediction = await make_prediction(model_id, data, model_info)
        
        # Log prediction
        n_instances = len(data.get("instances", [])) if isinstance(data, dict) and "instances" in data else 1
        logger.info("Prediction made", 
                   model_id=model_id,
                   n_instances=n_instances,
                   execution_mode=EXECUTION_MODE)
        
        return {
            "model_id": model_id,
            "prediction": prediction,
            "timestamp": datetime.utcnow().isoformat(),
            "execution_mode": EXECUTION_MODE
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction failed", 
                    model_id=model_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail="Prediction failed")

async def initialize_models():
    """Initialize models from the model registry"""
    try:
        logger.info("Initializing models from registry", 
                   table=MODEL_REGISTRY_TABLE)
        
        # Get available models from DynamoDB
        response_items = _scan_all(
            TableName=MODEL_REGISTRY_TABLE,
            FilterExpression="status = :status",
            ExpressionAttributeValues={":status": {"S": "ready"}},
            ProjectionExpression="model_id, model_path"
        )
        
        for item in response_items:
            model_id = item.get('model_id', {}).get('S')
            model_path = item.get('model_path', {}).get('S')
            
            if model_id and model_path:
                await load_model(model_id, model_path)
        
        logger.info("Model initialization complete", 
                   models_loaded=len(model_cache))
    
    except Exception as e:
        logger.error("Failed to initialize models", error=str(e))
        # Don't fail startup - models can be loaded on-demand

def create_regime_classifier(input_size: int, hidden_size: int = 64, n_regimes: int = 3,
                             dropout: float = 0.3, n_layers: int = 3):
    """Create RegimeClassifier with MLP architecture matching training"""
    import torch.nn as nn

    class RegimeClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            blocks = []
            in_f = input_size
            for i in range(n_layers - 1):
                blocks += [
                    nn.Linear(in_f, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.SiLU(),
                    nn.Dropout(dropout)
                ]
                in_f = hidden_size
            blocks += [nn.Linear(in_f, n_regimes)]   # logits (no softmax)
            self.net = nn.Sequential(*blocks)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, x):
            if x.ndim == 3:
                x = x.squeeze(1)
            return self.net(x)

    return RegimeClassifier()

def create_uncertainty_estimator(input_size: int):
    """Create UncertaintyEstimator with default architecture"""
    from torch import nn
    
    class UncertaintyEstimator(nn.Module):
        """Neural network for estimating uncertainty in causal effects"""
        
        def __init__(self, input_size: int = 20, hidden_size: int = 32, dropout: float = 0.3):
            super().__init__()
            self.uncertainty_net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1),
                nn.Softplus()  # Ensures positive uncertainty
            )
        
        def forward(self, x):
            return self.uncertainty_net(x)
    
    return UncertaintyEstimator(input_size=input_size)

async def load_model(model_id: str, model_path: str):
    """Load a hybrid causal inference model into memory"""
    try:
        logger.info("Loading hybrid causal model", model_id=model_id, model_path=model_path)
        
        # Download model from S3 if needed
        if model_path.startswith('s3://'):
            bucket, key = model_path.replace('s3://', '').split('/', 1)
            local_path = f"/tmp/{model_id}-{uuid.uuid4().hex}.pkl"
            
            s3.download_file(bucket, key, local_path)
            model_path = local_path
        
        # Load the pickled model artifacts
        with open(model_path, 'rb') as f:
            model_artifacts = joblib.load(f)
        
        # Extract components
        causal_model = model_artifacts['model_state']['causal_model']
        regime_classifier_state = model_artifacts['model_state']['regime_classifier_state']
        uncertainty_estimator_state = model_artifacts['model_state']['uncertainty_estimator_state']
        scaler = model_artifacts['scaler']
        feature_columns = model_artifacts['feature_columns']
        training_results = model_artifacts.get('training_results', {})
        
        # Reconstruct PyTorch models if states exist
        regime_classifier = None
        uncertainty_estimator = None
        
        if regime_classifier_state:
            # Reconstruct regime classifier with training hyperparams
            input_size = len(feature_columns)
            reg_cfg = training_results.get('pytorch_spec', {}).get('regime_config', {})
            regime_classifier = create_regime_classifier(
                input_size=input_size,
                hidden_size=reg_cfg.get('hidden_size', 64),
                n_regimes=reg_cfg.get('n_regimes', 3),
                dropout=reg_cfg.get('dropout', 0.3),
                n_layers=reg_cfg.get('n_layers', 3),
            )
            regime_classifier.load_state_dict(regime_classifier_state)
            regime_classifier.eval()
        
        if uncertainty_estimator_state:
            # Reconstruct uncertainty estimator
            input_size = len(feature_columns)
            uncertainty_estimator = create_uncertainty_estimator(input_size)
            uncertainty_estimator.load_state_dict(uncertainty_estimator_state)
            uncertainty_estimator.eval()
        
        # Check for version mismatches
        vers = model_artifacts.get("versions", {})
        mismatch = {k: (vers.get(k), cur) for k, cur in {
            "sklearn": sklearn.__version__,
            "econml": econml.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__
        }.items() if vers.get(k) and vers.get(k) != cur}
        if mismatch:
            logger.warning("Library version mismatch between train and serve", mismatch=mismatch)
        
        # Store in cache
        model_cache[model_id] = {
            "type": "hybrid_causal_model",
            "causal_model": causal_model,
            "regime_classifier": regime_classifier,
            "uncertainty_estimator": uncertainty_estimator,
            "scaler": scaler,
            "feature_columns": feature_columns,
            "training_results": training_results,
            "versions": model_artifacts.get("versions", {}),
            "loaded_at": datetime.utcnow().isoformat(),
            "status": "loaded"
        }
        
        logger.info("Hybrid causal model loaded successfully", 
                   model_id=model_id,
                   n_features=len(feature_columns))
        
        # Log model schema for client reference
        logger.info("Model schema", model_id=model_id, n_features=len(feature_columns),
                    first_features=feature_columns[:10])
    
    except Exception as e:
        logger.error("Failed to load hybrid causal model", 
                    model_id=model_id,
                    error=str(e))
        raise

async def make_prediction(model_id: str, data: Dict[str, Any], model_info: Dict[str, Any]):
    """Make prediction using the hybrid causal inference model"""
    try:
        model_type = model_info.get('type')
        
        if model_type == 'hybrid_causal_model':
            return await make_hybrid_causal_prediction(data, model_info)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    except Exception as e:
        logger.error("Prediction failed", 
                    model_id=model_id,
                    error=str(e))
        raise

async def make_hybrid_causal_prediction(data: Dict[str, Any], model_info: Dict[str, Any]):
    """Make prediction using hybrid causal inference model"""
    try:
        # Extract model components
        causal_model = model_info['causal_model']
        regime_classifier = model_info['regime_classifier']
        uncertainty_estimator = model_info['uncertainty_estimator']
        scaler = model_info['scaler']
        feature_columns = model_info['feature_columns']
        
        # Prepare input data
        input_data = prepare_input_data(data, feature_columns)
        
        # Guard against feature drift (order + count)
        if input_data.shape[1] != len(feature_columns):
            raise ValueError(f"Expected {len(feature_columns)} features, got {input_data.shape[1]}")
        
        # Validate scaler compatibility
        if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != len(feature_columns):
            raise ValueError(f"Scaler expects {scaler.n_features_in_} features, got {len(feature_columns)}")
        
        # Scale features
        X_scaled = scaler.transform(input_data)
        
        # Get causal effects from econml model
        causal_effects = causal_model.effect(X_scaled)
        
        # Get regime probabilities from PyTorch classifier
        regime_probs = None
        dominant_regime = None
        if regime_classifier:
            regime_classifier.eval()
            with torch.no_grad():
                X_tensor = torch.from_numpy(X_scaled).float()  # same result, slightly quicker
                regime_logits = regime_classifier(X_tensor)
                regime_probs = torch.softmax(regime_logits, dim=1).cpu().numpy()
                dominant_regime = np.argmax(regime_probs, axis=1)
        
        # Get uncertainty estimates from PyTorch estimator
        uncertainty = None
        if uncertainty_estimator:
            uncertainty_estimator.eval()
            with torch.no_grad():
                X_tensor = torch.from_numpy(X_scaled).float()  # same result, slightly quicker
                uncertainty = uncertainty_estimator(X_tensor).squeeze(-1).cpu().numpy()
        
        # Prepare response
        prediction_result = {
            'causal_effects': causal_effects.tolist() if hasattr(causal_effects, 'tolist') else float(causal_effects),
            'model_type': 'hybrid_causal_model',
            'n_samples': len(input_data),
            'n_features': len(feature_columns)
        }
        
        if regime_probs is not None:
            prediction_result['regime_probabilities'] = regime_probs.tolist()
            prediction_result['dominant_regime'] = dominant_regime.tolist()
        
        if uncertainty is not None:
            prediction_result['uncertainty'] = uncertainty.tolist()
        
        return prediction_result
    
    except Exception as e:
        logger.error("Hybrid causal prediction failed", error=str(e))
        raise

def _coerce(v):
    """Coerce input values to float, treating None/empty as missing"""
    # treat None/"" as missing
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return 0.0
    try:
        return float(v)
    except Exception:
        return 0.0

def prepare_input_data(data: Dict[str, Any], feature_columns: List[str]) -> np.ndarray:
    """Prepare input data for model inference"""
    try:
        # Accept either a dict (single row), {"instances": [dict, dict, ...]}, or raw list
        if isinstance(data, dict) and "instances" in data:
            rows = data["instances"]
        elif isinstance(data, list):
            rows = data
        else:
            rows = [data]
        
        mat = [[_coerce(row.get(col, 0.0)) for col in feature_columns] for row in rows]
        return np.asarray(mat, dtype=np.float32)
    
    except Exception as e:
        logger.error("Failed to prepare input data", error=str(e))
        raise

if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    )
