#!/usr/bin/env python3
"""
Model Serving Application
Serves Hybrid Causal Inference ML models via FastAPI endpoints
"""

import os
import json
import logging
import pickle
from typing import Dict, Any, Optional, List
from datetime import datetime

import boto3
import structlog
import numpy as np
import pandas as pd
import torch
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
        response = dynamodb.scan(
            TableName=MODEL_REGISTRY_TABLE,
            ProjectionExpression="model_id, model_name, model_version, status, created_at"
        )
        
        models = []
        for item in response.get('Items', []):
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
            "status": model_info.get('status')
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get model info", 
                    model_id=model_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve model info")

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
        logger.info("Prediction made", 
                   model_id=model_id,
                   input_size=len(str(data)),
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
        response = dynamodb.scan(
            TableName=MODEL_REGISTRY_TABLE,
            FilterExpression="status = :status",
            ExpressionAttributeValues={":status": {"S": "ready"}}
        )
        
        for item in response.get('Items', []):
            model_id = item.get('model_id', {}).get('S')
            model_path = item.get('model_path', {}).get('S')
            
            if model_id and model_path:
                await load_model(model_id, model_path)
        
        logger.info("Model initialization complete", 
                   models_loaded=len(model_cache))
    
    except Exception as e:
        logger.error("Failed to initialize models", error=str(e))
        # Don't fail startup - models can be loaded on-demand

def create_regime_classifier(input_size: int):
    """Create AttentionRegimeClassifier with default architecture"""
    from torch import nn
    
    class AttentionRegimeClassifier(nn.Module):
        """Self-attention regime classifier for identifying market states"""
        
        def __init__(self, input_size: int = 20, hidden_size: int = 32, n_regimes: int = 3, 
                     attention_heads: int = 4, dropout: float = 0.3):
            super().__init__()
            self.embedding = nn.Linear(input_size, hidden_size)
            self.attention = nn.MultiheadAttention(hidden_size, num_heads=attention_heads, 
                                                 dropout=dropout, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, n_regimes),
                nn.Softmax(dim=1)
            )
        
        def forward(self, x):
            # x shape: (batch_size, seq_len, input_size)
            if len(x.shape) == 2:
                x = x.unsqueeze(1)  # Add sequence dimension
            
            embedded = self.embedding(x)  # (batch_size, seq_len, hidden_size)
            attended, _ = self.attention(embedded, embedded, embedded)
            attended = self.dropout(attended)
            
            # Global average pooling over sequence dimension
            pooled = attended.mean(dim=1)  # (batch_size, hidden_size)
            return self.classifier(pooled)
    
    return AttentionRegimeClassifier(input_size=input_size)

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
            local_path = f"/tmp/{model_id}.pkl"
            
            s3.download_file(bucket, key, local_path)
            model_path = local_path
        
        # Load the pickled model artifacts
        with open(model_path, 'rb') as f:
            model_artifacts = pickle.load(f)
        
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
            # Reconstruct regime classifier
            input_size = len(feature_columns)
            regime_classifier = create_regime_classifier(input_size)
            regime_classifier.load_state_dict(regime_classifier_state)
            regime_classifier.eval()
        
        if uncertainty_estimator_state:
            # Reconstruct uncertainty estimator
            input_size = len(feature_columns)
            uncertainty_estimator = create_uncertainty_estimator(input_size)
            uncertainty_estimator.load_state_dict(uncertainty_estimator_state)
            uncertainty_estimator.eval()
        
        # Store in cache
        model_cache[model_id] = {
            "type": "hybrid_causal_model",
            "causal_model": causal_model,
            "regime_classifier": regime_classifier,
            "uncertainty_estimator": uncertainty_estimator,
            "scaler": scaler,
            "feature_columns": feature_columns,
            "training_results": training_results,
            "loaded_at": datetime.utcnow().isoformat(),
            "status": "loaded"
        }
        
        logger.info("Hybrid causal model loaded successfully", 
                   model_id=model_id,
                   n_features=len(feature_columns))
    
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
                X_tensor = torch.FloatTensor(X_scaled)
                regime_probs = regime_classifier(X_tensor).numpy()
                dominant_regime = np.argmax(regime_probs, axis=1)
        
        # Get uncertainty estimates from PyTorch estimator
        uncertainty = None
        if uncertainty_estimator:
            uncertainty_estimator.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled)
                uncertainty = uncertainty_estimator(X_tensor).numpy().flatten()
        
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

def prepare_input_data(data: Dict[str, Any], feature_columns: List[str]) -> np.ndarray:
    """Prepare input data for model inference"""
    try:
        # Extract features from input data
        features = []
        for col in feature_columns:
            if col in data:
                features.append(data[col])
            else:
                # Use default value for missing features
                features.append(0.0)
        
        # Convert to numpy array
        input_array = np.array(features).reshape(1, -1)
        
        return input_array
    
    except Exception as e:
        logger.error("Failed to prepare input data", error=str(e))
        raise

if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    )
