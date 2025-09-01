# Model Serving Pipeline

This pipeline stage handles model serving and inference using ECS Fargate and FastAPI.

## Overview

The Model Serving stage provides a REST API for making predictions using trained Hybrid Causal Inference ML models. It integrates with the ML workflow state machine and can serve multiple model types simultaneously. Currently supports:

- **Hybrid Causal Inference Models**: Combines econml causal models with PyTorch neural networks
- **Market Regime Classification**: Attention-based regime identification
- **Uncertainty Estimation**: Neural network uncertainty quantification
- **Causal Effect Estimation**: Double machine learning causal inference

## Architecture

- **Container**: Python 3.10 with FastAPI
- **Infrastructure**: ECS Fargate (no load balancer) - Persistent service using existing EcsFargateServiceConstruct
- **Integration**: Step Functions state machine for initialization, persistent service for inference
- **Storage**: S3 for models, DynamoDB for model registry

## Features

- **Hybrid Model Support**: Loads and serves econml + PyTorch hybrid models
- **Model Caching**: Models stay loaded in memory for fast inference
- **Feature Validation**: Automatic feature preparation and scaling
- **REST API endpoints**: Comprehensive inference and model management
- **Integration**: Works with model registry and S3 artifacts
- **Health monitoring**: Built-in health checks and logging
- **Step Functions**: Execution context awareness

## API Endpoints

- `GET /health` - Health check
- `GET /models` - List available models
- `GET /models/{model_id}/info` - Get detailed model information and feature requirements
- `POST /predict/{model_id}` - Make prediction using specified model

## Environment Variables

- `GOLD_BUCKET` - S3 bucket for processed data
- `ARTIFACTS_BUCKET` - S3 bucket for trained models
- `MODEL_REGISTRY_TABLE` - DynamoDB table for model metadata
- `EXECUTION_MODE` - Execution context (step-functions/standalone)
- `PIPELINE_EXECUTION_ID` - Step Functions execution ID
- `EXECUTION_START_TIME` - Step Functions execution start time

## Usage

The pipeline operates in two modes:

**Step Functions Mode (Initialization):**
1. Container starts as part of the ML workflow
2. Models are loaded from the model registry
3. Service is initialized and becomes ready for inference

**Persistent Service Mode:**
1. ECS Fargate service runs continuously
2. API endpoints are always available for inference requests
3. Models remain loaded in memory for fast inference
4. Results are logged and can be stored back to S3

## Prediction Format

### Input Data
The API expects input data with the same features used during training. Features should match the `feature_columns` from the model.

Example input:
```json
{
  "fred_GDP_lag_30d": 100.5,
  "fred_CPIAUCSL_lag_30d": 105.2,
  "fred_FEDFUNDS_lag_30d": 2.5,
  "yahoo_^GSPC_Close": 4500.0
}
```

### Output Format
For hybrid causal models, the API returns:
```json
{
  "causal_effects": [0.15, 0.23, 0.08],
  "model_type": "hybrid_causal_model",
  "n_samples": 1,
  "n_features": 20,
  "regime_probabilities": [[0.1, 0.7, 0.2]],
  "dominant_regime": [1],
  "uncertainty": [0.05]
}
```

## Development

To run locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

The API will be available at `http://localhost:8080`

## Integration

This stage integrates with:
- Data Lake Stack (S3 buckets)
- Model Registry (DynamoDB)
- ML Workflow State Machine (Step Functions)
- ECS Fargate infrastructure
