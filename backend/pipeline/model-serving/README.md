# Model Serving Pipeline

This pipeline stage handles model serving and inference using ECS Fargate and FastAPI.

## Overview

The Model Serving stage provides a REST API for making predictions using trained ML models. It integrates with the ML workflow state machine and can serve multiple model types simultaneously.

## Architecture

- **Container**: Python 3.10 with FastAPI
- **Infrastructure**: ECS Fargate (no load balancer) - Persistent service using existing EcsFargateServiceConstruct
- **Integration**: Step Functions state machine for initialization, persistent service for inference
- **Storage**: S3 for models, DynamoDB for model registry

## Features

- Model loading and caching from S3
- REST API endpoints for inference
- Integration with model registry
- Health monitoring and logging
- Step Functions execution context

## API Endpoints

- `GET /health` - Health check
- `GET /models` - List available models
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
