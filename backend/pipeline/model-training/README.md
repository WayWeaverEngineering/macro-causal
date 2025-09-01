# Model Training Stage

This stage implements distributed model training and hyperparameter tuning using Ray Train and Ray Tune for the macro causal inference model.

## Overview

The model training stage is the final stage in the ML pipeline and performs the following tasks:

1. **Load processed data** from S3 Gold bucket
2. **Train neural network models** using PyTorch and Ray Train
3. **Perform hyperparameter optimization** using Ray Tune with Optuna
4. **Save model artifacts** to S3 Artifacts bucket
5. **Register models** in DynamoDB model registry

## Architecture

- **EKS Cluster**: Hosts Ray cluster for distributed training
- **ECS Fargate**: Orchestrates Ray training jobs
- **Ray Train**: Distributed training framework
- **Ray Tune**: Hyperparameter optimization
- **PyTorch**: Deep learning framework
- **S3**: Model artifact storage
- **DynamoDB**: Model registry

## Model Architecture

The `MacroCausalModel` is a feedforward neural network with:
- Configurable hidden layers
- Dropout for regularization
- Batch normalization
- ReLU activation functions

## Hyperparameter Search Space

- **Learning rate**: 1e-4 to 1e-2 (log-uniform)
- **Batch size**: 32, 64, 128
- **Hidden dimensions**: [64, 32], [128, 64], [256, 128, 64]
- **Dropout**: 0.1 to 0.3 (uniform)
- **Weight decay**: 1e-5 to 1e-3 (log-uniform)

## Training Process

1. **Data Loading**: Load processed features from S3 Gold bucket
2. **Feature Preparation**: Scale features and handle missing values
3. **Data Splitting**: Chronological train/validation split (80/20)
4. **Model Training**: Distributed training with Ray Train
5. **Hyperparameter Tuning**: Ray Tune with ASHA scheduler
6. **Model Selection**: Best model based on validation loss
7. **Artifact Storage**: Save model state, scaler, and metadata
8. **Registry Update**: Register model in DynamoDB

## Environment Variables

- `EXECUTION_ID`: Pipeline execution identifier
- `GOLD_BUCKET`: S3 bucket containing processed data
- `ARTIFACTS_BUCKET`: S3 bucket for model artifacts
- `MODEL_REGISTRY_TABLE`: DynamoDB table for model registry

## Output

- **Model artifacts**: Saved to `s3://{artifacts-bucket}/models/{execution-id}/`
- **Model registry entry**: Stored in DynamoDB with metrics and metadata
- **Training logs**: Available in CloudWatch Logs

## Dependencies

- Ray 2.8.0
- PyTorch 2.1.0
- pandas 2.1.0
- scikit-learn 1.3.0
- boto3 1.34.0
- optuna 3.4.0
