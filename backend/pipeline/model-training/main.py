#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import json
import pickle
from datetime import datetime
from typing import Dict, Any, Optional

import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MacroCausalModel(nn.Module):
    """Neural network model for macro causal inference."""
    
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, dropout: float = 0.1):
        super(MacroCausalModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def load_data_from_s3(bucket_name: str, execution_id: str) -> pd.DataFrame:
    """Load processed data from S3 Gold bucket."""
    try:
        s3_client = boto3.client('s3')
        
        # Load gold data
        gold_key = f"gold/{execution_id}/processed_data.parquet"
        response = s3_client.get_object(Bucket=bucket_name, Key=gold_key)
        df = pd.read_parquet(response['Body'])
        
        logger.info(f"Loaded data with shape: {df.shape}")
        return df
        
    except ClientError as e:
        logger.error(f"Error loading data from S3: {e}")
        raise

def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare features and targets for training."""
    # Separate features and target
    feature_columns = [col for col in df.columns if col not in ['target', 'date', 'execution_id']]
    target_column = 'target'
    
    X = df[feature_columns].values
    y = df[target_column].values
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, feature_columns

def create_data_loaders(X: np.ndarray, y: np.ndarray, batch_size: int, train_ratio: float = 0.8) -> tuple:
    """Create train and validation data loaders."""
    # Split data chronologically
    split_idx = int(len(X) * train_ratio)
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_model(config: Dict[str, Any]):
    """Ray Train function for model training."""
    # Initialize Ray Train
    ray.init(address="auto")
    
    # Load data
    execution_id = os.environ.get('EXECUTION_ID')
    gold_bucket = os.environ.get('GOLD_BUCKET')
    
    df = load_data_from_s3(gold_bucket, execution_id)
    X, y, scaler, feature_columns = prepare_features(df)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        X, y, 
        batch_size=config['batch_size'],
        train_ratio=config['train_ratio']
    )
    
    # Initialize model
    model = MacroCausalModel(
        input_dim=len(feature_columns),
        hidden_dims=config['hidden_dims'],
        output_dim=1,
        dropout=config['dropout']
    )
    
    # Setup optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['max_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Report metrics to Ray Train
        train.report({
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'epoch': epoch
        })
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = Checkpoint.from_dict({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': scaler,
                'feature_columns': feature_columns,
                'config': config,
                'epoch': epoch,
                'val_loss': avg_val_loss
            })
            train.report({'checkpoint': checkpoint})
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                logger.info(f"Early stopping at epoch {epoch}")
                break

def save_model_to_s3(model_state: Dict, scaler: StandardScaler, feature_columns: list, 
                    config: Dict, metrics: Dict, execution_id: str, artifacts_bucket: str):
    """Save trained model and artifacts to S3."""
    try:
        s3_client = boto3.client('s3')
        
        # Create model artifacts
        model_artifacts = {
            'model_state': model_state,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'config': config,
            'metrics': metrics,
            'training_timestamp': datetime.now().isoformat(),
            'execution_id': execution_id
        }
        
        # Save to S3
        artifacts_key = f"models/{execution_id}/model_artifacts.pkl"
        s3_client.put_object(
            Bucket=artifacts_bucket,
            Key=artifacts_key,
            Body=pickle.dumps(model_artifacts)
        )
        
        logger.info(f"Model artifacts saved to s3://{artifacts_bucket}/{artifacts_key}")
        
    except ClientError as e:
        logger.error(f"Error saving model to S3: {e}")
        raise

def register_model_in_dynamodb(execution_id: str, model_metrics: Dict, model_uri: str, 
                             registry_table: str):
    """Register model in DynamoDB model registry."""
    try:
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table(registry_table)
        
        model_record = {
            'model_id': f"model_{execution_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'execution_id': execution_id,
            'model_uri': model_uri,
            'metrics': model_metrics,
            'status': 'TRAINED',
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0'
        }
        
        table.put_item(Item=model_record)
        logger.info(f"Model registered in DynamoDB: {model_record['model_id']}")
        
    except ClientError as e:
        logger.error(f"Error registering model in DynamoDB: {e}")
        raise

def main():

    # Eearly exit for testing
    return 0
    
    """Main training function with hyperparameter tuning."""
    parser = argparse.ArgumentParser(description='Ray-based model training for macro causal inference')
    parser.add_argument('--execution-id', required=True, help='Pipeline execution ID')
    parser.add_argument('--gold-bucket', required=True, help='S3 bucket containing gold data')
    parser.add_argument('--artifacts-bucket', required=True, help='S3 bucket for model artifacts')
    parser.add_argument('--model-registry-table', required=True, help='DynamoDB table for model registry')
    
    args = parser.parse_args()
    
    # Initialize Ray
    ray.init(address="auto")
    
    # Define hyperparameter search space
    config = {
        'learning_rate': tune.loguniform(1e-4, 1e-2),
        'batch_size': tune.choice([32, 64, 128]),
        'hidden_dims': tune.choice([[64, 32], [128, 64], [256, 128, 64]]),
        'dropout': tune.uniform(0.1, 0.3),
        'weight_decay': tune.loguniform(1e-5, 1e-3),
        'max_epochs': 100,
        'patience': 10,
        'train_ratio': 0.8
    }
    
    # Define scheduler and search algorithm
    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='val_loss',
        mode='min',
        max_t=100,
        grace_period=10,
        reduction_factor=2
    )
    
    search_alg = OptunaSearch(metric='val_loss', mode='min')
    
    # Run hyperparameter tuning
    analysis = tune.run(
        train_model,
        config=config,
        num_samples=20,  # Number of trials
        scheduler=scheduler,
        search_alg=search_alg,
        resources_per_trial={'cpu': 2, 'gpu': 0},  # Adjust based on cluster
        local_dir='/tmp/ray_results'
    )
    
    # Get best trial
    best_trial = analysis.get_best_trial('val_loss', 'min')
    best_config = best_trial.config
    best_metrics = best_trial.last_result
    
    logger.info(f"Best trial config: {best_config}")
    logger.info(f"Best trial final validation loss: {best_metrics['val_loss']}")
    
    # Save best model
    best_checkpoint = best_trial.checkpoint
    if best_checkpoint:
        checkpoint_data = best_checkpoint.to_dict()
        
        # Save to S3
        model_uri = f"s3://{args.artifacts_bucket}/models/{args.execution_id}/"
        save_model_to_s3(
            checkpoint_data['model_state_dict'],
            checkpoint_data['scaler'],
            checkpoint_data['feature_columns'],
            best_config,
            best_metrics,
            args.execution_id,
            args.artifacts_bucket
        )
        
        # Register in DynamoDB
        register_model_in_dynamodb(
            args.execution_id,
            best_metrics,
            model_uri,
            args.model_registry_table
        )
    
    logger.info("Model training completed successfully")
    ray.shutdown()

if __name__ == "__main__":
    main()
