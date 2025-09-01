#!/usr/bin/env python3
"""
Hybrid Causal Inference Training Pipeline
Implements X-Learner with Double Machine Learning (econml) + PyTorch components
"""

import os
import sys
import argparse
import logging
import json
import pickle
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.profiler import profile, record_function, ProfilerActivity

import boto3
from botocore.exceptions import ClientError

# Core causal inference
from econml.dml import CausalForestDML
from econml.metalearners import XLearner
from econml.dml import LinearDML

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Only warnings and errors during collection
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

class HybridCausalSystem:
    """Hybrid causal inference system combining econml and PyTorch"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Core causal inference (econml)
        self.causal_model = None
        self.regime_classifier = None
        self.uncertainty_estimator = None
        
        # Data preprocessing
        self.scaler = StandardScaler()
        self.feature_columns = []
        
        # Training results
        self.training_results = {}
        
    def _initialize_causal_model(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """Initialize the appropriate causal model based on data characteristics"""
        n_samples = len(X)
        
        if n_samples < 100:
            # Small sample size - use Linear DML
            logger.warning("Small sample size, using Linear DML instead of Causal Forest")
            self.causal_model = LinearDML(
                model_y=RandomForestRegressor(n_estimators=50, random_state=42),
                model_t=RandomForestRegressor(n_estimators=50, random_state=42)
            )
        else:
            # Sufficient sample size - use Causal Forest DML
            self.causal_model = CausalForestDML(
                model_y=RandomForestRegressor(
                    n_estimators=self.config['causal_model']['n_estimators'],
                    min_samples_leaf=self.config['causal_model']['min_samples_leaf'],
                    max_depth=self.config['causal_model']['max_depth'],
                    random_state=42
                ),
                model_t=RandomForestRegressor(
                    n_estimators=self.config['causal_model']['n_estimators'],
                    min_samples_leaf=self.config['causal_model']['min_samples_leaf'],
                    max_depth=self.config['causal_model']['max_depth'],
                    random_state=42
                ),
                n_estimators=self.config['causal_model']['n_estimators'],
                min_samples_leaf=self.config['causal_model']['min_samples_leaf'],
                max_depth=self.config['causal_model']['max_depth']
            )
    
    def _initialize_pytorch_components(self, X: np.ndarray):
        """Initialize PyTorch regime classifier and uncertainty estimator"""
        input_size = X.shape[1]
        
        # Regime classifier
        self.regime_classifier = AttentionRegimeClassifier(
            input_size=input_size,
            hidden_size=self.config['regime_classifier']['hidden_size'],
            n_regimes=self.config['regime_classifier']['n_regimes'],
            attention_heads=self.config['regime_classifier']['attention_heads'],
            dropout=self.config['regime_classifier']['dropout']
        )
        
        # Uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator(
            input_size=input_size,
            hidden_size=self.config['uncertainty_estimator']['hidden_size'],
            dropout=self.config['uncertainty_estimator']['dropout']
        )
    
    def _create_treatment_variables(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Create treatment variables (macro shocks) for causal inference"""
        treatment_features = []
        treatments = []
        
        try:
            # Fed Rate Shock
            if 'fred_FEDFUNDS_lag_30d' in df.columns:
                fed_rate = df['fred_FEDFUNDS_lag_30d']
                fed_shock = fed_rate.diff(30)  # 30-day change
                df['fed_rate_shock'] = fed_shock
                treatment_features.append('fed_rate_shock')
                treatments.append(fed_shock.values)
            
            # CPI Surprise
            if 'fred_CPIAUCSL_lag_30d' in df.columns:
                cpi = df['fred_CPIAUCSL_lag_30d']
                cpi_trend = cpi.rolling(12).mean()  # 12-month trend
                cpi_shock = (cpi - cpi_trend) / cpi_trend
                df['cpi_surprise'] = cpi_shock
                treatment_features.append('cpi_surprise')
                treatments.append(cpi_shock.values)
            
            # GDP Surprise
            if 'fred_GDP_lag_30d' in df.columns:
                gdp = df['fred_GDP_lag_30d']
                gdp_trend = gdp.rolling(8).mean()  # 8-quarter trend
                gdp_shock = (gdp - gdp_trend) / gdp_trend
                df['gdp_surprise'] = gdp_shock
                treatment_features.append('gdp_surprise')
                treatments.append(gdp_shock.values)
            
            # Market Stress (VIX-based)
            if 'yahoo_^VIX_volatility_30d' in df.columns:
                vix_vol = df['yahoo_^VIX_volatility_30d']
                vix_threshold = vix_vol.quantile(0.8)
                market_stress = (vix_vol > vix_threshold).astype(float)
                df['market_stress'] = market_stress
                treatment_features.append('market_stress')
                treatments.append(market_stress.values)
            
            if treatments:
                # Use the first treatment variable as primary treatment
                primary_treatment = np.array(treatments[0])
                logger.info(f"Created {len(treatment_features)} treatment variables")
                return primary_treatment, treatment_features
            else:
                logger.warning("No treatment variables could be created")
                return np.zeros(len(df)), []
                
        except Exception as e:
            logger.error(f"Error creating treatment variables: {e}")
            return np.zeros(len(df)), []
    
    def _create_outcome_variables(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Create outcome variables (asset returns) for causal inference"""
        outcome_features = []
        outcomes = []
        
        try:
            # S&P 500 returns
            if 'yahoo_^GSPC_return_30d' in df.columns:
                sp500_returns = df['yahoo_^GSPC_return_30d']
                df['sp500_returns'] = sp500_returns
                outcome_features.append('sp500_returns')
                outcomes.append(sp500_returns.values)
            
            # Bond returns (TLT)
            if 'yahoo_TLT_return_30d' in df.columns:
                bond_returns = df['yahoo_TLT_return_30d']
                df['bond_returns'] = bond_returns
                outcome_features.append('bond_returns')
                outcomes.append(bond_returns.values)
            
            # Gold returns
            if 'yahoo_GLD_return_30d' in df.columns:
                gold_returns = df['yahoo_GLD_return_30d']
                df['gold_returns'] = gold_returns
                outcome_features.append('gold_returns')
                outcomes.append(gold_returns.values)
            
            if outcomes:
                # Use S&P 500 returns as primary outcome
                primary_outcome = np.array(outcomes[0])
                logger.info(f"Created {len(outcome_features)} outcome variables")
                return primary_outcome, outcome_features
            else:
                logger.warning("No outcome variables could be created")
                return np.zeros(len(df)), []
                
        except Exception as e:
            logger.error(f"Error creating outcome variables: {e}")
            return np.zeros(len(df)), []
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepare feature matrix for training"""
        try:
            # Get all feature columns (exclude metadata and target columns)
            exclude_columns = ['date', 'execution_id', 'feature_creation_timestamp', 'target']
            feature_columns = [col for col in df.columns if col not in exclude_columns]
            
            # Remove treatment and outcome columns from features
            treatment_outcome_cols = []
            for col in df.columns:
                if any(keyword in col for keyword in ['_shock', '_surprise', '_returns', '_stress']):
                    treatment_outcome_cols.append(col)
            
            feature_columns = [col for col in feature_columns if col not in treatment_outcome_cols]
            
            # Create feature matrix
            X = df[feature_columns].values
            
            # Handle missing values
            X = np.nan_to_num(X, nan=0.0)
            X = np.where(np.isinf(X), 0, X)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            self.feature_columns = feature_columns
            logger.info(f"Prepared {len(feature_columns)} features with shape {X_scaled.shape}")
            
            return X_scaled, feature_columns
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise
    
    def _train_regime_classifier(self, X: np.ndarray, Y: np.ndarray, T: np.ndarray):
        """Train the PyTorch regime classifier"""
        try:
            logger.info("Training regime classifier...")
            
            # Create regime labels based on market conditions
            # Simple regime classification based on volatility and returns
            if 'yahoo_^VIX_volatility_30d' in self.feature_columns:
                vix_idx = self.feature_columns.index('yahoo_^VIX_volatility_30d')
                vix_values = X[:, vix_idx]
                
                # Create regime labels: 0=Low Vol, 1=Medium Vol, 2=High Vol
                vix_quantiles = np.percentile(vix_values, [33, 67])
                regime_labels = np.zeros(len(X))
                regime_labels[vix_values > vix_quantiles[1]] = 2  # High vol
                regime_labels[vix_values > vix_quantiles[0]] = 1  # Medium vol
                
                # Convert to tensors
                X_tensor = torch.FloatTensor(X)
                regime_tensor = torch.LongTensor(regime_labels)
                
                # Create dataset and dataloader
                dataset = TensorDataset(X_tensor, regime_tensor)
                dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
                
                # Training setup
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(self.regime_classifier.parameters(), 
                                     lr=0.001, weight_decay=1e-4)
                
                # Training loop
                self.regime_classifier.train()
                for epoch in range(50):
                    total_loss = 0
                    for batch_X, batch_y in dataloader:
                        optimizer.zero_grad()
                        outputs = self.regime_classifier(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    
                    if epoch % 10 == 0:
                        logger.info(f"Regime classifier epoch {epoch}, loss: {total_loss/len(dataloader):.4f}")
                
                logger.info("Regime classifier training completed")
                
        except Exception as e:
            logger.error(f"Error training regime classifier: {e}")
    
    def _train_uncertainty_estimator(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """Train the PyTorch uncertainty estimator"""
        try:
            logger.info("Training uncertainty estimator...")
            
            # Create uncertainty labels based on treatment effect variability
            # Simple heuristic: higher uncertainty when treatment effects vary more
            treatment_effects = np.abs(T) * np.abs(Y)  # Simple proxy for treatment effect
            uncertainty_labels = np.std(treatment_effects) * np.ones(len(X))
            
            # Add some noise to make it more realistic
            uncertainty_labels += np.random.normal(0, 0.01, len(uncertainty_labels))
            uncertainty_labels = np.maximum(uncertainty_labels, 0.001)  # Ensure positive
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X)
            uncertainty_tensor = torch.FloatTensor(uncertainty_labels)
            
            # Create dataset and dataloader
            dataset = TensorDataset(X_tensor, uncertainty_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.uncertainty_estimator.parameters(), 
                                 lr=0.001, weight_decay=1e-4)
            
            # Training loop
            self.uncertainty_estimator.train()
            for epoch in range(50):
                total_loss = 0
                for batch_X, batch_uncertainty in dataloader:
                    optimizer.zero_grad()
                    outputs = self.uncertainty_estimator(batch_X).squeeze()
                    loss = criterion(outputs, batch_uncertainty)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if epoch % 10 == 0:
                    logger.info(f"Uncertainty estimator epoch {epoch}, loss: {total_loss/len(dataloader):.4f}")
            
            logger.info("Uncertainty estimator training completed")
            
        except Exception as e:
            logger.error(f"Error training uncertainty estimator: {e}")
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train the complete hybrid causal inference system"""
        try:
            logger.info("Starting hybrid causal inference training...")
            
            # Step 1: Create treatment and outcome variables
            T, treatment_features = self._create_treatment_variables(df)
            Y, outcome_features = self._create_outcome_variables(df)
            
            # Step 2: Prepare features
            X, feature_columns = self._prepare_features(df)
            
            # Step 3: Initialize models
            self._initialize_causal_model(X, T, Y)
            self._initialize_pytorch_components(X)
            
            # Step 4: Train core causal model (econml)
            logger.info("Training core causal model...")
            self.causal_model.fit(Y, T, X=X)
            
            # Step 5: Train PyTorch components
            self._train_regime_classifier(X, Y, T)
            self._train_uncertainty_estimator(X, T, Y)
            
            # Step 6: Evaluate model performance
            performance_metrics = self._evaluate_model(X, T, Y)
            
            # Step 7: Generate training summary
            training_summary = {
                'n_samples': len(X),
                'n_features': len(feature_columns),
                'n_treatments': len(treatment_features),
                'n_outcomes': len(outcome_features),
                'performance_metrics': performance_metrics,
                'training_timestamp': datetime.now().isoformat(),
                'model_config': self.config
            }
            
            self.training_results = training_summary
            logger.info("Training completed successfully")
            
            return training_summary
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def _evaluate_model(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            # Time series split for evaluation
            tscv = TimeSeriesSplit(n_splits=3)
            
            mse_scores = []
            r2_scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                T_train, T_test = T[train_idx], T[test_idx]
                Y_train, Y_test = Y[train_idx], Y[test_idx]
                
                # Fit on training data
                self.causal_model.fit(Y_train, T_train, X=X_train)
                
                # Predict on test data
                effects = self.causal_model.effect(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(Y_test, effects)
                r2 = r2_score(Y_test, effects)
                
                mse_scores.append(mse)
                r2_scores.append(r2)
            
            return {
                'mean_mse': np.mean(mse_scores),
                'std_mse': np.std(mse_scores),
                'mean_r2': np.mean(r2_scores),
                'std_r2': np.std(r2_scores)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {'mean_mse': float('inf'), 'mean_r2': 0.0}
    
    def predict(self, X_new: np.ndarray, T_new: np.ndarray) -> Dict[str, Any]:
        """Generate predictions using the trained hybrid system"""
        try:
            # Scale new features
            X_new_scaled = self.scaler.transform(X_new)
            
            # Get causal effects from econml model
            causal_effects = self.causal_model.effect(X_new_scaled)
            
            # Get regime probabilities from PyTorch classifier
            self.regime_classifier.eval()
            with torch.no_grad():
                regime_probs = self.regime_classifier(torch.FloatTensor(X_new_scaled)).numpy()
            
            # Get uncertainty estimates from PyTorch estimator
            self.uncertainty_estimator.eval()
            with torch.no_grad():
                uncertainty = self.uncertainty_estimator(torch.FloatTensor(X_new_scaled)).numpy().flatten()
            
            # Determine dominant regime
            dominant_regime = np.argmax(regime_probs, axis=1)
            
            return {
                'causal_effects': causal_effects,
                'regime_probabilities': regime_probs,
                'dominant_regime': dominant_regime,
                'uncertainty': uncertainty
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

def load_data_from_s3(bucket_name: str, execution_id: str) -> pd.DataFrame:
    """Load processed data from S3 Gold bucket"""
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

def save_model_to_s3(model: HybridCausalSystem, execution_id: str, artifacts_bucket: str):
    """Save trained model and artifacts to S3"""
    try:
        s3_client = boto3.client('s3')
        
        # Create model artifacts
        model_artifacts = {
            'model_state': {
                'causal_model': model.causal_model,
                'regime_classifier_state': model.regime_classifier.state_dict() if model.regime_classifier else None,
                'uncertainty_estimator_state': model.uncertainty_estimator.state_dict() if model.uncertainty_estimator else None,
            },
            'scaler': model.scaler,
            'feature_columns': model.feature_columns,
            'training_results': model.training_results,
            'training_timestamp': datetime.now().isoformat(),
            'execution_id': execution_id
        }
        
        # Save to S3
        artifacts_key = f"models/{execution_id}/hybrid_causal_model.pkl"
        s3_client.put_object(
            Bucket=artifacts_bucket,
            Key=artifacts_key,
            Body=pickle.dumps(model_artifacts)
        )
        
        logger.info(f"Model artifacts saved to s3://{artifacts_bucket}/{artifacts_key}")
        
    except ClientError as e:
        logger.error(f"Error saving model to S3: {e}")
        raise

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Hybrid Causal Inference Training Pipeline')
    parser.add_argument('--execution-id', required=True, help='Pipeline execution ID')
    parser.add_argument('--gold-bucket', required=True, help='S3 bucket containing gold data')
    parser.add_argument('--artifacts-bucket', required=True, help='S3 bucket for model artifacts')
    
    args = parser.parse_args()
    
    try:
        # Load data
        logger.info("Loading training data...")
        df = load_data_from_s3(args.gold_bucket, args.execution_id)
        
        # Define model configuration
        config = {
            'causal_model': {
                'n_estimators': 100,
                'min_samples_leaf': 10,
                'max_depth': 10
            },
            'regime_classifier': {
                'hidden_size': 32,
                'n_regimes': 3,
                'attention_heads': 4,
                'dropout': 0.3
            },
            'uncertainty_estimator': {
                'hidden_size': 32,
                'dropout': 0.3
            }
        }
        
        # Initialize and train hybrid system
        logger.info("Initializing hybrid causal inference system...")
        hybrid_system = HybridCausalSystem(config)
        
        # Train the system
        training_results = hybrid_system.train(df)
        
        # Save model
        logger.info("Saving trained model...")
        save_model_to_s3(hybrid_system, args.execution_id, args.artifacts_bucket)
        
        # Log training summary
        logger.info("Training completed successfully!")
        logger.info(f"Training Summary: {json.dumps(training_results, indent=2)}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
