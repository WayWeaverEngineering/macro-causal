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
import io
import joblib
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
import sklearn
import econml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import boto3
from botocore.exceptions import ClientError

# Core causal inference
from econml.dml import CausalForestDML
from econml.dml import LinearDML

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Changed to INFO so training messages show up
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    # Set cuDNN determinism for bit-wise reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def validate_training_data_paths(gold_bucket: str, execution_id: str) -> bool:
    """Validate that training data exists in the expected S3 paths"""
    try:
        s3_client = boto3.client('s3')
        
        # Check if gold bucket is accessible
        try:
            s3_client.head_bucket(Bucket=gold_bucket)
            logger.info(f"Successfully accessed gold bucket: {gold_bucket}")
        except Exception as e:
            logger.error(f"Failed to access gold bucket {gold_bucket}: {e}")
            return False
        
        # Check if training data exists
        expected_data_key = f"gold/{execution_id}/processed_data.parquet"
        try:
            s3_client.head_object(Bucket=gold_bucket, Key=expected_data_key)
            logger.info(f"Training data found: {expected_data_key}")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.error(f"Training data not found: {expected_data_key}")
                logger.error("This may indicate that the data processing stage failed or data is missing")
                return False
            else:
                logger.error(f"Error checking training data: {e}")
                return False
                
    except Exception as e:
        logger.error(f"Error validating training data paths: {e}")
        return False

def validate_artifacts_bucket(artifacts_bucket: str) -> bool:
    """Validate that artifacts bucket is accessible and has required structure"""
    try:
        s3_client = boto3.client('s3')
        
        # Check if artifacts bucket is accessible
        try:
            s3_client.head_bucket(Bucket=artifacts_bucket)
            logger.info(f"Successfully accessed artifacts bucket: {artifacts_bucket}")
        except Exception as e:
            logger.error(f"Failed to access artifacts bucket {artifacts_bucket}: {e}")
            return False
        
        # Create models folder if it doesn't exist
        try:
            models_folder = f"models/{datetime.now().strftime('%Y%m%d')}/"
            s3_client.put_object(
                Bucket=artifacts_bucket,
                Key=f"{models_folder}.keep",
                Body=b'',  # Fixed: use bytes for .keep
                ContentType='text/plain'
            )
            logger.info(f"Ensured models folder exists: {models_folder}")
        except Exception as e:
            logger.warning(f"Could not create models folder: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating artifacts bucket: {e}")
        return False

# Note: We use MLP instead of attention because:
# 1. Each training example is a single feature vector (not a sequence)
# 2. Attention with seq_len=1 degenerates to an expensive identity transform
# 3. MLP provides the same expressive power for flat feature vectors
# 4. If you need temporal attention later, feed sliding windows of past data
#
# The improved MLP architecture provides:
# - Deeper network with configurable layers for better expressiveness
# - Layer normalization for stable training (batch-size agnostic)
# - Xavier weight initialization for better convergence
# - Graceful handling of both 2D and 3D inputs
# - SiLU activation for smoother gradients
class RegimeClassifier(nn.Module):
    """Regime classifier for identifying market states - efficient MLP architecture"""
    
    def __init__(self, input_size: int = 20, hidden_size: int = 64, n_regimes: int = 3, 
                 dropout: float = 0.3, n_layers: int = 3):
        super().__init__()
        
        # Build a deeper, more expressive MLP with constant hidden size
        blocks = []
        in_f = input_size
        
        for i in range(n_layers - 1):
            blocks += [
                nn.Linear(in_f, hidden_size), 
                nn.LayerNorm(hidden_size),  # More stable than BatchNorm for small datasets
                nn.SiLU(),  # Smoother than ReLU
                nn.Dropout(dropout)
            ]
            in_f = hidden_size
        
        # Final layer outputs logits (no activation, no normalization)
        blocks += [nn.Linear(in_f, n_regimes)]
        
        self.net = nn.Sequential(*blocks)
        
        # Initialize weights for better training
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization for better training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Handle both 2D and 3D inputs gracefully
        if x.ndim == 3:
            x = x.squeeze(1)  # Remove sequence dimension if present
        elif x.ndim != 2:
            raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D")
        
        return self.net(x)

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
        self.scaler = None  # Will be fitted during training
        self.feature_columns = []
        
        # Training results
        self.training_results = {}
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _assert_monotonic_dates(self, df: pd.DataFrame):
        """Assert that dates are monotonically increasing for safe rolling operations"""
        if 'date' in df.columns:
            d = pd.to_datetime(df['date'])
            if not d.is_monotonic_increasing:
                raise ValueError("Input not sorted by date; sort before rolling/diff.")
    

    
    def _initialize_causal_model(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """Initialize the appropriate causal model based on data characteristics"""
        n_samples = len(X)
        
        # Validate config parameters
        n_estimators = max(100, self.config['causal_model'].get('n_estimators', 100))
        min_samples_leaf = max(5, self.config['causal_model'].get('min_samples_leaf', 10))
        max_depth = self.config['causal_model'].get('max_depth', 10)
        
        logger.info(f"Initializing causal model with n_estimators={n_estimators}, min_samples_leaf={min_samples_leaf}, max_depth={max_depth}")
        
        if n_samples < 100:
            # Small sample size - use Linear DML
            logger.warning("Small sample size, using Linear DML instead of Causal Forest")
            rf_kwargs = dict(n_estimators=n_estimators, random_state=42, n_jobs=-1)
            self.causal_model = LinearDML(
                model_y=RandomForestRegressor(**rf_kwargs),
                model_t=RandomForestRegressor(**rf_kwargs),
                discrete_treatment=False,
                random_state=42
            )
        else:
            # Sufficient sample size - use Causal Forest DML
            rf_kwargs = dict(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                           max_depth=max_depth, max_features='sqrt', random_state=42, n_jobs=-1)
            self.causal_model = CausalForestDML(
                model_y=RandomForestRegressor(**rf_kwargs),
                model_t=RandomForestRegressor(**rf_kwargs),
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                max_depth=max_depth,
                discrete_treatment=False,
                random_state=42
            )
    
    def _initialize_pytorch_components(self, X: np.ndarray):
        """Initialize PyTorch regime classifier and uncertainty estimator"""
        input_size = X.shape[1]
        
        # Regime classifier (improved MLP)
        self.regime_classifier = RegimeClassifier(
            input_size=input_size,
            hidden_size=self.config['regime_classifier']['hidden_size'],
            n_regimes=self.config['regime_classifier']['n_regimes'],
            dropout=self.config['regime_classifier']['dropout'],
            n_layers=self.config['regime_classifier'].get('n_layers', 3)
        ).to(self.device)
        
        # Uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator(
            input_size=input_size,
            hidden_size=self.config['uncertainty_estimator']['hidden_size'],
            dropout=self.config['uncertainty_estimator']['dropout']
        ).to(self.device)
    
    def _create_treatment_variables(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Create treatment variables (macro shocks) for causal inference"""
        self._assert_monotonic_dates(df)
        
        treatment_features = []
        treatments = []
        
        try:
            # Fed Rate Shock
            if 'fred_FEDFUNDS_lag_30d' in df.columns:
                fed_rate = df['fred_FEDFUNDS_lag_30d'].astype(float)
                fed_shock = fed_rate.diff(30)  # 30-day change
                fed_shock = fed_shock.replace([np.inf, -np.inf], np.nan)
                df['fed_rate_shock'] = fed_shock
                treatment_features.append('fed_rate_shock')
                treatments.append(fed_shock.values)
            
            # CPI Surprise
            if 'fred_CPIAUCSL_lag_30d' in df.columns:
                cpi = df['fred_CPIAUCSL_lag_30d'].astype(float)
                cpi_trend = cpi.rolling(12).mean()  # 12-month trend
                cpi_shock = (cpi - cpi_trend) / (cpi_trend.replace(0, np.nan))
                cpi_shock = cpi_shock.replace([np.inf, -np.inf], np.nan)
                df['cpi_surprise'] = cpi_shock
                treatment_features.append('cpi_surprise')
                treatments.append(cpi_shock.values)
            
            # GDP Surprise
            if 'fred_GDP_lag_30d' in df.columns:
                gdp = df['fred_GDP_lag_30d'].astype(float)
                gdp_trend = gdp.rolling(8).mean()  # 8-quarter trend
                gdp_shock = (gdp - gdp_trend) / (gdp_trend.replace(0, np.nan))
                gdp_shock = gdp_shock.replace([np.inf, -np.inf], np.nan)
                df['gdp_surprise'] = gdp_shock
                treatment_features.append('gdp_surprise')
                treatments.append(gdp_shock.values)
            
            # Market Stress (VIX-based)
            if 'yahoo_^VIX_volatility_30d' in df.columns:
                vix_vol = df['yahoo_^VIX_volatility_30d'].astype(float)
                vix_threshold = vix_vol.quantile(0.8)
                market_stress = (vix_vol > vix_threshold).astype(float)
                df['market_stress'] = market_stress
                treatment_features.append('market_stress')
                treatments.append(market_stress.values)
            
            if treatments:
                # Use the first treatment variable as primary treatment
                primary_treatment = np.array(treatments[0])
                logger.info(f"Created {len(treatment_features)} treatment variables")
                logger.info(f"Treatments used: {treatment_features}")
                return primary_treatment, treatment_features
            else:
                raise ValueError("No treatment variables created; check column names and preprocessing.")
                
        except Exception as e:
            logger.error(f"Error creating treatment variables: {e}")
            return np.zeros(len(df)), []
    
    def _create_outcome_variables(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Create outcome variables (asset returns) for causal inference"""
        self._assert_monotonic_dates(df)
        
        outcome_features = []
        outcomes = []
        
        try:
            # S&P 500 returns
            if 'yahoo_^GSPC_return_30d' in df.columns:
                sp500_returns = df['yahoo_^GSPC_return_30d'].astype(float)
                df['sp500_returns'] = sp500_returns
                outcome_features.append('sp500_returns')
                outcomes.append(sp500_returns.values)
            
            # Bond returns (TLT)
            if 'yahoo_TLT_return_30d' in df.columns:
                bond_returns = df['yahoo_TLT_return_30d'].astype(float)
                df['bond_returns'] = bond_returns
                outcome_features.append('bond_returns')
                outcomes.append(bond_returns.values)
            
            # Gold returns
            if 'yahoo_GLD_return_30d' in df.columns:
                gold_returns = df['yahoo_GLD_return_30d'].astype(float)
                df['gold_returns'] = gold_returns
                outcome_features.append('gold_returns')
                outcomes.append(gold_returns.values)
            
            if outcomes:
                # Use S&P 500 returns as primary outcome
                primary_outcome = np.array(outcomes[0])
                logger.info(f"Created {len(outcome_features)} outcome variables")
                logger.info(f"Outcomes used: {outcome_features}")
                return primary_outcome, outcome_features
            else:
                raise ValueError("No outcome variables created; check column names and preprocessing.")
                
        except Exception as e:
            logger.error(f"Error creating outcome variables: {e}")
            return np.zeros(len(df)), []
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepare feature matrix for training - DO NOT fit scaler here"""
        try:
            # Get all feature columns (exclude metadata and target columns)
            exclude_columns = ['date', 'execution_id', 'feature_creation_timestamp', 'target']
            feature_columns = [col for col in df.columns if col not in exclude_columns]
            
            # Remove treatment and outcome columns from features
            drop_kw = ('_shock', '_surprise', '_stress', '_return', '_returns')
            # Also explicitly drop the engineered outcome aliases if present
            explicit_drop = {'sp500_returns', 'bond_returns', 'gold_returns'}
            feature_columns = [col for col in feature_columns 
                             if not any(k in col for k in drop_kw) and col not in explicit_drop]
            
            # Create feature matrix
            X = df[feature_columns].to_numpy(dtype=np.float64)
            
            # Handle missing values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            self.feature_columns = feature_columns
            logger.info(f"Prepared {len(feature_columns)} features with shape {X.shape}")
            
            return X, feature_columns
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise
    
    def _finalize_training_arrays(self, df: pd.DataFrame, T: np.ndarray, Y: np.ndarray, X: np.ndarray):
        """Enforce chronological order and align arrays to remove NaNs"""
        # enforce chronological order
        if 'date' in df.columns:
            order = np.argsort(pd.to_datetime(df['date']).values)
            df = df.iloc[order].reset_index(drop=True)
            T = T[order]
            Y = Y[order] 
            X = X[order]

        # valid mask: finite T & Y and finite X rows
        valid = np.isfinite(T) & np.isfinite(Y) & np.all(np.isfinite(X), axis=1)
        # also drop early rows lost to rolling windows (NaNs)
        
        # Log how many rows were dropped
        n_dropped = len(X) - np.sum(valid)
        if n_dropped > 0:
            logger.info(f"Dropped {n_dropped} rows due to NaN values in treatment/outcome/features")
        
        return X[valid], T[valid], Y[valid], valid
    
    def _policy_value(self, Y: np.ndarray, T: np.ndarray, tau: np.ndarray, q: float = 0.2, t_quant: float = 0.7) -> float:
        """Calculate policy value: realized outcome differential for top-q% by treatment effect"""
        # TODO: Consider implementing doubly-robust policy value with propensity/outcome models
        # for more robust evaluation of continuous treatments
        
        # Ensure inputs are arrays and handle NaNs
        tau = np.asarray(tau)
        m = np.isfinite(tau) & np.isfinite(T) & np.isfinite(Y)
        tau, T, Y = tau[m], T[m], Y[m]
        
        if len(tau) == 0:
            return np.nan
            
        # top-q by tau
        k = max(1, int(len(tau) * q))
        idx = np.argsort(tau)[-k:]

        # define 'treated' as large positive shocks using quantiles
        t_thr = np.quantile(T, t_quant)
        y1 = Y[idx][T[idx] >= t_thr]
        y0 = Y[idx][T[idx] < t_thr]
        
        if len(y1) == 0 or len(y0) == 0:
            return np.nan
        return float(np.nanmean(y1) - np.nanmean(y0))
    
    def _train_regime_classifier(self, X: np.ndarray, Y: np.ndarray, T: np.ndarray):
        """Train the PyTorch regime classifier with early stopping"""
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
                X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
                regime_tensor = torch.as_tensor(regime_labels, dtype=torch.long, device=self.device)
                
                # Split into train/validation for early stopping
                n = len(X_tensor)
                split = int(n * 0.85)
                train_ds = TensorDataset(X_tensor[:split], regime_tensor[:split])
                val_ds = TensorDataset(X_tensor[split:], regime_tensor[split:])
                
                # Create dataloaders with pin_memory for CUDA
                pin_memory = self.device.type == 'cuda'
                train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, pin_memory=pin_memory)
                val_dl = DataLoader(val_ds, batch_size=256, shuffle=False, pin_memory=pin_memory)
                
                # Training setup
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(self.regime_classifier.parameters(), 
                                     lr=0.001, weight_decay=1e-4)
                
                # Early stopping setup
                best_val_loss = float('inf')
                patience = 5
                bad_epochs = 0
                
                # Training loop with early stopping
                self.regime_classifier.train()
                for epoch in range(50):
                    # Training phase
                    total_loss = 0
                    for batch_X, batch_y in train_dl:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        optimizer.zero_grad()
                        outputs = self.regime_classifier(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    
                    # Validation phase
                    self.regime_classifier.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for batch_X, batch_y in val_dl:
                            batch_X = batch_X.to(self.device)
                            batch_y = batch_y.to(self.device)
                            outputs = self.regime_classifier(batch_X)
                            val_loss += criterion(outputs, batch_y).item()
                    
                    # Early stopping logic
                    if val_loss < best_val_loss - 1e-4:
                        best_val_loss = val_loss
                        bad_epochs = 0
                    else:
                        bad_epochs += 1
                    
                    if epoch % 10 == 0:
                        logger.info(f"Regime classifier epoch {epoch}, train_loss: {total_loss/len(train_dl):.4f}, val_loss: {val_loss/len(val_dl):.4f}")
                    
                    if bad_epochs >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
                    
                    self.regime_classifier.train()
                
                logger.info("Regime classifier training completed")
                
        except Exception as e:
            logger.error(f"Error training regime classifier: {e}")
    
    def _train_uncertainty_estimator(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """Train the PyTorch uncertainty estimator with improved uncertainty labels and early stopping"""
        try:
            logger.info("Training uncertainty estimator...")
            
            # Fit quick outcome model on train set, compute residuals
            base = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(np.c_[X, T], Y)
            resid = Y - base.predict(np.c_[X, T])
            # rolling std as uncertainty proxy (window=60)
            win = 60 if len(resid) > 60 else max(5, len(resid)//5)
            u = pd.Series(resid).rolling(win, min_periods=win//2).std().fillna(method='bfill').values
            u = np.clip(u, np.percentile(u, 5), np.percentile(u, 95))  # robustify
            
            # Convert to tensors
            X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            uncertainty_tensor = torch.as_tensor(u, dtype=torch.float32, device=self.device)
            
            # Split into train/validation for early stopping
            n = len(X_tensor)
            split = int(n * 0.85)
            train_ds = TensorDataset(X_tensor[:split], uncertainty_tensor[:split])
            val_ds = TensorDataset(X_tensor[split:], uncertainty_tensor[split:])
            
            # Create dataloaders with pin_memory for CUDA
            pin_memory = self.device.type == 'cuda'
            train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, pin_memory=pin_memory)
            val_dl = DataLoader(val_ds, batch_size=256, shuffle=False, pin_memory=pin_memory)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.uncertainty_estimator.parameters(), 
                                 lr=0.001, weight_decay=1e-4)
            
            # Early stopping setup
            best_val_loss = float('inf')
            patience = 5
            bad_epochs = 0
            
            # Training loop with early stopping
            self.uncertainty_estimator.train()
            for epoch in range(50):
                # Training phase
                total_loss = 0
                for batch_X, batch_uncertainty in train_dl:
                    batch_X = batch_X.to(self.device)
                    batch_uncertainty = batch_uncertainty.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.uncertainty_estimator(batch_X).squeeze()
                    loss = criterion(outputs, batch_uncertainty)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                # Validation phase
                self.uncertainty_estimator.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_uncertainty in val_dl:
                        batch_X = batch_X.to(self.device)
                        batch_uncertainty = batch_uncertainty.to(self.device)
                        outputs = self.uncertainty_estimator(batch_X).squeeze()
                        val_loss += criterion(outputs, batch_uncertainty).item()
                
                # Early stopping logic
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                
                if epoch % 10 == 0:
                    logger.info(f"Uncertainty estimator epoch {epoch}, train_loss: {total_loss/len(train_dl):.4f}, val_loss: {val_loss/len(val_dl):.4f}")
                
                if bad_epochs >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                self.uncertainty_estimator.train()
            
            logger.info("Uncertainty estimator training completed")
            
        except Exception as e:
            logger.error(f"Error training uncertainty estimator: {e}")
    
    def _evaluate_model(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance using policy value metric - uses fold-specific scalers"""
        try:
            tscv = TimeSeriesSplit(n_splits=3)
            pol20 = []
            
            for tr, te in tscv.split(X):
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(X[tr])
                Xte = scaler.transform(X[te])
                
                # Use clone or fallback to manual construction for econml compatibility
                try:
                    m = clone(self.causal_model)
                except Exception:
                    # Fallback for econml estimators that don't clone well
                    m = type(self.causal_model)(**self.causal_model.get_params())
                
                m.fit(Y[tr], T[tr], X=Xtr)
                tau = m.effect(Xte)
                pol20.append(self._policy_value(Y[te], T[te], tau, q=0.2))
            
            # CRITICAL: DO NOT overwrite self.scaler here - it's already set in train()
            # This prevents data leakage and ensures consistent scaling between training and inference
            return {
                "policy@20%_mean": float(np.nanmean(pol20)), 
                "policy@20%_std": float(np.nanstd(pol20))
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {"policy@20%_mean": np.nan, "policy@20%_std": np.nan}
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train the complete hybrid causal inference system"""
        try:
            logger.info("Starting hybrid causal inference training...")
            
            # Step 1: Create treatment and outcome variables
            T, treatment_features = self._create_treatment_variables(df)
            Y, outcome_features = self._create_outcome_variables(df)
            
            # Step 2: Prepare features (no scaling yet)
            X, feature_columns = self._prepare_features(df)
            
            # Step 3: Initialize models
            self._initialize_causal_model(X, T, Y)
            self._initialize_pytorch_components(X)
            
            # Step 4: Finalize arrays (sort by date and remove NaNs)
            X, T, Y, valid_mask = self._finalize_training_arrays(df, T, Y, X)
            
            # Fail fast if no data remains after filtering
            if len(X) == 0:
                raise ValueError("No rows left after alignment/NaN filtering. Check rolling windows and required columns.")
            
            # Step 5: Clip extreme values per-feature before scaling to reduce leverage
            # This avoids squashing small-variance columns
            lo = np.percentile(X, 0.1, axis=0)
            hi = np.percentile(X, 99.9, axis=0)
            X = np.clip(X, lo, hi)
            
            # Step 6: Fit global scaler and transform X for consistent training
            self.scaler = StandardScaler().fit(X)
            X_scaled = self.scaler.transform(X)
            
            # Step 6: Train core causal model (econml) - now with scaled data
            logger.info("Training core causal model...")
            self.causal_model.fit(Y, T, X=X_scaled)
            
            # Step 7: Train PyTorch components - also with scaled data
            self._train_regime_classifier(X_scaled, Y, T)
            self._train_uncertainty_estimator(X_scaled, T, Y)
            
            # Ensure models are in eval mode for inference
            self.regime_classifier.eval()
            self.uncertainty_estimator.eval()
            
            # Step 8: Evaluate model performance
            performance_metrics = self._evaluate_model(X, T, Y)
            
            # Step 9: Generate training summary
            training_summary = {
                'n_samples': len(X),
                'n_features': len(feature_columns),
                'n_treatments': len(treatment_features),
                'n_outcomes': len(outcome_features),
                'performance_metrics': performance_metrics,
                'training_timestamp': datetime.now().isoformat(),
                'model_config': self.config,
                'feature_columns': self.feature_columns,
                'training_mask': valid_mask.astype(bool).tolist(),  # Fixed: convert to list for JSON
                'dtypes': {c: str(df[c].dtype) for c in df.columns},
                'regime_names': {"0": "low_vol", "1": "med_vol", "2": "high_vol"},
                'pytorch_spec': {
                    'input_size': int(len(self.feature_columns)),
                    'regime_config': self.config['regime_classifier'],
                    'uncertainty_config': self.config['uncertainty_estimator'],
                    'architecture_note': 'MLP-based regime classifier (no attention needed for flat feature vectors)'
                }
            }
            
            self.training_results = training_summary
            logger.info("Training completed successfully")
            
            return training_summary
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def predict(self, X_new: np.ndarray) -> Dict[str, Any]:
        """Generate predictions using the trained hybrid system"""
        try:
            # Scale new features using fitted scaler
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Train the model first.")
            
            X_new_scaled = self.scaler.transform(X_new)
            
            # Get causal effects from econml model
            causal_effects = self.causal_model.effect(X_new_scaled)
            
            # Get regime probabilities from PyTorch classifier
            self.regime_classifier.eval()
            with torch.no_grad():
                X_tensor = torch.as_tensor(X_new_scaled, dtype=torch.float32, device=self.device)
                logits = self.regime_classifier(X_tensor)              # [B, n_regimes]
                regime_probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            # Get uncertainty estimates from PyTorch estimator
            self.uncertainty_estimator.eval()
            with torch.no_grad():
                uncertainty = self.uncertainty_estimator(X_tensor).cpu().numpy().flatten()
            
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
    """Load processed data from S3 Gold bucket - Fixed S3 parquet loading"""
    try:
        s3_client = boto3.client('s3')
        
        # Load gold data
        gold_key = f"gold/{execution_id}/processed_data.parquet"
        response = s3_client.get_object(Bucket=bucket_name, Key=gold_key)
        with io.BytesIO(response["Body"].read()) as buf:
            df = pd.read_parquet(buf)
        
        logger.info(f"Loaded data with shape: {df.shape}")
        return df
        
    except ClientError as e:
        logger.error(f"Error loading data from S3: {e}")
        raise

def save_model_to_s3(model: HybridCausalSystem, execution_id: str, artifacts_bucket: str):
    """Save trained model and artifacts to S3 with robust S3 writes"""
    try:
        s3_client = boto3.client('s3')
        
        # Create model artifacts
        model_artifacts = {
            'model_state': {
                'causal_model': model.causal_model,
                'regime_classifier_state': {k: v.detach().cpu() for k,v in model.regime_classifier.state_dict().items()} if model.regime_classifier else None,
                'uncertainty_estimator_state': {k: v.detach().cpu() for k,v in model.uncertainty_estimator.state_dict().items()} if model.uncertainty_estimator else None,
            },
            'scaler': model.scaler,
            'feature_columns': model.feature_columns,
            'training_results': model.training_results,
            'training_timestamp': datetime.now().isoformat(),
            'execution_id': execution_id,
            'versions': {
                'sklearn': sklearn.__version__,
                'econml': econml.__version__,
                'torch': torch.__version__,
                'numpy': np.__version__,
                'pandas': pd.__version__
            }
        }
        
        # Save to S3 using joblib for better compression
        artifacts_key = f"models/{execution_id}/hybrid_causal_model.pkl"
        buf_path = f"/tmp/hybrid_{execution_id}.pkl"
        joblib.dump(model_artifacts, buf_path, compress=3)
        
        with open(buf_path, "rb") as fh:
            s3_client.put_object(
                Bucket=artifacts_bucket,
                Key=artifacts_key,
                Body=fh.read(),
                ContentType='application/octet-stream'
            )
        
        # Clean up temp file
        os.remove(buf_path)
        
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
        # Validate S3 paths
        logger.info("Validating S3 paths...")
        if not validate_training_data_paths(args.gold_bucket, args.execution_id):
            return 1
        if not validate_artifacts_bucket(args.artifacts_bucket):
            return 1
        
        # Load data
        logger.info("Loading training data...")
        df = load_data_from_s3(args.gold_bucket, args.execution_id)
        
        # Define model configuration with validation
        config = {
            'causal_model': {
                'n_estimators': 200,  # Increased from 100
                'min_samples_leaf': 10,
                'max_depth': 10
            },
            'regime_classifier': {
                'hidden_size': 64,  # Increased from 32 for better expressiveness
                'n_regimes': 3,
                'dropout': 0.3,
                'n_layers': 3  # Configurable depth
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
