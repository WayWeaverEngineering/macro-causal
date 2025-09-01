#!/usr/bin/env python3
"""
Test script for the Hybrid Causal Inference Training Pipeline
Validates the training logic and model components
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add the current directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import HybridCausalSystem, AttentionRegimeClassifier, UncertaintyEstimator

# Configure logging for testing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_mock_data(n_samples: int = 500) -> pd.DataFrame:
    """Create mock training data for testing"""
    logger.info("Creating mock training data...")
    
    # Create date range
    start_date = datetime(2010, 1, 1)
    dates = [start_date + timedelta(days=i*30) for i in range(n_samples)]
    
    # Create mock FRED data
    np.random.seed(42)
    
    # Economic indicators
    gdp = 100 + np.cumsum(np.random.normal(0.5, 0.2, n_samples))
    cpi = 100 + np.cumsum(np.random.normal(0.1, 0.05, n_samples))
    fed_funds = 2 + np.cumsum(np.random.normal(0, 0.1, n_samples))
    unemployment = 5 + np.random.normal(0, 0.5, n_samples)
    
    # Financial data
    sp500_price = 1000 + np.cumsum(np.random.normal(5, 20, n_samples))
    bond_price = 100 + np.cumsum(np.random.normal(0.1, 1, n_samples))
    gold_price = 1500 + np.cumsum(np.random.normal(2, 10, n_samples))
    vix = 20 + np.random.normal(0, 5, n_samples)
    
    # Create feature columns
    data = {
        'date': dates,
        'fred_GDP_lag_30d': gdp,
        'fred_CPIAUCSL_lag_30d': cpi,
        'fred_FEDFUNDS_lag_30d': fed_funds,
        'fred_UNRATE_lag_30d': unemployment,
        'yahoo_^GSPC_Close': sp500_price,
        'yahoo_TLT_Close': bond_price,
        'yahoo_GLD_Close': gold_price,
        'yahoo_^VIX_Close': vix,
        'yahoo_^VIX_volatility_30d': np.abs(np.random.normal(0, 0.2, n_samples)),
        'yahoo_^GSPC_return_30d': np.random.normal(-0.01, 0.05, n_samples),
        'yahoo_TLT_return_30d': np.random.normal(0.005, 0.02, n_samples),
        'yahoo_GLD_return_30d': np.random.normal(0.002, 0.03, n_samples),
        'execution_id': 'test_execution_123',
        'feature_creation_timestamp': datetime.now().isoformat()
    }
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    logger.info(f"Created mock data with shape: {df.shape}")
    return df

def test_attention_regime_classifier():
    """Test the AttentionRegimeClassifier"""
    logger.info("Testing AttentionRegimeClassifier...")
    
    try:
        # Create model
        model = AttentionRegimeClassifier(
            input_size=20,
            hidden_size=32,
            n_regimes=3,
            attention_heads=4,
            dropout=0.3
        )
        
        # Create test input
        batch_size = 10
        input_size = 20
        X = torch.randn(batch_size, input_size)
        
        # Forward pass
        with torch.no_grad():
            output = model(X)
        
        # Check output shape
        assert output.shape == (batch_size, 3), f"Expected shape (10, 3), got {output.shape}"
        
        # Check probabilities sum to 1
        prob_sums = output.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-6), "Probabilities don't sum to 1"
        
        logger.info("✓ AttentionRegimeClassifier test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ AttentionRegimeClassifier test failed: {e}")
        return False

def test_uncertainty_estimator():
    """Test the UncertaintyEstimator"""
    logger.info("Testing UncertaintyEstimator...")
    
    try:
        # Create model
        model = UncertaintyEstimator(
            input_size=20,
            hidden_size=32,
            dropout=0.3
        )
        
        # Create test input
        batch_size = 10
        input_size = 20
        X = torch.randn(batch_size, input_size)
        
        # Forward pass
        with torch.no_grad():
            output = model(X)
        
        # Check output shape
        assert output.shape == (batch_size, 1), f"Expected shape (10, 1), got {output.shape}"
        
        # Check all outputs are positive (Softplus ensures this)
        assert torch.all(output > 0), "Uncertainty outputs should be positive"
        
        logger.info("✓ UncertaintyEstimator test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ UncertaintyEstimator test failed: {e}")
        return False

def test_hybrid_system_initialization():
    """Test HybridCausalSystem initialization"""
    logger.info("Testing HybridCausalSystem initialization...")
    
    try:
        # Test configuration
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
        
        # Create system
        hybrid_system = HybridCausalSystem(config)
        
        # Check attributes
        assert hybrid_system.config == config
        assert hybrid_system.causal_model is None
        assert hybrid_system.regime_classifier is None
        assert hybrid_system.uncertainty_estimator is None
        
        logger.info("✓ HybridCausalSystem initialization test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ HybridCausalSystem initialization test failed: {e}")
        return False

def test_feature_preparation():
    """Test feature preparation methods"""
    logger.info("Testing feature preparation...")
    
    try:
        # Create mock data
        df = create_mock_data(100)
        
        # Create system
        config = {
            'causal_model': {'n_estimators': 50, 'min_samples_leaf': 5, 'max_depth': 5},
            'regime_classifier': {'hidden_size': 16, 'n_regimes': 3, 'attention_heads': 2, 'dropout': 0.2},
            'uncertainty_estimator': {'hidden_size': 16, 'dropout': 0.2}
        }
        
        hybrid_system = HybridCausalSystem(config)
        
        # Test treatment variable creation
        T, treatment_features = hybrid_system._create_treatment_variables(df)
        assert len(T) == len(df), f"Treatment length mismatch: {len(T)} vs {len(df)}"
        assert len(treatment_features) > 0, "No treatment features created"
        
        # Test outcome variable creation
        Y, outcome_features = hybrid_system._create_outcome_variables(df)
        assert len(Y) == len(df), f"Outcome length mismatch: {len(Y)} vs {len(df)}"
        assert len(outcome_features) > 0, "No outcome features created"
        
        # Test feature preparation
        X, feature_columns = hybrid_system._prepare_features(df)
        assert X.shape[0] == len(df), f"Feature matrix row mismatch: {X.shape[0]} vs {len(df)}"
        assert len(feature_columns) > 0, "No features prepared"
        
        logger.info("✓ Feature preparation test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Feature preparation test failed: {e}")
        return False

def test_model_training():
    """Test the complete training pipeline"""
    logger.info("Testing complete training pipeline...")
    
    try:
        # Create mock data
        df = create_mock_data(200)  # Smaller dataset for faster testing
        
        # Create system with smaller models for testing
        config = {
            'causal_model': {
                'n_estimators': 20,  # Reduced for testing
                'min_samples_leaf': 5,
                'max_depth': 5
            },
            'regime_classifier': {
                'hidden_size': 16,
                'n_regimes': 3,
                'attention_heads': 2,
                'dropout': 0.2
            },
            'uncertainty_estimator': {
                'hidden_size': 16,
                'dropout': 0.2
            }
        }
        
        hybrid_system = HybridCausalSystem(config)
        
        # Train the system
        training_results = hybrid_system.train(df)
        
        # Check training results
        assert 'n_samples' in training_results
        assert 'n_features' in training_results
        assert 'performance_metrics' in training_results
        assert training_results['n_samples'] == len(df)
        
        # Check that models are trained
        assert hybrid_system.causal_model is not None
        assert hybrid_system.regime_classifier is not None
        assert hybrid_system.uncertainty_estimator is not None
        
        # Test prediction
        X_test = np.random.randn(5, len(hybrid_system.feature_columns))
        T_test = np.random.randn(5)
        
        predictions = hybrid_system.predict(X_test, T_test)
        
        # Check prediction structure
        assert 'causal_effects' in predictions
        assert 'regime_probabilities' in predictions
        assert 'dominant_regime' in predictions
        assert 'uncertainty' in predictions
        
        logger.info("✓ Complete training pipeline test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Complete training pipeline test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results"""
    logger.info("Starting comprehensive test suite...")
    
    tests = [
        ("AttentionRegimeClassifier", test_attention_regime_classifier),
        ("UncertaintyEstimator", test_uncertainty_estimator),
        ("HybridCausalSystem Initialization", test_hybrid_system_initialization),
        ("Feature Preparation", test_feature_preparation),
        ("Complete Training Pipeline", test_model_training)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Report results
    logger.info(f"\n{'='*50}")
    logger.info("TEST RESULTS SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! The training pipeline is ready for production.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please fix the issues before deployment.")
        return False

if __name__ == "__main__":
    # Check if we're in a test environment
    if os.environ.get('TESTING') == 'true':
        success = run_all_tests()
        sys.exit(0 if success else 1)
    else:
        logger.info("Test script loaded. Set TESTING=true to run tests.")
        logger.info("You can also run individual test functions:")
        logger.info("- test_attention_regime_classifier()")
        logger.info("- test_uncertainty_estimator()")
        logger.info("- test_hybrid_system_initialization()")
        logger.info("- test_feature_preparation()")
        logger.info("- test_model_training()")
