# Integration Guide: Hybrid Causal Inference Training Pipeline

This guide explains how to integrate and use the Hybrid Causal Inference Training Pipeline in your macro causal inference system.

## 🚀 Quick Start

### 1. Prerequisites

Ensure you have the following data pipeline components ready:
- **Data Collection**: FRED, Yahoo Finance, and World Bank data collectors
- **Data Processing**: Feature engineering pipeline completed
- **S3 Buckets**: Bronze, Silver, and Gold data buckets configured

### 2. Environment Setup

```bash
# Set required environment variables
export EXECUTION_ID="your_execution_id"
export GOLD_BUCKET="your-gold-bucket-name"
export ARTIFACTS_BUCKET="your-artifacts-bucket-name"
export AWS_REGION="us-east-1"
```

### 3. Run Training

```bash
# Run the training pipeline
python main.py \
    --execution-id $EXECUTION_ID \
    --gold-bucket $GOLD_BUCKET \
    --artifacts-bucket $ARTIFACTS_BUCKET
```

## 🏗️ Architecture Integration

### Data Flow Integration

```
Data Collection → Data Processing → Model Training → Model Artifacts
     ↓                ↓              ↓              ↓
  Bronze Bucket → Silver Bucket → Gold Bucket → Models Bucket
```

### Component Integration Points

#### 1. Data Processing Output
The training pipeline expects data from the data processing stage:

```python
# Expected data structure in gold bucket
gold/{execution_id}/
├── processed_data.parquet          # Main training dataset
└── pipeline_results/
    └── data_processing/
        └── {execution_id}_results.json
```

#### 2. Model Artifacts Output
The training pipeline produces:

```python
# Generated model artifacts
models/{execution_id}/
├── hybrid_causal_model.pkl         # Complete model state
├── training_results.json           # Performance metrics
├── feature_columns.txt             # Feature names
└── model_config.json              # Training configuration
```

## 🔧 Configuration

### Default Configuration

```python
default_config = {
    'causal_model': {
        'n_estimators': 100,        # Number of trees
        'min_samples_leaf': 10,     # Min samples per leaf
        'max_depth': 10             # Max tree depth
    },
    'regime_classifier': {
        'hidden_size': 32,          # Hidden layer size
        'n_regimes': 3,             # Market regimes
        'attention_heads': 4,       # Attention heads
        'dropout': 0.3              # Dropout rate
    },
    'uncertainty_estimator': {
        'hidden_size': 32,          # Hidden layer size
        'dropout': 0.3              # Dropout rate
    }
}
```

### Custom Configuration

```python
# Customize for your use case
custom_config = {
    'causal_model': {
        'n_estimators': 200,        # More trees for complex relationships
        'min_samples_leaf': 20,     # More robust to noise
        'max_depth': 15             # Deeper trees
    },
    'regime_classifier': {
        'hidden_size': 64,          # Larger network
        'n_regimes': 5,             # More granular regimes
        'attention_heads': 8,       # More attention heads
        'dropout': 0.4              # Higher dropout
    }
}
```

## 📊 Data Requirements

### Required Features

The training pipeline automatically creates these features from your data:

#### Economic Features (FRED)
- `fred_GDP_lag_30d`: GDP with 30-day lag
- `fred_CPIAUCSL_lag_30d`: CPI with 30-day lag
- `fred_FEDFUNDS_lag_30d`: Fed Funds Rate with 30-day lag
- `fred_UNRATE_lag_30d`: Unemployment Rate with 30-day lag

#### Financial Features (Yahoo Finance)
- `yahoo_^GSPC_Close`: S&P 500 closing prices
- `yahoo_TLT_Close`: TLT bond prices
- `yahoo_GLD_Close`: Gold ETF prices
- `yahoo_^VIX_Close`: VIX volatility index
- `yahoo_^VIX_volatility_30d`: 30-day VIX volatility

#### Derived Features
- Returns, volatility, technical indicators
- Rolling statistics, lagged features
- Regime classifications, interaction features

### Data Quality Requirements

- **Minimum Sample Size**: 100 observations
- **Feature Completeness**: >90% non-missing values
- **Data Frequency**: Monthly or higher frequency
- **Time Range**: At least 3 years of historical data

## 🔄 Training Workflow

### 1. Data Validation

```python
# The pipeline automatically validates:
# - Feature availability
# - Data completeness
# - Treatment variable creation
# - Outcome variable creation
```

### 2. Model Training

```python
# Training sequence:
# 1. Initialize hybrid system
# 2. Create treatment/outcome variables
# 3. Prepare feature matrix
# 4. Train causal model (econml)
# 5. Train regime classifier (PyTorch)
# 6. Train uncertainty estimator (PyTorch)
# 7. Evaluate performance
```

### 3. Model Persistence

```python
# Models are saved with:
# - Complete model state
# - Feature preprocessing
# - Training metrics
# - Configuration parameters
```

## 🎯 Inference Integration

### Loading Trained Models

```python
import pickle
from main import HybridCausalSystem

# Load trained model
with open('hybrid_causal_model.pkl', 'rb') as f:
    model_artifacts = pickle.load(f)

# Initialize system
config = model_artifacts['model_config']
hybrid_system = HybridCausalSystem(config)

# Load model state
hybrid_system.load_state(model_artifacts)
```

### Making Predictions

```python
# Prepare new data
X_new = prepare_features(new_data)
T_new = create_treatments(new_data)

# Generate predictions
predictions = hybrid_system.predict(X_new, T_new)

# Results include:
# - Causal effects
# - Regime probabilities
# - Uncertainty estimates
```

## 📈 Monitoring & Performance

### Training Metrics

```python
training_results = {
    'n_samples': 500,
    'n_features': 25,
    'performance_metrics': {
        'mean_mse': 0.0012,
        'std_mse': 0.0003,
        'mean_r2': 0.78,
        'std_r2': 0.05
    }
}
```

### Performance Benchmarks

- **Training Time**: 5-10 minutes for 500 samples
- **Memory Usage**: ~2-4GB for typical datasets
- **Prediction Speed**: <100ms per prediction
- **Accuracy**: R² > 0.7 for well-behaved data

## 🚨 Error Handling

### Common Issues

#### 1. Missing Dependencies
```bash
# Install econml
pip install econml==0.14.1

# Install PyTorch
pip install torch==2.1.0 torchvision==0.16.0
```

#### 2. Data Quality Issues
```python
# Check data completeness
df.isnull().sum()

# Verify feature names
expected_features = ['fred_GDP_lag_30d', 'yahoo_^GSPC_Close']
missing_features = [f for f in expected_features if f not in df.columns]
```

#### 3. Memory Issues
```python
# Reduce model complexity
config['causal_model']['n_estimators'] = 50
config['regime_classifier']['hidden_size'] = 16
```

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable PyTorch profiler
with torch.profiler.profile() as prof:
    results = hybrid_system.train(df)
```

## 🔄 CI/CD Integration

### AWS CodeBuild Integration

```yaml
# buildspec.yml
version: 0.2
phases:
  install:
    runtime-versions:
      python: 3.10
  build:
    commands:
      - pip install -r requirements.txt
      - python main.py --execution-id $EXECUTION_ID
```

### Docker Integration

```bash
# Build image
docker build -t macro-causal-training .

# Run training
docker run -e EXECUTION_ID=123 \
           -e GOLD_BUCKET=my-gold-bucket \
           -e ARTIFACTS_BUCKET=my-artifacts-bucket \
           macro-causal-training
```

## 🧪 Testing

### Run Test Suite

```bash
# Run all tests
export TESTING=true
python test_training.py

# Run individual tests
python -c "
from test_training import test_hybrid_system_initialization
test_hybrid_system_initialization()
"
```

### Test Data Generation

```python
# Generate mock data for testing
from test_training import create_mock_data
df = create_mock_data(n_samples=200)
```

## 📚 API Integration

### FastAPI Integration

```python
from fastapi import FastAPI
from main import HybridCausalSystem

app = FastAPI()

@app.post("/causal-analysis")
async def causal_analysis(query: dict):
    # Load model
    hybrid_system = load_trained_model()
    
    # Prepare data
    X_new, T_new = prepare_query_data(query)
    
    # Generate predictions
    predictions = hybrid_system.predict(X_new, T_new)
    
    return {
        "causal_effects": predictions['causal_effects'],
        "regime": predictions['dominant_regime'],
        "uncertainty": predictions['uncertainty']
    }
```

## 🔮 Future Enhancements

### Planned Features

1. **Incremental Training**: Update models with new data
2. **Model Ensembling**: Combine multiple causal models
3. **Advanced Regimes**: Dynamic regime detection
4. **Uncertainty Calibration**: Calibrated confidence intervals
5. **A/B Testing**: Model performance comparison

### Customization Points

- **Regime Definitions**: Customize market regime classifications
- **Treatment Variables**: Add new macro shock definitions
- **Feature Engineering**: Extend feature creation pipeline
- **Model Architecture**: Modify PyTorch component architectures

## 📞 Support

### Getting Help

1. **Check Logs**: Review training logs for error details
2. **Run Tests**: Validate system with test suite
3. **Review Data**: Ensure data quality and completeness
4. **Check Configuration**: Verify model parameters

### Common Patterns

- **Data Pipeline**: Ensure data flows correctly through all stages
- **Feature Engineering**: Verify feature creation and naming
- **Model Training**: Monitor training progress and performance
- **Model Persistence**: Confirm artifacts are saved correctly

---

**Built with ❤️ for Bridgewater Associates AI Lab**

*This integration guide provides comprehensive instructions for deploying and using the Hybrid Causal Inference Training Pipeline in production environments.*
