# Hybrid Causal Inference Training Pipeline

This directory contains the training pipeline for the Macro Causal Inference System, implementing **X-Learner with Double Machine Learning (DML)** using econml, combined with PyTorch components for regime classification and uncertainty estimation.

## 🎯 Overview

The training pipeline implements a **hybrid architecture** that combines:

1. **Core Causal Inference (econml)**: X-Learner with Double Machine Learning for robust causal effect estimation
2. **Regime Classification (PyTorch)**: Self-attention neural network for identifying market states
3. **Uncertainty Estimation (PyTorch)**: Neural network for estimating confidence in causal effects

## 🏗️ Architecture

### Hybrid System Components

```
Raw Features → Feature Engineering → Hybrid Training → Model Artifacts
                    ↓
            ┌─────────────────┬─────────────────┐
            │   econml Core   │  PyTorch Comp.  │
            │                 │                 │
            │ • X-Learner     │ • Regime Class. │
            │ • DML           │ • Uncertainty   │
            │ • Causal Forest │ • Self-Attention│
            └─────────────────┴─────────────────┘
                    ↓
            Combined Predictions + Regime Context
```

### Core Causal Inference (econml)

The system uses **Double Machine Learning (DML)** to address confounding variables:

1. **Stage 1**: Predict treatment (macro shock) from confounders
2. **Stage 1**: Predict outcome (asset returns) from confounders  
3. **Stage 2**: Estimate causal effect on residuals

**X-Learner** extends DML to handle heterogeneous treatment effects:
- **Learner 1**: Estimate outcome model for treated units
- **Learner 2**: Estimate outcome model for control units
- **Learner 3**: Estimate treatment effect model
- **Final Effect**: Weighted combination of all learners

### PyTorch Components

#### AttentionRegimeClassifier
- **Purpose**: Identifies different market states where causal relationships behave differently
- **Architecture**: Self-attention mechanism with multi-head attention
- **Regimes**: Low Volatility, Medium Volatility, High Volatility
- **Features**: Market volatility, economic indicators, regime interactions

#### UncertaintyEstimator
- **Purpose**: Provides confidence levels for causal effect estimates
- **Architecture**: Multi-layer perceptron with Softplus activation
- **Output**: Positive uncertainty values for each prediction
- **Features**: Economic conditions, market stress, treatment magnitude

## 📊 Training Data

### Input Features
The pipeline expects processed features from the data processing stage:

- **Economic Features**: FRED indicators with lags, rolling statistics, rate of change
- **Financial Features**: Asset returns, volatility, technical indicators
- **World Bank Features**: Cross-country comparisons, long-term trends
- **Regime Features**: Economic, inflation, monetary, and volatility regimes
- **Treatment Features**: Policy interventions, fiscal indicators, market stress

### Treatment Variables (Macro Shocks)
- **Fed Rate Shock**: 30-day change in Federal Funds Rate
- **CPI Surprise**: Deviation from 12-month trend
- **GDP Surprise**: Deviation from 8-quarter trend
- **Market Stress**: VIX-based volatility threshold

### Outcome Variables (Asset Returns)
- **S&P 500 Returns**: 30-day returns for ^GSPC
- **Bond Returns**: 30-day returns for TLT
- **Gold Returns**: 30-day returns for GLD

## 🚀 Training Process

### 1. Data Preparation
```python
# Load processed data from S3 Gold bucket
df = load_data_from_s3(gold_bucket, execution_id)

# Create treatment and outcome variables
T, treatment_features = create_treatment_variables(df)
Y, outcome_features = create_outcome_variables(df)

# Prepare feature matrix
X, feature_columns = prepare_features(df)
```

### 2. Model Initialization
```python
# Initialize hybrid system
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

hybrid_system = HybridCausalSystem(config)
```

### 3. Training Execution
```python
# Train the complete system
training_results = hybrid_system.train(df)

# Training includes:
# - Core causal model (econml)
# - Regime classifier (PyTorch)
# - Uncertainty estimator (PyTorch)
```

### 4. Model Persistence
```python
# Save model artifacts to S3
save_model_to_s3(hybrid_system, execution_id, artifacts_bucket)

# Artifacts include:
# - Trained causal model
# - PyTorch model states
# - Feature scaler
# - Training results
# - Configuration
```

## 📈 Model Performance

### Evaluation Metrics
- **Mean Squared Error (MSE)**: Causal effect prediction accuracy
- **R-squared (R²)**: Proportion of variance explained
- **Cross-validation**: Time-series split validation
- **Regime Classification**: Accuracy of market state identification

### Performance Characteristics
- **Sample Size**: Optimized for 300-500 monthly observations
- **Feature Count**: 15-20 engineered macro variables
- **Training Time**: ~5-10 minutes for typical datasets
- **Memory Usage**: Efficient for datasets up to 10GB

## 🔧 Configuration

### Causal Model Settings
```python
'causal_model': {
    'n_estimators': 100,        # Number of trees in forest
    'min_samples_leaf': 10,     # Minimum samples per leaf
    'max_depth': 10             # Maximum tree depth
}
```

### Regime Classifier Settings
```python
'regime_classifier': {
    'hidden_size': 32,          # Hidden layer size
    'n_regimes': 3,             # Number of market regimes
    'attention_heads': 4,       # Multi-head attention
    'dropout': 0.3              # Dropout rate
}
```

### Uncertainty Estimator Settings
```python
'uncertainty_estimator': {
    'hidden_size': 32,          # Hidden layer size
    'dropout': 0.3              # Dropout rate
}
```

## 📤 Output & Artifacts

### Model Artifacts
The pipeline generates comprehensive model artifacts:

```
models/{execution_id}/
├── hybrid_causal_model.pkl     # Complete model state
├── training_results.json       # Performance metrics
├── feature_columns.txt         # Feature names
└── model_config.json          # Training configuration
```

### Model State Contents
```python
model_artifacts = {
    'model_state': {
        'causal_model': trained_econml_model,
        'regime_classifier_state': pytorch_state_dict,
        'uncertainty_estimator_state': pytorch_state_dict,
    },
    'scaler': fitted_standard_scaler,
    'feature_columns': list_of_feature_names,
    'training_results': performance_metrics,
    'training_timestamp': iso_timestamp,
    'execution_id': execution_id
}
```

## 🎯 Inference Usage

### Prediction Interface
```python
# Load trained model
with open('hybrid_causal_model.pkl', 'rb') as f:
    model_artifacts = pickle.load(f)

# Initialize system
hybrid_system = HybridCausalSystem(config)
hybrid_system.load_state(model_artifacts)

# Generate predictions
predictions = hybrid_system.predict(X_new, T_new)

# Results include:
# - Causal effects
# - Regime probabilities
# - Dominant regime
# - Uncertainty estimates
```

### Example Response
```python
{
    'causal_effects': [-0.023, -0.018, -0.025],
    'regime_probabilities': [[0.1, 0.7, 0.2], ...],
    'dominant_regime': [1, 1, 1],
    'uncertainty': [0.15, 0.12, 0.18]
}
```

## 🚀 Deployment

### AWS Integration
- **S3**: Model artifacts storage
- **Lambda**: Training orchestration
- **EMR**: Distributed training (if needed)
- **DynamoDB**: Model registry (future enhancement)

### Container Deployment
```bash
# Build Docker image
docker build -t macro-causal-training .

# Run training
docker run -e EXECUTION_ID=123 \
           -e GOLD_BUCKET=my-gold-bucket \
           -e ARTIFACTS_BUCKET=my-artifacts-bucket \
           macro-causal-training
```

## 📊 Monitoring & Logging

### Logging Strategy
- **Collection Phase**: Only warnings and errors
- **Training Summary**: Single comprehensive summary at completion
- **No Progress Logs**: Minimal logging during execution

### Performance Monitoring
- **Training Metrics**: Loss curves, validation scores
- **Resource Usage**: Memory, CPU, GPU utilization
- **Error Tracking**: Comprehensive error collection and reporting

## 🔍 Troubleshooting

### Common Issues

#### Missing Dependencies
```bash
# Install econml
pip install econml==0.14.1

# Install PyTorch
pip install torch==2.1.0 torchvision==0.16.0
```

#### Memory Issues
- Reduce batch size in regime/uncertainty training
- Use smaller hidden layer sizes
- Process data in chunks

#### Training Convergence
- Adjust learning rates
- Increase training epochs
- Modify network architecture

### Debug Mode
```python
# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Enable PyTorch profiler
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    training_results = hybrid_system.train(df)
```

## 🔮 Future Enhancements

### Planned Improvements
- **Incremental Training**: Update models with new data
- **Model Ensembling**: Combine multiple causal models
- **Advanced Regimes**: Dynamic regime detection
- **Uncertainty Calibration**: Calibrated confidence intervals
- **A/B Testing**: Model performance comparison

### Research Directions
- **Causal Discovery**: Automatic treatment variable identification
- **Temporal Causality**: Time-varying causal effects
- **Multi-Output**: Simultaneous asset class modeling
- **Interpretability**: SHAP values for causal effects

## 📚 References

### Academic Papers
- Künzel, S. R., et al. (2019). "Metalearners for estimating heterogeneous treatment effects using machine learning." PNAS
- Chernozhukov, V., et al. (2018). "Double/debiased machine learning for treatment and structural parameters." The Econometrics Journal
- Vaswani, A., et al. (2017). "Attention is all you need." NIPS

### Technical Documentation
- [EconML Documentation](https://econml.azurewebsites.net/)
- [PyTorch Profiler](https://pytorch.org/tutorials/beginner/profiler.html)
- [Double Machine Learning Tutorial](https://econml.azurewebsites.net/spec/estimation/dml.html)

---

**Built with ❤️ for Bridgewater Associates AI Lab**

*This hybrid system demonstrates the power of combining proven causal inference methodology with cutting-edge PyTorch technology, providing the analytical rigor and technical sophistication that Bridgewater Associates values in their investment process.*
