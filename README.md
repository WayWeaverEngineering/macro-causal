# Macro Causal Inference System: X-Learner with Double Machine Learning

A production-ready causal inference system for analyzing macroeconomic relationships with asset returns, designed to generate investment insights based on scientifically rigorous causal analysis.

## 🎯 Project Overview

This system implements **X-Learner with Double Machine Learning (DML)** to identify causal relationships between macroeconomic shocks and asset returns. The approach separates correlation from causation, providing Bridgewater-style analytical rigor for macro-driven investment decisions.

### Key Features
- **Causal Inference**: True causal effects, not just statistical correlations
- **Regime-Dependent Analysis**: Effects that vary across different market conditions
- **Natural Language Explanations**: AI-generated insights using OpenAI
- **Real-time API**: FastAPI backend for interactive causal analysis
- **Professional Frontend**: React-based interface for investment professionals

## 🏗️ System Architecture

### Data Pipeline
```
FRED API → Lambda Collectors → S3 Bronze → ETL Processing → S3 Gold → X-Learner Training
Yahoo Finance → Lambda Collectors → S3 Bronze → ETL Processing → S3 Gold → X-Learner Training
World Bank → Lambda Collectors → S3 Bronze → ETL Processing → S3 Gold → X-Learner Training
```

### ML Pipeline
```
Raw Data → Feature Engineering → Treatment Definition → X-Learner Training → Model Registry → Inference API
```

### API Architecture
```
User Query → FastAPI → X-Learner Inference → OpenAI Text Generation → Structured Response
```

## 📊 Data Sources

### Economic Indicators (FRED)
- **GDP**: Gross Domestic Product (quarterly)
- **CPI**: Consumer Price Index (monthly)
- **Unemployment Rate**: Monthly unemployment data
- **Federal Funds Rate**: Central bank policy rate
- **Treasury Rates**: 2-year and 10-year yields
- **Oil Prices**: WTI crude oil prices
- **Exchange Rates**: USD/EUR, USD/CNY
- **Nonfarm Payrolls**: Employment data

### Asset Returns (Yahoo Finance)
- **S&P 500**: ^GSPC
- **Bonds**: TLT (20+ year Treasury)
- **Gold**: GLD
- **Oil**: USO
- **VIX**: Market volatility index

### Additional Macro Data (World Bank)
- **GDP Growth**: Annual percentage change
- **Inflation**: Consumer price inflation
- **Interest Rates**: Real interest rates
- **Trade Data**: Exports/imports as % of GDP

## 🔬 Causal Inference Methodology

### Hybrid Approach: econml + PyTorch

The system implements a **hybrid architecture** combining the reliability of **econml** for core causal inference with the flexibility of **PyTorch** for advanced components:

#### Core Causal Inference: X-Learner with DML (econml)
The system uses **X-Learner** combined with **Double Machine Learning (DML)** for robust causal inference:

**Double Machine Learning (DML)**
DML addresses confounding variables by using machine learning in two stages:

1. **Stage 1**: Predict treatment (macro shock) from confounders
2. **Stage 1**: Predict outcome (asset returns) from confounders  
3. **Stage 2**: Estimate causal effect on residuals

```python
# DML eliminates bias from confounding variables
fed_rate_residual = actual_fed_rate - predicted_fed_rate
sp500_residual = actual_sp500_returns - predicted_sp500_returns
causal_effect = regression(sp500_residual ~ fed_rate_residual)
```

**X-Learner**
X-Learner extends DML to handle heterogeneous treatment effects:

- **Learner 1**: Estimate outcome model for treated units
- **Learner 2**: Estimate outcome model for control units
- **Learner 3**: Estimate treatment effect model
- **Final Effect**: Weighted combination of all learners

#### Advanced Components: PyTorch

**Regime Classification**
Identifies different market states where causal relationships behave differently:

```python
class AttentionRegimeClassifier(nn.Module):
    def __init__(self, input_size=20, hidden_size=32, n_regimes=3):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, n_regimes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        embedded = self.embedding(x).unsqueeze(0)
        attended, _ = self.attention(embedded, embedded, embedded)
        return self.classifier(attended.squeeze(0))
```

**Uncertainty Estimation**
Provides confidence levels for causal effect estimates:

```python
class UncertaintyEstimator(nn.Module):
    def __init__(self, input_size=20, hidden_size=32):
        super().__init__()
        self.uncertainty_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Softplus()  # Ensures positive uncertainty
        )
    
    def forward(self, x):
        return self.uncertainty_net(x)
```

### Treatment Variables (Macro Shocks)

#### Fed Rate Shock
```python
fed_shock = fed_rate_current - fed_rate_previous
```

#### CPI Surprise
```python
cpi_trend = rolling_mean(cpi, 12_months)
cpi_shock = (cpi_actual - cpi_trend) / cpi_trend
```

#### GDP Surprise
```python
gdp_trend = rolling_mean(gdp, 8_quarters)
gdp_shock = (gdp_actual - gdp_trend) / gdp_trend
```

### Feature Engineering

#### Macro Features
- **Growth Rates**: GDP, CPI, employment growth
- **Interest Rates**: Real rates, yield curve slope
- **Volatility**: Rolling standard deviations
- **Lagged Variables**: 1-month and 3-month lags

#### Market Features
- **Asset Returns**: Monthly returns for major assets
- **Volatility**: VIX and asset-specific volatility
- **Correlations**: Rolling correlations between assets

## 🚀 Implementation Plan (4 Days)

### Day 1: Data Pipeline & Feature Engineering

#### Tasks
- [ ] Set up data collection from existing sources
- [ ] Implement feature engineering pipeline
- [ ] Create treatment variable definitions
- [ ] Align data frequencies (monthly)

#### Code Structure
```python
# data_preparation.py
class MacroCausalDataProcessor:
    def prepare_training_data(self):
        # Load FRED, Yahoo Finance, World Bank data
        # Create macro shocks (treatments)
        # Engineer features for DML
        # Generate asset returns (outcomes)
```

### Day 2: Hybrid Model Training

#### Tasks
- [ ] Train core X-Learner with DML (econml)
- [ ] Implement PyTorch regime classifier
- [ ] Implement PyTorch uncertainty estimator
- [ ] Integrate components and validate

#### Code Structure
```python
# hybrid_causal_system.py
class HybridCausalSystem:
    def __init__(self):
        # Core causal inference with econml
        self.causal_model = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100),
            model_t=RandomForestRegressor(n_estimators=100),
            n_estimators=100
        )
        
        # PyTorch components
        self.regime_classifier = AttentionRegimeClassifier()
        self.uncertainty_estimator = UncertaintyEstimator()
    
    def train(self, X, T, Y):
        # Train causal model (econml)
        self.causal_model.fit(Y, T, X=X)
        
        # Train PyTorch components with profiler
        self._train_regime_classifier(X, Y)
        self._train_uncertainty_estimator(X, T, Y)
```

### Day 3: FastAPI Integration & OpenAI Text Generation

#### Tasks
- [ ] Build FastAPI backend
- [ ] Integrate hybrid causal system
- [ ] Add OpenAI text generation with regime/uncertainty context
- [ ] Implement error handling

#### Code Structure
```python
# main.py
@app.post("/causal-analysis")
async def get_causal_analysis(query: CausalQuery):
    # Run hybrid inference (econml + PyTorch)
    inference = HybridCausalInference(causal_system)
    results = inference.get_comprehensive_analysis(X_new, T_new)
    
    # Generate enhanced explanation with regime/uncertainty
    explanation = generate_enhanced_explanation(query, results)
    
    return {
        "causal_effect": results['causal_effect'],
        "regime": results['dominant_regime'],
        "uncertainty": results['uncertainty'],
        "explanation": explanation
    }
```

### Day 4: Frontend & Deployment

#### Tasks
- [ ] Create React frontend
- [ ] Build query interface
- [ ] Add visualization components
- [ ] Deploy to AWS

## 📈 Model Performance & Validation

### Training Data
- **Sample Size**: ~300-500 monthly observations
- **Time Period**: 10+ years of historical data
- **Features**: 15-20 engineered macro variables
- **Treatments**: 3-5 macro shock types
- **Outcomes**: 5-8 asset classes

### Validation Strategy
- **Time-Series Split**: 80% training, 20% testing
- **Walk-Forward Validation**: No look-ahead bias
- **Cross-Validation**: K-fold for hyperparameter tuning
- **Backtesting**: Historical performance evaluation

### Performance Metrics
- **Causal Effect Accuracy**: Correlation between predicted and actual effects
- **Statistical Significance**: P-values and confidence intervals
- **Economic Significance**: Magnitude of effects relative to transaction costs
- **Regime Stability**: Effect consistency across different market conditions

## 🔧 Technical Implementation

### Dependencies
```python
# requirements.txt
# Core causal inference
econml==0.14.1          # X-Learner with DML
scikit-learn==1.3.0     # Traditional ML models

# PyTorch ecosystem
torch==2.1.0            # PyTorch for custom components
torchvision==0.16.0     # For torch.profiler
onnx==1.15.0            # ONNX export
onnxruntime==1.16.0     # ONNX inference

# Data and API
pandas==2.0.3           # Data manipulation
numpy==1.24.3           # Numerical computing
fastapi==0.104.1        # API framework
openai==1.0.0           # Text generation
fredapi==0.5.1          # FRED data access
yfinance==0.2.18        # Market data
```

### Model Configuration
```python
# Hybrid system configuration
hybrid_config = {
    # Core causal inference (econml)
    'causal_model': {
        'model_y': RandomForestRegressor(n_estimators=100, random_state=42),
        'model_t': RandomForestRegressor(n_estimators=100, random_state=42),
        'n_estimators': 100,
        'min_samples_leaf': 10,
        'max_depth': 10
    },
    
    # PyTorch regime classifier
    'regime_classifier': {
        'input_size': 20,
        'hidden_size': 32,
        'n_regimes': 3,  # High Vol/Recession, Low Vol/Expansion, Normal
        'attention_heads': 4,
        'dropout': 0.3
    },
    
    # PyTorch uncertainty estimator
    'uncertainty_estimator': {
        'input_size': 20,
        'hidden_size': 32,
        'dropout': 0.3
    }
}
```

### API Endpoints
```python
POST /causal-analysis
{
    "macro_variable": "fed_shock",
    "asset": "sp500_returns", 
    "magnitude": 1.0,
    "query_text": "What happens to S&P 500 when Fed raises rates by 1%?"
}

Response:
{
    "causal_analysis": {
        "effect": -0.023,
        "confidence_interval": [-0.035, -0.011],
        "regime": 1,
        "regime_probabilities": [0.1, 0.7, 0.2],
        "uncertainty": 0.15,
        "regime_effects": {
            "regime_0": -0.045,  # High Vol/Recession
            "regime_1": -0.018,  # Low Vol/Expansion
            "regime_2": -0.025   # Normal
        }
    },
    "explanation": "A 1% Fed rate hike causally reduces S&P 500 returns by 2.3%...",
    "methodology": "Hybrid: X-Learner DML + PyTorch Regime/Uncertainty"
}
```

## 🎯 Use Cases & Examples

### Example 1: Fed Rate Shock Analysis
**Query**: "What's the causal effect of a 1% Fed rate hike on S&P 500 returns?"

**Response**: 
- **Causal Effect**: -2.3% (significant at 5% level)
- **Confidence Interval**: [-3.5%, -1.1%]
- **Market Regime**: Low Volatility/Expansion (70% probability)
- **Uncertainty**: Low (0.15)
- **Regime-Dependent Effects**: 
  - High Vol/Recession: -4.5% (stronger effect)
  - Low Vol/Expansion: -1.8% (weaker effect)
  - Normal: -2.5% (baseline effect)
- **Explanation**: "A 1% Fed rate hike causally reduces S&P 500 returns by 2.3% in the current low-volatility expansion regime. The effect is stronger during high-volatility recession periods (-4.5%) and weaker during expansion periods (-1.8%)."

### Example 2: CPI Surprise Analysis
**Query**: "How do CPI surprises affect bond returns?"

**Response**:
- **Causal Effect**: -1.8% (significant at 5% level)
- **Confidence Interval**: [-2.9%, -0.7%]
- **Market Regime**: High Volatility/Recession (65% probability)
- **Uncertainty**: High (0.25)
- **Regime-Dependent Effects**:
  - High Vol/Recession: -2.8% (stronger effect)
  - Low Vol/Expansion: -1.2% (weaker effect)
- **Explanation**: "CPI surprises causally reduce bond returns by 1.8%, with stronger effects during high-volatility recession periods. Current uncertainty is elevated due to market stress."

### Example 3: Regime Classification Analysis
**Query**: "What market regime are we currently in and how does it affect causal relationships?"

**Response**:
- **Current Regime**: Low Volatility/Expansion (70% probability)
- **Regime Features**: Low VIX, positive GDP growth, stable inflation
- **Causal Implications**: Fed policy effects are muted in expansion regime
- **Risk Management**: Monitor for regime transition to high volatility
- **Explanation**: "The system identifies a low-volatility expansion regime where causal effects of macro shocks are typically smaller and more predictable. This suggests a favorable environment for risk assets."

## 🔍 Model Interpretation

### Causal Effects vs Correlations
- **Correlation**: Statistical association between variables
- **Causal Effect**: True causal relationship after controlling for confounders
- **DML Advantage**: Eliminates bias from omitted variables

### Heterogeneous Treatment Effects
- **Average Effect**: Overall causal effect across all conditions
- **Conditional Effects**: Effects that vary by market conditions
- **Regime-Dependent Effects**: Different causal relationships in different market states
- **X-Learner Advantage**: Captures regime-dependent relationships
- **PyTorch Enhancement**: Attention mechanisms identify regime-specific patterns

### Statistical Significance
- **P-values**: Probability of observing effect by chance
- **Confidence Intervals**: Range of plausible effect values
- **Economic Significance**: Magnitude relative to transaction costs

## 🚀 Deployment & Scaling

### AWS Infrastructure
- **S3**: Data lake (Bronze/Silver/Gold)
- **Lambda**: Data collection and processing
- **ECS**: FastAPI inference service
- **DynamoDB**: Model registry
- **CloudWatch**: Monitoring and logging

### Performance Optimization
- **Model Caching**: Pre-computed effects for common queries
- **Batch Processing**: Efficient handling of multiple requests
- **Async Processing**: Non-blocking inference pipeline
- **PyTorch Profiling**: torch.profiler for performance analysis
- **TorchScript Optimization**: Compiled models for faster inference
- **ONNX Deployment**: Cross-platform model serving

### Monitoring & Maintenance
- **Model Drift**: Monitor for changes in causal relationships
- **Data Quality**: Automated validation of incoming data
- **Performance Metrics**: Track inference speed and accuracy

## 📚 References & Further Reading

### Academic Papers
- Künzel, S. R., et al. (2019). "Metalearners for estimating heterogeneous treatment effects using machine learning." PNAS
- Chernozhukov, V., et al. (2018). "Double/debiased machine learning for treatment and structural parameters." The Econometrics Journal
- Vaswani, A., et al. (2017). "Attention is all you need." NIPS

### Technical Documentation
- [EconML Documentation](https://econml.azurewebsites.net/)
- [Double Machine Learning Tutorial](https://econml.azurewebsites.net/spec/estimation/dml.html)
- [X-Learner Implementation Guide](https://econml.azurewebsites.net/spec/estimation/dml.html#causal-forests)
- [PyTorch Profiler](https://pytorch.org/tutorials/beginner/profiler.html)
- [TorchScript Tutorial](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)
- [ONNX Export Guide](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)

### Bridgewater Resources
- Bridgewater Associates Research Papers
- Ray Dalio's "Principles for Navigating Big Debt Crises"
- Bridgewater's "Pure Alpha" methodology

## 🤝 Contributing

### Development Guidelines
1. **Causal Rigor**: Maintain scientific standards for causal inference
2. **Code Quality**: Follow PEP 8 and type hints
3. **Documentation**: Update docs for all new features
4. **Testing**: Add unit tests for new functionality

### Testing Strategy
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline testing
- **Causal Validation**: Statistical validation of causal effects
- **Performance Tests**: Load testing for API endpoints

## 📄 License

This project is proprietary and confidential. All rights reserved.

---

**Built with ❤️ for Bridgewater Associates AI Lab**

*This hybrid system demonstrates the power of combining proven causal inference methodology with cutting-edge PyTorch technology, providing the analytical rigor and technical sophistication that Bridgewater Associates values in their investment process.*
