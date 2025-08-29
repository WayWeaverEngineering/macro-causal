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

### X-Learner with Double Machine Learning

The system uses **X-Learner** combined with **Double Machine Learning (DML)** for robust causal inference:

#### Double Machine Learning (DML)
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

#### X-Learner
X-Learner extends DML to handle heterogeneous treatment effects:

- **Learner 1**: Estimate outcome model for treated units
- **Learner 2**: Estimate outcome model for control units
- **Learner 3**: Estimate treatment effect model
- **Final Effect**: Weighted combination of all learners

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

### Day 2: X-Learner with DML Model Training

#### Tasks
- [ ] Implement X-Learner with DML model
- [ ] Train model on historical data
- [ ] Implement time-series validation
- [ ] Add model performance metrics

#### Code Structure
```python
# causal_model.py
class XLearnerCausalModel:
    def __init__(self):
        self.model = CausalForestDML(
            model_y=RandomForestRegressor(),
            model_t=RandomForestRegressor(),
            n_estimators=100
        )
    
    def fit(self, X, T, Y):
        # Train X-Learner with DML
    
    def estimate_causal_effect(self, X_new=None):
        # Estimate causal effects
```

### Day 3: FastAPI Integration & OpenAI Text Generation

#### Tasks
- [ ] Build FastAPI backend
- [ ] Integrate X-Learner model
- [ ] Add OpenAI text generation
- [ ] Implement error handling

#### Code Structure
```python
# main.py
@app.post("/causal-analysis")
async def get_causal_analysis(query: CausalQuery):
    # Run X-Learner inference
    # Generate OpenAI explanation
    # Return structured response
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
econml==0.14.1          # Causal inference library
scikit-learn==1.3.0     # Machine learning
pandas==2.0.3           # Data manipulation
numpy==1.24.3           # Numerical computing
fastapi==0.104.1        # API framework
openai==1.0.0           # Text generation
fredapi==0.5.1          # FRED data access
yfinance==0.2.18        # Market data
```

### Model Configuration
```python
# X-Learner with DML settings
model_config = {
    'model_y': RandomForestRegressor(n_estimators=100, random_state=42),
    'model_t': RandomForestRegressor(n_estimators=100, random_state=42),
    'n_estimators': 100,
    'min_samples_leaf': 10,
    'max_depth': 10,
    'random_state': 42
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
    "causal_effect": -0.023,
    "confidence_interval": [-0.035, -0.011],
    "explanation": "A 1% Fed rate hike causally reduces S&P 500 returns by 2.3%...",
    "methodology": "X-Learner with Double Machine Learning"
}
```

## 🎯 Use Cases & Examples

### Example 1: Fed Rate Shock Analysis
**Query**: "What's the causal effect of a 1% Fed rate hike on S&P 500 returns?"

**Response**: 
- **Causal Effect**: -2.3% (significant at 5% level)
- **Confidence Interval**: [-3.5%, -1.1%]
- **Explanation**: "A 1% Fed rate hike causally reduces S&P 500 returns by 2.3% after controlling for inflation, GDP growth, and other confounding factors. This effect is statistically significant and economically meaningful."

### Example 2: CPI Surprise Analysis
**Query**: "How do CPI surprises affect bond returns?"

**Response**:
- **Causal Effect**: -1.8% (significant at 5% level)
- **Confidence Interval**: [-2.9%, -0.7%]
- **Explanation**: "Unexpected inflation increases causally reduce bond returns by 1.8%, reflecting the negative relationship between inflation and bond prices."

### Example 3: Regime-Dependent Effects
**Query**: "How do GDP shocks affect gold in high vs low volatility periods?"

**Response**:
- **High Volatility**: -0.5% effect
- **Low Volatility**: +0.2% effect
- **Explanation**: "GDP shocks have different causal effects on gold depending on market volatility, suggesting regime-dependent relationships."

## 🔍 Model Interpretation

### Causal Effects vs Correlations
- **Correlation**: Statistical association between variables
- **Causal Effect**: True causal relationship after controlling for confounders
- **DML Advantage**: Eliminates bias from omitted variables

### Heterogeneous Treatment Effects
- **Average Effect**: Overall causal effect across all conditions
- **Conditional Effects**: Effects that vary by market conditions
- **X-Learner Advantage**: Captures regime-dependent relationships

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

### Monitoring & Maintenance
- **Model Drift**: Monitor for changes in causal relationships
- **Data Quality**: Automated validation of incoming data
- **Performance Metrics**: Track inference speed and accuracy

## 📚 References & Further Reading

### Academic Papers
- Künzel, S. R., et al. (2019). "Metalearners for estimating heterogeneous treatment effects using machine learning." PNAS
- Chernozhukov, V., et al. (2018). "Double/debiased machine learning for treatment and structural parameters." The Econometrics Journal

### Technical Documentation
- [EconML Documentation](https://econml.azurewebsites.net/)
- [Double Machine Learning Tutorial](https://econml.azurewebsites.net/spec/estimation/dml.html)
- [X-Learner Implementation Guide](https://econml.azurewebsites.net/spec/estimation/dml.html#causal-forests)

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

*This system demonstrates the power of causal inference in macro-finance, providing the analytical rigor that Bridgewater Associates values in their investment process.*
