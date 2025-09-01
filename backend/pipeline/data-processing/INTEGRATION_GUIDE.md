# Data Processing Pipeline Integration Guide

This document explains how the data processing pipeline integrates with the model training stage to enable macro causal inference using X-learner with DML and regime classification with self-attention models.

## Pipeline Integration Overview

The data processing pipeline creates a bridge between raw data collection and model training by:

1. **Cleaning raw data** from the bronze bucket (data collection stage)
2. **Creating comprehensive features** for machine learning models
3. **Preparing structured datasets** for both causal inference and classification tasks

## Data Flow Integration

```
Data Collection (Bronze) → Data Processing (Silver + Gold) → Model Training
```

### Bronze Bucket (Input)
- Raw FRED economic data
- Raw World Bank development indicators  
- Raw Yahoo Finance market data

### Silver Bucket (Intermediate)
- Cleaned and validated data from all sources
- Consistent data types and formats
- Metadata and processing timestamps

### Gold Bucket (Output)
- **Training Features**: Comprehensive feature set for ML models
- **Target Variables**: GDP growth rates for causal inference
- **Regime Labels**: Economic regime classifications
- **Treatment Indicators**: Policy intervention markers

## Feature Engineering for Model Training

### X-Learner with DML Features

The pipeline creates features specifically designed for causal inference:

#### Treatment Features
- `fed_funds_increase`: Binary indicator for interest rate hikes
- `fed_funds_decrease`: Binary indicator for interest rate cuts
- `zero_lower_bound`: Indicator for zero interest rate policy
- `qe_proxy`: Quantitative easing proxy based on yield curve
- `high_debt_regime`: Fiscal policy stress indicator
- `market_stress`: Market volatility-based stress indicator

#### Outcome Features
- `target`: GDP growth rate (quarterly change)
- Economic indicators with various lags and transformations
- Financial market indicators and volatility measures

#### Control Features
- Lagged economic indicators (1, 7, 30, 90 days)
- Rolling statistics (mean, std, min, max)
- Rate of change measures (returns, percentage changes)
- Cross-country relative indicators

### Regime Classifier Features

Features designed for the self-attention regime classification model:

#### Regime Indicators
- `gdp_growth_regime`: Recession/Stable/Expansion
- `inflation_regime`: Deflation/Stable/High Inflation
- `monetary_regime`: Zero Lower Bound/Accommodative/Restrictive
- `volatility_regime`: Low/Medium/High Market Volatility

#### Regime Interactions
- Pairwise interactions between different regime types
- Temporal regime transition patterns
- Cross-regime correlation features

## Model Training Integration

### X-Learner with DML Training

The processed features enable training of:

```python
# Example usage in model training
from econml.dml import LinearDML
from econml.dr import DRLearner

# Load processed features from gold bucket
df = load_data_from_s3(gold_bucket, execution_id)

# Prepare features for causal inference
X = df[feature_columns]  # Control variables
T = df[treatment_columns]  # Treatment variables  
Y = df['target']  # Outcome variable

# Train X-learner with DML
estimator = LinearDML()
estimator.fit(Y, T, X=X)
```

### Regime Classifier Training

The regime features enable training of:

```python
# Example usage in model training
import torch
import torch.nn as nn

# Load processed features from gold bucket
df = load_data_from_s3(gold_bucket, execution_id)

# Prepare features for regime classification
X = df[feature_columns]  # All feature columns
y_regime = df['gdp_growth_regime']  # Regime labels

# Train self-attention regime classifier
model = MacroCausalModel(
    input_dim=len(feature_columns),
    hidden_dims=[128, 64, 32],
    output_dim=3  # 3 regime classes
)
```

## Data Quality Assurance

### Validation Checks
- **Completeness**: Ensures all required features are present
- **Data Types**: Validates numeric and categorical data types
- **Missing Values**: Handles missing data appropriately
- **Outliers**: Replaces infinite values and extreme outliers

### Consistency Measures
- **Date Alignment**: All features aligned to daily frequency
- **Scale Consistency**: Features normalized appropriately
- **Metadata Tracking**: Execution IDs and timestamps preserved

## Performance Optimization

### Memory Management
- **Batch Processing**: Processes data in manageable chunks
- **Efficient Storage**: Uses Parquet format for optimal I/O
- **S3 Optimization**: Minimizes S3 operations and costs

### Scalability Features
- **Modular Design**: Easy to extend with new data sources
- **Parallel Processing**: Supports concurrent data processing
- **Error Recovery**: Continues processing despite individual failures

## Monitoring and Debugging

### Pipeline Metrics
- **Processing Time**: Tracks time for each pipeline stage
- **Data Volume**: Monitors records processed at each stage
- **Error Rates**: Tracks processing failures and success rates
- **Feature Counts**: Reports number of features created

### Debugging Support
- **Detailed Logging**: Comprehensive logging at each step
- **Error Tracking**: Captures and reports all errors
- **Data Validation**: Checks data quality at each stage
- **S3 Results**: Saves pipeline results for analysis

## Future Enhancements

### Planned Improvements
- **Real-time Processing**: Stream processing for live data
- **Feature Store**: Centralized feature management
- **A/B Testing**: Support for feature experimentation
- **Auto-scaling**: Dynamic resource allocation

### Integration Extensions
- **Additional Data Sources**: More economic and financial data
- **Advanced Features**: Deep learning-based feature extraction
- **Model Monitoring**: Integration with model performance tracking
- **Automated Retraining**: Trigger model retraining on new data

## Troubleshooting

### Common Issues
1. **Missing Data**: Check data collection pipeline status
2. **Feature Errors**: Verify data types and transformations
3. **Memory Issues**: Reduce batch sizes or increase resources
4. **S3 Errors**: Check permissions and bucket access

### Debug Commands
```bash
# Test feature engineering locally
python test_pipeline.py

# Check pipeline logs
tail -f pipeline.log

# Validate S3 data
aws s3 ls s3://{bucket}/gold/{execution_id}/
```

## Conclusion

The data processing pipeline creates a robust foundation for macro causal inference by:

- **Standardizing data formats** across multiple sources
- **Creating comprehensive features** for both causal inference and classification
- **Ensuring data quality** through validation and cleaning
- **Providing monitoring** and debugging capabilities

This enables the model training stage to focus on algorithm development and hyperparameter tuning rather than data preparation and cleaning.
