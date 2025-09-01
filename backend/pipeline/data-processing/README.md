# Data Processing Pipeline

This directory contains the data processing pipeline that transforms raw data from the bronze bucket into cleaned data (silver) and training features (gold) for the macro causal inference models.

## Overview

The data processing pipeline consists of two main stages:

1. **Bronze to Silver Processing**: Cleans and validates raw data from multiple sources
2. **Silver to Gold Processing**: Creates comprehensive training features for machine learning models

## Architecture

The pipeline is designed to run on EMR Serverless and processes data from three main sources:

- **FRED (Federal Reserve Economic Data)**: Economic indicators with daily frequency
- **World Bank API**: Global development indicators with annual frequency
- **Yahoo Finance**: Financial market data with daily frequency

## Data Flow

```
Bronze Bucket (Raw Data)
    ↓
Bronze to Silver Processing
    ↓
Silver Bucket (Cleaned Data)
    ↓
Silver to Gold Processing
    ↓
Gold Bucket (Training Features)
```

## Features Created

### Economic Features (FRED)
- **Lagged Features**: 1, 7, 30, 90-day lags for all economic indicators
- **Rolling Statistics**: Mean, standard deviation, min/max over 7, 30, 90-day windows
- **Rate of Change**: Percentage change and log returns over various periods
- **Interaction Features**: Yield curve spread, Phillips curve proxy, labor productivity

### Financial Features (Yahoo Finance)
- **Returns**: Daily, weekly, monthly returns for all assets
- **Volatility**: Rolling and realized volatility measures
- **Technical Indicators**: Moving averages, price ratios
- **Market Features**: Market-wide volatility, correlation indices, VIX-based features

### World Bank Features
- **Cross-Country Comparisons**: Relative to US metrics, differences
- **Long-term Trends**: Rolling statistics over 1-3 year windows
- **Economic Ratios**: Debt-to-GDP, trade ratios, etc.

### Regime Features
- **Economic Regimes**: Recession, stable, expansion based on GDP growth
- **Inflation Regimes**: Deflation, stable, high inflation
- **Monetary Regimes**: Zero lower bound, accommodative, restrictive
- **Volatility Regimes**: Low, medium, high market volatility

### Treatment Features
- **Policy Interventions**: Interest rate changes, QE indicators
- **Fiscal Policy**: High debt regime indicators
- **Market Stress**: Volatility-based stress indicators

## File Structure

```
data-processing/
├── main.py                 # Main pipeline orchestration
├── feature_engineering.py  # Feature creation and engineering
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container definition
├── .dockerignore          # Docker build exclusions
└── README.md              # This file
```

## Usage

### Command Line Arguments

```bash
python main.py \
    --bronze-bucket <bronze-bucket-name> \
    --silver-bucket <silver-bucket-name> \
    --gold-bucket <gold-bucket-name> \
    --execution-id <unique-execution-id>
```

### Environment Variables

- `AWS_REGION`: AWS region for S3 operations
- `AWS_DEFAULT_REGION`: Default AWS region

## Output Structure

### Silver Bucket
```
silver/{execution_id}/
├── fred/
│   └── {timestamp}_{series_id}.parquet
├── worldbank/
│   └── {timestamp}_{country}_{indicator}.parquet
└── yahoo_finance/
    └── {timestamp}_{symbol}_price.parquet
```

### Gold Bucket
```
gold/{execution_id}/
├── processed_data.parquet          # Final training dataset
└── pipeline_results/
    └── data_processing/
        └── {execution_id}_results.json
```

## Data Quality

The pipeline implements several data quality measures:

- **Missing Value Handling**: Forward-filling for time series, zero-filling for features
- **Data Type Validation**: Ensures proper datetime and numeric types
- **Outlier Handling**: Replaces infinite values with zeros
- **Consistency Checks**: Validates date ranges and data completeness

## Performance Considerations

- **Batch Processing**: Processes files in batches to manage memory
- **Error Handling**: Continues processing even if individual files fail
- **Logging**: Comprehensive logging for monitoring and debugging
- **S3 Optimization**: Uses efficient S3 operations for large datasets

## Integration with Model Training

The processed features in the gold bucket are designed for:

1. **X-Learner with DML**: Treatment effects estimation using econml
2. **Regime Classifier**: Self-attention model for regime classification using PyTorch

## Monitoring and Logging

The pipeline provides:

- **Progress Logging**: Step-by-step progress updates
- **Error Tracking**: Comprehensive error collection and reporting
- **Performance Metrics**: Processing statistics and timing information
- **S3 Results**: Pipeline results saved to S3 for analysis

## Error Handling

The pipeline implements robust error handling:

- **Graceful Degradation**: Continues processing even with partial failures
- **Error Classification**: Categorizes errors by source and severity
- **Recovery Mechanisms**: Implements retry logic for transient failures
- **Status Reporting**: Provides detailed status information for each stage

## Future Enhancements

Potential improvements include:

- **Incremental Processing**: Process only new/changed data
- **Data Validation**: Schema validation and data quality scoring
- **Performance Optimization**: Parallel processing and caching
- **Monitoring Integration**: CloudWatch metrics and alarms
