# API Data Collection for Macro Causal Analysis

This document describes the API-based data collection system for economic indicators and market data used in the Macro Causal analysis project.

## Overview

The API data collection system automatically collects data from three major sources:
- **FRED (Federal Reserve Economic Data)** - Economic indicators
- **World Bank API** - Global economic indicators
- **Yahoo Finance** - Market data and financial instruments

## Architecture

### Components

1. **VPC Stack** (`VPCStack`) - Provides networking infrastructure
2. **API Data Collection Stack** (`APIDataCollectionStack`) - Orchestrates data collection
3. **Lambda Functions** - Individual collectors for each data source
4. **Step Functions** - Workflow orchestration
5. **EventBridge** - Scheduled triggers
6. **Secrets Manager** - API key management
7. **S3** - Data storage

### Data Flow

```
EventBridge (Scheduled) → Step Functions → Lambda Functions → S3 Bronze Bucket
```

## Data Sources

### FRED (Federal Reserve Economic Data)

**API Endpoint**: https://api.stlouisfed.org/fred/series/observations

**Economic Indicators Collected**:
- GDP (Gross Domestic Product)
- Unemployment Rate
- Consumer Price Index (CPI)
- Federal Funds Rate
- 10-Year Treasury Rate
- US/Euro Exchange Rate
- China/US Exchange Rate
- WTI Crude Oil Price
- 2-Year Treasury Rate
- Total Nonfarm Payrolls

**Rate Limits**: 120 requests per minute

### World Bank API

**API Endpoint**: https://api.worldbank.org/v2/country

**Economic Indicators Collected**:
- GDP (current US$)
- GDP growth (annual %)
- Unemployment rate
- Inflation, consumer prices (annual %)
- Real interest rate (%)
- Official exchange rate
- Exports/Imports of goods and services (% of GDP)
- Central government debt (% of GDP)
- Gross savings (% of GDP)

**Countries Covered**:
- US, CN, JP, DE, GB, FR, IN, IT, CA, BR

**Rate Limits**: 100 requests per minute

### Yahoo Finance

**API**: yfinance library (unofficial)

**Data Collected**:
- **Major US Indices**: S&P 500, Dow Jones, NASDAQ, VIX
- **Major ETFs**: SPY, QQQ, IWM, GLD, TLT
- **Major Stocks**: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, BRK-B, JPM, JNJ
- **Commodities**: Gold, Crude Oil, Natural Gas, Corn, Soybean Futures
- **Currencies**: EUR/USD, GBP/USD, USD/JPY, USD/CNY, USD/CAD

**Data Fields**: Open, High, Low, Close, Volume, Dividends, Stock Splits

## Setup Instructions

### 1. API Keys Configuration

Create API keys for the data sources and store them in AWS Secrets Manager:

```bash
# FRED API Key (Required)
aws secretsmanager update-secret \
  --secret-id "macro-causal-dev-api-secrets" \
  --secret-string '{"FRED_API_KEY":"your-fred-api-key"}'

# World Bank API Key (Optional - public access available)
aws secretsmanager update-secret \
  --secret-id "macro-causal-dev-api-secrets" \
  --secret-string '{"WORLD_BANK_API_KEY":"your-worldbank-api-key"}'

# Yahoo Finance API Key (Optional - public access available)
aws secretsmanager update-secret \
  --secret-id "macro-causal-dev-api-secrets" \
  --secret-string '{"YAHOO_FINANCE_API_KEY":"your-yahoo-finance-api-key"}'
```

### 2. Deploy Infrastructure

```bash
# Deploy VPC Stack
cd backend/cdk
npm run cdk deploy vpc-stack

# Deploy Data Lake Stack
npm run cdk deploy data-lake-stack

# Deploy API Data Collection Stack
npm run cdk deploy api-data-collection-stack
```

### 3. Verify Deployment

Check that the following resources are created:
- VPC with public/private subnets
- Lambda functions for each data collector
- Step Functions state machine
- EventBridge rules for scheduling
- Secrets Manager secret for API keys

## Usage

### Manual Trigger

Trigger data collection manually via AWS CLI:

```bash
# Trigger FRED data collection
aws lambda invoke \
  --function-name macro-causal-dev-fred-collector \
  --payload '{"start_date":"2024-01-01","end_date":"2024-01-31"}' \
  response.json

# Trigger World Bank data collection
aws lambda invoke \
  --function-name macro-causal-dev-worldbank-collector \
  --payload '{"start_date":"2020","end_date":"2024"}' \
  response.json

# Trigger Yahoo Finance data collection
aws lambda invoke \
  --function-name macro-causal-dev-yahoo-finance-collector \
  --payload '{"start_date":"2024-01-01","end_date":"2024-01-31"}' \
  response.json
```

### Scheduled Collection

Data collection runs automatically:
- **FRED**: Daily at 8 AM UTC (last 30 days)
- **World Bank**: Daily at 8 AM UTC (last 10 years)
- **Yahoo Finance**: Daily at 8 AM UTC (last 30 days)

### Step Functions Workflow

The complete workflow can be triggered via Step Functions:

```bash
aws stepfunctions start-execution \
  --state-machine-arn "arn:aws:states:region:account:stateMachine:macro-causal-dev-api-data-collection-workflow" \
  --input '{"collection_type":"api_data"}'
```

## Data Storage

### S3 Structure

Data is stored in the bronze bucket with the following structure:

```
s3://macro-causal-dev-bronze-bucket/
├── raw/
│   ├── fred/
│   │   ├── GDP/
│   │   │   └── 20240101_120000_GDP.json
│   │   ├── UNRATE/
│   │   └── ...
│   ├── worldbank/
│   │   ├── US/
│   │   │   ├── NY.GDP.MKTP.CD/
│   │   │   └── ...
│   │   ├── CN/
│   │   └── ...
│   └── yahoo_finance/
│       ├── ^GSPC/
│       │   ├── price/
│       │   └── info/
│       ├── AAPL/
│       └── ...
```

### Data Formats

#### FRED Data
```json
{
  "date": "2024-01-01T00:00:00.000Z",
  "value": 1234.56,
  "realtime_start": "2024-01-01",
  "realtime_end": "2024-01-01",
  "series_id": "GDP",
  "source": "FRED",
  "collection_timestamp": "2024-01-01T12:00:00.000Z"
}
```

#### World Bank Data
```json
{
  "date": "2024-01-01T00:00:00.000Z",
  "value": 1234.56,
  "country_code": "US",
  "indicator_code": "NY.GDP.MKTP.CD",
  "indicator_name": "GDP (current US$)",
  "unit": "US$",
  "obs_status": "",
  "decimal": 0,
  "source": "WorldBank",
  "collection_timestamp": "2024-01-01T12:00:00.000Z"
}
```

#### Yahoo Finance Data
```json
{
  "date": "2024-01-01T00:00:00.000Z",
  "open": 123.45,
  "high": 124.56,
  "low": 122.34,
  "close": 123.78,
  "volume": 1000000,
  "dividends": 0.0,
  "stock_splits": 0.0,
  "symbol": "^GSPC",
  "source": "YahooFinance",
  "collection_timestamp": "2024-01-01T12:00:00.000Z"
}
```

## Monitoring

### CloudWatch Metrics

The following metrics are available:
- Lambda function invocations and errors
- Step Functions execution success/failure
- S3 object creation events
- API response times and error rates

### CloudWatch Logs

Each Lambda function logs:
- API request details
- Data processing statistics
- Error messages and stack traces
- S3 upload confirmations

### Alerts

Configure CloudWatch alarms for:
- Lambda function errors
- Step Functions execution failures
- API rate limit violations
- Data collection timeouts

## Troubleshooting

### Common Issues

1. **API Rate Limits**: Functions include built-in rate limiting and retry logic
2. **Network Connectivity**: Lambda functions run in VPC with proper security groups
3. **Authentication**: Verify API keys are correctly stored in Secrets Manager
4. **Data Quality**: Check for missing or invalid data in S3 logs

### Debug Commands

```bash
# Check Lambda function logs
aws logs tail /aws/lambda/macro-causal-dev-fred-collector --follow

# Verify S3 data
aws s3 ls s3://macro-causal-dev-bronze-bucket/raw/fred/GDP/

# Test API connectivity
aws lambda invoke \
  --function-name macro-causal-dev-fred-collector \
  --payload '{"test":true}' \
  test-response.json
```

## Security

### Network Security
- Lambda functions run in private subnets
- VPC endpoints for AWS services
- Security groups restrict outbound traffic

### Data Security
- API keys stored in Secrets Manager
- S3 bucket encryption enabled
- IAM roles with least privilege access

### Compliance
- Data retention policies configured
- Audit logging enabled
- Access controls implemented

## Cost Optimization

### Lambda Configuration
- Memory: 1024 MB (optimized for pandas operations)
- Timeout: 15 minutes (sufficient for API calls)
- Reserved concurrency: Configure based on expected load

### S3 Storage
- Lifecycle policies for data retention
- Intelligent tiering for cost optimization
- Compression for historical data

### Monitoring Costs
- CloudWatch logs retention: 30 days
- Metrics retention: 30 days
- Alarm thresholds optimized for cost

## Future Enhancements

1. **Additional Data Sources**: Bloomberg, Reuters, custom APIs
2. **Real-time Streaming**: Kinesis integration for real-time data
3. **Data Quality Checks**: Automated validation and alerting
4. **Machine Learning**: Anomaly detection for data quality
5. **API Versioning**: Support for multiple API versions
6. **Geographic Distribution**: Multi-region data collection
