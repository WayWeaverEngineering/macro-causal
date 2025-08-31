# Data Collection Pipeline

This directory contains the consolidated data collection pipeline that consolidates data collection from multiple sources:

- **FRED (Federal Reserve Economic Data)**: Economic indicators and time series data
- **World Bank API**: Global development indicators for major economies
- **Yahoo Finance**: Financial market data including stocks, indices, ETFs, commodities, and currencies

## Architecture

The data collection pipeline is designed to run as a containerized application on ECS Fargate. It consolidates the functionality previously implemented as separate Lambda functions into a single, scalable application.

### Key Features

- **Unified Data Collection**: Single application handles all data sources
- **Fault Tolerance**: Individual source failures don't stop the entire pipeline
- **Rate Limiting**: Built-in rate limiting to respect API limits
- **Structured Output**: Consistent data format across all sources
- **S3 Storage**: All data is stored in the bronze bucket with organized folder structure
- **Comprehensive Logging**: Detailed logging for monitoring and debugging

## Data Sources

### FRED (Federal Reserve Economic Data)
- **Series**: GDP, Unemployment Rate, CPI, Federal Funds Rate, Treasury Rates, Exchange Rates, Oil Prices, Payrolls
- **Frequency**: Daily updates
- **API Rate Limit**: 120 requests per minute
- **Default Range**: Last 30 days

### World Bank API
- **Indicators**: GDP, GDP Growth, Unemployment, Inflation, Interest Rates, Exchange Rates, Trade, Debt, Savings
- **Countries**: US, China, Japan, Germany, UK, France, India, Italy, Canada, Brazil
- **Frequency**: Annual updates
- **API Rate Limit**: 100 requests per minute
- **Default Range**: Last 10 years

### Yahoo Finance
- **Assets**: Major indices, ETFs, stocks, commodities, currencies
- **Data Types**: Price data (OHLCV) and company information
- **Frequency**: Daily updates
- **API Rate Limit**: Respectful rate limiting (1 second between requests)
- **Default Range**: Last 30 days

## File Structure

```
data-collection/
├── main.py              # Main application code
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker image definition
├── .dockerignore       # Docker build exclusions
└── README.md           # This file
```

## Environment Variables

The following environment variables must be set:

- `BRONZE_BUCKET`: S3 bucket name for storing raw data
- `API_SECRETS_ARN`: ARN of Secrets Manager secret containing API keys
- `ENVIRONMENT`: Environment name (default: 'dev')

### API Keys Required

- `FRED_API_KEY`: FRED API key (required)
- `WORLD_BANK_API_KEY`: World Bank API key (optional, public access available)
- `YAHOO_FINANCE_API_KEY`: Yahoo Finance API key (optional, public access available)

## Usage

### Basic Usage

```bash
# Run with default settings
python main.py
```

**Note**: This application is designed to run in ECS Fargate containers. For local development and testing, you can run it directly with Python.

## Docker

### Building the Image

```bash
# Build the Docker image
docker build -t data-collection-pipeline .

# Build with specific tag
docker build -t data-collection-pipeline:v1.0.0 .
```

### Running the Container

```bash
# Run with environment variables
docker run -e BRONZE_BUCKET=my-bucket \
           -e API_SECRETS_ARN=arn:aws:secretsmanager:... \
           data-collection-pipeline
```

**Note**: In production, this container will be managed by ECS Fargate with environment variables configured through task definitions.

## Data Output

### S3 Structure

Data is organized in the following S3 structure:

```
s3://{BRONZE_BUCKET}/
├── raw/
│   ├── fred/
│   │   ├── GDP/
│   │   ├── UNRATE/
│   │   └── ...
│   ├── worldbank/
│   │   ├── US/
│   │   │   ├── NY.GDP.MKTP.CD/
│   │   │   └── ...
│   │   └── ...
│   └── yahoo_finance/
│       ├── ^GSPC/
│       │   ├── price/
│       │   └── info/
│       └── ...
└── pipeline_results/
    └── data_collection/
        └── {timestamp}_pipeline_results.json
```

### Data Format

All data is stored as JSON with consistent metadata:

```json
{
  "date": "2024-01-01T00:00:00",
  "value": 123.45,
  "source": "FRED",
  "collection_timestamp": "2024-01-01T12:00:00Z",
  "series_id": "GDP"
}
```

## Monitoring and Logging

### Log Levels

- **INFO**: General pipeline progress and successful operations
- **WARNING**: Non-critical issues (e.g., missing data for specific series)
- **ERROR**: Critical failures that prevent data collection

### Exit Codes

- **0**: Success - all data sources collected successfully
- **1**: Failure - one or more data sources failed

### Health Checks

The container includes a health check that can be used by ECS to monitor the application status.

## Deployment

### ECS Fargate

This application is designed to run on ECS Fargate with the following considerations:

- **CPU**: 1 vCPU (256 CPU units) minimum, 2 vCPU recommended for production
- **Memory**: 2 GB minimum, 4 GB recommended for production
- **Task Definition**: Should include environment variables and IAM roles
- **Auto Scaling**: Can be configured based on CloudWatch metrics

### IAM Permissions

The ECS task role requires the following permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::${BRONZE_BUCKET}",
        "arn:aws:s3:::${BRONZE_BUCKET}/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
      "Resource": "${API_SECRETS_ARN}"
    }
  ]
}
```

## Development

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export BRONZE_BUCKET=my-dev-bucket
export API_SECRETS_ARN=arn:aws:secretsmanager:...

# Run the application
python main.py
```

**Note**: For production deployment, this application will be built into a Docker image by CodeBuild and deployed to ECS Fargate automatically.

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure FRED_API_KEY is set in Secrets Manager
2. **S3 Permissions**: Verify the ECS task role has proper S3 permissions
3. **Rate Limiting**: Check logs for rate limit errors and adjust timing if needed
4. **Network Issues**: Ensure the ECS task has proper network access to external APIs

### Debug Mode

Enable debug logging by setting the log level:

```python
logging.basicConfig(level=logging.DEBUG)
```

## Performance Considerations

- **Parallel Processing**: Currently runs sequentially to respect API rate limits
- **Memory Usage**: Monitor memory usage for large datasets
- **Execution Time**: Typical runtime is 5-15 minutes depending on data volume
- **S3 Uploads**: Large datasets are uploaded in chunks to avoid memory issues

## Future Enhancements

- **Parallel Collection**: Implement parallel data collection for independent sources
- **Incremental Updates**: Only collect new/changed data
- **Data Validation**: Add data quality checks and validation
- **Metrics**: Enhanced CloudWatch metrics and dashboards
- **Alerting**: SNS notifications for failures and completion
