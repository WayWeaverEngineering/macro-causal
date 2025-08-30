import json
import boto3
import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import logging
from typing import Dict, List, Any
import time

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
secrets_client = boto3.client('secretsmanager')

# Environment variables
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'dev')
BRONZE_BUCKET = os.environ.get('BRONZE_BUCKET')
API_SECRETS_ARN = os.environ.get('API_SECRETS_ARN')

# World Bank API configuration
WORLDBANK_BASE_URL = 'https://api.worldbank.org/v2/country'
WORLDBANK_INDICATORS = [
    {'indicator': 'NY.GDP.MKTP.CD', 'name': 'GDP (current US$)'},
    {'indicator': 'NY.GDP.MKTP.KD.ZG', 'name': 'GDP growth (annual %)'},
    {'indicator': 'SL.UEM.TOTL.ZS', 'name': 'Unemployment, total (% of total labor force)'},
    {'indicator': 'FP.CPI.TOTL.ZG', 'name': 'Inflation, consumer prices (annual %)'},
    {'indicator': 'FR.INR.RINR', 'name': 'Real interest rate (%)'},
    {'indicator': 'PA.NUS.FCRF', 'name': 'Official exchange rate (LCU per US$, period average)'},
    {'indicator': 'NE.EXP.GNFS.ZS', 'name': 'Exports of goods and services (% of GDP)'},
    {'indicator': 'NE.IMP.GNFS.ZS', 'name': 'Imports of goods and services (% of GDP)'},
    {'indicator': 'GC.DOD.TOTL.GD.ZS', 'name': 'Central government debt, total (% of GDP)'},
    {'indicator': 'NY.GNS.ICTR.ZS', 'name': 'Gross savings (% of GDP)'}
]

# Target countries (major economies)
COUNTRIES = ['US', 'CN', 'JP', 'DE', 'GB', 'FR', 'IN', 'IT', 'CA', 'BR']

def get_api_key() -> str:
    """Retrieve World Bank API key from Secrets Manager (optional)"""
    try:
        response = secrets_client.get_secret_value(SecretId=API_SECRETS_ARN)
        secrets = json.loads(response['SecretString'])
        return secrets.get('WORLD_BANK_API_KEY', '')
    except Exception as e:
        logger.warning(f"World Bank API key not found, using public access: {e}")
        return ''

def fetch_worldbank_data(country_code: str, indicator: str, api_key: str = None, 
                        start_date: str = None, end_date: str = None) -> Dict[str, Any]:
    """Fetch data for a specific World Bank indicator and country"""
    try:
        # World Bank API doesn't require authentication for basic access
        url = f"{WORLDBANK_BASE_URL}/{country_code}/indicator/{indicator}"
        
        params = {
            'format': 'json',
            'per_page': 1000  # Maximum records per request
        }
        
        if start_date:
            params['date'] = f"{start_date}:{end_date or datetime.utcnow().year}"
        
        logger.info(f"Fetching data for country: {country_code}, indicator: {indicator}")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        return data
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data for country {country_code}, indicator {indicator}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error for country {country_code}, indicator {indicator}: {e}")
        raise

def process_worldbank_data(raw_data: List[Dict[str, Any]], country_code: str, 
                          indicator: str, indicator_name: str) -> pd.DataFrame:
    """Process raw World Bank data into a structured DataFrame"""
    try:
        if not raw_data or len(raw_data) < 2:
            logger.warning(f"No data found for country {country_code}, indicator {indicator}")
            return pd.DataFrame()
        
        # World Bank API returns metadata in first element, data in second
        data_points = raw_data[1] if len(raw_data) > 1 else []
        
        if not data_points:
            logger.warning(f"No data points found for country {country_code}, indicator {indicator}")
            return pd.DataFrame()
        
        # Extract data points
        processed_data = []
        for point in data_points:
            if point.get('value') is not None:  # Only include records with values
                processed_data.append({
                    'date': point.get('date'),
                    'value': point.get('value'),
                    'country_code': country_code,
                    'indicator_code': indicator,
                    'indicator_name': indicator_name,
                    'unit': point.get('unit', ''),
                    'obs_status': point.get('obs_status', ''),
                    'decimal': point.get('decimal', 0)
                })
        
        df = pd.DataFrame(processed_data)
        
        if df.empty:
            return df
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'], format='%Y')
        
        # Convert value column to numeric
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Add metadata
        df['source'] = 'WorldBank'
        df['collection_timestamp'] = datetime.utcnow().isoformat()
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing data for country {country_code}, indicator {indicator}: {e}")
        raise

def save_to_s3(df: pd.DataFrame, country_code: str, indicator: str, bucket: str) -> str:
    """Save DataFrame to S3 as JSON"""
    try:
        if df.empty:
            logger.warning(f"No data to save for country {country_code}, indicator {indicator}")
            return ""
        
        # Create filename with timestamp
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"raw/worldbank/{country_code}/{indicator}/{timestamp}_{country_code}_{indicator}.json"
        
        # Convert DataFrame to JSON
        json_data = df.to_json(orient='records', date_format='iso')
        
        # Upload to S3
        s3_client.put_object(
            Bucket=bucket,
            Key=filename,
            Body=json_data,
            ContentType='application/json'
        )
        
        logger.info(f"Successfully saved {len(df)} records for country {country_code}, indicator {indicator} to s3://{bucket}/{filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Error saving data for country {country_code}, indicator {indicator}: {e}")
        raise

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Main Lambda handler function"""
    try:
        logger.info("Starting World Bank data collection")
        
        # Get API key (optional for World Bank)
        api_key = get_api_key()
        
        # Determine date range (last 10 years by default)
        end_date = datetime.utcnow().year
        start_date = end_date - 10
        
        # Override with event parameters if provided
        if 'start_date' in event:
            start_date = int(event['start_date'])
        if 'end_date' in event:
            end_date = int(event['end_date'])
        
        results = {
            'success': [],
            'failed': [],
            'total_combinations': len(COUNTRIES) * len(WORLDBANK_INDICATORS),
            'start_date': start_date,
            'end_date': end_date,
            'collection_timestamp': datetime.utcnow().isoformat()
        }
        
        # Process each country and indicator combination
        for country_code in COUNTRIES:
            for indicator_info in WORLDBANK_INDICATORS:
                indicator = indicator_info['indicator']
                indicator_name = indicator_info['name']
                
                try:
                    logger.info(f"Processing country: {country_code}, indicator: {indicator}")
                    
                    # Fetch data from World Bank API
                    raw_data = fetch_worldbank_data(country_code, indicator, api_key, str(start_date), str(end_date))
                    
                    # Process the data
                    df = process_worldbank_data(raw_data, country_code, indicator, indicator_name)
                    
                    if not df.empty:
                        # Save to S3
                        s3_key = save_to_s3(df, country_code, indicator, BRONZE_BUCKET)
                        
                        results['success'].append({
                            'country_code': country_code,
                            'indicator_code': indicator,
                            'indicator_name': indicator_name,
                            'records_count': len(df),
                            's3_key': s3_key,
                            'date_range': f"{df['date'].min().year} to {df['date'].max().year}"
                        })
                    else:
                        results['failed'].append({
                            'country_code': country_code,
                            'indicator_code': indicator,
                            'error': 'No data returned'
                        })
                    
                    # Rate limiting - World Bank allows 100 requests per minute
                    time.sleep(0.6)
                    
                except Exception as e:
                    logger.error(f"Failed to process country {country_code}, indicator {indicator}: {e}")
                    results['failed'].append({
                        'country_code': country_code,
                        'indicator_code': indicator,
                        'error': str(e)
                    })
        
        logger.info(f"World Bank data collection completed. Success: {len(results['success'])}, Failed: {len(results['failed'])}")
        
        return {
            'statusCode': 200,
            'body': json.dumps(results),
            'headers': {
                'Content-Type': 'application/json'
            }
        }
        
    except Exception as e:
        logger.error(f"Error in World Bank data collection: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'message': 'Internal server error during World Bank data collection'
            }),
            'headers': {
                'Content-Type': 'application/json'
            }
        }
