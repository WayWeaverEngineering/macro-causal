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

# FRED API configuration
FRED_BASE_URL = 'https://api.stlouisfed.org/fred/series/observations'
FRED_SERIES = [
    'GDP',           # Gross Domestic Product
    'UNRATE',        # Unemployment Rate
    'CPIAUCSL',      # Consumer Price Index
    'FEDFUNDS',      # Federal Funds Rate
    'DGS10',         # 10-Year Treasury Rate
    'DEXUSEU',       # US/Euro Exchange Rate
    'DEXCHUS',       # China/US Exchange Rate
    'DCOILWTICO',    # WTI Crude Oil Price
    'DGS2',          # 2-Year Treasury Rate
    'PAYEMS'         # Total Nonfarm Payrolls
]

def get_api_key() -> str:
    """Retrieve FRED API key from Secrets Manager"""
    try:
        response = secrets_client.get_secret_value(SecretId=API_SECRETS_ARN)
        secrets = json.loads(response['SecretString'])
        return secrets.get('FRED_API_KEY', '')
    except Exception as e:
        logger.error(f"Error retrieving API key: {e}")
        raise

def fetch_fred_data(series_id: str, api_key: str, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
    """Fetch data for a specific FRED series"""
    try:
        params = {
            'series_id': series_id,
            'api_key': api_key,
            'file_type': 'json',
            'sort_order': 'asc'
        }
        
        if start_date:
            params['observation_start'] = start_date
        if end_date:
            params['observation_end'] = end_date
            
        logger.info(f"Fetching data for series: {series_id}")
        response = requests.get(FRED_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        return data
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data for series {series_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error for series {series_id}: {e}")
        raise

def process_fred_data(raw_data: Dict[str, Any], series_id: str) -> pd.DataFrame:
    """Process raw FRED data into a structured DataFrame"""
    try:
        observations = raw_data.get('observations', [])
        
        if not observations:
            logger.warning(f"No observations found for series {series_id}")
            return pd.DataFrame()
        
        # Extract data points
        data_points = []
        for obs in observations:
            data_points.append({
                'date': obs.get('date'),
                'value': obs.get('value'),
                'realtime_start': obs.get('realtime_start'),
                'realtime_end': obs.get('realtime_end')
            })
        
        df = pd.DataFrame(data_points)
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Convert value column to numeric, handling 'None' values
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Add metadata
        df['series_id'] = series_id
        df['source'] = 'FRED'
        df['collection_timestamp'] = datetime.utcnow().isoformat()
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing data for series {series_id}: {e}")
        raise

def save_to_s3(df: pd.DataFrame, series_id: str, bucket: str) -> str:
    """Save DataFrame to S3 as JSON"""
    try:
        if df.empty:
            logger.warning(f"No data to save for series {series_id}")
            return ""
        
        # Create filename with timestamp
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"raw/fred/{series_id}/{timestamp}_{series_id}.json"
        
        # Convert DataFrame to JSON
        json_data = df.to_json(orient='records', date_format='iso')
        
        # Upload to S3
        s3_client.put_object(
            Bucket=bucket,
            Key=filename,
            Body=json_data,
            ContentType='application/json'
        )
        
        logger.info(f"Successfully saved {len(df)} records for series {series_id} to s3://{bucket}/{filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Error saving data for series {series_id}: {e}")
        raise

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Main Lambda handler function"""
    try:
        logger.info("Starting FRED data collection")
        
        # Get API key
        api_key = get_api_key()
        if not api_key:
            raise ValueError("FRED API key not found in Secrets Manager")
        
        # Determine date range (last 30 days by default)
        end_date = datetime.utcnow().strftime('%Y-%m-%d')
        start_date = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        # Override with event parameters if provided
        if 'start_date' in event:
            start_date = event['start_date']
        if 'end_date' in event:
            end_date = event['end_date']
        
        results = {
            'success': [],
            'failed': [],
            'total_series': len(FRED_SERIES),
            'start_date': start_date,
            'end_date': end_date,
            'collection_timestamp': datetime.utcnow().isoformat()
        }
        
        # Process each series
        for series_id in FRED_SERIES:
            try:
                logger.info(f"Processing series: {series_id}")
                
                # Fetch data from FRED API
                raw_data = fetch_fred_data(series_id, api_key, start_date, end_date)
                
                # Process the data
                df = process_fred_data(raw_data, series_id)
                
                if not df.empty:
                    # Save to S3
                    s3_key = save_to_s3(df, series_id, BRONZE_BUCKET)
                    
                    results['success'].append({
                        'series_id': series_id,
                        'records_count': len(df),
                        's3_key': s3_key,
                        'date_range': f"{df['date'].min()} to {df['date'].max()}"
                    })
                else:
                    results['failed'].append({
                        'series_id': series_id,
                        'error': 'No data returned'
                    })
                
                # Rate limiting - FRED allows 120 requests per minute
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Failed to process series {series_id}: {e}")
                results['failed'].append({
                    'series_id': series_id,
                    'error': str(e)
                })
        
        logger.info(f"FRED data collection completed. Success: {len(results['success'])}, Failed: {len(results['failed'])}")
        
        return {
            'statusCode': 200,
            'body': json.dumps(results),
            'headers': {
                'Content-Type': 'application/json'
            }
        }
        
    except Exception as e:
        logger.error(f"Error in FRED data collection: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'message': 'Internal server error during FRED data collection'
            }),
            'headers': {
                'Content-Type': 'application/json'
            }
        }
