#!/usr/bin/env python3
"""
Base Data Collector Class
Provides common functionality for all data collectors
"""

import json
import boto3
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
import logging
import traceback
import time
from typing import Dict, Any, Optional, Union
import io
import requests
from functools import wraps

# Configure logging
logger = logging.getLogger(__name__)

# Initialize AWS clients
s3_client = boto3.client('s3')
secrets_client = boto3.client('secretsmanager')

# Environment variables
BRONZE_BUCKET = os.environ.get('BRONZE_BUCKET')
API_SECRETS_ARN = os.environ.get('API_SECRETS_ARN')

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff_factor: float = 2.0):
    """Decorator for retrying operations with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (backoff_factor ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {wait_time:.2f}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}: {e}")
            raise last_exception
        return wrapper
    return decorator

def log_api_call(func):
    """Decorator for logging API calls with detailed information"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.debug(f"API Call: {func_name} - Args: {args}, Kwargs: {kwargs}")
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(f"API Call Success: {func_name} - Duration: {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"API Call Failed: {func_name} - Duration: {duration:.2f}s - Error: {e}")
            logger.debug(f"API Call Error Details: {traceback.format_exc()}")
            raise
    return wrapper

class DataCollector:
    """Base class for data collectors with enhanced error handling and logging"""
    
    def __init__(self, collector_name: str = "Unknown"):
        self.collector_name = collector_name
        self.results = {
            'success': [],
            'failed': [],
            'collection_timestamp': datetime.now(timezone.utc).isoformat(),
            'collector_name': collector_name,
            'start_time': None,
            'end_time': None,
            'total_processed': 0,
            'total_success': 0,
            'total_failed': 0
        }
        logger.info(f"Initialized {collector_name} collector")
    
    def validate_date_range(self, start_date: str, end_date: str) -> tuple[str, str]:
        """Validate and fix date ranges to ensure they're valid"""
        try:
            # Parse dates
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Check if dates are in the future
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            
            if start_dt > now:
                logger.warning(f"Start date {start_date} is in the future, adjusting to 30 days ago")
                start_dt = now - timedelta(days=30)
                start_date = start_dt.strftime('%Y-%m-%d')
            
            if end_dt > now:
                logger.warning(f"End date {end_date} is in the future, adjusting to today")
                end_dt = now
                end_date = end_dt.strftime('%Y-%m-%d')
            
            # Ensure start_date <= end_date
            if start_dt > end_dt:
                logger.warning(f"Start date {start_date} is after end date {end_date}, swapping")
                start_date, end_date = end_date, start_date
            
            logger.info(f"Validated date range: {start_date} to {end_date}")
            return start_date, end_date
            
        except ValueError as e:
            logger.error(f"Invalid date format: {e}")
            # Return default date range
            now = datetime.now(timezone.utc)
            end_date = now.strftime('%Y-%m-%d')
            start_date = (now - timedelta(days=30)).strftime('%Y-%m-%d')
            logger.info(f"Using default date range: {start_date} to {end_date}")
            return start_date, end_date
    
    def get_default_date_range(self, days_back: int = 30) -> tuple[str, str]:
        """Get a default date range from today going back specified days"""
        now = datetime.now(timezone.utc)
        end_date = now.strftime('%Y-%m-%d')
        start_date = (now - timedelta(days=days_back)).strftime('%Y-%m-%d')
        return start_date, end_date
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def get_api_key(self, key_name: str) -> Optional[str]:
        """Retrieve API key from Secrets Manager with retry logic"""
        if not API_SECRETS_ARN:
            logger.warning(f"API_SECRETS_ARN environment variable not set. Cannot retrieve API key '{key_name}'")
            return None
            
        try:
            logger.debug(f"Retrieving API key '{key_name}' from Secrets Manager")
            response = secrets_client.get_secret_value(SecretId=API_SECRETS_ARN)
            secrets = json.loads(response['SecretString'])
            api_key = secrets.get(key_name, '')
            
            if api_key:
                logger.info(f"Successfully retrieved API key '{key_name}' from Secrets Manager")
                # Log partial key for debugging (first 4 chars)
                masked_key = api_key[:4] + '*' * (len(api_key) - 8) + api_key[-4:] if len(api_key) > 8 else '****'
                logger.debug(f"API key '{key_name}' (masked): {masked_key}")
            else:
                logger.warning(f"API key '{key_name}' not found in Secrets Manager secret")
                
            return api_key
            
        except Exception as e:
            logger.error(f"Error retrieving API key '{key_name}' from Secrets Manager: {e}")
            logger.debug(f"Secrets Manager error details: {traceback.format_exc()}")
            return None
    
    @log_api_call
    @retry_on_failure(max_retries=3, delay=1.0)
    def make_http_request(self, url: str, method: str = 'GET', params: Dict = None, 
                         headers: Dict = None, timeout: int = 30, **kwargs) -> requests.Response:
        """Make HTTP requests with comprehensive error handling and logging"""
        try:
            logger.debug(f"Making {method} request to {url}")
            if params:
                logger.debug(f"Request parameters: {params}")
            if headers:
                logger.debug(f"Request headers: {headers}")
            
            response = requests.request(
                method=method,
                url=url,
                params=params,
                headers=headers,
                timeout=timeout,
                **kwargs
            )
            
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {dict(response.headers)}")
            
            # Log response content for debugging (first 500 chars)
            if response.content:
                content_preview = response.text[:500]
                logger.debug(f"Response content preview: {content_preview}")
            
            response.raise_for_status()
            return response
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout for {url} after {timeout}s")
            raise
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error for {url}: {e}")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for {url}: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"Error response: {e.response.text}")
                # Try to parse as JSON for better formatting
                try:
                    error_json = e.response.json()
                    logger.error(f"Error response JSON: {error_json}")
                except:
                    pass
            if hasattr(e.response, 'status_code'):
                logger.error(f"HTTP Status Code: {e.response.status_code}")
            if hasattr(e.response, 'headers'):
                logger.error(f"Response Headers: {dict(e.response.headers)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error making request to {url}: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception details: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def save_to_s3(self, df: pd.DataFrame, s3_path: str, bucket: str) -> str:
        """Save DataFrame to S3 as parquet with retry logic"""
        try:
            if df.empty:
                logger.warning(f"No data to save to {s3_path}")
                return ""
            
            logger.debug(f"Saving {len(df)} records to s3://{bucket}/{s3_path}")
            
            # Convert DataFrame to parquet bytes
            parquet_buffer = io.BytesIO()
            df.to_parquet(parquet_buffer, index=False)
            parquet_buffer.seek(0)
            
            # Upload to S3
            s3_client.put_object(
                Bucket=bucket,
                Key=s3_path,
                Body=parquet_buffer.getvalue(),
                ContentType='application/octet-stream'
            )
            
            logger.info(f"Successfully saved {len(df)} records to s3://{bucket}/{s3_path}")
            return s3_path
            
        except Exception as e:
            logger.error(f"Error saving data to {s3_path}: {e}")
            logger.debug(f"S3 save error details: {traceback.format_exc()}")
            raise
    
    def log_collection_start(self, **kwargs):
        """Log the start of data collection with parameters"""
        self.results['start_time'] = datetime.now(timezone.utc).isoformat()
        logger.info(f"Starting {self.collector_name} data collection")
        logger.info(f"Collection parameters: {kwargs}")
    
    def log_collection_end(self):
        """Log the end of data collection with summary"""
        self.results['end_time'] = datetime.now(timezone.utc).isoformat()
        self.results['total_success'] = len(self.results['success'])
        self.results['total_failed'] = len(self.results['failed'])
        
        logger.info(f"Completed {self.collector_name} data collection")
        logger.info(f"Summary: {self.results['total_success']} successful, {self.results['total_failed']} failed")
        
        if self.results['failed']:
            logger.warning(f"Failed items: {self.results['failed']}")
    
    def add_success_result(self, item_id: str, **kwargs):
        """Add a successful result with standardized logging"""
        result = {'item_id': item_id, **kwargs}
        self.results['success'].append(result)
        logger.info(f"Successfully processed {item_id}")
        logger.debug(f"Success details for {item_id}: {kwargs}")
    
    def add_failed_result(self, item_id: str, error: str, **kwargs):
        """Add a failed result with standardized logging"""
        result = {'item_id': item_id, 'error': error, **kwargs}
        self.results['failed'].append(result)
        logger.error(f"Failed to process {item_id}: {error}")
        logger.debug(f"Failure details for {item_id}: {kwargs}")
    
    def validate_environment(self) -> bool:
        """Validate that required environment variables are set"""
        missing_vars = []
        
        if not BRONZE_BUCKET:
            missing_vars.append('BRONZE_BUCKET')
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            return False
        
        logger.info("Environment validation passed")
        return True
