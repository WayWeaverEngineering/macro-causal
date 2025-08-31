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
                        # Note: This decorator uses the global logger since it's not part of the class
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
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            return result
        except Exception as e:
            duration = time.time() - start_time
            # Note: This decorator uses the global logger since it's not part of the class
            logger.error(f"API Call Failed: {func_name} - Duration: {duration:.2f}s - Error: {e}")
            raise
    return wrapper

class DataCollector:
    """Base class for data collectors with enhanced error handling and logging"""
    
    def __init__(self, collector_name: str = "Unknown"):
        self.collector_name = collector_name
        # Create a logger specific to this collector instance
        self.logger = logging.getLogger(f"collectors.{collector_name.lower()}")
        
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
        self.logger.info(f"Initialized {collector_name} collector")
    
    def log_consolidated_error(self, context: str, error: Exception, response: Optional[requests.Response] = None, 
                              additional_info: Dict[str, Any] = None) -> None:
        """Consolidate error logging into a single line with key information"""
        error_parts = [f"{self.collector_name} {context} failed: {type(error).__name__}: {str(error)}"]
        
        if response:
            error_parts.append(f"HTTP {response.status_code}")
            if hasattr(response, 'url'):
                error_parts.append(f"URL: {response.url}")
        
        if additional_info:
            for key, value in additional_info.items():
                if value is not None:
                    error_parts.append(f"{key}: {value}")
        
        self.logger.error(" | ".join(error_parts))
    
    def validate_date_range(self, start_date: str, end_date: str) -> tuple[str, str]:
        """Validate and fix date ranges to ensure they're valid"""
        try:
            # Parse dates
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Check if dates are in the future
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            
            if start_dt > now:
                self.logger.warning(f"Start date {start_date} is in the future, adjusting to 50 years ago")
                start_dt = now - timedelta(days=18250)
                start_date = start_dt.strftime('%Y-%m-%d')
            
            if end_dt > now:
                self.logger.warning(f"End date {end_date} is in the future, adjusting to today")
                end_dt = now
                end_date = end_dt.strftime('%Y-%m-%d')
            
            # Ensure start_date <= end_date
            if start_dt > end_dt:
                self.logger.warning(f"Start date {start_date} is after end date {end_date}, swapping")
                start_date, end_date = end_date, start_date
            
            self.logger.info(f"Validated date range: {start_date} to {end_date}")
            return start_date, end_date
            
        except ValueError as e:
            self.logger.error(f"Invalid date format: {e}")
            # Return default date range
            now = datetime.now(timezone.utc)
            end_date = now.strftime('%Y-%m-%d')
            start_date = (now - timedelta(days=18250)).strftime('%Y-%m-%d')
            self.logger.info(f"Using default date range: {start_date} to {end_date}")
            return start_date, end_date
    
    def get_default_date_range(self, days_back: int = 18250) -> tuple[str, str]:
        """Get a default date range from today going back specified days (default: 50 years)"""
        now = datetime.now(timezone.utc)
        end_date = now.strftime('%Y-%m-%d')
        start_date = (now - timedelta(days=days_back)).strftime('%Y-%m-%d')
        return start_date, end_date
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def get_api_key(self, key_name: str) -> Optional[str]:
        """Retrieve API key from Secrets Manager with retry logic"""
        if not API_SECRETS_ARN:
            self.logger.warning(f"API_SECRETS_ARN environment variable not set. Cannot retrieve API key '{key_name}'")
            return None
            
        try:
            response = secrets_client.get_secret_value(SecretId=API_SECRETS_ARN)
            secrets = json.loads(response['SecretString'])
            api_key = secrets.get(key_name, '')
            
            if not api_key:
                self.logger.warning(f"API key '{key_name}' not found in Secrets Manager secret")
                
            return api_key
            
        except Exception as e:
            self.log_consolidated_error(f"API key retrieval '{key_name}'", e)
            return None
    
    @log_api_call
    @retry_on_failure(max_retries=3, delay=1.0)
    def make_http_request(self, url: str, method: str = 'GET', params: Dict = None, 
                         headers: Dict = None, timeout: int = 30, **kwargs) -> requests.Response:
        """Make HTTP requests with comprehensive error handling and logging"""
        try:
            response = requests.request(
                method=method,
                url=url,
                params=params,
                headers=headers,
                timeout=timeout,
                **kwargs
            )
            
            response.raise_for_status()
            return response
            
        except requests.exceptions.Timeout:
            self.log_consolidated_error("HTTP request timeout", Exception(f"Timeout after {timeout}s"), 
                                      additional_info={"url": url})
            raise
        except requests.exceptions.ConnectionError as e:
            self.log_consolidated_error("HTTP connection", e, additional_info={"url": url})
            raise
        except requests.exceptions.HTTPError as e:
            additional_info = {"url": url}
            if hasattr(e.response, 'status_code'):
                additional_info["status_code"] = e.response.status_code
            if hasattr(e.response, 'text') and e.response.text:
                additional_info["response_preview"] = e.response.text[:200]
            
            self.log_consolidated_error("HTTP request", e, e.response, additional_info)
            raise
        except Exception as e:
            self.log_consolidated_error("HTTP request", e, additional_info={"url": url})
            raise
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def save_to_s3(self, df: pd.DataFrame, s3_path: str, bucket: str = None) -> str:
        """Save DataFrame to S3 with retry logic"""
        if not bucket:
            bucket = BRONZE_BUCKET
            
        try:
            # Convert DataFrame to Parquet format in memory
            buffer = io.BytesIO()
            df.to_parquet(buffer, index=False)
            buffer.seek(0)
            
            # Upload to S3
            s3_client.put_object(
                Bucket=bucket,
                Key=s3_path,
                Body=buffer.getvalue()
            )
            
            return s3_path
            
        except Exception as e:
            self.log_consolidated_error("S3 save", e, additional_info={"bucket": bucket, "path": s3_path, "records": len(df)})
            raise
    
    def add_success_result(self, item_id: str, records_count: int, s3_key: str, **kwargs) -> None:
        """Add a successful result to the collection results"""
        result = {
            'item_id': item_id,
            'status': 'success',
            'records_count': records_count,
            's3_key': s3_key,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        result.update(kwargs)
        
        self.results['success'].append(result)
        self.results['total_success'] += 1
        self.results['total_processed'] += 1
    
    def add_failed_result(self, item_id: str, error: str, **kwargs) -> None:
        """Add a failed result to the collection results"""
        result = {
            'item_id': item_id,
            'status': 'failed',
            'error': error,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        result.update(kwargs)
        
        self.results['failed'].append(result)
        self.results['total_failed'] += 1
        self.results['total_processed'] += 1
        
        self.logger.error(f"Failed to process {item_id}: {error}")
    
    def validate_environment(self) -> bool:
        """Validate that required environment variables are set"""
        required_vars = ['BRONZE_BUCKET']
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            self.logger.error(f"Missing required environment variables: {missing_vars}")
            return False
        
        return True
    
    def log_collection_start(self, **kwargs) -> None:
        """Log the start of data collection"""
        self.results['start_time'] = datetime.now(timezone.utc).isoformat()
        self.logger.info(f"Starting {self.collector_name} data collection")
    
    def log_collection_end(self) -> None:
        """Log the end of data collection with comprehensive summary"""
        self.results['end_time'] = datetime.now(timezone.utc).isoformat()
        
        # Calculate summary statistics
        total_records = sum(result.get('records_count', 0) for result in self.results['success'])
        success_rate = (self.results['total_success'] / self.results['total_processed'] * 100) if self.results['total_processed'] > 0 else 0
        
        # Log basic completion info
        self.logger.info(f"Completed {self.collector_name} data collection: {self.results['total_success']} successful, {self.results['total_failed']} failed")
        
        # Log detailed summary
        self.logger.info(f"{self.collector_name} Collection Summary:")
        self.logger.info(f"   • Total items processed: {self.results['total_processed']}")
        self.logger.info(f"   • Successful collections: {self.results['total_success']}")
        self.logger.info(f"   • Failed collections: {self.results['total_failed']}")
        self.logger.info(f"   • Success rate: {success_rate:.1f}%")
        self.logger.info(f"   • Total records collected: {total_records:,}")
        
        # Log successful items with record counts
        if self.results['success']:
            self.logger.info(f"   • Successful items:")
            for result in self.results['success']:
                item_id = result.get('item_id', 'Unknown')
                records = result.get('records_count', 0)
                date_range = result.get('date_range', '')
                if date_range:
                    self.logger.info(f"     - {item_id}: {records:,} records ({date_range})")
                else:
                    self.logger.info(f"     - {item_id}: {records:,} records")
        
        # Log failed items
        if self.results['failed']:
            self.logger.info(f"   • Failed items:")
            for result in self.results['failed']:
                item_id = result.get('item_id', 'Unknown')
                error = result.get('error', 'Unknown error')
                self.logger.info(f"     - {item_id}: {error}")

    def get_collection_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the collection results"""
        total_records = sum(result.get('records_count', 0) for result in self.results['success'])
        success_rate = (self.results['total_success'] / self.results['total_processed'] * 100) if self.results['total_processed'] > 0 else 0
        
        return {
            'collector_name': self.collector_name,
            'total_processed': self.results['total_processed'],
            'total_success': self.results['total_success'],
            'total_failed': self.results['total_failed'],
            'total_records_collected': total_records,
            'success_rate_percent': round(success_rate, 1),
            'start_time': self.results['start_time'],
            'end_time': self.results['end_time'],
            'successful_items': [
                {
                    'item_id': result.get('item_id'),
                    'records_count': result.get('records_count', 0),
                    'date_range': result.get('date_range', ''),
                    's3_key': result.get('s3_key', '')
                }
                for result in self.results['success']
            ],
            'failed_items': [
                {
                    'item_id': result.get('item_id'),
                    'error': result.get('error', 'Unknown error')
                }
                for result in self.results['failed']
            ]
        }
