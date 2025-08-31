#!/usr/bin/env python3
"""
Base Data Collector Class
Provides common functionality for all data collectors
"""

import json
import boto3
import pandas as pd
from datetime import datetime, timezone
import os
import logging
from typing import Dict, Any, Optional
import io

# Configure logging
logger = logging.getLogger(__name__)

# Initialize AWS clients
s3_client = boto3.client('s3')
secrets_client = boto3.client('secretsmanager')

# Environment variables
BRONZE_BUCKET = os.environ.get('BRONZE_BUCKET')
API_SECRETS_ARN = os.environ.get('API_SECRETS_ARN')

class DataCollector:
    """Base class for data collectors"""
    
    def __init__(self):
        self.results = {
            'success': [],
            'failed': [],
            'collection_timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def get_api_key(self, key_name: str) -> Optional[str]:
        """Retrieve API key from Secrets Manager"""
        if not API_SECRETS_ARN:
            logger.warning(f"API_SECRETS_ARN environment variable not set. Cannot retrieve API key '{key_name}'")
            return None
            
        try:
            response = secrets_client.get_secret_value(SecretId=API_SECRETS_ARN)
            secrets = json.loads(response['SecretString'])
            api_key = secrets.get(key_name, '')
            
            if api_key:
                logger.info(f"Successfully retrieved API key '{key_name}' from Secrets Manager")
            else:
                logger.warning(f"API key '{key_name}' not found in Secrets Manager secret")
                
            return api_key
            
        except Exception as e:
            logger.warning(f"Error retrieving API key '{key_name}' from Secrets Manager: {e}")
            return None
    
    def save_to_s3(self, df: pd.DataFrame, s3_path: str, bucket: str) -> str:
        """Save DataFrame to S3 as parquet"""
        try:
            if df.empty:
                logger.warning(f"No data to save to {s3_path}")
                return ""
            
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
            raise
