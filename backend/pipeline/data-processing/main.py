#!/usr/bin/env python3
"""
Data Processing Pipeline - EMR Serverless Application
Processes raw data from bronze bucket to create cleaned data (silver) and training features (gold)
"""

import argparse
import logging
import sys
import os
import io
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np
import boto3
from botocore.exceptions import ClientError
import json
from feature_engineering import FeatureEngineer
from pandas.api.types import (
    is_integer_dtype, is_float_dtype,
    is_datetime64_dtype, is_datetime64tz_dtype
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_s3_paths(bronze_bucket: str, silver_bucket: str, gold_bucket: str) -> bool:
    """Validate that all required S3 buckets and paths are accessible"""
    try:
        s3_client = boto3.client('s3')
        
        # Test bucket access
        for bucket_name in [bronze_bucket, silver_bucket, gold_bucket]:
            try:
                s3_client.head_bucket(Bucket=bucket_name)
                logger.info(f"Successfully accessed bucket: {bucket_name}")
            except Exception as e:
                logger.error(f"Failed to access bucket {bucket_name}: {e}")
                return False
        
        # Check if bronze bucket has expected data structure
        expected_prefixes = ['raw/fred/', 'raw/worldbank/', 'raw/yahoo_finance/']
        missing_prefixes = []
        
        for prefix in expected_prefixes:
            try:
                resp = s3_client.list_objects_v2(Bucket=bronze_bucket, Prefix=prefix, MaxKeys=1)
                if resp.get('KeyCount', 0) == 0:
                    missing_prefixes.append(prefix)
            except Exception as e:
                logger.warning(f"Could not check prefix {prefix}: {e}")
                missing_prefixes.append(prefix)
        
        if missing_prefixes:
            logger.warning(f"Missing expected data prefixes in bronze bucket: {missing_prefixes}")
            logger.warning("Data processing may fail if no data exists in these paths")
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating S3 paths: {e}")
        return False

def ensure_output_folders(silver_bucket: str, gold_bucket: str, execution_id: str) -> None:
    """Ensure output folders exist in silver and gold buckets"""
    try:
        s3_client = boto3.client('s3')
        
        # Create silver bucket folders
        silver_folders = [
            f'silver/{execution_id}/fred/',
            f'silver/{execution_id}/worldbank/',
            f'silver/{execution_id}/yahoo_finance/'
        ]
        
        for folder in silver_folders:
            try:
                s3_client.put_object(
                    Bucket=silver_bucket,
                    Key=f"{folder}.keep",
                    Body=b'',
                    ContentType='text/plain'
                )
                logger.info(f"Created silver folder: {folder}")
            except Exception as e:
                logger.warning(f"Could not create silver folder {folder}: {e}")
        
        # Create gold bucket folders
        gold_folders = [
            f'gold/{execution_id}/',
            f'pipeline_results/data_processing/'
        ]
        
        for folder in gold_folders:
            try:
                s3_client.put_object(
                    Bucket=gold_bucket,
                    Key=f"{folder}.keep",
                    Body=b'',
                    ContentType='text/plain'
                )
                logger.info(f"Created gold folder: {folder}")
            except Exception as e:
                logger.warning(f"Could not create gold folder {folder}: {e}")
                
    except Exception as e:
        logger.error(f"Error in ensure_output_folders: {e}")
        raise

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Data Processing Pipeline')
    
    parser.add_argument('--bronze-bucket', required=True, help='S3 bucket for raw data')
    parser.add_argument('--silver-bucket', required=True, help='S3 bucket for processed data')
    parser.add_argument('--gold-bucket', required=True, help='S3 bucket for feature data')
    parser.add_argument('--execution-id', required=True, help='Pipeline execution ID')
    
    return parser.parse_args()

def log_pipeline_summary(results: Dict[str, Any]) -> None:
    """Log a comprehensive summary of the data processing pipeline"""
    logger.info("=" * 80)
    logger.info("DATA PROCESSING PIPELINE SUMMARY")
    logger.info("=" * 80)
    
    logger.info(f"Pipeline Execution ID: {results['execution_id']}")
    logger.info(f"Start Time: {results['start_time']}")
    logger.info(f"End Time: {results['end_time']}")
    logger.info(f"Overall Status: {results['overall_status']}")
    
    logger.info("\nBronze to Silver Processing:")
    logger.info(f"  - FRED Records: {results['bronze_to_silver']['fred']['records_processed']}")
    logger.info(f"  - World Bank Records: {results['bronze_to_silver']['worldbank']['records_processed']}")
    logger.info(f"  - Yahoo Finance Records: {results['bronze_to_silver']['yahoo_finance']['records_processed']}")
    
    logger.info("\nSilver to Gold Processing:")
    logger.info(f"  - Final Dataset Shape: {results['silver_to_gold']['final_shape']}")
    logger.info(f"  - Feature Columns: {results['silver_to_gold']['feature_count']}")
    logger.info(f"  - Date Range: {results['silver_to_gold']['date_range']}")
    
    logger.info("=" * 80)

class DataProcessor:
    """Main data processing class for the pipeline"""
    
    def __init__(self, bronze_bucket: str, silver_bucket: str, gold_bucket: str, execution_id: str):
        self.bronze_bucket = bronze_bucket
        self.silver_bucket = silver_bucket
        self.gold_bucket = gold_bucket
        self.execution_id = execution_id
        
        # Initialize S3 client
        self.s3_client = boto3.client('s3')
        
        # Results tracking
        self.results = {
            'execution_id': execution_id,
            'start_time': datetime.now(timezone.utc).isoformat(),
            'overall_status': 'success',
            'bronze_to_silver': {},
            'silver_to_gold': {},
            'errors': []
        }
    
    def _now_utc_iso(self) -> str:
        """Get current UTC time as ISO string."""
        return datetime.now(timezone.utc).isoformat()
    
    def list_s3_objects(self, bucket: str, prefix: str) -> List[str]:
        """List all objects in S3 bucket with given prefix"""
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
            
            objects = []
            for page in pages:
                if 'Contents' in page:
                    objects.extend([obj['Key'] for obj in page['Contents']])
            
            return objects
        except Exception as e:
            logger.error(f"Error listing S3 objects: {e}")
            raise
    
    def load_parquet_from_s3(self, bucket: str, key: str) -> pd.DataFrame:
        """Load Parquet file from S3 safely via BytesIO."""
        try:
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            with io.BytesIO(obj['Body'].read()) as buf:
                df = pd.read_parquet(buf)  # relies on pyarrow in EMR
            return df
        except Exception as e:
            logger.error(f"Error loading Parquet from s3://{bucket}/{key}: {e}")
            raise
    
    def save_parquet_to_s3(self, df: pd.DataFrame, bucket: str, key: str) -> str:
        """Save DataFrame as Parquet to S3 using BytesIO."""
        try:
            buf = io.BytesIO()
            df.to_parquet(buf, index=False, compression="snappy", engine="pyarrow")
            buf.seek(0)
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=buf.getvalue(),
                ContentType='application/x-parquet'
            )
            return key
        except Exception as e:
            logger.error(f"Error saving Parquet to S3 s3://{bucket}/{key}: {e}")
            raise
    
    def process_fred_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean FRED data"""
        try:
            # Ensure date column is datetime
            df['date'] = pd.to_datetime(df['date'], utc=True)
            
            # Convert value to numeric, handling missing values
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Remove rows with missing values
            df = df.dropna(subset=['value', 'date'])
            
            # Sort by date
            df = df.sort_values('date')
            
            # Add processing metadata
            df['processed_timestamp'] = self._now_utc_iso()
            df['execution_id'] = self.execution_id
            
            return df
        except Exception as e:
            logger.error(f"Error processing FRED data: {e}")
            raise
    
    def process_worldbank_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean World Bank data"""
        try:
            # Ensure date column is datetime
            df['date'] = pd.to_datetime(df['date'], utc=True)
            
            # Convert value to numeric, handling missing values
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Remove rows with missing values
            df = df.dropna(subset=['value', 'date'])
            
            # Sort by date
            df = df.sort_values('date')
            
            # Add processing metadata
            df['processed_timestamp'] = self._now_utc_iso()
            df['execution_id'] = self.execution_id
            
            return df
        except Exception as e:
            logger.error(f"Error processing World Bank data: {e}")
            raise
    
    def process_yahoo_finance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean Yahoo Finance data"""
        try:
            # Check if required columns exist
            required_columns = ['Close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing required columns: {missing_columns}. Skipping Yahoo Finance data point.")
                return pd.DataFrame()
            
            # Pick/normalize a date column with more options including epoch timestamps
            date_column = None
            possible_date_columns = ['Date', 'date', 'DATE', 'Date_Time', 'datetime', 'asOfDate', 'asOfDateTime', 'timestamp']
            for col in possible_date_columns:
                if col in df.columns:
                    date_column = col
                    break
            
            if date_column is None:
                logger.warning(f"No date column found in Yahoo Finance data. Available columns: {list(df.columns)}. Skipping data point.")
                return pd.DataFrame()
            
            # Standardize column names
            rename_map = {date_column: 'date', 'Adj Close': 'Adj_Close', 'AdjClose': 'Adj_Close'}
            df = df.rename(columns=rename_map)

            # Ensure a symbol column exists
            symbol_col = None
            for c in ['symbol', 'Symbol', 'ticker', 'Ticker']:
                if c in df.columns:
                    symbol_col = c
                    break
            if symbol_col is None:
                logger.warning("No symbol/ticker column found in Yahoo Finance file. Skipping data point.")
                return pd.DataFrame()
            # normalize to 'symbol' as string
            df['symbol'] = df[symbol_col].astype(str)

            # Date normalization using proper pandas type checking
            date_s = df['date']
            if is_integer_dtype(date_s) or is_float_dtype(date_s):
                # epoch seconds (or ms if that is your schema—adjust unit as needed)
                df['date'] = pd.to_datetime(date_s, unit='s', utc=True, errors='coerce')
            elif is_datetime64tz_dtype(date_s):
                # keep in UTC but make naive if you want a naive UTC column
                df['date'] = date_s.dt.tz_convert('UTC').dt.tz_localize(None)
            elif is_datetime64_dtype(date_s):
                # naive datetimes — treat as UTC
                df['date'] = pd.to_datetime(date_s, utc=True, errors='coerce')
            else:
                # strings/objects
                df['date'] = pd.to_datetime(date_s, utc=True, errors='coerce')
            
            # Check for invalid dates
            if df['date'].isna().all():
                logger.warning("All date values are invalid after conversion. Skipping Yahoo Finance data point.")
                return pd.DataFrame()

            # Convert numeric columns safely
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Better dtype for volume to keep NA support
            if 'Volume' in df.columns:
                df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').astype('Int64')

            # Remove rows with missing values in key columns
            df = df.dropna(subset=['Close', 'date'])
            
            # Safety check: ensure we have usable data after cleaning
            if df[['date', 'symbol', 'Close']].dropna().empty:
                logger.info("Yahoo Finance frame has no usable rows (date/symbol/Close NA).")
                return pd.DataFrame()
            
            # Sort by date
            df = df.sort_values('date')
            
            # Add processing metadata
            df['processed_timestamp'] = self._now_utc_iso()
            df['execution_id'] = self.execution_id
            
            return df
        except Exception as e:
            logger.error(f"Error processing Yahoo Finance data: {e}")
            raise
    
    def bronze_to_silver_processing(self) -> Dict[str, Any]:
        """Process raw data from bronze bucket to silver bucket"""
        logger.info("Starting Bronze to Silver processing...")
        
        results = {
            'fred': {'records_processed': 0, 'files_processed': 0},
            'worldbank': {'records_processed': 0, 'files_processed': 0},
            'yahoo_finance': {'records_processed': 0, 'files_processed': 0}
        }
        
        try:
            # Process FRED data
            logger.info("Processing FRED data...")
            fred_files = self.list_s3_objects(self.bronze_bucket, 'raw/fred/')
            
            if not fred_files:
                logger.warning("No FRED data files found in bronze bucket")
            else:
                for file_key in fred_files:
                    if file_key.endswith('.parquet'):
                        try:
                            df = self.load_parquet_from_s3(self.bronze_bucket, file_key)
                            processed_df = self.process_fred_data(df)
                            
                            # Save to silver bucket
                            silver_key = f"silver/{self.execution_id}/fred/{file_key.split('/')[-1]}"
                            self.save_parquet_to_s3(processed_df, self.silver_bucket, silver_key)
                            
                            results['fred']['records_processed'] += len(processed_df)
                            results['fred']['files_processed'] += 1
                            
                        except Exception as e:
                            logger.warning(f"Failed to process FRED file {file_key}: {e}")
                            self.results['errors'].append(f"FRED processing error: {e}")
            
            # Process World Bank data
            logger.info("Processing World Bank data...")
            worldbank_files = self.list_s3_objects(self.bronze_bucket, 'raw/worldbank/')
            
            if not worldbank_files:
                logger.warning("No World Bank data files found in bronze bucket")
            else:
                for file_key in worldbank_files:
                    if file_key.endswith('.parquet'):
                        try:
                            df = self.load_parquet_from_s3(self.bronze_bucket, file_key)
                            processed_df = self.process_worldbank_data(df)
                            
                            # Save to silver bucket
                            silver_key = f"silver/{self.execution_id}/worldbank/{file_key.split('/')[-1]}"
                            self.save_parquet_to_s3(processed_df, self.silver_bucket, silver_key)
                            
                            results['worldbank']['records_processed'] += len(processed_df)
                            results['worldbank']['files_processed'] += 1
                            
                        except Exception as e:
                            logger.warning(f"Failed to process World Bank file {file_key}: {e}")
                            self.results['errors'].append(f"World Bank processing error: {e}")
            
            # Process Yahoo Finance data
            logger.info("Processing Yahoo Finance data...")
            yahoo_files = self.list_s3_objects(self.bronze_bucket, 'raw/yahoo_finance/')
            
            if not yahoo_files:
                logger.warning("No Yahoo Finance data files found in bronze bucket")
            else:
                for file_key in yahoo_files:
                    if not file_key.endswith('.parquet'):
                        continue
                    # Skip non-price schemas like "info"
                    if "/info/" in file_key.lower():
                        logger.warning(f"Skipping non-price Yahoo file: {file_key}")
                        continue
                    try:
                        df = self.load_parquet_from_s3(self.bronze_bucket, file_key)
                        processed_df = self.process_yahoo_finance_data(df)
                        
                        # Save to silver bucket
                        silver_key = f"silver/{self.execution_id}/yahoo_finance/{file_key.split('/')[-1]}"
                        self.save_parquet_to_s3(processed_df, self.silver_bucket, silver_key)
                        
                        results['yahoo_finance']['records_processed'] += len(processed_df)
                        results['yahoo_finance']['files_processed'] += 1
                        
                    except Exception as e:
                        cols = list(df.columns) if 'df' in locals() else None
                        logger.warning(f"Failed to process Yahoo Finance file {file_key}: {e}; columns={cols}")
                        self.results['errors'].append(f"Yahoo Finance processing error: {e}")
                        # Continue processing other files instead of failing completely
                        continue
            
            # Check if any data was processed
            total_files = sum(r['files_processed'] for r in results.values())
            if total_files == 0:
                logger.warning("No data files were processed. This may indicate missing input data.")
                self.results['errors'].append("No input data files found to process")
            
            logger.info("Bronze to Silver processing completed successfully")
            
        except Exception as e:
            logger.error(f"Bronze to Silver processing failed: {e}")
            self.results['overall_status'] = 'partial_failure'
            self.results['errors'].append(f"Bronze to Silver processing error: {e}")
        
        return results
    
    def silver_to_gold_processing(self) -> Dict[str, Any]:
        """Process cleaned data from silver bucket to create training features in gold bucket"""
        logger.info("Starting Silver to Gold processing...")
        
        results = {
            'final_shape': (0, 0),
            'feature_count': 0,
            'date_range': '',
            'records_processed': 0
        }
        
        try:
            # Load processed data from silver bucket
            logger.info("Loading processed data from silver bucket...")
            
            # Load FRED data
            fred_files = self.list_s3_objects(self.silver_bucket, f'silver/{self.execution_id}/fred/')
            fred_data = []
            for file_key in fred_files:
                if file_key.endswith('.parquet'):
                    df = self.load_parquet_from_s3(self.silver_bucket, file_key)
                    fred_data.append(df)
            
            fred_df = pd.concat(fred_data, ignore_index=True) if fred_data else pd.DataFrame()
            logger.info(f"Loaded {len(fred_df)} FRED records")
            
            # Load World Bank data
            worldbank_files = self.list_s3_objects(self.silver_bucket, f'silver/{self.execution_id}/worldbank/')
            worldbank_data = []
            for file_key in worldbank_files:
                if file_key.endswith('.parquet'):
                    df = self.load_parquet_from_s3(self.silver_bucket, file_key)
                    worldbank_data.append(df)
            
            worldbank_df = pd.concat(worldbank_data, ignore_index=True) if worldbank_data else pd.DataFrame()
            logger.info(f"Loaded {len(worldbank_df)} World Bank records")
            
            # Load Yahoo Finance data
            yahoo_files = self.list_s3_objects(self.silver_bucket, f'silver/{self.execution_id}/yahoo_finance/')
            yahoo_data = []
            for file_key in yahoo_files:
                if file_key.endswith('.parquet'):
                    df = self.load_parquet_from_s3(self.silver_bucket, file_key)
                    yahoo_data.append(df)
            
            yahoo_df = pd.concat(yahoo_data, ignore_index=True) if yahoo_data else pd.DataFrame()
            logger.info(f"Loaded {len(yahoo_df)} Yahoo Finance records")
            
            # Create training features using the feature engineering module
            feature_engineer = FeatureEngineer(self.execution_id)
            features_df, feature_names = feature_engineer.create_final_features(fred_df, worldbank_df, yahoo_df)
            
            # Save features if non-empty
            gold_key = f"gold/{self.execution_id}/processed_data.parquet"
            if not features_df.empty:
                self.save_parquet_to_s3(features_df, self.gold_bucket, gold_key)
            else:
                logger.warning("No features produced; skipping Parquet write.")
            
            # Update results safely
            results['final_shape'] = features_df.shape
            results['feature_count'] = len(feature_names)
            
            if not features_df.empty and 'date' in features_df.columns and pd.api.types.is_datetime64_any_dtype(features_df['date']):
                dmin = features_df['date'].min()
                dmax = features_df['date'].max()
                results['date_range'] = f"{dmin.strftime('%Y-%m-%d')} to {dmax.strftime('%Y-%m-%d')}"
            else:
                results['date_range'] = 'N/A'
            results['records_processed'] = len(features_df)
            
            logger.info("Silver to Gold processing completed successfully")
            
        except Exception as e:
            logger.error(f"Silver to Gold processing failed: {e}")
            self.results['overall_status'] = 'partial_failure'
            self.results['errors'].append(f"Silver to Gold processing error: {e}")
        
        return results
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete data processing pipeline"""
        try:
            logger.info("Starting Data Processing Pipeline...")
            
            # Step 1: Bronze to Silver processing
            bronze_to_silver_results = self.bronze_to_silver_processing()
            self.results['bronze_to_silver'] = bronze_to_silver_results
            
            # Step 2: Silver to Gold processing
            silver_to_gold_results = self.silver_to_gold_processing()
            self.results['silver_to_gold'] = silver_to_gold_results
            
            # Record completion time
            self.results['end_time'] = datetime.now(timezone.utc).isoformat()
            
            # Determine overall status
            if self.results['errors']:
                self.results['overall_status'] = 'partial_failure'
            
            logger.info("Data Processing Pipeline completed successfully")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Data Processing Pipeline failed: {e}")
            self.results['overall_status'] = 'failure'
            self.results['errors'].append(f"Pipeline execution error: {e}")
            self.results['end_time'] = datetime.now(timezone.utc).isoformat()
            return self.results

def main() -> int:
    """Main entry point for the data processing pipeline."""

    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Log startup information
        logger.info('=' * 60)
        logger.info('DATA PROCESSING PIPELINE STARTED')
        logger.info('=' * 60)
        logger.info(f'Timestamp: {datetime.now(timezone.utc).isoformat()}')
        logger.info(f'Bronze bucket: {args.bronze_bucket}')
        logger.info(f'Silver bucket: {args.silver_bucket}')
        logger.info(f'Gold bucket: {args.gold_bucket}')
        logger.info(f'Execution ID: {args.execution_id}')
        logger.info('=' * 60)
        
        # Validate S3 paths before processing
        if not validate_s3_paths(args.bronze_bucket, args.silver_bucket, args.gold_bucket):
            logger.error("S3 path validation failed. Exiting.")
            return 1
        
        # Ensure output folders exist
        ensure_output_folders(args.silver_bucket, args.gold_bucket, args.execution_id)
        
        # Initialize data processor
        processor = DataProcessor(
            bronze_bucket=args.bronze_bucket,
            silver_bucket=args.silver_bucket,
            gold_bucket=args.gold_bucket,
            execution_id=args.execution_id
        )
        
        # Run the pipeline
        results = processor.run_pipeline()
        
        # Log pipeline summary
        log_pipeline_summary(results)
        
        # Save results to S3
        try:
            results_key = f"pipeline_results/data_processing/{args.execution_id}_results.json"
            processor.s3_client.put_object(
                Bucket=args.gold_bucket,
                Key=results_key,
                Body=json.dumps(results, indent=2, default=str),
                ContentType='application/json'
            )
            logger.info(f"Pipeline results saved to s3://{args.gold_bucket}/{results_key}")
        except Exception as e:
            logger.warning(f"Failed to save pipeline results: {e}")
        
        # Exit with appropriate code
        if results['overall_status'] == 'success':
            logger.info("Data Processing Pipeline completed successfully")
            return 0
        elif results['overall_status'] == 'partial_failure':
            logger.warning("Data Processing Pipeline completed with partial failures")
            return 1
        else:
            logger.error("Data Processing Pipeline failed")
            return 1
            
    except Exception as e:
        logger.error(f'Data Processing Pipeline failed with error: {str(e)}', exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
