#!/usr/bin/env python3
"""
Data Processing Pipeline - EMR Serverless Application
Processes raw data from bronze bucket to create cleaned data (silver) and training features (gold)
"""

import argparse
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np
import boto3
from botocore.exceptions import ClientError
import json
from feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
            'start_time': datetime.now().isoformat(),
            'overall_status': 'success',
            'bronze_to_silver': {},
            'silver_to_gold': {},
            'errors': []
        }
    
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
        """Load Parquet file from S3"""
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            df = pd.read_parquet(response['Body'])
            return df
        except Exception as e:
            logger.error(f"Error loading Parquet from S3 {bucket}/{key}: {e}")
            raise
    
    def save_parquet_to_s3(self, df: pd.DataFrame, bucket: str, key: str) -> str:
        """Save DataFrame as Parquet to S3"""
        try:
            buffer = df.to_parquet(index=False)
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=buffer,
                ContentType='application/octet-stream'
            )
            return key
        except Exception as e:
            logger.error(f"Error saving Parquet to S3 {bucket}/{key}: {e}")
            raise
    
    def process_fred_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean FRED data"""
        try:
            # Ensure date column is datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Convert value to numeric, handling missing values
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Remove rows with missing values
            df = df.dropna(subset=['value', 'date'])
            
            # Sort by date
            df = df.sort_values('date')
            
            # Add processing metadata
            df['processed_timestamp'] = datetime.now().isoformat()
            df['execution_id'] = self.execution_id
            
            return df
        except Exception as e:
            logger.error(f"Error processing FRED data: {e}")
            raise
    
    def process_worldbank_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean World Bank data"""
        try:
            # Ensure date column is datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Convert value to numeric, handling missing values
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Remove rows with missing values
            df = df.dropna(subset=['value', 'date'])
            
            # Sort by date
            df = df.sort_values('date')
            
            # Add processing metadata
            df['processed_timestamp'] = datetime.now().isoformat()
            df['execution_id'] = self.execution_id
            
            return df
        except Exception as e:
            logger.error(f"Error processing World Bank data: {e}")
            raise
    
    def process_yahoo_finance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean Yahoo Finance data"""
        try:
            # Ensure date column is datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Convert OHLCV columns to numeric
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with missing values in key columns
            df = df.dropna(subset=['Close', 'Date'])
            
            # Sort by date
            df = df.sort_values('Date')
            
            # Rename Date column to date for consistency
            df = df.rename(columns={'Date': 'date'})
            
            # Add processing metadata
            df['processed_timestamp'] = datetime.now().isoformat()
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
            
            for file_key in yahoo_files:
                if file_key.endswith('.parquet'):
                    try:
                        df = self.load_parquet_from_s3(self.bronze_bucket, file_key)
                        processed_df = self.process_yahoo_finance_data(df)
                        
                        # Save to silver bucket
                        silver_key = f"silver/{self.execution_id}/yahoo_finance/{file_key.split('/')[-1]}"
                        self.save_parquet_to_s3(processed_df, self.silver_bucket, silver_key)
                        
                        results['yahoo_finance']['records_processed'] += len(processed_df)
                        results['yahoo_finance']['files_processed'] += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to process Yahoo Finance file {file_key}: {e}")
                        self.results['errors'].append(f"Yahoo Finance processing error: {e}")
            
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
            
            # Save features to gold bucket
            gold_key = f"gold/{self.execution_id}/processed_data.parquet"
            self.save_parquet_to_s3(features_df, self.gold_bucket, gold_key)
            
            # Update results
            results['final_shape'] = features_df.shape
            results['feature_count'] = len([col for col in features_df.columns if col not in ['date', 'target', 'execution_id', 'feature_creation_timestamp']])
            results['date_range'] = f"{features_df['date'].min().strftime('%Y-%m-%d')} to {features_df['date'].max().strftime('%Y-%m-%d')}"
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
            self.results['end_time'] = datetime.now().isoformat()
            
            # Determine overall status
            if self.results['errors']:
                self.results['overall_status'] = 'partial_failure'
            
            logger.info("Data Processing Pipeline completed successfully")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Data Processing Pipeline failed: {e}")
            self.results['overall_status'] = 'failure'
            self.results['errors'].append(f"Pipeline execution error: {e}")
            self.results['end_time'] = datetime.now().isoformat()
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
        logger.info(f'Timestamp: {datetime.now().isoformat()}')
        logger.info(f'Bronze bucket: {args.bronze_bucket}')
        logger.info(f'Silver bucket: {args.silver_bucket}')
        logger.info(f'Gold bucket: {args.gold_bucket}')
        logger.info(f'Execution ID: {args.execution_id}')
        logger.info('=' * 60)
        
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
