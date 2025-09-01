#!/usr/bin/env python3
import argparse
import logging
import sys
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='EMR Serverless Test Application')
    
    parser.add_argument('--bronze-bucket', required=True, help='S3 bucket for raw data')
    parser.add_argument('--silver-bucket', required=True, help='S3 bucket for processed data')
    parser.add_argument('--gold-bucket', required=True, help='S3 bucket for feature data')
    parser.add_argument('--execution-id', required=True, help='Pipeline execution ID')
    
    return parser.parse_args()

def log_environment_info() -> None:
    """Log information about the execution environment."""
    logger.info('=' * 60)
    logger.info('EMR SERVERLESS TEST APPLICATION STARTED')
    logger.info('=' * 60)
    logger.info(f'Timestamp: {datetime.now().isoformat()}')
    logger.info(f'Python version: {sys.version}')
    logger.info(f'Platform: {sys.platform}')
    logger.info(f'AWS Region: {os.environ.get("AWS_REGION", "Not set")}')
    logger.info(f'Working directory: {os.getcwd()}')
    logger.info(f'User: {os.environ.get("USER", "Unknown")}')
    logger.info(f'Home directory: {os.environ.get("HOME", "Not set")}')
    logger.info('=' * 60)

def test_imports() -> None:
    """Test that all required packages can be imported."""
    logger.info('Testing package imports...')
    
    try:
        import pandas as pd
        logger.info(f'pandas imported successfully (version: {pd.__version__})')
    except ImportError as e:
        logger.error(f'Failed to import pandas: {e}')
    
    try:
        import numpy as np
        logger.info(f'numpy imported successfully (version: {np.__version__})')
    except ImportError as e:
        logger.error(f'Failed to import numpy: {e}')
    
    try:
        import boto3
        logger.info(f'boto3 imported successfully (version: {boto3.__version__})')
    except ImportError as e:
        logger.error(f'Failed to import boto3: {e}')
    
    try:
        import pyarrow
        logger.info(f'pyarrow imported successfully (version: {pyarrow.__version__})')
    except ImportError as e:
        logger.error(f'Failed to import pyarrow: {e}')

def test_spark_environment() -> None:
    """Test Spark environment if available."""
    logger.info('Testing Spark environment...')
    
    try:
        from pyspark.sql import SparkSession
        logger.info('PySpark is available')
        
        # Try to create a Spark session
        spark = SparkSession.builder \
            .appName("EMR-Serverless-Test") \
            .getOrCreate()
        
        logger.info(f'Spark session created successfully')
        logger.info(f'   - Spark version: {spark.version}')
        logger.info(f'   - Spark UI: {spark.sparkContext.uiWebUrl}')
        
        # Create a simple test DataFrame
        test_data = [("Hello", "World"), ("EMR", "Serverless")]
        df = spark.createDataFrame(test_data, ["col1", "col2"])
        logger.info(f'Test DataFrame created with {df.count()} rows')
        
        # Show the data
        logger.info('Test DataFrame content:')
        df.show()
        
        spark.stop()
        logger.info('Spark session stopped successfully')
        
    except ImportError:
        logger.warning('PySpark not available - this is normal for some EMR configurations')
    except Exception as e:
        logger.error(f'Spark test failed: {e}')

def main() -> int:
    """Main entry point for the test application."""
    try:

        # Exit early for now for testing
        return 0

        # Parse command line arguments
        args = parse_arguments()
        
        # Log startup information
        log_environment_info()
        
        # Log the input parameters
        logger.info('Application Parameters:')
        logger.info(f'   - Bronze bucket: {args.bronze_bucket}')
        logger.info(f'   - Silver bucket: {args.silver_bucket}')
        logger.info(f'   - Gold bucket: {args.gold_bucket}')
        logger.info(f'   - Execution ID: {args.execution_id}')
        logger.info('')
        
        # Test package imports
        test_imports()
        logger.info('')
        
        # Test Spark environment
        test_spark_environment()
        logger.info('')
        
        # Success message
        logger.info('EMR SERVERLESS TEST COMPLETED SUCCESSFULLY!')
        logger.info('=' * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f'Application failed with error: {str(e)}', exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())