#!/usr/bin/env python3
"""
Data Collection Pipeline - ECS Fargate Application
Consolidates data collection from multiple sources:
- FRED (Federal Reserve Economic Data)
- World Bank API
- Yahoo Finance
"""

import json
import boto3
from datetime import datetime
import os
import logging
import sys

# Import collector classes
from collectors import FREDCollector, WorldBankCollector, YahooFinanceCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize AWS clients
s3_client = boto3.client('s3')

# Environment variables
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'dev')
BRONZE_BUCKET = os.environ.get('BRONZE_BUCKET')

def main():
    """Main function to run data collection"""
    try:
        logger.info("Starting data collection pipeline")
        
        # Check required environment variables
        if not BRONZE_BUCKET:
            raise ValueError("BRONZE_BUCKET environment variable is required")
        
        # Initialize collectors
        fred_collector = FREDCollector()
        worldbank_collector = WorldBankCollector()
        yahoo_collector = YahooFinanceCollector()
        
        # Collect data from all sources
        all_results = {
            'pipeline_start_time': datetime.utcnow().isoformat(),
            'fred': None,
            'worldbank': None,
            'yahoo_finance': None,
            'pipeline_end_time': None,
            'overall_status': 'success'
        }
        
        # Collect FRED data
        try:
            logger.info("Collecting FRED data...")
            all_results['fred'] = fred_collector.collect()
        except Exception as e:
            logger.error(f"FRED data collection failed: {e}")
            all_results['fred'] = {'error': str(e)}
            all_results['overall_status'] = 'partial_failure'
        
        # Collect World Bank data
        try:
            logger.info("Collecting World Bank data...")
            all_results['worldbank'] = worldbank_collector.collect()
        except Exception as e:
            logger.error(f"World Bank data collection failed: {e}")
            all_results['worldbank'] = {'error': str(e)}
            all_results['overall_status'] = 'partial_failure'
        
        # Collect Yahoo Finance data
        try:
            logger.info("Collecting Yahoo Finance data...")
            all_results['yahoo_finance'] = yahoo_collector.collect()
        except Exception as e:
            logger.error(f"Yahoo Finance data collection failed: {e}")
            all_results['yahoo_finance'] = {'error': str(e)}
            all_results['overall_status'] = 'partial_failure'
        
        # Record completion time
        all_results['pipeline_end_time'] = datetime.utcnow().isoformat()
        
        # Save overall results to S3
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        results_s3_path = f"pipeline_results/data_collection/{timestamp}_pipeline_results.json"
        
        s3_client.put_object(
            Bucket=BRONZE_BUCKET,
            Key=results_s3_path,
            Body=json.dumps(all_results, indent=2),
            ContentType='application/json'
        )
        
        logger.info(f"Data collection pipeline completed. Results saved to s3://{BRONZE_BUCKET}/{results_s3_path}")
        logger.info(f"Overall status: {all_results['overall_status']}")
        
        # Exit with appropriate code
        if all_results['overall_status'] == 'success':
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Data collection pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
