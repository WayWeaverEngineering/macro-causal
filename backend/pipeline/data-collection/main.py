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
from datetime import datetime, timezone
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

BRONZE_BUCKET = os.environ.get('BRONZE_BUCKET')

def ensure_s3_folder_structure(bucket_name: str) -> None:
    """Ensure the expected S3 folder structure exists for downstream processing"""
    try:
        # Create placeholder files to establish folder structure
        required_folders = [
            'raw/fred/',
            'raw/worldbank/',
            'raw/yahoo_finance/',
            'pipeline_results/data_collection/'
        ]
        
        for folder in required_folders:
            # Create a placeholder file to establish the folder
            placeholder_key = f"{folder}.keep"
            try:
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=placeholder_key,
                    Body='',
                    ContentType='text/plain'
                )
                logger.info(f"Created folder structure: {folder}")
            except Exception as e:
                logger.warning(f"Could not create folder {folder}: {e}")
                
    except Exception as e:
        logger.warning(f"Could not ensure S3 folder structure: {e}")

def validate_bronze_bucket_access(bucket_name: str) -> bool:
    """Validate that we can access the bronze bucket and it has the expected structure"""
    try:
        # Test bucket access
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"Successfully accessed bronze bucket: {bucket_name}")
        
        # Check if bucket has expected structure
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix='raw/',
            MaxKeys=1
        )
        
        if 'Contents' in response:
            logger.info("Bronze bucket has existing data structure")
        else:
            logger.info("Bronze bucket is empty, will create new structure")
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to access bronze bucket {bucket_name}: {e}")
        return False

def log_pipeline_summary(all_results: dict) -> None:
    """Log a comprehensive summary of the entire pipeline"""
    logger.info("=" * 80)
    logger.info("DATA COLLECTION PIPELINE SUMMARY")
    logger.info("=" * 80)
    
    # Calculate overall statistics
    total_records = 0
    total_success = 0
    total_failed = 0
    collector_summaries = []
    
    for collector_name, result in all_results.items():
        if collector_name in ['pipeline_start_time', 'pipeline_end_time', 'overall_status']:
            continue
            
        if isinstance(result, dict) and 'error' not in result:
            # Calculate summary from result dictionary
            total_processed = result.get('total_processed', 0)
            total_success = result.get('total_success', 0)
            total_failed = result.get('total_failed', 0)
            total_records_collected = sum(r.get('records_count', 0) for r in result.get('success', []))
            
            # Calculate success rate properly
            success_rate_percent = (total_success / total_processed * 100) if total_processed > 0 else 0
            
            summary = {
                'collector_name': result.get('collector_name', collector_name),
                'total_processed': total_processed,
                'total_success': total_success,
                'total_failed': total_failed,
                'total_records_collected': total_records_collected,
                'success_rate_percent': round(success_rate_percent, 1)
            }
            
            total_records += summary['total_records_collected']
            total_success += summary['total_success']
            total_failed += summary['total_failed']
            collector_summaries.append(summary)
        else:
            logger.warning(f"{collector_name.upper()}: Collection failed or no results available")
    
    # Log overall statistics
    logger.info(f"OVERALL STATISTICS:")
    logger.info(f"   • Total records collected: {total_records:,}")
    logger.info(f"   • Total successful collections: {total_success}")
    logger.info(f"   • Total failed collections: {total_failed}")
    logger.info(f"   • Overall success rate: {(total_success / (total_success + total_failed) * 100):.1f}%" if (total_success + total_failed) > 0 else "N/A")
    
    # Log individual collector summaries
    logger.info(f"COLLECTOR BREAKDOWN:")
    for summary in collector_summaries:
        logger.info(f"   • {summary['collector_name']}: {summary['total_records_collected']:,} records ({summary['success_rate_percent']}% success)")
    
    # Log timing information
    if all_results.get('pipeline_start_time') and all_results.get('pipeline_end_time'):
        start_time = datetime.fromisoformat(all_results['pipeline_start_time'].replace('Z', '+00:00'))
        end_time = datetime.fromisoformat(all_results['pipeline_end_time'].replace('Z', '+00:00'))
        duration = end_time - start_time
        logger.info(f"PIPELINE TIMING:")
        logger.info(f"   • Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info(f"   • End time: {end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info(f"   • Total duration: {duration}")
    
    logger.info(f"OVERALL STATUS: {all_results.get('overall_status', 'unknown').upper()}")
    logger.info("=" * 80)

def main():
    """Main function to run data collection"""
    try:
        logger.info("Starting data collection pipeline")
        
        # Check required environment variables
        if not BRONZE_BUCKET:
            raise ValueError("BRONZE_BUCKET environment variable is required")
        
        # Ensure S3 folder structure
        ensure_s3_folder_structure(BRONZE_BUCKET)
        
        # Validate bronze bucket access
        if not validate_bronze_bucket_access(BRONZE_BUCKET):
            raise ValueError("Could not access or validate BRONZE_BUCKET. Exiting.")
        
        # Initialize collectors
        fred_collector = FREDCollector()
        worldbank_collector = WorldBankCollector()
        yahoo_collector = YahooFinanceCollector()
        
        # Collect data from all sources
        all_results = {
            'pipeline_start_time': datetime.now(timezone.utc).isoformat(),
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
        all_results['pipeline_end_time'] = datetime.now(timezone.utc).isoformat()
        
        # Log comprehensive pipeline summary
        log_pipeline_summary(all_results)
        
        # Save overall results to S3
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        results_s3_path = f"pipeline_results/data_collection/{timestamp}_pipeline_results.json"
        
        try:
            s3_client.put_object(
                Bucket=BRONZE_BUCKET,
                Key=results_s3_path,
                Body=json.dumps(all_results, indent=2),
                ContentType='application/json'
            )
            logger.info(f"Data collection pipeline completed. Results saved to s3://{BRONZE_BUCKET}/{results_s3_path}")
        except Exception as e:
            logger.error(f"Failed to save results to S3: {e}")
            all_results['overall_status'] = 'failed'
        
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
