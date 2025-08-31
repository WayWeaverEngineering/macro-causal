#!/usr/bin/env python3
"""
FRED Data Collector
Collects economic data from Federal Reserve Economic Data (FRED) API
"""

import pandas as pd
from datetime import datetime, timedelta, timezone
import logging
import time
from typing import Dict, List, Any
from .base_collector import DataCollector, BRONZE_BUCKET

# Configure logging
logger = logging.getLogger(__name__)

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

class FREDCollector(DataCollector):
    """FRED data collector with enhanced error handling"""
    
    def __init__(self):
        super().__init__(collector_name="FRED")
    
    def fetch_fred_data(self, series_id: str, api_key: str, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Fetch data for a specific FRED series with enhanced error handling"""
        try:
            # Validate and fix date range
            if not start_date or not end_date:
                start_date, end_date = self.get_default_date_range(30)
            else:
                start_date, end_date = self.validate_date_range(start_date, end_date)
            
            params = {
                'series_id': series_id,
                'file_type': 'json',
                'sort_order': 'asc'
            }
            
            # Add API key if available
            if api_key:
                params['api_key'] = api_key
            
            # Add date parameters
            params['observation_start'] = start_date
            params['observation_end'] = end_date
                
            logger.info(f"Fetching FRED data for series: {series_id} from {start_date} to {end_date}")
            logger.debug(f"FRED API parameters: {params}")
            
            # Use the enhanced HTTP request method from base collector
            response = self.make_http_request(FRED_BASE_URL, params=params, timeout=30)
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error fetching FRED data for series {series_id}: {e}")
            logger.debug(f"FRED API error details for {series_id}: {e}")
            raise
    
    def process_fred_data(self, raw_data: Dict[str, Any], series_id: str) -> pd.DataFrame:
        """Process raw FRED data into a structured DataFrame"""
        try:
            observations = raw_data.get('observations', [])
            
            if not observations:
                logger.warning(f"No observations found for FRED series {series_id}")
                return pd.DataFrame()
            
            logger.debug(f"Processing {len(observations)} observations for FRED series {series_id}")
            
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
            df['collection_timestamp'] = datetime.now(timezone.utc).isoformat()
            
            logger.debug(f"Processed {len(df)} records for FRED series {series_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error processing FRED data for series {series_id}: {e}")
            logger.debug(f"FRED data processing error details: {e}")
            raise
    
    def collect(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Collect FRED data with comprehensive error handling"""
        try:
            # Validate environment
            if not self.validate_environment():
                raise ValueError("Environment validation failed")
            
            # Log collection start
            self.log_collection_start(start_date=start_date, end_date=end_date)
            
            # Get API key (optional)
            api_key = self.get_api_key('FRED_API_KEY')
            if not api_key:
                logger.warning("FRED API key not found. Using limited access mode (120 requests/minute).")
                logger.warning("For production use, please obtain a free FRED API key from https://fred.stlouisfed.org/docs/api/api_key.html")
                logger.warning("Set API_SECRETS_ARN environment variable and add FRED_API_KEY to the secret for full access.")
                # Continue without API key (limited rate)
                api_key = None
            
            # Determine date range (last 30 days by default)
            if not end_date or not start_date:
                start_date, end_date = self.get_default_date_range(30)
            else:
                start_date, end_date = self.validate_date_range(start_date, end_date)
            
            self.results['start_date'] = start_date
            self.results['end_date'] = end_date
            self.results['total_series'] = len(FRED_SERIES)
            
            logger.info(f"Processing {len(FRED_SERIES)} FRED series from {start_date} to {end_date}")
            
            # Process each series
            for series_id in FRED_SERIES:
                try:
                    logger.info(f"Processing FRED series: {series_id}")
                    
                    # Fetch data from FRED API
                    raw_data = self.fetch_fred_data(series_id, api_key, start_date, end_date)
                    
                    # Process the data
                    df = self.process_fred_data(raw_data, series_id)
                    
                    if not df.empty:
                        # Create S3 path
                        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
                        s3_path = f"raw/fred/{series_id}/{timestamp}_{series_id}.parquet"
                        
                        # Save to S3
                        s3_key = self.save_to_s3(df, s3_path, BRONZE_BUCKET)
                        
                        # Add success result
                        self.add_success_result(
                            item_id=series_id,
                            records_count=len(df),
                            s3_key=s3_key,
                            date_range=f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
                        )
                    else:
                        self.add_failed_result(
                            item_id=series_id,
                            error='No data returned from FRED API'
                        )
                    
                    # Rate limiting - FRED allows 120 requests per minute without API key, 1200 with API key
                    if api_key:
                        time.sleep(0.5)  # 1200 requests per minute = 0.5 seconds between requests
                    else:
                        time.sleep(0.5)  # 120 requests per minute = 0.5 seconds between requests (same for now)
                    
                except Exception as e:
                    logger.error(f"Failed to process FRED series {series_id}: {e}")
                    self.add_failed_result(
                        item_id=series_id,
                        error=str(e)
                    )
            
            # Log collection end
            self.log_collection_end()
            
            return self.results
            
        except Exception as e:
            logger.error(f"FRED data collection failed: {e}")
            logger.debug(f"FRED collection error details: {e}")
            self.add_failed_result(
                item_id="collection",
                error=f"Collection failed: {str(e)}"
            )
            return self.results
