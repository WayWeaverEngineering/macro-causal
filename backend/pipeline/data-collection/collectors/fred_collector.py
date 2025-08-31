#!/usr/bin/env python3
"""
FRED Data Collector
Collects economic data from Federal Reserve Economic Data (FRED) API
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
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
    """FRED data collector"""
    
    def fetch_fred_data(self, series_id: str, api_key: str, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
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
                
            logger.info(f"Fetching FRED data for series: {series_id}")
            response = requests.get(FRED_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching FRED data for series {series_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error for FRED series {series_id}: {e}")
            raise
    
    def process_fred_data(self, raw_data: Dict[str, Any], series_id: str) -> pd.DataFrame:
        """Process raw FRED data into a structured DataFrame"""
        try:
            observations = raw_data.get('observations', [])
            
            if not observations:
                logger.warning(f"No observations found for FRED series {series_id}")
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
            logger.error(f"Error processing FRED data for series {series_id}: {e}")
            raise
    
    def collect(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Collect FRED data"""
        logger.info("Starting FRED data collection")
        
        # Get API key
        api_key = self.get_api_key('FRED_API_KEY')
        if not api_key:
            raise ValueError("FRED API key not found in Secrets Manager")
        
        # Determine date range (last 30 days by default)
        if not end_date:
            end_date = datetime.utcnow().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        self.results['start_date'] = start_date
        self.results['end_date'] = end_date
        self.results['total_series'] = len(FRED_SERIES)
        
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
                    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                    s3_path = f"raw/fred/{series_id}/{timestamp}_{series_id}.json"
                    
                    # Save to S3
                    s3_key = self.save_to_s3(df, s3_path, BRONZE_BUCKET)
                    
                    self.results['success'].append({
                        'series_id': series_id,
                        'records_count': len(df),
                        's3_key': s3_key,
                        'date_range': f"{df['date'].min()} to {df['date'].max()}"
                    })
                else:
                    self.results['failed'].append({
                        'series_id': series_id,
                        'error': 'No data returned'
                    })
                
                # Rate limiting - FRED allows 120 requests per minute
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Failed to process FRED series {series_id}: {e}")
                self.results['failed'].append({
                    'series_id': series_id,
                    'error': str(e)
                })
        
        logger.info(f"FRED data collection completed. Success: {len(self.results['success'])}, Failed: {len(self.results['failed'])}")
        return self.results
