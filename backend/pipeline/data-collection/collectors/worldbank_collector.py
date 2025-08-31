#!/usr/bin/env python3
"""
World Bank Data Collector
Collects economic indicators from World Bank API
"""

import requests
import pandas as pd
from datetime import datetime
import logging
import time
from typing import Dict, List, Any
from .base_collector import DataCollector, BRONZE_BUCKET

# Configure logging
logger = logging.getLogger(__name__)

# World Bank API configuration
WORLDBANK_BASE_URL = 'https://api.worldbank.org/v2/country'
WORLDBANK_INDICATORS = [
    {'indicator': 'NY.GDP.MKTP.CD', 'name': 'GDP (current US$)'},
    {'indicator': 'NY.GDP.MKTP.KD.ZG', 'name': 'GDP growth (annual %)'},
    {'indicator': 'SL.UEM.TOTL.ZS', 'name': 'Unemployment, total (% of total labor force)'},
    {'indicator': 'FP.CPI.TOTL.ZG', 'name': 'Inflation, consumer prices (annual %)'},
    {'indicator': 'FR.INR.RINR', 'name': 'Real interest rate (%)'},
    {'indicator': 'PA.NUS.FCRF', 'name': 'Official exchange rate (LCU per US$, period average)'},
    {'indicator': 'NE.EXP.GNFS.ZS', 'name': 'Exports of goods and services (% of GDP)'},
    {'indicator': 'NE.IMP.GNFS.ZS', 'name': 'Imports of goods and services (% of GDP)'},
    {'indicator': 'GC.DOD.TOTL.GD.ZS', 'name': 'Central government debt, total (% of GDP)'},
    {'indicator': 'NY.GNS.ICTR.ZS', 'name': 'Gross savings (% of GDP)'}
]

# Target countries (major economies)
COUNTRIES = ['US', 'CN', 'JP', 'DE', 'GB', 'FR', 'IN', 'IT', 'CA', 'BR']

class WorldBankCollector(DataCollector):
    """World Bank data collector"""
    
    def fetch_worldbank_data(self, country_code: str, indicator: str, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Fetch data for a specific World Bank indicator and country"""
        try:
            url = f"{WORLDBANK_BASE_URL}/{country_code}/indicator/{indicator}"
            
            params = {
                'format': 'json',
                'per_page': 1000  # Maximum records per request
            }
            
            if start_date:
                params['date'] = f"{start_date}:{end_date or datetime.utcnow().year}"
            
            logger.info(f"Fetching World Bank data for country: {country_code}, indicator: {indicator}")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching World Bank data for country {country_code}, indicator {indicator}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error for World Bank country {country_code}, indicator {indicator}: {e}")
            raise
    
    def process_worldbank_data(self, raw_data: List[Dict[str, Any]], country_code: str, 
                              indicator: str, indicator_name: str) -> pd.DataFrame:
        """Process raw World Bank data into a structured DataFrame"""
        try:
            if not raw_data or len(raw_data) < 2:
                logger.warning(f"No data found for World Bank country {country_code}, indicator {indicator}")
                return pd.DataFrame()
            
            # World Bank API returns metadata in first element, data in second
            data_points = raw_data[1] if len(raw_data) > 1 else []
            
            if not data_points:
                logger.warning(f"No data points found for World Bank country {country_code}, indicator {indicator}")
                return pd.DataFrame()
            
            # Extract data points
            processed_data = []
            for point in data_points:
                if point.get('value') is not None:  # Only include records with values
                    processed_data.append({
                        'date': point.get('date'),
                        'value': point.get('value'),
                        'country_code': country_code,
                        'indicator_code': indicator,
                        'indicator_name': indicator_name,
                        'unit': point.get('unit', ''),
                        'obs_status': point.get('obs_status', ''),
                        'decimal': point.get('decimal', 0)
                    })
            
            df = pd.DataFrame(processed_data)
            
            if df.empty:
                return df
            
            # Convert date column
            df['date'] = pd.to_datetime(df['date'], format='%Y')
            
            # Convert value column to numeric
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Add metadata
            df['source'] = 'WorldBank'
            df['collection_timestamp'] = datetime.utcnow().isoformat()
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing World Bank data for country {country_code}, indicator {indicator}: {e}")
            raise
    
    def collect(self, start_date: int = None, end_date: int = None) -> Dict[str, Any]:
        """Collect World Bank data"""
        logger.info("Starting World Bank data collection")
        
        # Determine date range (last 10 years by default)
        if not end_date:
            end_date = datetime.utcnow().year
        if not start_date:
            start_date = end_date - 10
        
        self.results['start_date'] = start_date
        self.results['end_date'] = end_date
        self.results['total_combinations'] = len(COUNTRIES) * len(WORLDBANK_INDICATORS)
        
        # Process each country and indicator combination
        for country_code in COUNTRIES:
            for indicator_info in WORLDBANK_INDICATORS:
                indicator = indicator_info['indicator']
                indicator_name = indicator_info['name']
                
                try:
                    logger.info(f"Processing World Bank country: {country_code}, indicator: {indicator}")
                    
                    # Fetch data from World Bank API
                    raw_data = self.fetch_worldbank_data(country_code, indicator, str(start_date), str(end_date))
                    
                    # Process the data
                    df = self.process_worldbank_data(raw_data, country_code, indicator, indicator_name)
                    
                    if not df.empty:
                        # Create S3 path
                        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                        s3_path = f"raw/worldbank/{country_code}/{indicator}/{timestamp}_{country_code}_{indicator}.json"
                        
                        # Save to S3
                        s3_key = self.save_to_s3(df, s3_path, BRONZE_BUCKET)
                        
                        self.results['success'].append({
                            'country_code': country_code,
                            'indicator_code': indicator,
                            'indicator_name': indicator_name,
                            'records_count': len(df),
                            's3_key': s3_key,
                            'date_range': f"{df['date'].min().year} to {df['date'].max().year}"
                        })
                    else:
                        self.results['failed'].append({
                            'country_code': country_code,
                            'indicator_code': indicator,
                            'error': 'No data returned'
                        })
                    
                    # Rate limiting - World Bank allows 100 requests per minute
                    time.sleep(0.6)
                    
                except Exception as e:
                    logger.error(f"Failed to process World Bank country {country_code}, indicator {indicator}: {e}")
                    self.results['failed'].append({
                        'country_code': country_code,
                        'indicator_code': indicator,
                        'error': str(e)
                    })
        
        logger.info(f"World Bank data collection completed. Success: {len(self.results['success'])}, Failed: {len(self.results['failed'])}")
        return self.results
