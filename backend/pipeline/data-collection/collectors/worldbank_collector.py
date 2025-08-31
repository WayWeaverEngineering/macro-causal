#!/usr/bin/env python3
"""
World Bank Data Collector
Collects global development indicators from World Bank API
"""

import pandas as pd
from datetime import datetime, timezone
import logging
import time
from typing import Dict, List, Any
from .base_collector import DataCollector, BRONZE_BUCKET

# Configure logging
logger = logging.getLogger(__name__)

# World Bank API configuration
WORLDBANK_BASE_URL = 'https://api.worldbank.org/v2/countries'

# World Bank indicators (major economic indicators)
WORLDBANK_INDICATORS = [
    'NY.GDP.MKTP.CD',      # GDP (current US$)
    'NY.GDP.MKTP.KD.ZG',   # GDP growth (annual %)
    'SL.UEM.TOTL.ZS',      # Unemployment, total (% of total labor force)
    'FP.CPI.TOTL.ZG',      # Inflation, consumer prices (annual %)
    'FR.INR.RINR',         # Real interest rate (%)
    'PA.NUS.FCRF',         # Official exchange rate (LCU per US$, period average)
    'NE.TRD.GNFS.ZS',      # Trade (% of GDP)
    'GC.DOD.TOTL.GD.ZS',   # Central government debt, total (% of GDP)
    'NY.GNS.ICTR.ZS',      # Gross savings (% of GNI)
    'SE.ADT.LITR.ZS'       # Literacy rate, adult total (% of people ages 15 and above)
]

# Target countries (major economies)
COUNTRIES = ['US', 'CN', 'JP', 'DE', 'GB', 'FR', 'IN', 'IT', 'CA', 'BR']

class WorldBankCollector(DataCollector):
    """World Bank data collector with enhanced error handling"""
    
    def __init__(self):
        super().__init__(collector_name="WorldBank")
    
    def fetch_worldbank_data(self, country_code: str, indicator: str, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Fetch data for a specific World Bank indicator and country with enhanced error handling"""
        try:
            url = f"{WORLDBANK_BASE_URL}/{country_code}/indicator/{indicator}"
            
            params = {
                'format': 'json',
                'per_page': 1000  # Maximum records per request
            }
            
            # Handle date range for World Bank API (uses years)
            if start_date and end_date:
                try:
                    start_year = int(start_date)
                    end_year = int(end_date)
                    params['date'] = f"{start_year}:{end_year}"
                except ValueError:
                    # If dates are not years, use current year
                    current_year = datetime.now(timezone.utc).year
                    params['date'] = f"{current_year-50}:{current_year}"
            else:
                # Default to last 50 years
                current_year = datetime.now(timezone.utc).year
                params['date'] = f"{current_year-50}:{current_year}"
            
            # Use the enhanced HTTP request method from base collector
            response = self.make_http_request(url, params=params, timeout=30)
            
            return response.json()
            
        except Exception as e:
            self.log_consolidated_error(f"World Bank data fetch for {country_code}/{indicator}", e,
                                      additional_info={"start_date": start_date, "end_date": end_date})
            raise
    
    def process_worldbank_data(self, raw_data: List[Dict[str, Any]], country_code: str, 
                              indicator: str, indicator_name: str) -> pd.DataFrame:
        """Process raw World Bank data into a structured DataFrame"""
        try:
            if not raw_data or len(raw_data) < 2:
                self.logger.warning(f"No data found for World Bank {country_code}/{indicator}")
                return pd.DataFrame()
            
            # World Bank API returns metadata in first element, data in second element
            data_points = raw_data[1] if len(raw_data) > 1 else []
            
            if not data_points:
                self.logger.warning(f"No data points found for World Bank {country_code}/{indicator}")
                return pd.DataFrame()
            
            # Extract data points
            processed_data = []
            for point in data_points:
                processed_data.append({
                    'date': point.get('date'),
                    'value': point.get('value'),
                    'country': point.get('country', {}).get('value', country_code),
                    'indicator': point.get('indicator', {}).get('value', indicator_name)
                })
            
            df = pd.DataFrame(processed_data)
            
            # Convert date column (World Bank uses years)
            df['date'] = pd.to_datetime(df['date'] + '-01-01', errors='coerce')
            
            # Convert value column to numeric, handling 'None' values
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Add metadata
            df['country_code'] = country_code
            df['indicator_code'] = indicator
            df['source'] = 'WorldBank'
            df['collection_timestamp'] = datetime.now(timezone.utc).isoformat()
            
            return df
            
        except Exception as e:
            self.log_consolidated_error(f"World Bank data processing for {country_code}/{indicator}", e,
                                      additional_info={"raw_data_length": len(raw_data) if raw_data else 0})
            raise
    
    def collect(self, start_date: int = None, end_date: int = None) -> Dict[str, Any]:
        """Collect World Bank data with comprehensive error handling"""
        try:
            # Validate environment
            if not self.validate_environment():
                raise ValueError("Environment validation failed")
            
            # Log collection start
            self.log_collection_start(start_date=start_date, end_date=end_date)
            
            # Determine date range (last 50 years by default)
            if not end_date:
                end_date = datetime.now(timezone.utc).year
            if not start_date:
                start_date = end_date - 50
            
            self.results['start_date'] = start_date
            self.results['end_date'] = end_date
            self.results['total_combinations'] = len(COUNTRIES) * len(WORLDBANK_INDICATORS)
            
            # Process each country and indicator combination
            combination_count = 0
            total_combinations = len(COUNTRIES) * len(WORLDBANK_INDICATORS)
            
            for country_code in COUNTRIES:
                for indicator in WORLDBANK_INDICATORS:
                    combination_count += 1
                    try:
                        # Fetch data from World Bank API
                        raw_data = self.fetch_worldbank_data(country_code, indicator, start_date, end_date)
                        
                        # Get indicator name from metadata if available
                        indicator_name = indicator
                        if raw_data and len(raw_data) > 0:
                            metadata = raw_data[0] if isinstance(raw_data[0], dict) else {}
                            if 'indicator' in metadata:
                                indicator_name = metadata['indicator'].get('value', indicator)
                        
                        # Process the data
                        df = self.process_worldbank_data(raw_data, country_code, indicator, indicator_name)
                        
                        if not df.empty:
                            # Create S3 path
                            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
                            s3_path = f"raw/worldbank/{country_code}/{indicator}/{timestamp}_{country_code}_{indicator}.parquet"
                            
                            # Save to S3
                            s3_key = self.save_to_s3(df, s3_path, BRONZE_BUCKET)
                            
                            # Add success result
                            self.add_success_result(
                                item_id=f"{country_code}/{indicator}",
                                records_count=len(df),
                                s3_key=s3_key,
                                date_range=f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
                            )
                        else:
                            self.add_failed_result(
                                item_id=f"{country_code}/{indicator}",
                                error='No data returned from World Bank API'
                            )
                        
                        # Rate limiting - World Bank allows 100 requests per minute
                        time.sleep(0.6)  # 100 requests per minute = 0.6 seconds between requests
                        
                    except Exception as e:
                        self.log_consolidated_error(f"World Bank processing {country_code}/{indicator}", e)
                        self.add_failed_result(
                            item_id=f"{country_code}/{indicator}",
                            error=str(e)
                        )
            
            # Log collection end
            self.log_collection_end()
            
            return self.results
            
        except Exception as e:
            self.log_consolidated_error("World Bank data collection", e)
            self.add_failed_result(
                item_id="collection",
                error=f"Collection failed: {str(e)}"
            )
            return self.results
