#!/usr/bin/env python3
"""
World Bank Data Collector
Collects economic indicators from World Bank API
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
                    params['date'] = f"{current_year-10}:{current_year}"
            else:
                # Default to last 10 years
                current_year = datetime.now(timezone.utc).year
                params['date'] = f"{current_year-10}:{current_year}"
            
            logger.info(f"Fetching World Bank data for country: {country_code}, indicator: {indicator}")
            logger.debug(f"World Bank API URL: {url}")
            logger.debug(f"World Bank API parameters: {params}")
            
            # Use the enhanced HTTP request method from base collector
            response = self.make_http_request(url, params=params, timeout=30)
            
            # Log detailed error information if the request failed
            if response.status_code >= 400:
                logger.error(f"World Bank API request failed for {country_code}/{indicator}")
                logger.error(f"Status Code: {response.status_code}")
                logger.error(f"URL: {response.url}")
                logger.error(f"Request Parameters: {params}")
                try:
                    error_body = response.text
                    logger.error(f"Response Body: {error_body}")
                    # Try to parse as JSON for better formatting
                    try:
                        error_json = response.json()
                        logger.error(f"Response JSON: {error_json}")
                    except:
                        pass
                except Exception as e:
                    logger.error(f"Could not read response body: {e}")
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error fetching World Bank data for country {country_code}, indicator {indicator}: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception details: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
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
            
            logger.debug(f"Processing {len(data_points)} data points for World Bank {country_code}/{indicator}")
            
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
            df['collection_timestamp'] = datetime.now(timezone.utc).isoformat()
            
            logger.debug(f"Processed {len(df)} records for World Bank {country_code}/{indicator}")
            return df
            
        except Exception as e:
            logger.error(f"Error processing World Bank data for country {country_code}, indicator {indicator}: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception details: {str(e)}")
            logger.error(f"Raw data structure: {type(raw_data)}")
            if isinstance(raw_data, list):
                logger.error(f"Raw data length: {len(raw_data)}")
                if raw_data:
                    logger.error(f"First element type: {type(raw_data[0])}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def collect(self, start_date: int = None, end_date: int = None) -> Dict[str, Any]:
        """Collect World Bank data with comprehensive error handling"""
        try:
            # Validate environment
            if not self.validate_environment():
                raise ValueError("Environment validation failed")
            
            # Log collection start
            self.log_collection_start(start_date=start_date, end_date=end_date)
            
            # Determine date range (last 10 years by default)
            if not end_date:
                end_date = datetime.now(timezone.utc).year
            if not start_date:
                start_date = end_date - 10
            
            self.results['start_date'] = start_date
            self.results['end_date'] = end_date
            self.results['total_combinations'] = len(COUNTRIES) * len(WORLDBANK_INDICATORS)
            
            logger.info(f"Processing {len(COUNTRIES)} countries and {len(WORLDBANK_INDICATORS)} indicators from {start_date} to {end_date}")
            
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
                            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
                            s3_path = f"raw/worldbank/{country_code}/{indicator}/{timestamp}_{country_code}_{indicator}.parquet"
                            
                            # Save to S3
                            s3_key = self.save_to_s3(df, s3_path, BRONZE_BUCKET)
                            
                            # Add success result
                            self.add_success_result(
                                item_id=f"{country_code}_{indicator}",
                                country_code=country_code,
                                indicator_code=indicator,
                                indicator_name=indicator_name,
                                records_count=len(df),
                                s3_key=s3_key,
                                date_range=f"{df['date'].min().year} to {df['date'].max().year}"
                            )
                        else:
                            self.add_failed_result(
                                item_id=f"{country_code}_{indicator}",
                                error='No data returned from World Bank API',
                                country_code=country_code,
                                indicator_code=indicator
                            )
                        
                        # Rate limiting - World Bank allows 100 requests per minute
                        time.sleep(0.6)
                        
                    except Exception as e:
                        logger.error(f"Failed to process World Bank country {country_code}, indicator {indicator}: {e}")
                        logger.error(f"Exception type: {type(e).__name__}")
                        logger.error(f"Exception details: {str(e)}")
                        import traceback
                        logger.error(f"Full traceback: {traceback.format_exc()}")
                        self.add_failed_result(
                            item_id=f"{country_code}_{indicator}",
                            error=str(e),
                            country_code=country_code,
                            indicator_code=indicator
                        )
            
            # Log collection end
            self.log_collection_end()
            
            return self.results
            
        except Exception as e:
            logger.error(f"World Bank data collection failed: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception details: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            self.add_failed_result(
                item_id="collection",
                error=f"Collection failed: {str(e)}"
            )
            return self.results
