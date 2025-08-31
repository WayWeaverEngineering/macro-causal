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
import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import logging
import time
from typing import Dict, List, Any
import yfinance as yf
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize AWS clients
s3_client = boto3.client('s3')
secrets_client = boto3.client('secretsmanager')

# Environment variables
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'dev')
BRONZE_BUCKET = os.environ.get('BRONZE_BUCKET')
API_SECRETS_ARN = os.environ.get('API_SECRETS_ARN')

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

# Yahoo Finance configuration
YAHOO_SYMBOLS = [
    # Major US Indices
    '^GSPC',    # S&P 500
    '^DJI',     # Dow Jones Industrial Average
    '^IXIC',    # NASDAQ Composite
    '^VIX',     # CBOE Volatility Index
    
    # Major ETFs
    'SPY',      # SPDR S&P 500 ETF
    'QQQ',      # Invesco QQQ Trust
    'IWM',      # iShares Russell 2000 ETF
    'GLD',      # SPDR Gold Shares
    'TLT',      # iShares 20+ Year Treasury Bond ETF
    
    # Major Stocks
    'AAPL',     # Apple Inc.
    'MSFT',     # Microsoft Corporation
    'GOOGL',    # Alphabet Inc.
    'AMZN',     # Amazon.com Inc.
    'TSLA',     # Tesla Inc.
    'NVDA',     # NVIDIA Corporation
    'META',     # Meta Platforms Inc.
    'BRK-B',    # Berkshire Hathaway Inc.
    'JPM',      # JPMorgan Chase & Co.
    'JNJ',      # Johnson & Johnson
    
    # Commodities
    'GC=F',     # Gold Futures
    'CL=F',     # Crude Oil Futures
    'NG=F',     # Natural Gas Futures
    'ZC=F',     # Corn Futures
    'ZS=F',     # Soybean Futures
    
    # Currencies
    'EURUSD=X', # Euro/US Dollar
    'GBPUSD=X', # British Pound/US Dollar
    'USDJPY=X', # US Dollar/Japanese Yen
    'USDCNY=X', # US Dollar/Chinese Yuan
    'USDCAD=X', # US Dollar/Canadian Dollar
]

class DataCollector:
    """Base class for data collectors"""
    
    def __init__(self):
        self.results = {
            'success': [],
            'failed': [],
            'collection_timestamp': datetime.utcnow().isoformat()
        }
    
    def get_api_key(self, key_name: str) -> str:
        """Retrieve API key from Secrets Manager"""
        try:
            response = secrets_client.get_secret_value(SecretId=API_SECRETS_ARN)
            secrets = json.loads(response['SecretString'])
            return secrets.get(key_name, '')
        except Exception as e:
            logger.warning(f"API key {key_name} not found in Secrets Manager: {e}")
            return ''
    
    def save_to_s3(self, df: pd.DataFrame, s3_path: str, bucket: str) -> str:
        """Save DataFrame to S3 as JSON"""
        try:
            if df.empty:
                logger.warning(f"No data to save to {s3_path}")
                return ""
            
            # Convert DataFrame to JSON
            json_data = df.to_json(orient='records', date_format='iso')
            
            # Upload to S3
            s3_client.put_object(
                Bucket=bucket,
                Key=s3_path,
                Body=json_data,
                ContentType='application/json'
            )
            
            logger.info(f"Successfully saved {len(df)} records to s3://{bucket}/{s3_path}")
            return s3_path
            
        except Exception as e:
            logger.error(f"Error saving data to {s3_path}: {e}")
            raise

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

class YahooFinanceCollector(DataCollector):
    """Yahoo Finance data collector"""
    
    def fetch_yahoo_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Fetch data for a specific Yahoo Finance symbol"""
        try:
            logger.info(f"Fetching Yahoo Finance data for symbol: {symbol}")
            
            # Use yfinance library to fetch data
            ticker = yf.Ticker(symbol)
            
            # Determine date range
            if not start_date:
                start_date = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.utcnow().strftime('%Y-%m-%d')
            
            # Fetch historical data
            df = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if df.empty:
                logger.warning(f"No data returned for Yahoo Finance symbol {symbol}")
                return pd.DataFrame()
            
            # Reset index to make date a column
            df = df.reset_index()
            
            # Rename columns for consistency
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Dividends': 'dividends',
                'Stock Splits': 'stock_splits'
            })
            
            # Add metadata
            df['symbol'] = symbol
            df['source'] = 'YahooFinance'
            df['collection_timestamp'] = datetime.utcnow().isoformat()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data for symbol {symbol}: {e}")
            raise
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get additional information about a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', ''),
                'exchange': info.get('exchange', ''),
                'country': info.get('country', ''),
                'website': info.get('website', ''),
                'description': info.get('longBusinessSummary', '')
            }
            
        except Exception as e:
            logger.warning(f"Could not fetch info for Yahoo Finance symbol {symbol}: {e}")
            return {
                'symbol': symbol,
                'name': '',
                'sector': '',
                'industry': '',
                'market_cap': 0,
                'currency': '',
                'exchange': '',
                'country': '',
                'website': '',
                'description': ''
            }
    
    def collect(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Collect Yahoo Finance data"""
        logger.info("Starting Yahoo Finance data collection")
        
        # Determine date range (last 30 days by default)
        if not end_date:
            end_date = datetime.utcnow().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        self.results['start_date'] = start_date
        self.results['end_date'] = end_date
        self.results['total_symbols'] = len(YAHOO_SYMBOLS)
        
        # Process each symbol
        for symbol in YAHOO_SYMBOLS:
            try:
                logger.info(f"Processing Yahoo Finance symbol: {symbol}")
                
                # Fetch price data
                df = self.fetch_yahoo_data(symbol, start_date, end_date)
                
                if not df.empty:
                    # Create S3 paths
                    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                    price_s3_path = f"raw/yahoo_finance/{symbol}/price/{timestamp}_{symbol}_price.json"
                    info_s3_path = f"raw/yahoo_finance/{symbol}/info/{timestamp}_{symbol}_info.json"
                    
                    # Save price data to S3
                    price_s3_key = self.save_to_s3(df, price_s3_path, BRONZE_BUCKET)
                    
                    # Get symbol information
                    symbol_info = self.get_symbol_info(symbol)
                    
                    # Save symbol info to S3
                    info_df = pd.DataFrame([symbol_info])
                    info_s3_key = self.save_to_s3(info_df, info_s3_path, BRONZE_BUCKET)
                    
                    self.results['success'].append({
                        'symbol': symbol,
                        'price_records_count': len(df),
                        'price_s3_key': price_s3_key,
                        'info_s3_key': info_s3_key,
                        'date_range': f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
                        'symbol_name': symbol_info.get('name', ''),
                        'sector': symbol_info.get('sector', '')
                    })
                else:
                    self.results['failed'].append({
                        'symbol': symbol,
                        'error': 'No data returned'
                    })
                
                # Rate limiting - be respectful to Yahoo Finance
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to process Yahoo Finance symbol {symbol}: {e}")
                self.results['failed'].append({
                    'symbol': symbol,
                    'error': str(e)
                })
        
        logger.info(f"Yahoo Finance data collection completed. Success: {len(self.results['success'])}, Failed: {len(self.results['failed'])}")
        return self.results

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
