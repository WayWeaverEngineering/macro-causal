import json
import boto3
import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import logging
from typing import Dict, List, Any
import time
import yfinance as yf

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
secrets_client = boto3.client('secretsmanager')

# Environment variables
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'dev')
BRONZE_BUCKET = os.environ.get('BRONZE_BUCKET')
API_SECRETS_ARN = os.environ.get('API_SECRETS_ARN')

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

def get_api_key() -> str:
    """Retrieve Yahoo Finance API key from Secrets Manager (optional)"""
    try:
        response = secrets_client.get_secret_value(SecretId=API_SECRETS_ARN)
        secrets = json.loads(response['SecretString'])
        return secrets.get('YAHOO_FINANCE_API_KEY', '')
    except Exception as e:
        logger.warning(f"Yahoo Finance API key not found, using public access: {e}")
        return ''

def fetch_yahoo_data(symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Fetch data for a specific Yahoo Finance symbol"""
    try:
        logger.info(f"Fetching data for symbol: {symbol}")
        
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
            logger.warning(f"No data returned for symbol {symbol}")
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
        logger.error(f"Error fetching data for symbol {symbol}: {e}")
        raise

def get_symbol_info(symbol: str) -> Dict[str, Any]:
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
        logger.warning(f"Could not fetch info for symbol {symbol}: {e}")
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

def save_to_s3(df: pd.DataFrame, symbol: str, bucket: str, data_type: str = 'price') -> str:
    """Save DataFrame to S3 as JSON"""
    try:
        if df.empty:
            logger.warning(f"No data to save for symbol {symbol}")
            return ""
        
        # Create filename with timestamp
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"raw/yahoo_finance/{symbol}/{data_type}/{timestamp}_{symbol}_{data_type}.json"
        
        # Convert DataFrame to JSON
        json_data = df.to_json(orient='records', date_format='iso')
        
        # Upload to S3
        s3_client.put_object(
            Bucket=bucket,
            Key=filename,
            Body=json_data,
            ContentType='application/json'
        )
        
        logger.info(f"Successfully saved {len(df)} records for symbol {symbol} to s3://{bucket}/{filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Error saving data for symbol {symbol}: {e}")
        raise

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Main Lambda handler function"""
    try:
        logger.info("Starting Yahoo Finance data collection")
        
        # Get API key (optional for Yahoo Finance)
        api_key = get_api_key()
        
        # Determine date range (last 30 days by default)
        end_date = datetime.utcnow().strftime('%Y-%m-%d')
        start_date = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        # Override with event parameters if provided
        if 'start_date' in event:
            start_date = event['start_date']
        if 'end_date' in event:
            end_date = event['end_date']
        
        results = {
            'success': [],
            'failed': [],
            'total_symbols': len(YAHOO_SYMBOLS),
            'start_date': start_date,
            'end_date': end_date,
            'collection_timestamp': datetime.utcnow().isoformat()
        }
        
        # Process each symbol
        for symbol in YAHOO_SYMBOLS:
            try:
                logger.info(f"Processing symbol: {symbol}")
                
                # Fetch price data
                df = fetch_yahoo_data(symbol, start_date, end_date)
                
                if not df.empty:
                    # Save price data to S3
                    price_s3_key = save_to_s3(df, symbol, BRONZE_BUCKET, 'price')
                    
                    # Get symbol information
                    symbol_info = get_symbol_info(symbol)
                    
                    # Save symbol info to S3
                    info_df = pd.DataFrame([symbol_info])
                    info_s3_key = save_to_s3(info_df, symbol, BRONZE_BUCKET, 'info')
                    
                    results['success'].append({
                        'symbol': symbol,
                        'price_records_count': len(df),
                        'price_s3_key': price_s3_key,
                        'info_s3_key': info_s3_key,
                        'date_range': f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
                        'symbol_name': symbol_info.get('name', ''),
                        'sector': symbol_info.get('sector', '')
                    })
                else:
                    results['failed'].append({
                        'symbol': symbol,
                        'error': 'No data returned'
                    })
                
                # Rate limiting - be respectful to Yahoo Finance
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to process symbol {symbol}: {e}")
                results['failed'].append({
                    'symbol': symbol,
                    'error': str(e)
                })
        
        logger.info(f"Yahoo Finance data collection completed. Success: {len(results['success'])}, Failed: {len(results['failed'])}")
        
        return {
            'statusCode': 200,
            'body': json.dumps(results),
            'headers': {
                'Content-Type': 'application/json'
            }
        }
        
    except Exception as e:
        logger.error(f"Error in Yahoo Finance data collection: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'message': 'Internal server error during Yahoo Finance data collection'
            }),
            'headers': {
                'Content-Type': 'application/json'
            }
        }
