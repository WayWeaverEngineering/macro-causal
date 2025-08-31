#!/usr/bin/env python3
"""
Yahoo Finance Data Collector
Collects financial market data from Yahoo Finance using direct API calls
"""

import pandas as pd
from datetime import datetime, timedelta, timezone
import logging
import time
import requests
import json
from typing import Dict, Any, Optional
from .base_collector import DataCollector, BRONZE_BUCKET

# Configure logging
logger = logging.getLogger(__name__)

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

# Alternative symbols for problematic ones
SYMBOL_ALTERNATIVES = {
    'JNJ': ['JNJ', 'JNJ.N'],  # Try alternative symbols for JNJ
}

class YahooFinanceCollector(DataCollector):
    """Yahoo Finance data collector using direct API calls"""
    
    def __init__(self):
        super().__init__(collector_name="YahooFinance")
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def _get_yahoo_crumb(self, symbol: str) -> str:
        """Get Yahoo Finance crumb for API authentication"""
        try:
            # First get the quote page to extract crumb
            quote_url = f"https://finance.yahoo.com/quote/{symbol}"
            response = self.session.get(quote_url, timeout=10)
            
            response.raise_for_status()
            
            # Extract crumb from the page content
            content = response.text
            crumb_start = content.find('"CrumbStore":{"crumb":"')
            if crumb_start != -1:
                crumb_start += len('"CrumbStore":{"crumb":"')
                crumb_end = content.find('"', crumb_start)
                if crumb_end != -1:
                    return content[crumb_start:crumb_end]
            
            logger.warning(f"Could not extract crumb for {symbol}, proceeding without authentication")
            return ""
            
        except Exception as e:
            logger.warning(f"Could not get crumb for {symbol}: {e}, proceeding without authentication")
            return ""
    
    def _date_to_timestamp(self, date_str: str) -> int:
        """Convert date string to timestamp"""
        try:
            # Parse date string (YYYY-MM-DD format)
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            return int(date_obj.timestamp())
        except ValueError:
            # If date parsing fails, use current date
            return int(datetime.now().timestamp())
    
    def fetch_yahoo_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Fetch data for a specific Yahoo Finance symbol using direct API calls"""
        try:
            # Validate and fix date range
            if not start_date or not end_date:
                start_date, end_date = self.get_default_date_range(30)
            else:
                start_date, end_date = self.validate_date_range(start_date, end_date)
            
            logger.info(f"Fetching Yahoo Finance data for symbol: {symbol} from {start_date} to {end_date}")
            
            # Convert dates to timestamps
            start_timestamp = self._date_to_timestamp(start_date)
            end_timestamp = self._date_to_timestamp(end_date)
            
            # Get crumb for authentication
            crumb = self._get_yahoo_crumb(symbol)
            
            # Construct URL and fetch data
            url = self._get_historical_data_url(symbol, start_timestamp, end_timestamp, crumb)
            
            response = self.session.get(url, timeout=15)
            
            # If we get a 401, try without crumb
            if response.status_code == 401:
                logger.warning(f"Yahoo Finance API returned 401 for {symbol}, retrying without crumb")
                url = self._get_historical_data_url(symbol, start_timestamp, end_timestamp, "")
                response = self.session.get(url, timeout=15)
            
            response.raise_for_status()
            
            # Parse CSV data
            data = response.text.strip().split('\n')
            if len(data) < 2:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Parse CSV headers and data
            headers = data[0].split(',')
            rows = []
            for line in data[1:]:
                if line.strip():
                    values = line.split(',')
                    if len(values) == len(headers):
                        rows.append(dict(zip(headers, values)))
            
            if not rows:
                logger.warning(f"No valid data rows for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(rows)
            
            # Convert date column
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Convert numeric columns
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Add metadata
            df['symbol'] = symbol
            df['source'] = 'YahooFinance'
            df['collection_timestamp'] = datetime.now(timezone.utc).isoformat()
            
            logger.info(f"Fetched {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            self.log_consolidated_error(f"Yahoo Finance data fetch for {symbol}", e,
                                      additional_info={"start_date": start_date, "end_date": end_date})
            raise
    
    def _get_historical_data_url(self, symbol: str, start_timestamp: int, end_timestamp: int, crumb: str) -> str:
        """Construct Yahoo Finance historical data URL"""
        base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"
        params = f"?symbol={symbol}&period1={start_timestamp}&period2={end_timestamp}&interval=1d"
        
        if crumb:
            params += f"&crumb={crumb}"
        
        return base_url + symbol + params
    
    def fetch_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Fetch basic information about a symbol"""
        try:
            # Try the quote API first
            quote_url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}"
            response = self.session.get(quote_url, timeout=10)
            
            response.raise_for_status()
            data = response.json()
            
            if 'quoteResponse' in data and 'result' in data['quoteResponse']:
                result = data['quoteResponse']['result']
                if result and len(result) > 0:
                    quote = result[0]
                    return {
                        'symbol': quote.get('symbol', symbol),
                        'shortName': quote.get('shortName', ''),
                        'longName': quote.get('longName', ''),
                        'market': quote.get('market', ''),
                        'quoteType': quote.get('quoteType', ''),
                        'currency': quote.get('currency', ''),
                        'exchange': quote.get('exchange', ''),
                        'marketState': quote.get('marketState', ''),
                        'regularMarketPrice': quote.get('regularMarketPrice', 0),
                        'regularMarketVolume': quote.get('regularMarketVolume', 0),
                        'regularMarketTime': quote.get('regularMarketTime', 0)
                    }
            
            # If quote API fails, try alternative method
            logger.warning(f"Quote API failed for {symbol}, trying alternative method")
            return self._fetch_symbol_info_alternative(symbol)
            
        except Exception as e:
            self.log_consolidated_error(f"Yahoo Finance symbol info fetch for {symbol}", e)
            return {}
    
    def _fetch_symbol_info_alternative(self, symbol: str) -> Dict[str, Any]:
        """Alternative method to fetch symbol information"""
        try:
            # Try scraping from the quote page
            quote_url = f"https://finance.yahoo.com/quote/{symbol}"
            response = self.session.get(quote_url, timeout=10)
            
            response.raise_for_status()
            
            # Extract basic info from page title and meta tags
            content = response.text
            title_start = content.find('<title>')
            title_end = content.find('</title>')
            
            title = ""
            if title_start != -1 and title_end != -1:
                title = content[title_start + 7:title_end].strip()
            
            return {
                'symbol': symbol,
                'shortName': title.replace(' (', ' - ').replace(')', '') if title else '',
                'longName': title if title else '',
                'market': 'Unknown',
                'quoteType': 'Unknown',
                'currency': 'USD',
                'exchange': 'Unknown',
                'marketState': 'Unknown',
                'regularMarketPrice': 0,
                'regularMarketVolume': 0,
                'regularMarketTime': 0
            }
            
        except Exception as e:
            self.log_consolidated_error(f"Yahoo Finance alternative symbol info for {symbol}", e)
            return {}
    
    def collect(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Collect Yahoo Finance data with comprehensive error handling"""
        try:
            # Validate environment
            if not self.validate_environment():
                raise ValueError("Environment validation failed")
            
            # Log collection start
            self.log_collection_start(start_date=start_date, end_date=end_date)
            
            # Determine date range (last 30 days by default)
            if not end_date or not start_date:
                start_date, end_date = self.get_default_date_range(30)
            else:
                start_date, end_date = self.validate_date_range(start_date, end_date)
            
            self.results['start_date'] = start_date
            self.results['end_date'] = end_date
            self.results['total_symbols'] = len(YAHOO_SYMBOLS)
            
            logger.info(f"Processing {len(YAHOO_SYMBOLS)} Yahoo Finance symbols from {start_date} to {end_date}")
            
            # Process each symbol
            for symbol in YAHOO_SYMBOLS:
                try:
                    logger.info(f"Processing Yahoo Finance symbol: {symbol}")
                    
                    # Fetch price data
                    df = self.fetch_yahoo_data(symbol, start_date, end_date)
                    
                    if not df.empty:
                        # Create S3 path for price data
                        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
                        s3_path = f"raw/yahoo_finance/{symbol}/price/{timestamp}_{symbol}_price.parquet"
                        
                        # Save price data to S3
                        s3_key = self.save_to_s3(df, s3_path, BRONZE_BUCKET)
                        
                        # Add success result for price data
                        self.add_success_result(
                            item_id=f"{symbol}/price",
                            records_count=len(df),
                            s3_key=s3_key,
                            date_range=f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}"
                        )
                        
                        # Fetch symbol information
                        symbol_info = self.fetch_symbol_info(symbol)
                        
                        if symbol_info:
                            # Create DataFrame for symbol info
                            info_df = pd.DataFrame([symbol_info])
                            info_df['collection_timestamp'] = datetime.now(timezone.utc).isoformat()
                            
                            # Create S3 path for symbol info
                            info_s3_path = f"raw/yahoo_finance/{symbol}/info/{timestamp}_{symbol}_info.parquet"
                            
                            # Save symbol info to S3
                            info_s3_key = self.save_to_s3(info_df, info_s3_path, BRONZE_BUCKET)
                            
                            # Add success result for symbol info
                            self.add_success_result(
                                item_id=f"{symbol}/info",
                                records_count=1,
                                s3_key=info_s3_key
                            )
                        else:
                            self.add_failed_result(
                                item_id=f"{symbol}/info",
                                error='No symbol information returned'
                            )
                    else:
                        self.add_failed_result(
                            item_id=f"{symbol}/price",
                            error='No price data returned from Yahoo Finance API'
                        )
                    
                    # Rate limiting - be respectful to Yahoo Finance
                    time.sleep(1)  # 1 second between requests
                    
                except Exception as e:
                    self.log_consolidated_error(f"Yahoo Finance symbol processing {symbol}", e)
                    self.add_failed_result(
                        item_id=symbol,
                        error=str(e)
                    )
            
            # Log collection end
            self.log_collection_end()
            
            return self.results
            
        except Exception as e:
            self.log_consolidated_error("Yahoo Finance data collection", e)
            self.add_failed_result(
                item_id="collection",
                error=f"Collection failed: {str(e)}"
            )
            return self.results
