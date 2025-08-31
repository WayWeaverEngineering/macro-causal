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
    
    # Currencies (using alternative symbols that work better)
    'EURUSD=X', # Euro/US Dollar
    'GBPUSD=X', # British Pound/US Dollar
    'USDJPY=X', # US Dollar/Japanese Yen
    'USDCNY=X', # US Dollar/Chinese Yuan
    'USDCAD=X', # US Dollar/Canadian Dollar
]

DEFAULT_DAYS_BACK = 27375 # 75 years = 365 * 75 days

class YahooFinanceCollector(DataCollector):
    """Yahoo Finance data collector using direct API calls with workarounds for API restrictions"""
    
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
            
            self.logger.warning(f"Could not extract crumb for {symbol}, proceeding without authentication")
            return ""
            
        except Exception as e:
            self.logger.warning(f"Could not get crumb for {symbol}: {e}, proceeding without authentication")
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
                start_date, end_date = self.get_default_date_range(DEFAULT_DAYS_BACK)
            else:
                start_date, end_date = self.validate_date_range(start_date, end_date)
            
            # Convert dates to timestamps
            start_timestamp = self._date_to_timestamp(start_date)
            end_timestamp = self._date_to_timestamp(end_date)
            
            # Get crumb for authentication
            crumb = self._get_yahoo_crumb(symbol)
            
            # Try multiple API endpoints for historical data
            df = self._try_historical_data_endpoints(symbol, start_timestamp, end_timestamp, crumb)
            
            if df.empty:
                self.logger.warning(f"No data returned for {symbol} from any endpoint")
                return pd.DataFrame()
            
            # Add metadata
            df['symbol'] = symbol
            df['source'] = 'YahooFinance'
            df['collection_timestamp'] = datetime.now(timezone.utc).isoformat()
            
            return df
            
        except Exception as e:
            self.log_consolidated_error(f"Yahoo Finance data fetch for {symbol}", e,
                                      additional_info={"start_date": start_date, "end_date": end_date})
            raise
    
    def _try_historical_data_endpoints(self, symbol: str, start_timestamp: int, end_timestamp: int, crumb: str) -> pd.DataFrame:
        """Try multiple endpoints to get historical data"""
        endpoints = [
            # Primary endpoint - chart API
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?symbol={symbol}&period1={start_timestamp}&period2={end_timestamp}&interval=1d",
            # Alternative endpoint
            f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?symbol={symbol}&period1={start_timestamp}&period2={end_timestamp}&interval=1d",
            # CSV download endpoint (most reliable)
            f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1={start_timestamp}&period2={end_timestamp}&interval=1d&events=history"
        ]
        
        for i, url in enumerate(endpoints):
            try:
                # Add crumb if available
                if crumb and "crumb=" not in url:
                    url += f"&crumb={crumb}"
                
                response = self.session.get(url, timeout=15)
                
                if response.status_code == 200:
                    # Try to parse as CSV first (most common format)
                    if 'download' in url or 'text/csv' in response.headers.get('content-type', ''):
                        return self._parse_csv_data(response.text, symbol)
                    else:
                        # Try to parse as JSON
                        return self._parse_json_data(response.json(), symbol)
                
                self.logger.warning(f"Endpoint {i+1} returned status {response.status_code} for {symbol}")
                
            except Exception as e:
                self.logger.warning(f"Endpoint {i+1} failed for {symbol}: {e}")
                continue
        
        return pd.DataFrame()
    
    def _parse_csv_data(self, csv_text: str, symbol: str) -> pd.DataFrame:
        """Parse CSV data from Yahoo Finance"""
        try:
            lines = csv_text.strip().split('\n')
            if len(lines) < 2:
                return pd.DataFrame()
            
            # Parse headers and data
            headers = lines[0].split(',')
            rows = []
            
            for line in lines[1:]:
                if line.strip():
                    values = line.split(',')
                    if len(values) == len(headers):
                        rows.append(dict(zip(headers, values)))
            
            if not rows:
                return pd.DataFrame()
            
            df = pd.DataFrame(rows)
            
            # Convert date column
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            
            # Convert numeric columns
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            self.logger.warning(f"Failed to parse CSV data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _parse_json_data(self, json_data: Dict, symbol: str) -> pd.DataFrame:
        """Parse JSON data from Yahoo Finance chart API"""
        try:
            if 'chart' not in json_data or 'result' not in json_data['chart'] or not json_data['chart']['result']:
                return pd.DataFrame()
            
            result = json_data['chart']['result'][0]
            timestamps = result.get('timestamp', [])
            quote = result.get('indicators', {}).get('quote', [{}])[0]
            
            # Extract OHLCV data
            opens = quote.get('open', [])
            highs = quote.get('high', [])
            lows = quote.get('low', [])
            closes = quote.get('close', [])
            volumes = quote.get('volume', [])
            
            # Create DataFrame
            df_data = []
            for i, timestamp in enumerate(timestamps):
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                df_data.append({
                    'Date': dt,
                    'Open': opens[i] if i < len(opens) and opens[i] is not None else None,
                    'High': highs[i] if i < len(highs) and highs[i] is not None else None,
                    'Low': lows[i] if i < len(lows) and lows[i] is not None else None,
                    'Close': closes[i] if i < len(closes) and closes[i] is not None else None,
                    'Volume': volumes[i] if i < len(volumes) and volumes[i] is not None else None
                })
            
            return pd.DataFrame(df_data)
            
        except Exception as e:
            self.logger.warning(f"Failed to parse JSON data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_basic_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get basic symbol information without using restricted APIs"""
        try:
            # Try to get basic info from the quote page
            quote_url = f"https://finance.yahoo.com/quote/{symbol}"
            response = self.session.get(quote_url, timeout=10)
            
            if response.status_code == 200:
                content = response.text
                
                # Extract title
                title_start = content.find('<title>')
                title_end = content.find('</title>')
                title = ""
                if title_start != -1 and title_end != -1:
                    title = content[title_start + 7:title_end].strip()
                
                # Extract basic info from meta tags
                currency = 'USD'  # Default
                if 'currency' in content.lower():
                    # Try to extract currency from meta tags
                    currency_match = content.find('"currency":"')
                    if currency_match != -1:
                        currency_start = currency_match + 12
                        currency_end = content.find('"', currency_start)
                        if currency_end != -1:
                            currency = content[currency_start:currency_end]
                
                return {
                    'symbol': symbol,
                    'shortName': title.replace(' (', ' - ').replace(')', '') if title else symbol,
                    'longName': title if title else symbol,
                    'market': 'Unknown',
                    'quoteType': 'Unknown',
                    'currency': currency,
                    'exchange': 'Unknown',
                    'marketState': 'Unknown',
                    'regularMarketPrice': 0,
                    'regularMarketVolume': 0,
                    'regularMarketTime': 0
                }
            
            # Return minimal info if page access fails
            return {
                'symbol': symbol,
                'shortName': symbol,
                'longName': symbol,
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
            self.logger.warning(f"Could not get basic info for {symbol}: {e}")
            return {
                'symbol': symbol,
                'shortName': symbol,
                'longName': symbol,
                'market': 'Unknown',
                'quoteType': 'Unknown',
                'currency': 'USD',
                'exchange': 'Unknown',
                'marketState': 'Unknown',
                'regularMarketPrice': 0,
                'regularMarketVolume': 0,
                'regularMarketTime': 0
            }
    
    def collect(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Collect Yahoo Finance data with comprehensive error handling"""
        try:
            # Validate environment
            if not self.validate_environment():
                raise ValueError("Environment validation failed")
            
            # Log collection start
            self.log_collection_start(start_date=start_date, end_date=end_date)
            
            # Determine date range (last DEFAULT_DAYS_BACK years by default)
            if not end_date or not start_date:
                start_date, end_date = self.get_default_date_range(DEFAULT_DAYS_BACK)
            else:
                start_date, end_date = self.validate_date_range(start_date, end_date)
            
            self.results['start_date'] = start_date
            self.results['end_date'] = end_date
            self.results['total_symbols'] = len(YAHOO_SYMBOLS)
            
            # Process each symbol
            for i, symbol in enumerate(YAHOO_SYMBOLS, 1):
                try:
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
                        
                        # Get basic symbol information (without using restricted APIs)
                        symbol_info = self.get_basic_symbol_info(symbol)
                        
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
