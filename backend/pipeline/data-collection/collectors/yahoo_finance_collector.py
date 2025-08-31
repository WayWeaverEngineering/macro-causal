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
            
            # Log detailed error information if the request failed
            if response.status_code >= 400:
                logger.error(f"Yahoo Finance crumb extraction request failed for {symbol}")
                logger.error(f"Status Code: {response.status_code}")
                logger.error(f"URL: {response.url}")
                logger.error(f"Request Headers: {dict(self.session.headers)}")
                try:
                    error_body = response.text[:1000]  # Limit to first 1000 chars for HTML content
                    logger.error(f"Response Body (first 1000 chars): {error_body}")
                except Exception as e:
                    logger.error(f"Could not read response body: {e}")
            
            response.raise_for_status()
            
            # Extract crumb from the page
            content = response.text
            
            # Try multiple patterns for crumb extraction
            crumb_patterns = [
                '"CrumbStore":{"crumb":"',
                '"crumb":"',
                'CrumbStore":{"crumb":"'
            ]
            
            crumb = ""
            for pattern in crumb_patterns:
                if pattern in content:
                    crumb_start = content.find(pattern) + len(pattern)
                    crumb_end = content.find('"', crumb_start)
                    if crumb_end > crumb_start:
                        crumb = content[crumb_start:crumb_end]
                        break
            
            if not crumb:
                logger.debug(f"Could not extract crumb for {symbol}, proceeding without authentication")
            
            return crumb
        except Exception as e:
            logger.debug(f"Could not get crumb for {symbol}: {e}, proceeding without authentication")
            return ""
    
    def _get_historical_data_url(self, symbol: str, start_date: int, end_date: int, crumb: str = "") -> str:
        """Construct Yahoo Finance historical data URL"""
        base_url = "https://query2.finance.yahoo.com/v8/finance/chart/"
        
        # URL parameters
        params = {
            'symbol': symbol,
            'period1': start_date,
            'period2': end_date,
            'interval': '1d',
            'includePrePost': 'false',
            'events': 'div,split',
            'lang': 'en-US',
            'region': 'US'
        }
        
        if crumb:
            params['crumb'] = crumb
        
        # Build query string
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return f"{base_url}{symbol}?{query_string}"
    
    def _get_quote_data_url(self, symbol: str) -> str:
        """Construct Yahoo Finance quote data URL"""
        base_url = "https://query2.finance.yahoo.com/v7/finance/quote"
        params = {
            'symbols': symbol,
            'lang': 'en-US',
            'region': 'US'
        }
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return f"{base_url}?{query_string}"
    
    def _date_to_timestamp(self, date_str: str) -> int:
        """Convert date string to Unix timestamp"""
        try:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            return int(dt.timestamp())
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
            
            # Log detailed error information if the request failed
            if response.status_code >= 400:
                logger.error(f"Yahoo Finance API request failed for {symbol}")
                logger.error(f"Status Code: {response.status_code}")
                logger.error(f"URL: {response.url}")
                logger.error(f"Request Headers: {dict(self.session.headers)}")
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
            
            # If we get a 401, try without crumb
            if response.status_code == 401:
                logger.info(f"Historical data API returned 401 for {symbol}, trying without authentication")
                url = self._get_historical_data_url(symbol, start_timestamp, end_timestamp, "")
                response = self.session.get(url, timeout=15)
                
                # Log error details for the retry attempt as well
                if response.status_code >= 400:
                    logger.error(f"Yahoo Finance API retry request also failed for {symbol}")
                    logger.error(f"Retry Status Code: {response.status_code}")
                    logger.error(f"Retry URL: {response.url}")
                    try:
                        error_body = response.text
                        logger.error(f"Retry Response Body: {error_body}")
                        try:
                            error_json = response.json()
                            logger.error(f"Retry Response JSON: {error_json}")
                        except:
                            pass
                    except Exception as e:
                        logger.error(f"Could not read retry response body: {e}")
            
            response.raise_for_status()
            
            data = response.json()
            
            # Parse the response
            if 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
                logger.warning(f"No data returned for Yahoo Finance symbol {symbol}")
                return pd.DataFrame()
            
            result = data['chart']['result'][0]
            
            # Extract timestamps and OHLCV data
            timestamps = result.get('timestamp', [])
            quote = result.get('indicators', {}).get('quote', [{}])[0]
            
            # Extract OHLCV data
            opens = quote.get('open', [])
            highs = quote.get('high', [])
            lows = quote.get('low', [])
            closes = quote.get('close', [])
            volumes = quote.get('volume', [])
            
            # Extract dividends and splits
            events = result.get('events', {})
            dividends = events.get('dividends', {})
            splits = events.get('splits', {})
            
            # Create DataFrame
            df_data = []
            for i, timestamp in enumerate(timestamps):
                # Convert timestamp to datetime
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                
                # Get dividend and split data for this date
                dividend = dividends.get(str(timestamp), {}).get('amount', 0) if str(timestamp) in dividends else 0
                split = splits.get(str(timestamp), {}).get('splitRatio', 0) if str(timestamp) in splits else 0
                
                df_data.append({
                    'date': dt,
                    'open': opens[i] if i < len(opens) and opens[i] is not None else None,
                    'high': highs[i] if i < len(highs) and highs[i] is not None else None,
                    'low': lows[i] if i < len(lows) and lows[i] is not None else None,
                    'close': closes[i] if i < len(closes) and closes[i] is not None else None,
                    'volume': volumes[i] if i < len(volumes) and volumes[i] is not None else None,
                    'dividends': dividend,
                    'stock_splits': split
                })
            
            df = pd.DataFrame(df_data)
            
            if df.empty:
                logger.warning(f"No data returned for Yahoo Finance symbol {symbol}")
                return pd.DataFrame()
            
            # Add metadata
            df['symbol'] = symbol
            df['source'] = 'YahooFinance'
            df['collection_timestamp'] = datetime.now(timezone.utc).isoformat()
            
            logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data for symbol {symbol}: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception details: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get additional information about a symbol using direct API calls"""
        try:
            # Try the quote API first
            url = self._get_quote_data_url(symbol)
            
            response = self.session.get(url, timeout=10)
            
            # Log detailed error information if the request failed
            if response.status_code >= 400:
                logger.error(f"Yahoo Finance Quote API request failed for {symbol}")
                logger.error(f"Status Code: {response.status_code}")
                logger.error(f"URL: {response.url}")
                logger.error(f"Request Headers: {dict(self.session.headers)}")
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
            
            # If we get a 401, try alternative approach
            if response.status_code == 401:
                logger.info(f"Quote API returned 401 for {symbol}, trying alternative method")
                return self._get_symbol_info_alternative(symbol)
            
            response.raise_for_status()
            
            data = response.json()
            
            if 'quoteResponse' not in data or 'result' not in data['quoteResponse'] or not data['quoteResponse']['result']:
                logger.warning(f"No quote data returned for {symbol}")
                return self._get_default_symbol_info(symbol)
            
            quote = data['quoteResponse']['result'][0]
            
            return {
                'symbol': symbol,
                'name': quote.get('longName', quote.get('shortName', '')),
                'sector': quote.get('sector', ''),
                'industry': quote.get('industry', ''),
                'market_cap': quote.get('marketCap', 0),
                'currency': quote.get('currency', ''),
                'exchange': quote.get('fullExchangeName', ''),
                'country': quote.get('marketState', ''),
                'website': quote.get('website', ''),
                'description': quote.get('longBusinessSummary', '')
            }
            
        except Exception as e:
            logger.warning(f"Could not fetch info for Yahoo Finance symbol {symbol}: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception details: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return self._get_default_symbol_info(symbol)
    
    def _get_symbol_info_alternative(self, symbol: str) -> Dict[str, Any]:
        """Alternative method to get symbol info when quote API fails"""
        try:
            # Try to get basic info from the chart API response
            # This is more reliable as it's the same endpoint we use for price data
            start_date, end_date = self.get_default_date_range(1)  # Just get today's data
            start_timestamp = self._date_to_timestamp(start_date)
            end_timestamp = self._date_to_timestamp(end_date)
            
            url = self._get_historical_data_url(symbol, start_timestamp, end_timestamp)
            
            response = self.session.get(url, timeout=10)
            
            # Log detailed error information if the request failed
            if response.status_code >= 400:
                logger.error(f"Yahoo Finance Alternative API request failed for {symbol}")
                logger.error(f"Status Code: {response.status_code}")
                logger.error(f"URL: {response.url}")
                logger.error(f"Request Headers: {dict(self.session.headers)}")
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
            
            response.raise_for_status()
            
            data = response.json()
            
            if 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
                return self._get_default_symbol_info(symbol)
            
            result = data['chart']['result'][0]
            meta = result.get('meta', {})
            
            return {
                'symbol': symbol,
                'name': meta.get('symbol', symbol),
                'sector': '',
                'industry': '',
                'market_cap': 0,
                'currency': meta.get('currency', ''),
                'exchange': meta.get('exchangeName', ''),
                'country': '',
                'website': '',
                'description': ''
            }
            
        except Exception as e:
            logger.debug(f"Alternative symbol info method also failed for {symbol}: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception details: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return self._get_default_symbol_info(symbol)
    
    def _get_default_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Return default symbol info when API call fails"""
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
    
    def try_alternative_symbols(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Try alternative symbols if the main symbol fails"""
        if symbol in SYMBOL_ALTERNATIVES:
            alternatives = SYMBOL_ALTERNATIVES[symbol]
            logger.info(f"Trying alternative symbols for {symbol}: {alternatives}")
            
            for alt_symbol in alternatives:
                try:
                    logger.info(f"Attempting alternative symbol: {alt_symbol}")
                    df = self.fetch_yahoo_data(alt_symbol, start_date, end_date)
                    if not df.empty:
                        # Update the symbol in the dataframe to the original symbol
                        df['symbol'] = symbol
                        logger.info(f"Successfully fetched data using alternative symbol {alt_symbol} for {symbol}")
                        return df
                except Exception as e:
                    logger.warning(f"Alternative symbol {alt_symbol} also failed: {e}")
                    continue
        
        return None
    
    def collect(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Collect Yahoo Finance data with improved error handling"""
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
                    
                    # If main symbol fails, try alternatives
                    if df.empty and symbol in SYMBOL_ALTERNATIVES:
                        logger.info(f"Main symbol {symbol} failed, trying alternatives...")
                        df = self.try_alternative_symbols(symbol, start_date, end_date)
                    
                    if not df.empty:
                        # Create S3 paths
                        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
                        price_s3_path = f"raw/yahoo_finance/{symbol}/price/{timestamp}_{symbol}_price.parquet"
                        info_s3_path = f"raw/yahoo_finance/{symbol}/info/{timestamp}_{symbol}_info.parquet"
                        
                        # Save price data to S3
                        price_s3_key = self.save_to_s3(df, price_s3_path, BRONZE_BUCKET)
                        
                        # Get symbol information
                        symbol_info = self.get_symbol_info(symbol)
                        
                        # Save symbol info to S3
                        info_df = pd.DataFrame([symbol_info])
                        info_s3_key = self.save_to_s3(info_df, info_s3_path, BRONZE_BUCKET)
                        
                        # Add success result
                        self.add_success_result(
                            item_id=symbol,
                            price_records_count=len(df),
                            price_s3_key=price_s3_key,
                            info_s3_key=info_s3_key,
                            date_range=f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
                            symbol_name=symbol_info.get('name', ''),
                            sector=symbol_info.get('sector', '')
                        )
                    else:
                        self.add_failed_result(
                            item_id=symbol,
                            error='No data returned after trying main and alternative symbols'
                        )
                    
                    # Rate limiting - be respectful to Yahoo Finance
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Failed to process Yahoo Finance symbol {symbol}: {e}")
                    self.add_failed_result(
                        item_id=symbol,
                        error=str(e)
                    )
            
            # Log collection end
            self.log_collection_end()
            
            return self.results
            
        except Exception as e:
            logger.error(f"Yahoo Finance data collection failed: {e}")
            logger.debug(f"Yahoo Finance collection error details: {e}")
            self.add_failed_result(
                item_id="collection",
                error=f"Collection failed: {str(e)}"
            )
            return self.results
