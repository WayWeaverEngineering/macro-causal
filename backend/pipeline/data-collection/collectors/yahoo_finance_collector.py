#!/usr/bin/env python3
"""
Yahoo Finance Data Collector
Collects financial market data from Yahoo Finance
"""

import pandas as pd
from datetime import datetime, timedelta, timezone
import logging
import time
import yfinance as yf
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
    'JNJ',      # Johnson & Johnson (may have timezone issues)
    
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
    """Yahoo Finance data collector with enhanced error handling"""
    
    def __init__(self):
        super().__init__(collector_name="YahooFinance")
    
    def fetch_yahoo_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Fetch data for a specific Yahoo Finance symbol with enhanced error handling"""
        try:
            # Validate and fix date range
            if not start_date or not end_date:
                start_date, end_date = self.get_default_date_range(30)
            else:
                start_date, end_date = self.validate_date_range(start_date, end_date)
            
            logger.info(f"Fetching Yahoo Finance data for symbol: {symbol} from {start_date} to {end_date}")
            
            # Use yfinance library to fetch data
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data with error handling
            try:
                df = ticker.history(start=start_date, end=end_date, interval='1d')
            except Exception as history_error:
                logger.warning(f"Error fetching history for {symbol}: {history_error}")
                # Try with different parameters
                try:
                    df = ticker.history(period="30d", interval='1d')
                    logger.info(f"Successfully fetched data for {symbol} using period parameter")
                except Exception as period_error:
                    logger.error(f"Failed to fetch data for {symbol} with both methods: {period_error}")
                    return pd.DataFrame()
            
            if df.empty:
                logger.warning(f"No data returned for Yahoo Finance symbol {symbol}")
                return pd.DataFrame()
            
            # Reset index to make date a column
            df = df.reset_index()
            
            # Handle timezone issues by converting to UTC
            if 'Date' in df.columns:
                try:
                    # Convert timezone-aware dates to UTC
                    if df['Date'].dt.tz is not None:
                        df['Date'] = df['Date'].dt.tz_convert('UTC')
                    else:
                        # If no timezone, assume UTC
                        df['Date'] = df['Date'].dt.tz_localize('UTC')
                except Exception as tz_error:
                    logger.warning(f"Timezone conversion failed for {symbol}: {tz_error}")
                    # Continue without timezone conversion
            
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
            df['collection_timestamp'] = datetime.now(timezone.utc).isoformat()
            
            logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data for symbol {symbol}: {e}")
            logger.debug(f"Yahoo Finance error details for {symbol}: {e}")
            raise
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get additional information about a symbol with error handling"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Add timeout and retry logic for info fetching
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    info = ticker.info
                    break
                except Exception as info_error:
                    if attempt == max_retries - 1:
                        logger.warning(f"Failed to fetch info for {symbol} after {max_retries} attempts: {info_error}")
                        raise
                    logger.debug(f"Attempt {attempt + 1} failed for {symbol}, retrying...")
                    time.sleep(1)
            
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
