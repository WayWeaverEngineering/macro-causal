#!/usr/bin/env python3
"""
Yahoo Finance Data Collector
Collects financial market data from Yahoo Finance
"""

import pandas as pd
from datetime import datetime, timedelta
import logging
import time
import yfinance as yf
from typing import Dict, Any
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
