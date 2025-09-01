#!/usr/bin/env python3
"""
Test script for the data processing pipeline
Tests individual components without requiring S3 access
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data():
    """Create test data for testing the pipeline"""
    
    # Create test FRED data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # FRED data
    fred_data = []
    for date in dates:
        fred_data.extend([
            {'date': date, 'series_id': 'GDP', 'value': 20000 + np.random.normal(0, 100)},
            {'date': date, 'series_id': 'UNRATE', 'value': 5 + np.random.normal(0, 0.5)},
            {'date': date, 'series_id': 'CPIAUCSL', 'value': 250 + np.random.normal(0, 2)},
            {'date': date, 'series_id': 'FEDFUNDS', 'value': 2 + np.random.normal(0, 0.3)},
            {'date': date, 'series_id': 'DGS10', 'value': 3 + np.random.normal(0, 0.5)}
        ])
    
    fred_df = pd.DataFrame(fred_data)
    
    # World Bank data (annual frequency)
    worldbank_data = []
    for year in range(2020, 2024):
        date = datetime(year, 1, 1)
        worldbank_data.extend([
            {'date': date, 'country_code': 'US', 'indicator_code': 'NY.GDP.MKTP.CD', 'value': 20000 + np.random.normal(0, 1000)},
            {'date': date, 'country_code': 'CN', 'indicator_code': 'NY.GDP.MKTP.CD', 'value': 15000 + np.random.normal(0, 800)},
            {'date': date, 'country_code': 'JP', 'indicator_code': 'NY.GDP.MKTP.CD', 'value': 5000 + np.random.normal(0, 300)}
        ])
    
    worldbank_df = pd.DataFrame(worldbank_data)
    
    # Yahoo Finance data
    yahoo_data = []
    for date in dates:
        yahoo_data.extend([
            {'date': date, 'symbol': '^GSPC', 'Close': 4000 + np.random.normal(0, 50)},
            {'date': date, 'symbol': '^VIX', 'Close': 20 + np.random.normal(0, 5)},
            {'date': date, 'symbol': 'AAPL', 'Close': 150 + np.random.normal(0, 10)}
        ])
    
    yahoo_df = pd.DataFrame(yahoo_data)
    
    return fred_df, worldbank_df, yahoo_df

def test_feature_engineering():
    """Test the feature engineering module"""
    logger.info("Testing feature engineering module...")
    
    try:
        # Create test data
        fred_df, worldbank_df, yahoo_df = create_test_data()
        
        # Initialize feature engineer
        feature_engineer = FeatureEngineer("test_execution_123")
        
        # Test economic features
        logger.info("Testing economic features creation...")
        fred_features_df, fred_feature_names = feature_engineer.create_economic_features(fred_df)
        logger.info(f"Created {len(fred_feature_names)} economic features")
        
        # Test financial features
        logger.info("Testing financial features creation...")
        yahoo_features_df, yahoo_feature_names = feature_engineer.create_financial_features(yahoo_df)
        logger.info(f"Created {len(yahoo_feature_names)} financial features")
        
        # Test World Bank features
        logger.info("Testing World Bank features creation...")
        worldbank_features_df, worldbank_feature_names = feature_engineer.create_world_bank_features(worldbank_df)
        logger.info(f"Created {len(worldbank_feature_names)} World Bank features")
        
        # Test final feature creation
        logger.info("Testing final feature creation...")
        final_df, all_features = feature_engineer.create_final_features(fred_df, worldbank_df, yahoo_df)
        
        logger.info(f"Final dataset shape: {final_df.shape}")
        logger.info(f"Total features created: {len(all_features)}")
        logger.info(f"Feature columns: {list(final_df.columns)[:10]}...")  # Show first 10 columns
        
        # Test data quality
        logger.info("Testing data quality...")
        missing_values = final_df.isnull().sum().sum()
        infinite_values = np.isinf(final_df.select_dtypes(include=[np.number])).sum().sum()
        
        logger.info(f"Missing values: {missing_values}")
        logger.info(f"Infinite values: {infinite_values}")
        
        # Check target variable
        if 'target' in final_df.columns:
            target_stats = final_df['target'].describe()
            logger.info(f"Target variable statistics:\n{target_stats}")
        
        logger.info("Feature engineering tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Feature engineering test failed: {e}")
        return False

def test_data_processing():
    """Test the data processing pipeline components"""
    logger.info("Testing data processing components...")
    
    try:
        # Create test data
        fred_df, worldbank_df, yahoo_df = create_test_data()
        
        # Test data cleaning
        logger.info("Testing data cleaning...")
        
        # Test FRED data processing
        from main import DataProcessor
        processor = DataProcessor("test_bucket", "test_bucket", "test_bucket", "test_execution")
        
        cleaned_fred = processor.process_fred_data(fred_df)
        logger.info(f"Cleaned FRED data shape: {cleaned_fred.shape}")
        
        cleaned_worldbank = processor.process_worldbank_data(worldbank_df)
        logger.info(f"Cleaned World Bank data shape: {cleaned_worldbank.shape}")
        
        cleaned_yahoo = processor.process_yahoo_finance_data(yahoo_df)
        logger.info(f"Cleaned Yahoo Finance data shape: {cleaned_yahoo.shape}")
        
        logger.info("Data processing tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Data processing test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting data processing pipeline tests...")
    
    # Test feature engineering
    feature_test_passed = test_feature_engineering()
    
    # Test data processing
    processing_test_passed = test_data_processing()
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Feature Engineering: {'PASSED' if feature_test_passed else 'FAILED'}")
    logger.info(f"Data Processing: {'PASSED' if processing_test_passed else 'FAILED'}")
    
    if feature_test_passed and processing_test_passed:
        logger.info("All tests passed! Data processing pipeline is ready.")
        return 0
    else:
        logger.error("Some tests failed. Please check the logs above.")
        return 1

if __name__ == "__main__":
    exit(main())
