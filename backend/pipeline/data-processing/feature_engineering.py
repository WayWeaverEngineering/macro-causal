#!/usr/bin/env python3
"""
Feature Engineering Module for Macro Causal Inference
Creates features for X-learner with DML and regime classifier training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import logging
import pytz

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering class for macro causal inference"""
    
    def __init__(self, execution_id: str):
        self.execution_id = execution_id
    
    def _force_utc_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Force DataFrame index to be UTC timezone-aware DatetimeIndex"""
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.to_datetime(idx, errors='coerce', utc=True)
            logger.debug("Converted non-DatetimeIndex to UTC timezone-aware")
        else:
            if idx.tz is None:
                idx = idx.tz_localize('UTC')
                logger.debug("Made timezone-naive DatetimeIndex UTC timezone-aware")
            else:
                idx = idx.tz_convert('UTC')
                logger.debug(f"Converted DatetimeIndex from {idx.tz} to UTC")
        
        out = df.copy()
        out.index = idx
        return out
        
    def create_economic_features(self, fred_df: pd.DataFrame) -> pd.DataFrame:
        """Create economic features from FRED data"""
        try:
            if fred_df.empty or 'date' not in fred_df.columns:
                logger.info("No FRED data available")
                return pd.DataFrame(), []
                
            logger.info("Creating economic features from FRED data...")
            
            # Pivot FRED data to wide format
            fred_pivot = fred_df.pivot_table(
                index='date',
                columns='series_id',
                values='value',
                aggfunc='first'
            )
            
            # Force UTC timezone-aware index before resampling
            logger.debug("Forcing FRED pivot index to UTC timezone-aware")
            fred_pivot = self._force_utc_index(fred_pivot)
            fred_pivot = fred_pivot.sort_index().resample('D').ffill()
            logger.debug(f"FRED pivot index timezone after normalization: {fred_pivot.index.tz}")
            
            # Collect all new features in a dictionary to avoid DataFrame fragmentation
            new_features = {}
            economic_features = []
            
            for series_id in fred_pivot.columns:
                if series_id in fred_pivot.columns:
                    series_data = fred_pivot[series_id]
                    
                    # Basic lagged features
                    for lag in [1, 7, 30, 90]:
                        if len(series_data) > lag:
                            lagged_col = f"fred_{series_id}_lag_{lag}d"
                            new_features[lagged_col] = series_data.shift(lag)
                            economic_features.append(lagged_col)
                    
                    # Rolling statistics
                    for window in [7, 30, 90]:
                        if len(series_data) > window:
                            # Rolling mean
                            mean_col = f"fred_{series_id}_rolling_mean_{window}d"
                            new_features[mean_col] = series_data.rolling(window).mean()
                            economic_features.append(mean_col)
                            
                            # Rolling standard deviation
                            std_col = f"fred_{series_id}_rolling_std_{window}d"
                            new_features[std_col] = series_data.rolling(window).std()
                            economic_features.append(std_col)
                            
                            # Rolling min/max
                            min_col = f"fred_{series_id}_rolling_min_{window}d"
                            max_col = f"fred_{series_id}_rolling_max_{window}d"
                            new_features[min_col] = series_data.rolling(window).min()
                            new_features[max_col] = series_data.rolling(window).max()
                            economic_features.extend([min_col, max_col])
                    
                    # Rate of change features
                    for period in [7, 30, 90]:
                        if len(series_data) > period:
                            # Percentage change
                            pct_col = f"fred_{series_id}_pct_change_{period}d"
                            new_features[pct_col] = series_data.pct_change(period, fill_method=None)
                            economic_features.append(pct_col)
                            
                            # Log returns
                            log_ret_col = f"fred_{series_id}_log_return_{period}d"
                            new_features[log_ret_col] = np.log(series_data / series_data.shift(period))
                            economic_features.append(log_ret_col)
            
            # Create interaction features between related indicators
            interaction_features = self._create_economic_interactions(fred_pivot)
            economic_features.extend(interaction_features)
            
            # Add all new features at once using concat to avoid fragmentation
            if new_features:
                new_features_df = pd.DataFrame(new_features, index=fred_pivot.index)
                fred_pivot = pd.concat([fred_pivot, new_features_df], axis=1)
            
            logger.info(f"Created {len(economic_features)} economic features")
            return fred_pivot, economic_features
            
        except Exception as e:
            logger.error(f"Error creating economic features: {e}")
            raise
    
    def _create_economic_interactions(self, fred_pivot: pd.DataFrame) -> List[str]:
        """Create interaction features between economic indicators"""
        interaction_features = []
        
        try:
            # Interest rate interactions
            if 'FEDFUNDS' in fred_pivot.columns and 'DGS10' in fred_pivot.columns:
                # Yield curve spread (10Y - Fed Funds)
                fred_pivot['yield_curve_spread'] = fred_pivot['DGS10'] - fred_pivot['FEDFUNDS']
                interaction_features.append('yield_curve_spread')
                
                # Yield curve slope (normalized)
                fred_pivot['yield_curve_slope'] = (fred_pivot['DGS10'] - fred_pivot['FEDFUNDS']) / fred_pivot['FEDFUNDS']
                interaction_features.append('yield_curve_slope')
            
            # Inflation and unemployment interactions
            if 'CPIAUCSL' in fred_pivot.columns and 'UNRATE' in fred_pivot.columns:
                # Phillips curve proxy
                fred_pivot['phillips_curve_proxy'] = fred_pivot['CPIAUCSL'].pct_change(12, fill_method=None) - fred_pivot['UNRATE']
                interaction_features.append('phillips_curve_proxy')
            
            # GDP and employment interactions
            if 'GDP' in fred_pivot.columns and 'PAYEMS' in fred_pivot.columns:
                # Labor productivity proxy
                fred_pivot['labor_productivity_proxy'] = fred_pivot['GDP'] / fred_pivot['PAYEMS']
                interaction_features.append('labor_productivity_proxy')
            
            # Exchange rate interactions
            if 'DEXUSEU' in fred_pivot.columns and 'DEXCHUS' in fred_pivot.columns:
                # Currency strength index
                fred_pivot['currency_strength_index'] = (fred_pivot['DEXUSEU'] + fred_pivot['DEXCHUS']) / 2
                interaction_features.append('currency_strength_index')
            
            logger.info(f"Created {len(interaction_features)} interaction features")
            
        except Exception as e:
            logger.warning(f"Error creating interaction features: {e}")
        
        return interaction_features
    
    def create_financial_features(self, yahoo_df: pd.DataFrame) -> pd.DataFrame:
        """Create financial market features from Yahoo Finance data"""
        try:
            if yahoo_df.empty or 'date' not in yahoo_df.columns:
                logger.info("No Yahoo Finance data available")
                return pd.DataFrame(), []
                
            logger.info("Creating financial market features from Yahoo Finance data...")
            
            # Pivot Yahoo Finance data to wide format
            yahoo_pivot = yahoo_df.pivot_table(
                index='date',
                columns='symbol',
                values='Close',
                aggfunc='first'
            )
            
            # Force UTC timezone-aware index before resampling
            logger.debug("Forcing Yahoo pivot index to UTC timezone-aware")
            yahoo_pivot = self._force_utc_index(yahoo_pivot)
            yahoo_pivot = yahoo_pivot.sort_index().resample('D').ffill()
            logger.debug(f"Yahoo pivot index timezone after normalization: {yahoo_pivot.index.tz}")
            
            # Collect all new features in a dictionary to avoid DataFrame fragmentation
            financial_features, feat_dict = [], {}
            
            for symbol in yahoo_pivot.columns:
                if symbol in yahoo_pivot.columns:
                    symbol_data = yahoo_pivot[symbol]
                    
                    # Returns features
                    for period in [1, 7, 30]:
                        if len(symbol_data) > period:
                            ret_col = f"yahoo_{symbol}_return_{period}d"
                            log_ret_col = f"yahoo_{symbol}_log_return_{period}d"
                            feat_dict[ret_col] = symbol_data.pct_change(period, fill_method=None)
                            feat_dict[log_ret_col] = np.log(symbol_data / symbol_data.shift(period))
                            financial_features += [ret_col, log_ret_col]
                    
                    # Volatility features
                    for window in [7, 30, 90]:
                        if len(symbol_data) > window:
                            vol_col = f"yahoo_{symbol}_volatility_{window}d"
                            realized_vol_col = f"yahoo_{symbol}_realized_vol_{window}d"
                            r = symbol_data.pct_change(fill_method=None)
                            feat_dict[vol_col] = r.rolling(window).std()
                            feat_dict[realized_vol_col] = np.sqrt((r ** 2).rolling(window).sum())
                            financial_features += [vol_col, realized_vol_col]
                    
                    # Technical indicators
                    for window in [7, 14, 30]:
                        if len(symbol_data) > window:
                            ma_col = f"yahoo_{symbol}_ma_{window}d"
                            ma_ratio_col = f"yahoo_{symbol}_ma_ratio_{window}d"
                            roll = symbol_data.rolling(window)
                            feat_dict[ma_col] = roll.mean()
                            feat_dict[ma_ratio_col] = symbol_data / roll.mean()
                            financial_features += [ma_col, ma_ratio_col]
            
            # Create market-wide features
            market_features, market_feat_dict = self._create_market_features(yahoo_pivot)
            financial_features.extend(market_features)
            
            # One-shot add to avoid fragmentation (combine all features)
            if feat_dict or market_feat_dict:
                all_feat_dict = {**feat_dict, **market_feat_dict}
                yahoo_pivot = pd.concat([yahoo_pivot, pd.DataFrame(all_feat_dict, index=yahoo_pivot.index)], axis=1).copy()
            
            logger.info(f"Created {len(financial_features)} financial features")
            return yahoo_pivot, financial_features
            
        except Exception as e:
            logger.error(f"Error creating financial features: {e}")
            raise
    
    def _create_market_features(self, yahoo_pivot: pd.DataFrame) -> Tuple[List[str], Dict[str, pd.Series]]:
        """Create market-wide financial features"""
        market_features = []
        market_feat_dict = {}
        
        try:
            # Market indices (if available)
            market_indices = ['^GSPC', '^DJI', '^IXIC']
            available_indices = [idx for idx in market_indices if idx in yahoo_pivot.columns]
            
            if available_indices:
                # Market volatility index
                market_returns = yahoo_pivot[available_indices].pct_change(fill_method=None)
                market_vol = market_returns.rolling(30).std().mean(axis=1)
                market_feat_dict['market_volatility_30d'] = market_vol
                market_features.append('market_volatility_30d')
                
                # Market correlation
                if len(available_indices) > 1:
                    # Ensure we have enough data for rolling correlation
                    min_data = market_returns.notna().sum().min()
                    if min_data >= 2:
                        market_corr = market_returns.rolling(30).corr()
                        # Average correlation between indices
                        market_feat_dict['market_correlation_30d'] = market_corr.groupby(level=0).mean().mean(axis=1)
                        market_features.append('market_correlation_30d')
            
            # VIX-based features (if available)
            if '^VIX' in yahoo_pivot.columns:
                vix_data = yahoo_pivot['^VIX']
                
                # VIX lagged features
                for lag in [1, 7, 30]:
                    if len(vix_data) > lag:
                        vix_lag_col = f"vix_lag_{lag}d"
                        market_feat_dict[vix_lag_col] = vix_data.shift(lag)
                        market_features.append(vix_lag_col)
                
                # VIX rolling statistics
                for window in [7, 30]:
                    if len(vix_data) > window:
                        vix_mean_col = f"vix_rolling_mean_{window}d"
                        vix_std_col = f"vix_rolling_std_{window}d"
                        market_feat_dict[vix_mean_col] = vix_data.rolling(window).mean()
                        market_feat_dict[vix_std_col] = vix_data.rolling(window).std()
                        market_features.extend([vix_mean_col, vix_std_col])
            
            logger.info(f"Created {len(market_features)} market features")
            
        except Exception as e:
            logger.warning(f"Error creating market features: {e}")
        
        return market_features, market_feat_dict
    
    def create_world_bank_features(self, worldbank_df: pd.DataFrame) -> pd.DataFrame:
        """Create features from World Bank data"""
        try:
            logger.info("Creating World Bank features...")
            
            if worldbank_df.empty:
                logger.info("No World Bank data available")
                return pd.DataFrame(), []
            
            # Pivot World Bank data to wide format
            worldbank_pivot = worldbank_df.pivot_table(
                index='date',
                columns=['country_code', 'indicator_code'],
                values='value',
                aggfunc='first'
            )
            
            # Force UTC timezone-aware index before resampling
            logger.debug("Forcing World Bank pivot index to UTC timezone-aware")
            worldbank_pivot = self._force_utc_index(worldbank_pivot)
            worldbank_pivot = worldbank_pivot.sort_index().resample('D').ffill()
            logger.debug(f"World Bank pivot index timezone after normalization: {worldbank_pivot.index.tz}")
            
            # Snapshot original tuple columns to avoid re-looping over newly added ones
            orig_tuple_cols = [c for c in worldbank_pivot.columns
                              if isinstance(c, tuple) and len(c) == 2]
            
            # Collect all new features in a dictionary to avoid DataFrame fragmentation
            new_features = {}
            worldbank_features = []
            
            # Create features for each indicator-country combination (only original columns)
            for col in orig_tuple_cols:
                country_code, indicator_code = col
                indicator_data = worldbank_pivot[col]
                
                # Basic lagged features
                for lag in [1, 7, 30, 90]:
                    if len(indicator_data) > lag:
                        lagged_col = f"wb_{country_code}_{indicator_code}_lag_{lag}d"
                        new_features[lagged_col] = indicator_data.shift(lag)
                        worldbank_features.append(lagged_col)
                
                # Rolling statistics (for annual data, use longer windows)
                for window in [365, 730, 1095]:  # 1, 2, 3 years
                    if len(indicator_data) > window:
                        mean_col = f"wb_{country_code}_{indicator_code}_rolling_mean_{window}d"
                        std_col = f"wb_{country_code}_{indicator_code}_rolling_std_{window}d"
                        roll = indicator_data.rolling(window)
                        new_features[mean_col] = roll.mean()
                        new_features[std_col] = roll.std()
                        worldbank_features.extend([mean_col, std_col])
            
            # Create cross-country features
            cross_country_features, cross_country_new_features = self._create_cross_country_features(worldbank_pivot)
            worldbank_features.extend(cross_country_features)
            
            # Add all new features at once using concat to avoid fragmentation
            if new_features or cross_country_new_features:
                all_new_features = {**new_features, **cross_country_new_features}
                new_features_df = pd.DataFrame(all_new_features, index=worldbank_pivot.index)
                worldbank_pivot = pd.concat(
                    [worldbank_pivot, new_features_df], axis=1
                ).copy()
            
            # Flatten leftover MultiIndex columns to strings to avoid tuple columns downstream
            if isinstance(worldbank_pivot.columns, pd.MultiIndex):
                flat = [f"wb_{a}_{b}" if isinstance(c, tuple) else str(c) for c in worldbank_pivot.columns]
                # make unique to handle potential collisions
                seen = {}
                unique = []
                for name in flat:
                    if name not in seen:
                        seen[name] = 0
                        unique.append(name)
                    else:
                        seen[name] += 1
                        unique.append(f"{name}__{seen[name]}")
                worldbank_pivot.columns = unique
            
            logger.info(f"Created {len(worldbank_features)} World Bank features")
            return worldbank_pivot, worldbank_features
            
        except Exception as e:
            logger.error(f"Error creating World Bank features: {e}")
            raise
    
    def _create_cross_country_features(self, worldbank_pivot: pd.DataFrame) -> Tuple[List[str], Dict[str, pd.Series]]:
        """Create cross-country comparison features"""
        cross_country_features = []
        new_features = {}
        
        try:
            columns = worldbank_pivot.columns
            
            # Only derive countries/indicators from tuple (multiindex-origin) columns
            tuple_cols = [col for col in columns if isinstance(col, tuple) and len(col) == 2]
            if not tuple_cols:
                logger.info("No tuple columns found in World Bank pivot; skipping cross-country features.")
                return cross_country_features, new_features
            
            countries = list({col[0] for col in tuple_cols})
            indicators = list({col[1] for col in tuple_cols})
            
            if 'US' in countries:
                for indicator in indicators:
                    us_col = ('US', indicator)
                    if us_col in columns:
                        us_data = worldbank_pivot[us_col]
                        for country in countries:
                            if country == 'US':
                                continue
                            country_col = (country, indicator)
                            if country_col in columns:
                                relative_col = f"wb_{country}_{indicator}_relative_us"
                                diff_col = f"wb_{country}_{indicator}_diff_us"
                                new_features[relative_col] = worldbank_pivot[country_col] / us_data
                                new_features[diff_col] = worldbank_pivot[country_col] - us_data
                                cross_country_features.extend([relative_col, diff_col])
            
            logger.info(f"Created {len(cross_country_features)} cross-country features")
            
        except Exception as e:
            logger.warning(f"Error creating cross-country features: {e}")
        
        return cross_country_features, new_features
    
    def create_regime_features(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        """Create regime classification features for the self-attention model"""
        try:
            logger.info("Creating regime classification features...")
            
            regime_features = []
            
            # Economic regime indicators
            if 'fred_GDP_lag_30d' in combined_df.columns:
                # GDP growth regime
                gdp_growth = combined_df['fred_GDP_lag_30d'].pct_change(90, fill_method=None)
                combined_df['gdp_growth_regime'] = pd.cut(
                    gdp_growth, 
                    bins=[-np.inf, -0.02, 0.02, np.inf], 
                    labels=['recession', 'stable', 'expansion']
                )
                regime_features.append('gdp_growth_regime')
            
            # Inflation regime
            if 'fred_CPIAUCSL_lag_30d' in combined_df.columns:
                inflation_rate = combined_df['fred_CPIAUCSL_lag_30d'].pct_change(365, fill_method=None)
                combined_df['inflation_regime'] = pd.cut(
                    inflation_rate,
                    bins=[-np.inf, 0.01, 0.03, np.inf],
                    labels=['deflation', 'stable', 'high_inflation']
                )
                regime_features.append('inflation_regime')
            
            # Monetary policy regime
            if 'fred_FEDFUNDS_lag_30d' in combined_df.columns:
                fed_funds = combined_df['fred_FEDFUNDS_lag_30d']
                combined_df['monetary_regime'] = pd.cut(
                    fed_funds,
                    bins=[-np.inf, 0.01, 0.05, np.inf],
                    labels=['zero_lower_bound', 'accommodative', 'restrictive']
                )
                regime_features.append('monetary_regime')
            
            # Market volatility regime
            if 'market_volatility_30d' in combined_df.columns:
                market_vol = combined_df['market_volatility_30d']
                if market_vol.notna().sum() >= 3:
                    vol_quantiles = market_vol.quantile([0.33, 0.67])
                    if np.isfinite(vol_quantiles.iloc[0]) and np.isfinite(vol_quantiles.iloc[1]):
                        combined_df['volatility_regime'] = pd.cut(
                            market_vol,
                            bins=[-np.inf, vol_quantiles.iloc[0], vol_quantiles.iloc[1], np.inf],
                            labels=['low_vol', 'medium_vol', 'high_vol']
                        )
                        regime_features.append('volatility_regime')
            
            # Create regime interaction features
            regime_interactions = self._create_regime_interactions(combined_df)
            regime_features.extend(regime_interactions)
            
            logger.info(f"Created {len(regime_features)} regime features")
            return combined_df, regime_features
            
        except Exception as e:
            logger.error(f"Error creating regime features: {e}")
            raise
    
    def _create_regime_interactions(self, df: pd.DataFrame) -> List[str]:
        """Create interaction features between different regimes"""
        regime_interactions = []
        
        try:
            regime_columns = [col for col in df.columns if isinstance(col, str) and col.endswith('_regime')]
            
            # Create pairwise regime interactions
            for i, regime1 in enumerate(regime_columns):
                for regime2 in regime_columns[i+1:]:
                    if regime1 in df.columns and regime2 in df.columns:
                        # Create interaction feature
                        interaction_col = f"regime_interaction_{regime1.replace('_regime', '')}_{regime2.replace('_regime', '')}"
                        
                        # Convert regimes to numeric for interaction
                        regime1_numeric = pd.Categorical(df[regime1]).codes
                        regime2_numeric = pd.Categorical(df[regime2]).codes
                        
                        df[interaction_col] = regime1_numeric * regime2_numeric
                        regime_interactions.append(interaction_col)
            
            logger.info(f"Created {len(regime_interactions)} regime interaction features")
            
        except Exception as e:
            logger.warning(f"Error creating regime interactions: {e}")
        
        return regime_interactions
    
    def create_treatment_features(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        """Create treatment features for causal inference"""
        try:
            logger.info("Creating treatment features for causal inference...")
            
            treatment_features = []
            
            # Policy intervention indicators
            if 'fred_FEDFUNDS_lag_30d' in combined_df.columns:
                fed_funds = combined_df['fred_FEDFUNDS_lag_30d']
                
                # Interest rate change indicators
                fed_funds_change = fed_funds.diff(30)
                combined_df['fed_funds_increase'] = (fed_funds_change > 0.25).astype(int)
                combined_df['fed_funds_decrease'] = (fed_funds_change < -0.25).astype(int)
                treatment_features.extend(['fed_funds_increase', 'fed_funds_decrease'])
                
                # Zero lower bound indicator
                combined_df['zero_lower_bound'] = (fed_funds < 0.01).astype(int)
                treatment_features.append('zero_lower_bound')
            
            # Quantitative easing indicators (proxy based on balance sheet)
            if 'fred_DGS10_lag_30d' in combined_df.columns and 'fred_FEDFUNDS_lag_30d' in combined_df.columns:
                # Yield curve compression as QE proxy
                yield_spread = combined_df['fred_DGS10_lag_30d'] - combined_df['fred_FEDFUNDS_lag_30d']
                combined_df['qe_proxy'] = (yield_spread < 1.0).astype(int)
                treatment_features.append('qe_proxy')
            
            # Fiscal policy indicators (proxy based on debt)
            if 'wb_US_GC.DOD.TOTL.GD.ZS_lag_30d' in combined_df.columns:
                debt_gdp = combined_df['wb_US_GC.DOD.TOTL.GD.ZS_lag_30d']
                combined_df['high_debt_regime'] = (debt_gdp > 100).astype(int)
                treatment_features.append('high_debt_regime')
            
            # Market stress indicators
            if 'market_volatility_30d' in combined_df.columns:
                market_vol = combined_df['market_volatility_30d']
                vol_threshold = market_vol.quantile(0.9)
                combined_df['market_stress'] = (market_vol > vol_threshold).astype(int)
                treatment_features.append('market_stress')
            
            logger.info(f"Created {len(treatment_features)} treatment features")
            return combined_df, treatment_features
            
        except Exception as e:
            logger.error(f"Error creating treatment features: {e}")
            raise
    
    def create_final_features(self, fred_df: pd.DataFrame, worldbank_df: pd.DataFrame, 
                            yahoo_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Create the final feature set for training"""
        try:
            logger.info("Creating final feature set...")
            
            # Step 1: Create economic features
            try:
                fred_features_df, fred_feature_names = self.create_economic_features(fred_df)
                logger.info(f"Economic features created successfully: {len(fred_feature_names)} features")
            except Exception as e:
                logger.error(f"Error creating economic features: {e}")
                fred_features_df, fred_feature_names = pd.DataFrame(), []
            
            # Step 2: Create financial features
            try:
                yahoo_features_df, yahoo_feature_names = self.create_financial_features(yahoo_df)
                logger.info(f"Financial features created successfully: {len(yahoo_feature_names)} features")
            except Exception as e:
                logger.error(f"Error creating financial features: {e}")
                yahoo_features_df, yahoo_feature_names = pd.DataFrame(), []
            
            # Step 3: Create World Bank features
            try:
                worldbank_features_df, worldbank_feature_names = self.create_world_bank_features(worldbank_df)
                logger.info(f"World Bank features created successfully: {len(worldbank_feature_names)} features")
            except Exception as e:
                logger.error(f"Error creating World Bank features: {e}")
                worldbank_features_df, worldbank_feature_names = pd.DataFrame(), []
            
            # Step 4: Check if we have any data to work with
            if all(df.empty for df in [fred_features_df, yahoo_features_df, worldbank_features_df]):
                logger.error("All feature DataFrames are empty, cannot proceed")
                raise ValueError("No features could be created from any data source")
            
            # Step 4: Normalize timezone-aware indexes before combining
            logger.info("Normalizing timezone indexes before combining features...")
            normalized_dfs = self._normalize_timezone_indexes([
                fred_features_df,
                yahoo_features_df,
                worldbank_features_df
            ])
            
            # Log timezone info for debugging
            for i, df_name in enumerate(['FRED', 'Yahoo', 'World Bank']):
                if not normalized_dfs[i].empty:
                    logger.info(f"{df_name} features timezone: {normalized_dfs[i].index.tz}")
                    logger.info(f"{df_name} features shape: {normalized_dfs[i].shape}")
                    logger.info(f"{df_name} features index range: {normalized_dfs[i].index.min()} to {normalized_dfs[i].index.max()}")
                else:
                    logger.info(f"{df_name} features: empty DataFrame")
            
            # Validate date range compatibility
            non_empty_dfs = [df for df in normalized_dfs if not df.empty]
            if len(non_empty_dfs) > 1:
                min_dates = [df.index.min() for df in non_empty_dfs]
                max_dates = [df.index.max() for df in non_empty_dfs]
                logger.info(f"Date range validation - Min dates: {min_dates}")
                logger.info(f"Date range validation - Max dates: {max_dates}")
                
                # Check for significant date range mismatches
                min_date_range = max(min_dates) - min(min_dates)
                max_date_range = max(max_dates) - min(max_dates)
                if min_date_range > pd.Timedelta(days=30) or max_date_range > pd.Timedelta(days=30):
                    logger.warning(f"Significant date range mismatch detected: min_diff={min_date_range}, max_diff={max_date_range}")
            
            # Step 5: Combine all features using normalized dataframes
            logger.info("Combining normalized feature dataframes...")
            try:
                combined_df = pd.concat(normalized_dfs, axis=1)
            except Exception as e:
                logger.error(f"Error during concatenation: {e}")
                # Try to identify which DataFrame is causing the issue
                for i, df in enumerate(normalized_dfs):
                    if not df.empty:
                        logger.info(f"DataFrame {i} index type: {type(df.index)}, timezone: {getattr(df.index, 'tz', 'N/A')}")
                        logger.info(f"DataFrame {i} index sample: {df.index[:5]}")
                
                # Try fallback approach with common timezone index
                logger.info("Attempting fallback concatenation with common timezone index...")
                try:
                    common_index = self._create_common_timezone_index(normalized_dfs)
                    # Reindex all dataframes to the common index
                    reindexed_dfs = []
                    for df in normalized_dfs:
                        if not df.empty:
                            reindexed_df = df.reindex(common_index, method='ffill')
                            reindexed_dfs.append(reindexed_df)
                        else:
                            reindexed_dfs.append(df)
                    
                    combined_df = pd.concat(reindexed_dfs, axis=1)
                    logger.info("Fallback concatenation successful")
                except Exception as fallback_e:
                    logger.error(f"Fallback concatenation also failed: {fallback_e}")
                    logger.error("Attempting final fallback with empty DataFrame...")
                    # Create a minimal empty DataFrame to avoid complete failure
                    try:
                        combined_df = pd.DataFrame(index=pd.date_range(
                            start=pd.Timestamp.now() - pd.Timedelta(days=365),
                            end=pd.Timestamp.now(),
                            freq='D',
                            tz='UTC'
                        ))
                        logger.warning("Created minimal empty DataFrame as final fallback")
                    except Exception as final_fallback_e:
                        logger.error(f"Final fallback also failed: {final_fallback_e}")
                        raise
            
            # Final validation: ensure combined DataFrame has consistent timezone-aware index
            if not combined_df.empty:
                if combined_df.index.tz is None:
                    logger.warning("Combined DataFrame has timezone-naive index, converting to UTC")
                    combined_df.index = combined_df.index.tz_localize('UTC')
                elif str(combined_df.index.tz) != 'UTC':
                    logger.info(f"Converting combined DataFrame index from {combined_df.index.tz} to UTC")
                    combined_df.index = combined_df.index.tz_convert('UTC')
                
                logger.info(f"Combined DataFrame timezone: {combined_df.index.tz}")
                logger.info(f"Combined DataFrame shape: {combined_df.shape}")
                logger.info(f"Combined DataFrame index range: {combined_df.index.min()} to {combined_df.index.max()}")
            else:
                logger.warning("Combined DataFrame is empty after concatenation")
            
            # Remove duplicate columns and de-fragment in one shot
            combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()].copy()
            
            # Validate the combined DataFrame
            combined_df = self._validate_combined_dataframe(combined_df)
            
            # Step 6: Create regime features
            logger.info("Creating regime features...")
            try:
                combined_df, regime_feature_names = self.create_regime_features(combined_df)
                logger.info(f"Regime features created successfully: {len(regime_feature_names)} features")
            except Exception as e:
                logger.error(f"Error creating regime features: {e}")
                regime_feature_names = []
            
            # Step 7: Create treatment features
            logger.info("Creating treatment features...")
            try:
                combined_df, treatment_feature_names = self.create_treatment_features(combined_df)
                logger.info(f"Treatment features created successfully: {len(treatment_feature_names)} features")
            except Exception as e:
                logger.error(f"Error creating treatment features: {e}")
                treatment_feature_names = []
            
            # Step 8: Create target variable (GDP growth)
            logger.info("Creating target variable...")
            try:
                if 'fred_GDP_lag_30d' in combined_df.columns:
                    tgt = combined_df['fred_GDP_lag_30d'].pct_change(90, fill_method=None)
                    combined_df['target'] = tgt.replace([np.inf, -np.inf], np.nan)
                    logger.info("Target variable created successfully")
                else:
                    logger.warning("GDP lag feature not found, skipping target variable creation")
            except Exception as e:
                logger.error(f"Error creating target variable: {e}")
                logger.warning("Target variable creation failed, continuing without target")
            
            # Step 9: Clean up the dataset
            logger.info("Cleaning final dataset...")
            combined_df = self._clean_final_dataset(combined_df)
            
            # Final timezone validation
            if not combined_df.empty:
                if hasattr(combined_df.index, 'tz'):
                    if combined_df.index.tz is None:
                        logger.warning("Final dataset has timezone-naive index, converting to UTC")
                        combined_df.index = combined_df.index.tz_localize('UTC')
                    elif str(combined_df.index.tz) != 'UTC':
                        logger.info(f"Converting final dataset index from {combined_df.index.tz} to UTC")
                        combined_df.index = combined_df.index.tz_convert('UTC')
                
                logger.info(f"Final dataset validation complete: shape={combined_df.shape}, timezone={getattr(combined_df.index, 'tz', 'N/A')}")
                
                # Additional validation: check for any remaining timezone issues in columns
                datetime_columns = [col for col in combined_df.columns if pd.api.types.is_datetime64_any_dtype(combined_df[col])]
                if datetime_columns:
                    logger.info(f"Found {len(datetime_columns)} datetime columns, ensuring timezone consistency")
                    for col in datetime_columns:
                        if combined_df[col].dt.tz is None:
                            logger.debug(f"Column {col} is timezone-naive, converting to UTC")
                            combined_df[col] = combined_df[col].dt.tz_localize('UTC')
                        elif str(combined_df[col].dt.tz) != 'UTC':
                            logger.debug(f"Column {col} timezone: {combined_df[col].dt.tz}, converting to UTC")
                            combined_df[col] = combined_df[col].dt.tz_convert('UTC')
            else:
                logger.warning("Final dataset is empty after all processing steps")
            
            # Collect all feature names
            all_features = (fred_feature_names + yahoo_feature_names + 
                          worldbank_feature_names + regime_feature_names + 
                          treatment_feature_names)
            
            # Remove duplicates
            all_features = list(set(all_features))
            
            # Log feature creation summary
            logger.info("=== Feature Creation Summary ===")
            logger.info(f"Economic features: {len(fred_feature_names)}")
            logger.info(f"Financial features: {len(yahoo_feature_names)}")
            logger.info(f"World Bank features: {len(worldbank_feature_names)}")
            logger.info(f"Regime features: {len(regime_feature_names)}")
            logger.info(f"Treatment features: {len(treatment_feature_names)}")
            logger.info(f"Total unique features: {len(all_features)}")
            logger.info(f"Final dataset shape: {combined_df.shape}")
            logger.info(f"Final dataset timezone: {getattr(combined_df.index, 'tz', 'N/A')}")
            logger.info("=== End Feature Creation Summary ===")
            
            # Final validation before return
            if not combined_df.empty:
                try:
                    # Ensure the final dataset is properly formatted
                    combined_df = combined_df.copy()  # Defragment
                    
                    # Final timezone check
                    if hasattr(combined_df.index, 'tz') and combined_df.index.tz is not None:
                        if str(combined_df.index.tz) != 'UTC':
                            logger.info(f"Final timezone conversion: {combined_df.index.tz} -> UTC")
                            combined_df.index = combined_df.index.tz_convert('UTC')
                    
                    # Final check for any remaining timezone issues
                    if hasattr(combined_df.index, 'tz') and combined_df.index.tz is None:
                        logger.warning("Final dataset index is still timezone-naive, converting to UTC")
                        combined_df.index = combined_df.index.tz_localize('UTC')
                    
                    # Final validation: ensure all datetime columns are timezone-consistent
                    datetime_columns = [col for col in combined_df.columns if pd.api.types.is_datetime64_any_dtype(combined_df[col])]
                    if datetime_columns:
                        logger.info(f"Final validation: ensuring timezone consistency in {len(datetime_columns)} datetime columns")
                        for col in datetime_columns:
                            try:
                                if combined_df[col].dt.tz is None:
                                    combined_df[col] = combined_df[col].dt.tz_localize('UTC')
                                elif str(combined_df[col].dt.tz) != 'UTC':
                                    combined_df[col] = combined_df[col].dt.tz_convert('UTC')
                            except Exception as col_e:
                                logger.warning(f"Could not normalize timezone for column {col}: {col_e}")
                    
                    # Final check: ensure the DataFrame is not fragmented
                    if hasattr(combined_df, '_is_copy') and combined_df._is_copy is not None:
                        combined_df = combined_df.copy()
                    
                    # Final validation: ensure the DataFrame has a valid timezone-aware index
                    if not hasattr(combined_df.index, 'tz') or combined_df.index.tz is None:
                        logger.warning("Final dataset index is still timezone-naive, converting to UTC")
                        combined_df.index = combined_df.index.tz_localize('UTC')
                    
                    logger.info(f"Final dataset ready: shape={combined_df.shape}, timezone={getattr(combined_df.index, 'tz', 'N/A')}")
                except Exception as e:
                    logger.error(f"Error in final validation: {e}")
                    # Continue with the dataset as-is rather than failing completely
            
            return combined_df, all_features
            
        except Exception as e:
            logger.error(f"Error creating final features: {e}")
            logger.error(f"Error details: {type(e).__name__}: {str(e)}")
            
            # Try to provide more context about what failed
            if "Cannot join tz-naive with tz-aware" in str(e):
                logger.error("Timezone mismatch detected. This suggests inconsistent timezone handling in the input data.")
                logger.error("Please ensure all input data has consistent timezone information.")
                logger.error("The fix implemented should handle this automatically by normalizing all timezones to UTC.")
            elif "timezone" in str(e).lower():
                logger.error("Timezone-related error detected. This suggests timezone conversion issues.")
                logger.error("Please check the timezone information in your input data.")
                logger.error("The fix implemented should handle this automatically by normalizing all timezones to UTC.")
            else:
                logger.error("Unknown error occurred during feature creation.")
                logger.error("Please check the logs for more details about what failed.")
            
            raise
    
    def _normalize_timezone_indexes(self, dataframes: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Normalize all dataframe indexes to be timezone-aware UTC before concatenation"""
        normalized_dfs = []
        
        for i, df in enumerate(dataframes):
            if df.empty:
                normalized_dfs.append(df)
                continue
                
            try:
                # Ensure index is datetime
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index, errors='coerce')
                
                # Handle any NaT values that might cause issues
                if df.index.isna().any():
                    logger.warning(f"DataFrame {i} has NaT values in index, removing them")
                    df = df.loc[~df.index.isna()]
                
                # Normalize to UTC timezone-aware
                if df.index.tz is None:
                    # If timezone-naive, assume UTC and make timezone-aware
                    df.index = df.index.tz_localize('UTC')
                    logger.debug(f"DataFrame {i} index made timezone-aware UTC")
                else:
                    # If already timezone-aware, convert to UTC
                    df.index = df.index.tz_convert('UTC')
                    logger.debug(f"DataFrame {i} index converted to UTC")
                
                # Also check if there's a 'date' column that might have timezone issues
                if 'date' in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df['date']):
                        if df['date'].dt.tz is None:
                            df['date'] = df['date'].dt.tz_localize('UTC')
                        else:
                            df['date'] = df['date'].dt.tz_convert('UTC')
                
                normalized_dfs.append(df)
                
            except Exception as e:
                logger.error(f"Error normalizing timezone for DataFrame {i}: {e}")
                # Try to create a minimal valid DataFrame to avoid breaking the pipeline
                if not df.empty:
                    # Create a simple index with UTC timezone
                    if not df.index.empty:
                        start_date = df.index.min()
                        end_date = df.index.max()
                        # Ensure dates are timezone-aware UTC
                        if getattr(start_date, 'tz', None) is None:
                            start_date = start_date.tz_localize('UTC')
                            end_date = end_date.tz_localize('UTC')
                        else:
                            start_date = start_date.tz_convert('UTC')
                            end_date = end_date.tz_convert('UTC')
                    else:
                        start_date = pd.Timestamp.now()
                        end_date = pd.Timestamp.now()
                    
                    simple_index = pd.date_range(
                        start=start_date,
                        end=end_date,
                        freq='D',
                        tz='UTC'
                    )
                    # Create a minimal DataFrame with the same columns but simple index
                    minimal_df = pd.DataFrame(index=simple_index, columns=df.columns)
                    minimal_df = minimal_df.fillna(0)
                    normalized_dfs.append(minimal_df)
                else:
                    normalized_dfs.append(df)
        
        return normalized_dfs
    
    def _create_common_timezone_index(self, dataframes: List[pd.DataFrame]) -> pd.DatetimeIndex:
        """Create a common timezone-aware UTC index for all dataframes"""
        try:
            # Find the common date range across all non-empty dataframes
            non_empty_dfs = [df for df in dataframes if not df.empty]
            if not non_empty_dfs:
                # If all are empty, create a default index
                return pd.date_range(
                    start=pd.Timestamp.now() - pd.Timedelta(days=365),
                    end=pd.Timestamp.now(),
                    freq='D',
                    tz='UTC'
                )
            
            # Get the min and max dates across all dataframes
            all_dates = []
            for df in non_empty_dfs:
                if not df.index.empty:
                    i_min, i_max = df.index.min(), df.index.max()
                    if getattr(i_min, 'tz', None) is None:
                        i_min = i_min.tz_localize('UTC')
                        i_max = i_max.tz_localize('UTC')
                    else:
                        i_min = i_min.tz_convert('UTC')
                        i_max = i_max.tz_convert('UTC')
                    all_dates.extend([i_min, i_max])
            
            if not all_dates:
                # Fallback to default index
                return pd.date_range(
                    start=pd.Timestamp.now() - pd.Timedelta(days=365),
                    end=pd.Timestamp.now(),
                    freq='D',
                    tz='UTC'
                )
            
            min_date = min(all_dates)
            max_date = max(all_dates)
            
            # Create a daily index covering the full range
            common_index = pd.date_range(
                start=min_date,
                end=max_date,
                freq='D',
                tz='UTC'
            )
            
            logger.info(f"Created common timezone-aware index: {min_date} to {max_date}")
            return common_index
            
        except Exception as e:
            logger.error(f"Error creating common timezone index: {e}")
            # Fallback to default index
            return pd.date_range(
                start=pd.Timestamp.now() - pd.Timedelta(days=365),
                end=pd.Timestamp.now(),
                freq='D',
                tz='UTC'
            )
    
    def _ensure_timezone_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all datetime columns in the DataFrame are timezone-consistent"""
        try:
            if df.empty:
                return df
            
            # Check all columns for datetime types
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    if df[col].dt.tz is None:
                        # If timezone-naive, assume UTC
                        df[col] = df[col].dt.tz_localize('UTC')
                    elif str(df[col].dt.tz) != 'UTC':
                        # If different timezone, convert to UTC
                        df[col] = df[col].dt.tz_convert('UTC')
            
            return df
            
        except Exception as e:
            logger.warning(f"Error ensuring timezone consistency: {e}")
            return df
    
    def _validate_combined_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the combined DataFrame"""
        try:
            if df.empty:
                logger.warning("Combined DataFrame is empty")
                return df
            
            # Check for any remaining timezone issues
            if hasattr(df.index, 'tz'):
                if df.index.tz is None:
                    logger.warning("Combined DataFrame index is timezone-naive, converting to UTC")
                    df.index = df.index.tz_localize('UTC')
                elif str(df.index.tz) != 'UTC':
                    logger.info(f"Converting combined DataFrame index from {df.index.tz} to UTC")
                    df.index = df.index.tz_convert('UTC')
            
            # Check for any infinite values
            if df.isin([np.inf, -np.inf]).any().any():
                logger.warning("Found infinite values in combined DataFrame, replacing with NaN")
                df = df.replace([np.inf, -np.inf], np.nan)
            
            # Check for any remaining NaN values in the index
            if df.index.isna().any():
                logger.warning("Found NaN values in index, removing affected rows")
                df = df.dropna(subset=[df.index.name] if df.index.name else None)
            
            # Ensure the DataFrame is not fragmented
            df = df.copy()
            
            logger.info(f"Combined DataFrame validation complete: shape={df.shape}, timezone={getattr(df.index, 'tz', 'N/A')}")
            return df
            
        except Exception as e:
            logger.error(f"Error validating combined DataFrame: {e}")
            return df
    
    def _clean_final_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare the final dataset"""
        try:
            # Reset index to get date as column
            df = df.reset_index()
            
            # Ensure timezone consistency in all datetime columns
            df = self._ensure_timezone_consistency(df)
            
            # Remove rows with missing target values
            if 'target' in df.columns:
                df = df.dropna(subset=['target'])
            
            # Fill missing values in features with 0, handling categorical columns properly
            feature_columns = [col for col in df.columns if col not in ['date', 'target', 'execution_id', 'feature_creation_timestamp']]
            
            for col in feature_columns:
                if col not in df.columns:
                    continue
                if isinstance(df[col].dtype, pd.CategoricalDtype):
                    # ensure a safe string category
                    fill_value = 'unknown'
                    if fill_value not in df[col].cat.categories:
                        df[col] = df[col].cat.add_categories([fill_value])
                    df[col] = df[col].fillna(fill_value)
                else:
                    df[col] = df[col].fillna(0)
            
            # Remove infinite values
            df = df.replace([np.inf, -np.inf], 0)
            
            # Add execution metadata
            df['execution_id'] = self.execution_id
            df['feature_creation_timestamp'] = datetime.now().isoformat()
            
            # Sort by date
            df = df.sort_values('date')
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning final dataset: {e}")
            raise
