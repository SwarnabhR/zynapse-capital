"""
NSE Analytics Engine
Technical indicators, momentum scoring, and pattern detection
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path


class AnalyticsEngine:
    """
    Analytics engine for generating trading signals and insights
    """
    
    def __init__(self):
        self.logger = logging.getLogger('AnalyticsEngine')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def calculate_technical_indicators(self, df: pd.DataFrame, 
                                       symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate technical indicators for time-series data
        
        Args:
            df: DataFrame with OHLCV data (must be sorted by date)
            symbol: Optional symbol filter for single-stock analysis
        
        Returns:
            DataFrame with technical indicators added
        """
        self.logger.info("📈 Calculating technical indicators...")
        
        if symbol:
            # Single symbol analysis
            df = df[df['symbol'] == symbol].copy()
            df = df.sort_values('date').reset_index(drop=True)
            df = self._add_indicators_single(df)
        else:
            # Multi-symbol analysis
            results = []
            for sym in df['symbol'].unique():
                sym_df = df[df['symbol'] == sym].copy()
                sym_df = sym_df.sort_values('date').reset_index(drop=True)
                sym_df = self._add_indicators_single(sym_df)
                results.append(sym_df)
            df = pd.concat(results, ignore_index=True)
        
        self.logger.info(f"✅ Technical indicators calculated")
        return df
    
    def _add_indicators_single(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators for a single symbol"""
        
        if len(df) < 2:
            return df
        
        # 1. Moving Averages
        if len(df) >= 5:
            df['ma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
        if len(df) >= 10:
            df['ma_10'] = df['close'].rolling(window=10, min_periods=1).mean()
        if len(df) >= 20:
            df['ma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        
        # 2. Exponential Moving Averages
        if len(df) >= 12:
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        if len(df) >= 26:
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # 3. MACD (Moving Average Convergence Divergence)
        if 'ema_12' in df.columns and 'ema_26' in df.columns:
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # 4. RSI (Relative Strength Index)
        if len(df) >= 14:
            df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        
        # 5. Bollinger Bands
        if len(df) >= 20:
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / df['bb_middle']) * 100
            df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])) * 100
        
        # 6. ATR (Average True Range) - Volatility
        if len(df) >= 14:
            df['atr_14'] = self._calculate_atr(df, 14)
        
        # 7. Volume Moving Average
        if len(df) >= 20:
            df['volume_ma_20'] = df['volume'].rolling(window=20, min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calculate_momentum_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum score combining returns, delivery, and validation flags
        Score: 0-100 (higher = better momentum)
        """
        self.logger.info("⚡ Calculating momentum scores...")
        
        df = df.copy()
        df['momentum_score'] = 0.0
        
        # Component 1: Price momentum (0-30 points)
        if 'day_return_pct' in df.columns:
            # Normalize returns to 0-30 scale (cap at ±20%)
            df['return_component'] = df['day_return_pct'].clip(-20, 20)
            df['return_component'] = ((df['return_component'] + 20) / 40) * 30
            df['momentum_score'] += df['return_component']
        
        # Component 2: Delivery strength (0-30 points)
        if 'delivery_pct' in df.columns:
            df['delivery_component'] = (df['delivery_pct'] / 100) * 30
            df['momentum_score'] += df['delivery_component']
        
        # Component 3: Volume strength (0-20 points)
        if 'volume_ratio' in df.columns:
            # High volume = strong momentum
            df['volume_component'] = df['volume_ratio'].clip(0, 3)
            df['volume_component'] = (df['volume_component'] / 3) * 20
            df['momentum_score'] += df['volume_component']
        elif 'volume' in df.columns:
            # Fallback: use relative volume
            volume_percentile = df['volume'].rank(pct=True)
            df['volume_component'] = volume_percentile * 20
            df['momentum_score'] += df['volume_component']
        
        # Component 4: Validation quality (0-20 points, penalties for flags)
        df['quality_component'] = 20.0
        
        if 'validation_flags' in df.columns:
            # Penalize risky flags
            df.loc[df['validation_flags'].str.contains('POSSIBLE_SPLIT|POSSIBLE_BONUS', na=False), 
                   'quality_component'] -= 10
            df.loc[df['validation_flags'].str.contains('VOLUME_SPIKE', na=False) & 
                   (df['delivery_pct'] < 20), 'quality_component'] -= 5
            df.loc[df['validation_flags'].str.contains('ILLIQUID|LOW_VOLUME', na=False), 
                   'quality_component'] -= 5
            df.loc[df['validation_flags'].str.contains('PENNY_STOCK', na=False), 
                   'quality_component'] -= 3
        
        df['momentum_score'] += df['quality_component']
        
        # Round to 2 decimals
        df['momentum_score'] = df['momentum_score'].round(2)
        
        self.logger.info(f"✅ Momentum scores calculated (range: {df['momentum_score'].min():.2f} - {df['momentum_score'].max():.2f})")
        
        return df
    
    def calculate_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk score based on validation flags and metrics
        Score: 0-100 (higher = riskier)
        """
        self.logger.info("⚠️  Calculating risk scores...")
        
        df = df.copy()
        df['risk_score'] = 0.0
        
        if 'validation_flags' not in df.columns:
            return df
        
        # High risk flags
        df.loc[df['validation_flags'].str.contains('POSSIBLE_SPLIT|POSSIBLE_BONUS', na=False), 
               'risk_score'] += 30
        df.loc[df['validation_flags'].str.contains('EXTREME_RETURN', na=False), 
               'risk_score'] += 20
        df.loc[df['validation_flags'].str.contains('VOLUME_SPIKE', na=False) & 
               (df['delivery_pct'] < 20), 'risk_score'] += 25
        
        # Medium risk flags
        df.loc[df['validation_flags'].str.contains('UPPER_CIRCUIT_10|LOWER_CIRCUIT_10', na=False), 
               'risk_score'] += 15
        df.loc[df['validation_flags'].str.contains('ILLIQUID', na=False), 
               'risk_score'] += 10
        df.loc[df['validation_flags'].str.contains('PENNY_STOCK', na=False), 
               'risk_score'] += 10
        
        # Low risk flags
        df.loc[df['validation_flags'].str.contains('LOW_VOLUME', na=False), 
               'risk_score'] += 5
        df.loc[df['validation_flags'].str.contains('UPPER_CIRCUIT_5|LOWER_CIRCUIT_5', na=False), 
               'risk_score'] += 5
        
        # Volatility-based risk
        if 'day_range_pct' in df.columns:
            # High intraday range = higher risk
            high_volatility = df['day_range_pct'] > 10
            df.loc[high_volatility, 'risk_score'] += 10
        
        # Low delivery = higher risk
        if 'delivery_pct' in df.columns:
            low_delivery = df['delivery_pct'] < 30
            df.loc[low_delivery, 'risk_score'] += 10
        
        df['risk_score'] = df['risk_score'].clip(0, 100).round(2)
        
        self.logger.info(f"✅ Risk scores calculated (range: {df['risk_score'].min():.2f} - {df['risk_score'].max():.2f})")
        
        return df
    
    def detect_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect chart patterns and trading signals
        """
        self.logger.info("🔍 Detecting patterns...")
        
        df = df.copy()
        df['pattern'] = ''
        df['signal'] = 'HOLD'
        
        # Pattern 1: Golden Cross (bullish)
        if 'ma_5' in df.columns and 'ma_20' in df.columns:
            golden_cross = (df['ma_5'] > df['ma_20']) & (df['ma_5'].shift(1) <= df['ma_20'].shift(1))
            df.loc[golden_cross, 'pattern'] += 'GOLDEN_CROSS;'
            df.loc[golden_cross, 'signal'] = 'BUY'
        
        # Pattern 2: Death Cross (bearish)
        if 'ma_5' in df.columns and 'ma_20' in df.columns:
            death_cross = (df['ma_5'] < df['ma_20']) & (df['ma_5'].shift(1) >= df['ma_20'].shift(1))
            df.loc[death_cross, 'pattern'] += 'DEATH_CROSS;'
            df.loc[death_cross, 'signal'] = 'SELL'
        
        # Pattern 3: RSI Oversold (potential buy)
        if 'rsi_14' in df.columns:
            oversold = df['rsi_14'] < 30
            df.loc[oversold, 'pattern'] += 'RSI_OVERSOLD;'
            df.loc[oversold & (df['signal'] == 'HOLD'), 'signal'] = 'BUY'
        
        # Pattern 4: RSI Overbought (potential sell)
        if 'rsi_14' in df.columns:
            overbought = df['rsi_14'] > 70
            df.loc[overbought, 'pattern'] += 'RSI_OVERBOUGHT;'
            df.loc[overbought & (df['signal'] == 'HOLD'), 'signal'] = 'SELL'
        
        # Pattern 5: Bollinger Breakout
        if 'bb_position' in df.columns:
            bb_breakout_upper = df['bb_position'] > 100
            bb_breakout_lower = df['bb_position'] < 0
            df.loc[bb_breakout_upper, 'pattern'] += 'BB_BREAKOUT_UPPER;'
            df.loc[bb_breakout_lower, 'pattern'] += 'BB_BREAKOUT_LOWER;'
        
        # Pattern 6: Volume Breakout with Quality
        if 'volume_ratio' in df.columns and 'delivery_pct' in df.columns:
            quality_breakout = (df['volume_ratio'] > 2) & (df['delivery_pct'] > 70) & (df['day_return_pct'] > 5)
            df.loc[quality_breakout, 'pattern'] += 'QUALITY_BREAKOUT;'
            df.loc[quality_breakout, 'signal'] = 'STRONG_BUY'
        
        # Pattern 7: MACD Bullish Crossover
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd_bullish = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
            df.loc[macd_bullish, 'pattern'] += 'MACD_BULLISH;'
            df.loc[macd_bullish & (df['signal'] == 'HOLD'), 'signal'] = 'BUY'
        
        pattern_count = (df['pattern'] != '').sum()
        self.logger.info(f"✅ Detected patterns in {pattern_count} records")
        
        return df
    
    def generate_ranking(self, df: pd.DataFrame, date: Optional[str] = None) -> pd.DataFrame:
        """
        Generate stock rankings based on momentum and risk scores
        """
        self.logger.info("🏆 Generating rankings...")
        
        if date:
            df = df[df['date'] == date].copy()
        
        # Calculate composite score (momentum - risk)
        df['composite_score'] = df['momentum_score'] - (df['risk_score'] * 0.5)
        
        # Rank stocks
        df['momentum_rank'] = df['momentum_score'].rank(ascending=False, method='min')
        df['risk_rank'] = df['risk_score'].rank(ascending=True, method='min')
        df['composite_rank'] = df['composite_score'].rank(ascending=False, method='min')
        
        # Add percentile ranks
        df['momentum_percentile'] = df['momentum_score'].rank(pct=True) * 100
        df['risk_percentile'] = df['risk_score'].rank(pct=True) * 100
        
        self.logger.info(f"✅ Rankings generated for {len(df)} stocks")
        
        return df