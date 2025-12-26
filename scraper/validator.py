"""
NSE Data Validation Engine
Production-grade validation for quant trading systems
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class ValidationEngine:
    """
    Validates NSE equity data for production use
    Detects anomalies, corporate actions, and data quality issues
    """
    
    def __init__(self):
        self.logger = logging.getLogger('ValidationEngine')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # NSE circuit breaker limits
        self.stock_circuit_limits = [2, 5, 10, 20]  # Percentage limits
        self.index_circuit_limits = [10, 15, 20]
        
        # Thresholds for anomaly detection
        self.min_volume = 100  # Minimum volume for valid trade
        self.min_price = 0.01  # Minimum price (Re 0.01)
        self.max_day_return = 20  # Max normal daily return (%)
        self.min_delivery_pct = 0  # Min delivery percentage
        self.max_delivery_pct = 100  # Max delivery percentage
    
    def validate_dataframe(self, df: pd.DataFrame, 
                          validation_level: str = 'comprehensive') -> Tuple[pd.DataFrame, Dict]:
        """
        Main validation method
        
        Args:
            df: DataFrame to validate
            validation_level: 'basic', 'standard', or 'comprehensive'
        
        Returns:
            Tuple of (validated_df, validation_report)
        """
        self.logger.info(f"🔍 Starting {validation_level} validation on {len(df)} records")
        
        report = {
            'total_records': len(df),
            'validation_level': validation_level,
            'issues_found': [],
            'warnings': [],
            'flags_added': []
        }
        
        # Add validation flag columns
        df['is_valid'] = True
        df['validation_flags'] = ''
        
        # Basic validation (always run)
        df = self._validate_schema(df, report)
        df = self._validate_business_rules(df, report)
        df = self._validate_price_ranges(df, report)
        
        if validation_level in ['standard', 'comprehensive']:
            # Standard validation
            df = self._detect_circuit_breakers(df, report)
            df = self._detect_suspicious_patterns(df, report)
            df = self._validate_delivery_data(df, report)
            df = self._detect_liquidity_issues(df, report)
        
        if validation_level == 'comprehensive':
            # Advanced validation
            df = self._detect_corporate_actions(df, report)
            df = self._detect_price_anomalies(df, report)
            df = self._validate_cross_sectional(df, report)
        
        # Calculate validation statistics
        invalid_count = (~df['is_valid']).sum()
        report['invalid_records'] = int(invalid_count)
        report['valid_records'] = int(len(df) - invalid_count)
        report['validation_rate'] = round((report['valid_records'] / report['total_records']) * 100, 2)
        
        self.logger.info(f"✅ Validation complete: {report['validation_rate']}% valid")
        self.logger.info(f"   Valid: {report['valid_records']}, Invalid: {report['invalid_records']}")
        
        return df, report
    
    def _validate_schema(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Validate required columns exist and have correct types"""
        required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            report['issues_found'].append(f"Missing columns: {missing_cols}")
            self.logger.error(f"❌ Missing required columns: {missing_cols}")
        
        return df
    
    def _validate_business_rules(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Validate fundamental business rules"""
        issues = 0
        
        # Rule 1: High >= Low
        invalid_range = df['high'] < df['low']
        if invalid_range.any():
            count = invalid_range.sum()
            df.loc[invalid_range, 'is_valid'] = False
            df.loc[invalid_range, 'validation_flags'] += 'HIGH<LOW;'
            issues += count
            report['issues_found'].append(f"High < Low: {count} records")
        
        # Rule 2: Close within High-Low range
        invalid_close = (df['close'] > df['high']) | (df['close'] < df['low'])
        if invalid_close.any():
            count = invalid_close.sum()
            df.loc[invalid_close, 'is_valid'] = False
            df.loc[invalid_close, 'validation_flags'] += 'CLOSE_OUT_OF_RANGE;'
            issues += count
            report['issues_found'].append(f"Close out of range: {count} records")
        
        # Rule 3: Open within High-Low range (allow slight deviation for gaps)
        invalid_open = (df['open'] > df['high'] * 1.01) | (df['open'] < df['low'] * 0.99)
        if invalid_open.any():
            count = invalid_open.sum()
            df.loc[invalid_open, 'validation_flags'] += 'OPEN_OUT_OF_RANGE;'
            report['warnings'].append(f"Open out of range: {count} records (may be valid gaps)")
        
        # Rule 4: Volume >= 0
        invalid_volume = df['volume'] < 0
        if invalid_volume.any():
            count = invalid_volume.sum()
            df.loc[invalid_volume, 'is_valid'] = False
            df.loc[invalid_volume, 'validation_flags'] += 'NEGATIVE_VOLUME;'
            issues += count
            report['issues_found'].append(f"Negative volume: {count} records")
        
        # Rule 5: Prices > 0
        invalid_price = (df['open'] <= 0) | (df['high'] <= 0) | (df['low'] <= 0) | (df['close'] <= 0)
        if invalid_price.any():
            count = invalid_price.sum()
            df.loc[invalid_price, 'is_valid'] = False
            df.loc[invalid_price, 'validation_flags'] += 'INVALID_PRICE;'
            issues += count
            report['issues_found'].append(f"Invalid price (<=0): {count} records")
        
        if issues > 0:
            self.logger.warning(f"⚠️  Business rule violations: {issues} records")
        
        return df
    
    def _validate_price_ranges(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Validate prices are within reasonable ranges"""
        
        # Check for penny stocks (< ₹1) - flag but don't invalidate
        penny_stocks = df['close'] < 1.0
        if penny_stocks.any():
            count = penny_stocks.sum()
            df.loc[penny_stocks, 'validation_flags'] += 'PENNY_STOCK;'
            report['warnings'].append(f"Penny stocks (<₹1): {count} records")
        
        # Check for extremely low volume
        low_volume = df['volume'] < self.min_volume
        if low_volume.any():
            count = low_volume.sum()
            df.loc[low_volume, 'validation_flags'] += 'LOW_VOLUME;'
            report['warnings'].append(f"Low volume (<{self.min_volume}): {count} records")
        
        return df
    
    def _detect_circuit_breakers(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Detect potential circuit breaker hits"""
        
        if 'day_return_pct' not in df.columns:
            return df
        
        # Check for circuit breaker patterns
        for limit in self.stock_circuit_limits:
            # Upper circuit
            upper_circuit = df['day_return_pct'] >= limit
            if upper_circuit.any():
                count = upper_circuit.sum()
                df.loc[upper_circuit, 'validation_flags'] += f'UPPER_CIRCUIT_{limit}%;'
                report['flags_added'].append(f"Upper circuit {limit}%: {count} stocks")
            
            # Lower circuit
            lower_circuit = df['day_return_pct'] <= -limit
            if lower_circuit.any():
                count = lower_circuit.sum()
                df.loc[lower_circuit, 'validation_flags'] += f'LOWER_CIRCUIT_{limit}%;'
                report['flags_added'].append(f"Lower circuit {limit}%: {count} stocks")
        
        return df
    
    def _detect_corporate_actions(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Detect potential corporate actions (splits, bonuses)"""
        
        if 'day_return_pct' not in df.columns or 'volume' not in df.columns:
            return df
        
        # Pattern 1: Large negative return with very high volume (potential split)
        # Example: -50% price drop with 10x average volume
        split_pattern = (df['day_return_pct'] < -40) & (df['volume'] > df['volume'].median() * 5)
        if split_pattern.any():
            count = split_pattern.sum()
            df.loc[split_pattern, 'validation_flags'] += 'POSSIBLE_SPLIT;'
            report['flags_added'].append(f"Possible stock split: {count} stocks")
            self.logger.warning(f"⚠️  Detected {count} possible stock splits")
        
        # Pattern 2: Large negative return with high delivery (bonus issue)
        if 'delivery_pct' in df.columns:
            bonus_pattern = (df['day_return_pct'] < -30) & (df['delivery_pct'] > 80)
            if bonus_pattern.any():
                count = bonus_pattern.sum()
                df.loc[bonus_pattern, 'validation_flags'] += 'POSSIBLE_BONUS;'
                report['flags_added'].append(f"Possible bonus issue: {count} stocks")
                self.logger.warning(f"⚠️  Detected {count} possible bonus issues")
        
        # Pattern 3: Unusual volume spike (10x+ median) - potential corporate action
        median_volume = df['volume'].median()
        volume_spike = df['volume'] > median_volume * 10
        if volume_spike.any():
            count = volume_spike.sum()
            df.loc[volume_spike, 'validation_flags'] += 'VOLUME_SPIKE;'
            report['warnings'].append(f"Unusual volume spike: {count} stocks")
        
        return df
    
    def _detect_suspicious_patterns(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Detect suspicious trading patterns"""
        
        # Pattern 1: Zero volume but price change
        if 'volume' in df.columns and 'day_return_pct' in df.columns:
            suspicious = (df['volume'] == 0) & (df['day_return_pct'] != 0)
            if suspicious.any():
                count = suspicious.sum()
                df.loc[suspicious, 'is_valid'] = False
                df.loc[suspicious, 'validation_flags'] += 'ZERO_VOLUME_PRICE_CHANGE;'
                report['issues_found'].append(f"Price change with zero volume: {count} records")
        
        # Pattern 2: High = Low = Close (no trading, only theoretical price)
        no_trading = (df['high'] == df['low']) & (df['volume'] == 0)
        if no_trading.any():
            count = no_trading.sum()
            df.loc[no_trading, 'validation_flags'] += 'NO_TRADING;'
            report['warnings'].append(f"No trading activity: {count} stocks")
        
        return df
    
    def _validate_delivery_data(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Validate delivery percentage data"""
        
        if 'delivery_pct' not in df.columns:
            return df
        
        # Check for invalid delivery percentages
        invalid_delivery = (df['delivery_pct'] < 0) | (df['delivery_pct'] > 100)
        if invalid_delivery.any():
            count = invalid_delivery.sum()
            df.loc[invalid_delivery, 'is_valid'] = False
            df.loc[invalid_delivery, 'validation_flags'] += 'INVALID_DELIVERY_PCT;'
            report['issues_found'].append(f"Invalid delivery %: {count} records")
        
        # Check for suspicious: volume > 0 but delivery_pct = 0
        suspicious_delivery = (df['volume'] > 1000) & (df['delivery_pct'] == 0)
        if suspicious_delivery.any():
            count = suspicious_delivery.sum()
            df.loc[suspicious_delivery, 'validation_flags'] += 'ZERO_DELIVERY;'
            report['warnings'].append(f"High volume but zero delivery: {count} stocks")
        
        return df
    
    def _detect_liquidity_issues(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Flag illiquid stocks"""
        
        # Very low volume stocks (potential manipulation risk)
        if 'volume' in df.columns:
            illiquid = df['volume'] < 1000
            if illiquid.any():
                count = illiquid.sum()
                df.loc[illiquid, 'validation_flags'] += 'ILLIQUID;'
                report['flags_added'].append(f"Illiquid stocks: {count} records")
        
        # Very low trade count
        if 'trades' in df.columns:
            low_trades = df['trades'] < 10
            if low_trades.any():
                count = low_trades.sum()
                df.loc[low_trades, 'validation_flags'] += 'LOW_TRADES;'
                report['warnings'].append(f"Low trade count (<10): {count} stocks")
        
        return df
    
    def _detect_price_anomalies(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Detect price anomalies using statistical methods"""
        
        if 'day_return_pct' not in df.columns:
            return df
        
        # Z-score based anomaly detection
        returns = df['day_return_pct'].replace([np.inf, -np.inf], np.nan).dropna()
        if len(returns) > 0:
            mean_return = returns.mean()
            std_return = returns.std()
            
            if std_return > 0:
                # Flag returns > 3 standard deviations
                df['return_zscore'] = (df['day_return_pct'] - mean_return) / std_return
                extreme_returns = df['return_zscore'].abs() > 3
                
                if extreme_returns.any():
                    count = extreme_returns.sum()
                    df.loc[extreme_returns, 'validation_flags'] += 'EXTREME_RETURN;'
                    report['flags_added'].append(f"Extreme returns (>3σ): {count} stocks")
        
        return df
    
    def _validate_cross_sectional(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Cross-sectional validation across all stocks"""
        
        # Check for duplicate symbols on same date
        if 'symbol' in df.columns and 'date' in df.columns:
            duplicates = df.duplicated(subset=['symbol', 'date'], keep=False)
            if duplicates.any():
                count = duplicates.sum()
                df.loc[duplicates, 'is_valid'] = False
                df.loc[duplicates, 'validation_flags'] += 'DUPLICATE;'
                report['issues_found'].append(f"Duplicate records: {count} entries")
        
        return df
    
    def generate_report(self, report: Dict, save_path: Optional[Path] = None) -> str:
        """Generate human-readable validation report"""
        
        report_text = f"""
{'='*80}
VALIDATION REPORT
{'='*80}
Validation Level: {report['validation_level']}
Total Records: {report['total_records']:,}
Valid Records: {report['valid_records']:,}
Invalid Records: {report['invalid_records']:,}
Validation Rate: {report['validation_rate']}%

"""
        
        if report['issues_found']:
            report_text += f"\n🚨 CRITICAL ISSUES FOUND:\n"
            for issue in report['issues_found']:
                report_text += f"   • {issue}\n"
        
        if report['warnings']:
            report_text += f"\n⚠️  WARNINGS:\n"
            for warning in report['warnings']:
                report_text += f"   • {warning}\n"
        
        if report['flags_added']:
            report_text += f"\n🏁 FLAGS ADDED:\n"
            for flag in report['flags_added']:
                report_text += f"   • {flag}\n"
        
        report_text += f"\n{'='*80}\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"📄 Report saved to {save_path}")
        
        return report_text