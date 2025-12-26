"""
NSE Data Parser - Phase 1: Basic Reader
Parses NSE equity bhavcopy (pd*.csv files)
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Optional, Tuple
import argparse

from validator import ValidationEngine

class NSEParser:
    """
    NSE Data Parser - Starting with equity bhavcopy
    Will expand to other file types incrementally
    """
    
    def __init__(self, raw_data_dir='data/raw', output_dir='data/processed'):
        """
        Initialize parser
        
        Args:
            raw_data_dir: Where downloader saved files
            output_dir: Where to save parsed data
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger('NSEParser')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def parse_delivery_data(self, date: str) -> Optional[pd.DataFrame]:
        """
        Parse NSE delivery data (MTO files)
        Returns DataFrame with delivery qty and percentage
        """
        try:
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            formatted_date = date_obj.strftime('%d%m%Y')
            
            possible_paths = [
                self.raw_data_dir / date / 'delivery' / f'MTO_{formatted_date}.DAT',
                self.raw_data_dir / date / 'delivery' / f'mto_{formatted_date}.dat',
                self.raw_data_dir / date / 'delivery' / f'MTO_{formatted_date}.csv',
            ]
            
            file_path = None
            for test_path in possible_paths:
                if test_path.exists():
                    file_path = test_path
                    break
            
            if not file_path:
                logging.warning(f"⚠️ Delivery file not found for {date}")
                return None
            
            logging.info(f"📦 Reading delivery data from {file_path.name}")
            
            # MTO files have 3 header lines, then data starts
            # Format: Record Type,Sr No,Name of Security,Quantity Traded,Deliverable Quantity,%
            df = pd.read_csv(
                file_path,
                skiprows=3,  # Skip the 3 header lines
                names=['record_type', 'sr_no', 'symbol', 'series', 'qty_traded', 'delivery_qty', 'delivery_pct'],
                on_bad_lines='skip',
                engine='python'
            )
            
            logging.info(f"✅ Read {len(df)} rows from delivery file")
            
            # Filter for data rows (record_type == '20')
            df = df[df['record_type'] == '20'].copy()
            logging.info(f"✅ Filtered to {len(df)} data records")
            
            # Clean symbol names (remove spaces and convert to uppercase)
            df['symbol'] = df['symbol'].str.strip().str.upper()
            
            # Filter for equity series (EQ, N*, etc.)
            # In MTO files, series can be EQ, N1, N2, N3, NF, etc. for equity
            # Let's keep only those that match common equity patterns
            equity_series = ['EQ', 'BE', 'BZ', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'NF', 'NR']
            df = df[df['series'].isin(equity_series)]
            logging.info(f"✅ Filtered to {len(df)} equity securities")
            
            # Select and clean relevant columns
            df = df[['symbol', 'delivery_qty', 'delivery_pct']].copy()
            
            # Convert to numeric
            df['delivery_qty'] = pd.to_numeric(df['delivery_qty'], errors='coerce')
            df['delivery_pct'] = pd.to_numeric(df['delivery_pct'], errors='coerce')
            
            # Drop rows with missing data
            before_drop = len(df)
            df.dropna(inplace=True)
            if before_drop > len(df):
                logging.info(f"⚠️  Dropped {before_drop - len(df)} rows with missing data")
            
            logging.info(f"✅ Parsed {len(df)} delivery records")
            
            return df
            
        except Exception as e:
            logging.error(f"❌ Error parsing delivery data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


    def parse_and_join(self, date: str) -> Optional[pd.DataFrame]:
        """
        Parse bhavcopy and delivery data, then join them
        This creates the complete equity dataset
        """
        logging.info(f"🔗 Parsing and joining data for {date}")
        
        # Parse both datasets
        bhavcopy_df = self.parse_bhavcopy(date)
        delivery_df = self.parse_delivery_data(date)
        
        if bhavcopy_df is None:
            logging.error("❌ Bhavcopy parsing failed")
            return None
        
        if delivery_df is None:
            logging.warning("⚠️ Proceeding without delivery data")
            return bhavcopy_df
        
        # Join on symbol
        logging.info(f"🔗 Joining {len(bhavcopy_df)} bhavcopy + {len(delivery_df)} delivery records")
        
        merged_df = pd.merge(
            bhavcopy_df,
            delivery_df,
            on='symbol',
            how='left'  # Keep all bhavcopy records
        )
        
        logging.info(f"✅ Merged dataset: {len(merged_df)} records")
        
        # Calculate delivery insights
        if 'delivery_pct' in merged_df.columns:
            # Categorize delivery strength
            merged_df['delivery_strength'] = pd.cut(
                merged_df['delivery_pct'],
                bins=[0, 30, 50, 70, 100],
                labels=['Speculative', 'Mixed', 'Strong', 'Very Strong']
            )
            
            # Flag high delivery stocks
            merged_df['high_delivery'] = merged_df['delivery_pct'] >= 70
            
            logging.info(f"📊 Delivery distribution:")
            logging.info(f"   High delivery (>70%): {merged_df['high_delivery'].sum()} stocks")
            logging.info(f"   Low delivery (<30%): {(merged_df['delivery_pct'] < 30).sum()} stocks")
        
        return merged_df

    
    def parse_bhavcopy(self, date: str) -> Optional[pd.DataFrame]:
        """
        Parse equity bhavcopy for a given date
        
        Args:
            date: Date string (YYYY-MM-DD)
        
        Returns:
            DataFrame with cleaned equity data
        """
        try:
            self.logger.info(f"📊 Parsing bhavcopy for {date}")
            
            # Build path to price data file (pd*.csv)
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            date_str = date_obj.strftime('%d%m%Y')  # DDMMYYYY format
            
            bhavcopy_dir = self.raw_data_dir / date / 'bhavcopy'
            bhavcopy_file = bhavcopy_dir / f'pd{date_str}.csv'
            
            # Check if file exists
            if not bhavcopy_file.exists():
                self.logger.error(f"❌ File not found: {bhavcopy_file}")
                return None
            
            # Read CSV
            df = pd.read_csv(bhavcopy_file)
            
            self.logger.info(f"✅ Read {len(df)} rows from bhavcopy")
            
            # Filter for equity only (SERIES == 'EQ')
            # Also filter out indices (rows with empty SYMBOL or SERIES)
            df_equity = df[
                (df['SERIES'] == 'EQ') & 
                (df['SYMBOL'].notna()) & 
                (df['SYMBOL'] != '')
            ].copy()
            
            self.logger.info(f"✅ Filtered to {len(df_equity)} equity securities")
            
            if len(df_equity) == 0:
                self.logger.warning("⚠️  No equity securities found after filtering")
                return None
            
            # Select and rename columns
            df_clean = df_equity[[
                'SYMBOL',
                'SERIES',
                'SECURITY',
                'PREV_CL_PR',
                'OPEN_PRICE',
                'HIGH_PRICE',
                'LOW_PRICE',
                'CLOSE_PRICE',
                'NET_TRDVAL',
                'NET_TRDQTY',
                'TRADES',
                'HI_52_WK',
                'LO_52_WK'
            ]].copy()
            
            # Rename columns to standard format
            df_clean.columns = [
                'symbol',
                'series',
                'security_name',
                'prev_close',
                'open',
                'high',
                'low',
                'close',
                'turnover',
                'volume',
                'trades',
                'high_52w',
                'low_52w'
            ]
            
            # Add date column
            df_clean['date'] = date
            
            # Convert numeric columns
            numeric_cols = ['prev_close', 'open', 'high', 'low', 'close', 
                           'turnover', 'volume', 'trades', 'high_52w', 'low_52w']
            
            for col in numeric_cols:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Drop rows with missing critical data
            df_clean = df_clean.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
            
            # Sort by volume (descending)
            df_clean = df_clean.sort_values('volume', ascending=False).reset_index(drop=True)
            
            # Calculate derived metrics
            df_clean = self.calculate_derived_metrics(df_clean)
            
            self.logger.info(f"✅ Cleaned data ready: {len(df_clean)} rows, {len(df_clean.columns)} columns")
            
            return df_clean

            
        except Exception as e:
            self.logger.error(f"❌ Error parsing bhavcopy: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived trading metrics
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with additional calculated columns
        """
        self.logger.info("🧮 Calculating derived metrics...")
        
        # 1. VWAP (Volume Weighted Average Price)
        df['vwap'] = df['turnover'] / df['volume']
        
        # 2. Day Return % (Close vs Previous Close)
        df['day_return_pct'] = ((df['close'] - df['prev_close']) / df['prev_close']) * 100
        
        # 3. Intraday Range % (High-Low range)
        df['day_range_pct'] = ((df['high'] - df['low']) / df['low']) * 100
        
        # 4. Close Position in Range (where close is vs high/low)
        # 0% = closed at low, 100% = closed at high
        df['close_position_pct'] = ((df['close'] - df['low']) / (df['high'] - df['low'])) * 100
        
        # 5. Distance from 52-week High/Low
        df['dist_from_52w_high_pct'] = ((df['high_52w'] - df['close']) / df['high_52w']) * 100
        df['dist_from_52w_low_pct'] = ((df['close'] - df['low_52w']) / df['low_52w']) * 100
        
        # 6. Average Trade Size
        df['avg_trade_size'] = df['volume'] / df['trades']
        
        # 7. Average Trade Value
        df['avg_trade_value'] = df['turnover'] / df['trades']
        
        # 8. Price Change (absolute)
        df['price_change'] = df['close'] - df['prev_close']
        
        # 9. Upper Shadow % (High - Close, shows selling pressure)
        df['upper_shadow_pct'] = ((df['high'] - df['close']) / df['close']) * 100
        
        # 10. Lower Shadow % (Close - Low, shows buying support)
        df['lower_shadow_pct'] = ((df['close'] - df['low']) / df['close']) * 100
        
        # Handle infinite/NaN values (division by zero cases)
        df = df.replace([float('inf'), -float('inf')], pd.NA)
        
        # Round to 2 decimal places for readability
        metric_cols = [
            'vwap', 'day_return_pct', 'day_range_pct', 'close_position_pct',
            'dist_from_52w_high_pct', 'dist_from_52w_low_pct', 
            'avg_trade_size', 'avg_trade_value', 'price_change',
            'upper_shadow_pct', 'lower_shadow_pct'
        ]
        
        for col in metric_cols:
            if col in df.columns:
                df[col] = df[col].round(2)
        
        self.logger.info(f"✅ Added {len(metric_cols)} derived metrics")
        
        return df
    
    def save_csv(self, df: pd.DataFrame, date: str, filename: str):
        """Save DataFrame to CSV"""
        output_path = self.output_dir / f'{filename}_{date}.csv'
        df.to_csv(output_path, index=False)
        self.logger.info(f"💾 Saved to {output_path}")
        return output_path
    
    def validate_data(self, df: pd.DataFrame, validation_level: str = 'standard') -> Tuple[pd.DataFrame, Dict]:
        """
        Validate parsed data
        
        Args:
            df: DataFrame to validate
            validation_level: 'basic', 'standard', or 'comprehensive'
        
        Returns:
            Tuple of (validated_df, validation_report)
        """
        validator = ValidationEngine()
        validated_df, report = validator.validate_dataframe(df, validation_level)
        
        # Generate and print report
        report_text = validator.generate_report(report)
        print(report_text)
        
        return validated_df, report

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Parse NSE data')
    
    # Date arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--date', help='Single date in YYYY-MM-DD format')
    group.add_argument('--start', help='Start date for range (YYYY-MM-DD)')
    
    parser.add_argument('--end', help='End date for range (YYYY-MM-DD), use with --start')
    
    # Validation arguments
    parser.add_argument('--validate', choices=['basic', 'standard', 'comprehensive'],
                       default='standard', help='Validation level (default: standard)')
    parser.add_argument('--skip-invalid', action='store_true',
                       help='Skip invalid records in output')
    
    args = parser.parse_args()
    
    # Initialize parser
    nse_parser = NSEParser()
    
    # Handle single date vs date range
    if args.date:
        process_single_date(nse_parser, args.date, args.validate, args.skip_invalid)
    else:
        if not args.end:
            print("❌ Error: --end is required when using --start")
            return
        process_date_range(nse_parser, args.start, args.end, args.validate, args.skip_invalid)

def process_single_date(nse_parser, date: str, validation_level: str = 'standard', 
                       skip_invalid: bool = False):
    """Process a single date with validation"""
    df = nse_parser.parse_and_join(date)
    
    if df is not None:
        # Validate data
        df, validation_report = nse_parser.validate_data(df, validation_level)
        
        # Optionally filter out invalid records
        if skip_invalid:
            valid_count = len(df)
            df = df[df['is_valid']].copy()
            removed = valid_count - len(df)
            if removed > 0:
                print(f"\n🗑️  Removed {removed} invalid records")
        
        # Save to CSV
        nse_parser.save_csv(df, date, 'equity_full')
        
        # Print summary statistics
        print_summary(df, date)
        
        print(f"\n✅ Success! Saved to data/processed/equity_full_{date}.csv")
        print(f"📊 Total columns: {len(df.columns)}")
        
        # Save validation report
        report_path = nse_parser.output_dir / f'validation_report_{date}.txt'
        from validator import ValidationEngine
        validator = ValidationEngine()
        validator.generate_report(validation_report, report_path)
    else:
        print(f"\n❌ Parsing failed for {date}")

def process_date_range(nse_parser, start_date: str, end_date: str, 
                      validation_level: str = 'standard', skip_invalid: bool = False):
    """Process multiple dates in a range with validation"""
    from datetime import datetime, timedelta
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    if start > end:
        print("❌ Error: Start date must be before end date")
        return
    
    # Generate date list (skip weekends)
    dates = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    print(f"{'='*80}")
    print(f"📅 BATCH PROCESSING: {start_date} to {end_date}")
    print(f"{'='*80}")
    print(f"Total dates to process: {len(dates)} (excluding weekends)")
    print(f"Validation level: {validation_level}")
    print(f"{'='*80}\n")
    
    # Track statistics
    successful = 0
    failed = 0
    all_data = []
    all_validation_reports = []
    
    # Process each date
    for i, date in enumerate(dates, 1):
        print(f"\n[{i}/{len(dates)}] Processing {date}...")
        
        df = nse_parser.parse_and_join(date)
        
        if df is not None:
            # Validate
            df, validation_report = nse_parser.validate_data(df, validation_level)
            all_validation_reports.append(validation_report)
            
            # Filter invalid if requested
            if skip_invalid:
                df = df[df['is_valid']].copy()
            
            # Save individual file
            nse_parser.save_csv(df, date, 'equity_full')
            all_data.append(df)
            successful += 1
            print(f"✅ Success - {len(df)} securities")
        else:
            failed += 1
            print(f"❌ Failed (file may not exist)")
    
    # Create combined dataset
    if all_data:
        print(f"\n{'='*80}")
        print(f"📊 BATCH PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"✅ Successful: {successful}/{len(dates)} dates")
        print(f"❌ Failed: {failed}/{len(dates)} dates")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Save combined dataset
        combined_file = nse_parser.output_dir / f'equity_full_{start_date}_to_{end_date}.csv'
        combined_df.to_csv(combined_file, index=False)
        
        print(f"\n📦 COMBINED DATASET:")
        print(f"   Total records: {len(combined_df):,}")
        print(f"   Valid records: {(combined_df['is_valid']).sum():,}")
        print(f"   Invalid records: {(~combined_df['is_valid']).sum():,}")
        print(f"   Unique symbols: {combined_df['symbol'].nunique()}")
        print(f"   Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        print(f"   Saved to: {combined_file}")
        
        # Aggregate validation statistics
        total_records = sum(r['total_records'] for r in all_validation_reports)
        total_valid = sum(r['valid_records'] for r in all_validation_reports)
        print(f"\n📋 AGGREGATE VALIDATION:")
        print(f"   Overall validation rate: {(total_valid/total_records*100):.2f}%")
        
        # Summary statistics
        if 'delivery_pct' in combined_df.columns:
            print(f"\n📈 AGGREGATE STATISTICS:")
            print(f"   Avg delivery %: {combined_df['delivery_pct'].mean():.2f}%")
            print(f"   Avg volume: {combined_df['volume'].mean():,.0f}")
            print(f"   Avg turnover: ₹{combined_df['turnover'].mean():,.0f}")
        
        print(f"\n{'='*80}")
    else:
        print(f"\n❌ No data processed successfully")

def print_summary(df, date):
    """Print summary statistics for a single date"""
    print(f"\n{'='*80}")
    print(f"📊 EQUITY DATA SUMMARY - {date}")
    print(f"{'='*80}")
    print(f"Total Securities: {len(df)}")
    
    # Check if delivery data is available
    has_delivery = 'delivery_pct' in df.columns
    
    if has_delivery:
        print(f"\n🎯 DELIVERY ANALYSIS:")
        print(f"   Very Strong (>70%): {(df['delivery_pct'] >= 70).sum()} stocks")
        print(f"   Strong (50-70%):    {((df['delivery_pct'] >= 50) & (df['delivery_pct'] < 70)).sum()} stocks")
        print(f"   Mixed (30-50%):     {((df['delivery_pct'] >= 30) & (df['delivery_pct'] < 50)).sum()} stocks")
        print(f"   Speculative (<30%): {(df['delivery_pct'] < 30).sum()} stocks")
        
        print(f"\n📈 TOP 10 HIGH DELIVERY STOCKS:")
        high_del = df.nlargest(10, 'delivery_pct')[['symbol', 'close', 'delivery_pct', 'day_return_pct']]
        print(high_del.to_string(index=False))
        
        print(f"\n📉 TOP 10 SPECULATIVE STOCKS (Low Delivery):")
        low_del = df[df['delivery_pct'] < 30].nlargest(10, 'volume')[['symbol', 'close', 'delivery_pct', 'volume', 'day_return_pct']]
        print(low_del.to_string(index=False))
    
    # Print top gainers/losers (conditional on delivery data)
    print(f"\n🚀 TOP 10 GAINERS:")
    gainer_cols = ['symbol', 'close', 'day_return_pct', 'volume']
    if has_delivery:
        gainer_cols.insert(3, 'delivery_pct')
    gainers = df.nlargest(10, 'day_return_pct')[gainer_cols]
    print(gainers.to_string(index=False))
    
    print(f"\n📉 TOP 10 LOSERS:")
    loser_cols = ['symbol', 'close', 'day_return_pct', 'volume']
    if has_delivery:
        loser_cols.insert(3, 'delivery_pct')
    losers = df.nsmallest(10, 'day_return_pct')[loser_cols]
    print(losers.to_string(index=False))


if __name__ == '__main__':
    main()